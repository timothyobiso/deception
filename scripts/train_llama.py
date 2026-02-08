#!/usr/bin/env python3
"""Training script for Llama 3.1 8B with LoRA/QLoRA for deception detection."""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer
from datasets import Dataset
import wandb
from tqdm import tqdm

from deception_interpretability.models.llama_deception_model import LlamaDeceptionModel, LlamaDeceptionTrainer
from deception_interpretability.data.hf_dataset_loaders import UnifiedHFDataset
from deception_interpretability.utils.training_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Llama 3.1 8B for deception detection')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, 
                       default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Llama model to use')
    parser.add_argument('--use_qlora', action='store_true', default=True,
                       help='Use QLoRA (4-bit quantization + LoRA)')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./checkpoints/llama-deception',
                       help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                       help='Warmup ratio')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='unified-hf',
                       choices=['unified-hf', 'socialmaze', 'werewolf-amongus'],
                       help='Which dataset to use')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='llama-deception',
                       help='W&B project name')
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Push model to HuggingFace Hub')
    parser.add_argument('--hub_model_id', type=str, default='llama-3.1-8b-deception',
                       help='HuggingFace Hub model ID')
    
    return parser.parse_args()


class DeceptionDataCollator:
    """Data collator for deception detection with Llama."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        """Collate batch of examples."""
        # Convert lists to tensors if needed, otherwise stack existing tensors
        def to_tensor(values):
            if isinstance(values[0], torch.Tensor):
                return torch.stack(values)
            else:
                return torch.tensor(values)
        
        input_ids = to_tensor([ex['input_ids'] for ex in examples])
        attention_mask = to_tensor([ex['attention_mask'] for ex in examples])
        labels = to_tensor([ex['labels'] for ex in examples])
        
        # Add deception labels if available
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        if 'deception_labels' in examples[0]:
            batch['deception_labels'] = torch.tensor([ex['deception_labels'] for ex in examples])
        
        return batch


class DeceptionLlamaTrainer(Trainer):
    """HF Trainer subclass that adds deception classification loss."""

    def __init__(self, deception_head, deception_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.deception_head = deception_head
        self.deception_weight = deception_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        deception_labels = inputs.pop("deception_labels", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )
        lm_loss = outputs.loss

        # Compute deception loss when labels are present
        deception_loss = torch.tensor(0.0, device=lm_loss.device)
        if deception_labels is not None:
            hidden = outputs.hidden_states[-1]  # (B, seq, D)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            # Move pooled to deception head device in case of multi-GPU
            head_device = next(self.deception_head.parameters()).device
            deception_out = self.deception_head(pooled.to(head_device))
            deception_logits = deception_out["deception_logits"].squeeze(-1)
            deception_loss = nn.functional.binary_cross_entropy_with_logits(
                deception_logits, deception_labels.float().to(head_device)
            )

        loss = lm_loss + self.deception_weight * deception_loss.to(lm_loss.device)
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.deception_head.eval()
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.deception_head.train()
        return result

    def create_optimizer(self):
        """Include deception head parameters in the optimizer."""
        super().create_optimizer()
        # Add deception head params to the existing optimizer param groups
        self.optimizer.add_param_group({
            "params": list(self.deception_head.parameters()),
            "lr": self.args.learning_rate,
        })
        return self.optimizer


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize wandb
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=f"llama-deception-{args.dataset}",
            config=vars(args)
        )
    
    print("=" * 50)
    print("Llama 3.1 8B Deception Training")
    print("=" * 50)
    
    # Load tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with LoRA/QLoRA
    print(f"Loading model with {'QLoRA' if args.use_qlora else 'LoRA'}...")
    model_wrapper = LlamaDeceptionModel(
        model_name=args.model_name,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.use_qlora,
        add_deception_head=True,
    )
    
    # Create trainer
    trainer = LlamaDeceptionTrainer(model_wrapper, tokenizer)
    
    # Load datasets
    print(f"Loading {args.dataset} dataset...")
    
    if args.dataset == 'socialmaze':
        from deception_interpretability.data.hf_dataset_loaders import SocialMazeDataset
        dataset = SocialMazeDataset(tokenizer, max_length=args.max_length)
        dataset_dict = dataset.process_for_training()
    elif args.dataset == 'werewolf-amongus':
        from deception_interpretability.data.hf_dataset_loaders import WerewolfAmongUsDataset
        dataset = WerewolfAmongUsDataset(tokenizer, max_length=args.max_length)
        dataset_dict = dataset.process_for_training()
    else:  # unified-hf
        dataset = UnifiedHFDataset(tokenizer, max_length=args.max_length)
        dataset_dict = dataset.load_all_datasets()
    
    # Process datasets for Llama format
    print("Processing datasets for Llama format...")
    
    def process_examples(examples):
        """Process batch of examples."""
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        all_deception_labels = []
        
        for i in range(len(examples['text'])):
            ex = {
                'text': examples['text'][i],
                'game_type': examples.get('game_type', ['unknown'] * len(examples['text']))[i],
                'speaker': examples.get('speaker', ['unknown'] * len(examples['text']))[i],
                'is_deceptive': bool(examples.get('deception_labels', [0] * len(examples['text']))[i]),
                'deception_type': examples.get('deception_type', ['unknown'] * len(examples['text']))[i],
            }
            processed_ex = trainer.prepare_training_example(ex, max_length=args.max_length)
            
            # Extract tensors and convert to lists for batched processing
            all_input_ids.append(processed_ex['input_ids'].tolist())
            all_attention_masks.append(processed_ex['attention_mask'].tolist())
            all_labels.append(processed_ex['labels'].tolist())
            if 'deception_label' in processed_ex:
                all_deception_labels.append(processed_ex['deception_label'])
        
        # Return as lists for HuggingFace datasets
        batch = {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks,
            'labels': all_labels,
        }
        
        if all_deception_labels:
            batch['deception_labels'] = all_deception_labels
        
        return batch
    
    # Process train/val/test splits
    if 'train' not in dataset_dict:
        print(f"Available splits: {list(dataset_dict.keys())}")
        raise ValueError("No 'train' split found in dataset")
        
    train_dataset = dataset_dict['train'].map(
        process_examples,
        batched=True,
        batch_size=100,
        remove_columns=dataset_dict['train'].column_names,
        desc="Processing training data"
    )
    
    eval_dataset = None
    if 'validation' in dataset_dict:
        eval_dataset = dataset_dict['validation'].map(
            process_examples,
            batched=True,
            batch_size=100,
            remove_columns=dataset_dict['validation'].column_names,
            desc="Processing validation data"
        )
    elif 'test' in dataset_dict:
        eval_dataset = dataset_dict['test'].map(
            process_examples,
            batched=True,
            batch_size=100,
            remove_columns=dataset_dict['test'].column_names,
            desc="Processing test data as validation"
        )
    
    # Print dataset statistics
    print(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation examples: {len(eval_dataset)}")
    
    # Get training arguments
    training_args = trainer.get_training_args(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
    )
    
    # Create data collator
    data_collator = DeceptionDataCollator(tokenizer, max_length=args.max_length)
    
    # Create Trainer with multi-task deception loss
    hf_trainer = DeceptionLlamaTrainer(
        deception_head=model_wrapper.deception_head,
        deception_weight=1.0,
        model=model_wrapper.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    print(f"Total optimization steps: {len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.num_epochs}")
    
    train_result = hf_trainer.train()
    
    # Save model
    print(f"Saving model to {args.output_dir}...")
    model_wrapper.save_model(args.output_dir)
    hf_trainer.save_model(args.output_dir)
    
    # Save training results
    with open(os.path.join(args.output_dir, "train_results.txt"), "w") as f:
        f.write(str(train_result))
    
    # Push to hub if requested
    if args.push_to_hub:
        print(f"Pushing model to HuggingFace Hub as {args.hub_model_id}...")
        hf_trainer.push_to_hub(args.hub_model_id)
    
    # Evaluate
    if eval_dataset:
        print("Running evaluation...")
        eval_results = hf_trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
            f.write(str(eval_results))
    
    print("Training complete!")
    
    # Log final results to wandb
    if wandb.run:
        wandb.log({
            "final_train_loss": train_result.training_loss,
            "final_eval_loss": eval_results.get('eval_loss', None) if eval_dataset else None,
        })
        wandb.finish()


if __name__ == '__main__':
    # Check if user has access to Llama model
    print("Note: You need access to meta-llama/Llama-3.1-8B-Instruct on HuggingFace.")
    print("Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print()
    
    main()