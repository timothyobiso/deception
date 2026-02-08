#!/usr/bin/env python3
"""Training script for deception models."""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from deception_interpretability.models.deception_model import DeceptionModel, SmallDeceptionModel, DeceptionConfig
from deception_interpretability.data.dataset_loader import MafiaDataset, AmongUsDataset, SecretHitlerDataset
from deception_interpretability.data.hf_dataset_loaders import SocialMazeDataset, WerewolfAmongUsDataset, UnifiedHFDataset
from deception_interpretability.utils.training_utils import set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train deception model')
    parser.add_argument('--model_type', type=str, default='small', choices=['small', 'full'])
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--dataset', type=str, default='unified-hf', 
                       choices=['mafia', 'amongus', 'secrethitler', 'socialmaze', 'werewolf-amongus', 'unified-hf'])
    parser.add_argument('--data_path', type=str, required=False)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_project', type=str, default='deception-interpretability')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=500)
    return parser.parse_args()


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    gradient_accumulation_steps=1
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch.get('labels', input_ids).to(device)
        deception_labels = batch.get('deception_labels').to(device) if 'deception_labels' in batch else None
        role_labels = batch.get('role_labels').to(device) if 'role_labels' in batch else None
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            deception_labels=deception_labels,
            role_labels=role_labels
        )
        
        loss = outputs['loss'] / gradient_accumulation_steps
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        progress_bar.set_postfix({'loss': total_loss / num_batches})
        
        if wandb.run is not None:
            wandb.log({
                'train_loss': loss.item() * gradient_accumulation_steps,
                'learning_rate': scheduler.get_last_lr()[0]
            })
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate model performance."""
    model.eval()
    total_loss = 0
    total_deception_acc = 0
    total_role_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.get('labels', input_ids).to(device)
            deception_labels = batch.get('deception_labels').to(device) if 'deception_labels' in batch else None
            role_labels = batch.get('role_labels').to(device) if 'role_labels' in batch else None
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                deception_labels=deception_labels,
                role_labels=role_labels
            )
            
            total_loss += outputs['loss'].item()
            
            if 'deception_score' in outputs and deception_labels is not None:
                preds = (outputs['deception_score'].squeeze(-1) > 0.5).float()
                total_deception_acc += (preds == deception_labels).float().mean().item()
            
            if 'role_logits' in outputs and role_labels is not None:
                preds = outputs['role_logits'].argmax(dim=-1)
                total_role_acc += (preds == role_labels).float().mean().item()
            
            num_batches += 1
    
    metrics = {
        'eval_loss': total_loss / num_batches,
        'eval_deception_acc': total_deception_acc / num_batches if total_deception_acc > 0 else 0,
        'eval_role_acc': total_role_acc / num_batches if total_role_acc > 0 else 0
    }
    
    return metrics


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    if args.wandb_run_name:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset: {args.dataset}")
    
    # Load HuggingFace datasets
    if args.dataset in ['socialmaze', 'werewolf-amongus', 'unified-hf']:
        if args.dataset == 'socialmaze':
            dataset = SocialMazeDataset(tokenizer, args.max_length)
            dataset_dict = dataset.process_for_training()
        elif args.dataset == 'werewolf-amongus':
            dataset = WerewolfAmongUsDataset(tokenizer, args.max_length)
            dataset_dict = dataset.process_for_training()
        else:  # unified-hf
            dataset = UnifiedHFDataset(tokenizer, args.max_length)
            dataset_dict = dataset.load_all_datasets()
        
        # Create dataloaders
        train_loader = DataLoader(
            dataset_dict.get('train', dataset_dict.get('training')),
            batch_size=args.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset_dict.get('validation', dataset_dict.get('valid', dataset_dict.get('test'))),
            batch_size=args.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            dataset_dict.get('test', dataset_dict.get('validation')),
            batch_size=args.batch_size,
            shuffle=False
        )
        
        # Print statistics
        if args.dataset == 'unified-hf':
            stats = dataset.get_statistics(dataset_dict)
            print(f"Dataset statistics: {stats}")
            
    # Load local datasets (legacy)
    else:
        if args.data_path is None:
            raise ValueError(f"--data_path required for {args.dataset} dataset")
            
        if args.dataset == 'mafia':
            dataset = MafiaDataset(args.dataset, tokenizer, args.max_length)
        elif args.dataset == 'amongus':
            dataset = AmongUsDataset(args.dataset, tokenizer, args.max_length)
        else:
            dataset = SecretHitlerDataset(args.dataset, tokenizer, args.max_length)
        
        dataset.load_raw_data(args.data_path)
        processed_data = dataset.preprocess()
        
        train_loader, val_loader, test_loader = dataset.create_dataloaders(
            batch_size=args.batch_size,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1
        )
    
    print(f"Creating model: {args.model_type}")
    if args.model_type == 'small':
        model = SmallDeceptionModel(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=256,
            num_layers=4
        )
    else:
        config = DeceptionConfig(
            base_model=args.base_model,
            hidden_size=768,
            num_deception_heads=4
        )
        model = DeceptionModel(config)
    
    model = model.to(args.device)
    
    num_training_steps = len(train_loader) * args.num_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print("Starting training...")
    global_step = 0
    best_eval_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            args.device, args.gradient_accumulation_steps
        )
        
        print(f"Average training loss: {train_loss:.4f}")
        
        eval_metrics = evaluate(model, val_loader, args.device)
        print(f"Evaluation metrics: {eval_metrics}")
        
        if wandb.run is not None:
            wandb.log(eval_metrics)
        
        if eval_metrics['eval_loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['eval_loss']
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                os.path.join(args.output_dir, 'best_model.pt')
            )
            print(f"Saved best model with eval loss: {best_eval_loss:.4f}")
        
        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
    
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(model, test_loader, args.device)
    print(f"Test metrics: {test_metrics}")
    
    if wandb.run is not None:
        wandb.log({'test_metrics': test_metrics})
        wandb.finish()
    
    save_checkpoint(
        model, optimizer, scheduler, args.num_epochs, global_step,
        os.path.join(args.output_dir, 'final_model.pt')
    )
    print("Training complete!")


if __name__ == '__main__':
    main()