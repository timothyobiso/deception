"""Llama-based models for deception detection with LoRA/QLoRA for efficient training."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    LlamaForCausalLM,
    LlamaConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from typing import Dict, Optional, List, Tuple
import einops


class LlamaDeceptionModel:
    """Llama 3.1 8B with LoRA for deception detection.
    
    Uses QLoRA (4-bit quantization + LoRA) for efficient fine-tuning.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        use_qlora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        load_in_4bit: bool = True,
        device_map: str = "auto",
        add_deception_head: bool = True,
    ):
        self.model_name = model_name
        self.use_qlora = use_qlora
        self.device_map = device_map
        self.add_deception_head = add_deception_head
        
        # Configure quantization for QLoRA
        if use_qlora and load_in_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            self.bnb_config = None
        
        # Load model
        self.model = self._load_model()
        
        # Configure LoRA
        if use_qlora:
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # Prepare model for training
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, self.lora_config)
            
            print("LoRA configuration:")
            self.model.print_trainable_parameters()
        
        # Add custom deception detection head
        if add_deception_head:
            self.deception_head = DeceptionClassificationHead(
                self.model.config.hidden_size
            ).to(self.model.device)
    
    def _load_model(self):
        """Load Llama model with optional quantization."""
        print(f"Loading {self.model_name}...")
        
        model_kwargs = {
            "device_map": self.device_map,
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        
        if self.bnb_config:
            model_kwargs["quantization_config"] = self.bnb_config
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        return model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        extract_deception_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Llama with optional deception detection."""
        
        # Get base model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=extract_deception_features,
        )
        
        result = {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
        
        # Extract deception features if requested
        if extract_deception_features and self.add_deception_head:
            # Use last hidden state for deception detection
            hidden_states = outputs.hidden_states[-1]
            
            # Pool hidden states (mean pooling over sequence)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
            
            # Get deception predictions
            deception_outputs = self.deception_head(pooled)
            result.update(deception_outputs)
        
        return result
    
    def save_model(self, path: str):
        """Save LoRA adapter weights."""
        if self.use_qlora:
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), path + "/model.pt")
        
        # Save deception head separately
        if self.add_deception_head:
            torch.save(
                self.deception_head.state_dict(),
                path + "/deception_head.pt"
            )
    
    def load_model(self, path: str):
        """Load LoRA adapter weights."""
        if self.use_qlora:
            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            self.model.load_state_dict(torch.load(path + "/model.pt"))
        
        # Load deception head
        if self.add_deception_head and os.path.exists(path + "/deception_head.pt"):
            self.deception_head.load_state_dict(
                torch.load(path + "/deception_head.pt")
            )


class DeceptionClassificationHead(nn.Module):
    """Multi-task classification head for deception detection."""
    
    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        
        # Deception detection (binary)
        self.deception_classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        
        # Role prediction (impostor/crewmate/werewolf/villager/etc)
        self.role_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 8),  # 8 possible roles
        )
        
        # Deception type (if deceptive)
        self.deception_type_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 7),  # 7 deception types
        )
    
    def forward(self, pooled_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through classification heads."""
        
        deception_logits = self.deception_classifier(pooled_hidden)
        deception_probs = torch.sigmoid(deception_logits)
        
        role_logits = self.role_classifier(pooled_hidden)
        role_probs = F.softmax(role_logits, dim=-1)
        
        deception_type_logits = self.deception_type_classifier(pooled_hidden)
        deception_type_probs = F.softmax(deception_type_logits, dim=-1)
        
        return {
            "deception_logits": deception_logits,
            "deception_probs": deception_probs,
            "role_logits": role_logits,
            "role_probs": role_probs,
            "deception_type_logits": deception_type_logits,
            "deception_type_probs": deception_type_probs,
        }


class LlamaDeceptionTrainer:
    """Trainer for Llama deception model with game-specific prompting."""
    
    def __init__(
        self,
        model: LlamaDeceptionModel,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # Set up tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for batch generation
    
    def format_game_prompt(
        self,
        game_type: str,
        speaker: str,
        message: str,
        context: Optional[str] = None,
        instruction: str = "Analyze if this message contains deception.",
    ) -> str:
        """Format input for Llama 3.1 Instruct format."""
        
        system_prompt = """You are analyzing social deduction game conversations for deceptive behavior.
        
Deception indicators include:
- False claims about one's role (e.g., impostor claiming to be crewmate)
- False accusations against innocent players
- Fabricated alibis or activities
- Misleading information about game events
- Trust manipulation tactics

Game context is provided to help determine if statements are truthful or deceptive."""
        
        user_content = f"Game: {game_type}\n"
        if context:
            user_content += f"Context: {context}\n"
        user_content += f"Speaker: {speaker}\n"
        user_content += f"Message: {message}\n\n"
        user_content += instruction
        
        # Llama 3.1 Instruct format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def prepare_training_example(
        self,
        example: Dict,
        max_length: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """Prepare a single training example."""
        
        # Format prompt
        prompt = self.format_game_prompt(
            game_type=example.get("game_type", "unknown"),
            speaker=example.get("speaker", "unknown"),
            message=example.get("text", ""),
            context=example.get("context", None),
        )
        
        # Expected response based on labels
        if example.get("is_deceptive", False):
            response = f"This message contains deception. Type: {example.get('deception_type', 'unknown')}"
        else:
            response = "This message does not contain deception."
        
        # Full text for training
        full_text = prompt + response + "<|eot_id|>"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels (mask prompt, only train on response)
        labels = encoding["input_ids"].clone()
        prompt_length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[0, :prompt_length] = -100  # Mask prompt tokens
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "deception_label": int(example.get("is_deceptive", False)),
        }
    
    def get_training_args(
        self,
        output_dir: str = "./llama-deception",
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
    ) -> TrainingArguments:
        """Get training arguments optimized for Llama 3.1 8B."""
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="wandb",
            run_name="llama-3.1-8b-deception",
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            remove_unused_columns=False,
        )