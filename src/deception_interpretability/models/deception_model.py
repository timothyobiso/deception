"""Models for learning deceptive behavior in social deduction games."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from typing import Optional, Dict, List, Tuple
import einops


class DeceptionConfig(PretrainedConfig):
    """Configuration for deception-aware models."""
    
    model_type = "deception_model"
    
    def __init__(
        self,
        base_model: str = "gpt2",
        hidden_size: int = 768,
        num_deception_heads: int = 4,
        deception_hidden_dim: int = 256,
        role_embedding_dim: int = 64,
        max_players: int = 10,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.num_deception_heads = num_deception_heads
        self.deception_hidden_dim = deception_hidden_dim
        self.role_embedding_dim = role_embedding_dim
        self.max_players = max_players
        self.dropout = dropout


class DeceptionHead(nn.Module):
    """Multi-task head for deception-related predictions."""
    
    def __init__(self, config: DeceptionConfig):
        super().__init__()
        self.config = config
        
        self.deception_detector = nn.Sequential(
            nn.Linear(config.hidden_size, config.deception_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.deception_hidden_dim, config.deception_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.deception_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.role_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.deception_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.deception_hidden_dim, 5)
        )
        
        self.suspicion_scorer = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.deception_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.deception_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.intent_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.deception_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.deception_hidden_dim, 7)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through deception heads."""
        outputs = {}
        
        outputs['deception_score'] = self.deception_detector(hidden_states)
        
        outputs['role_logits'] = self.role_predictor(hidden_states)
        
        if target_hidden_states is not None:
            combined = torch.cat([hidden_states, target_hidden_states], dim=-1)
            outputs['suspicion_score'] = self.suspicion_scorer(combined)
        
        outputs['intent_logits'] = self.intent_classifier(hidden_states)
        
        return outputs


class DeceptionModel(PreTrainedModel):
    """Model for learning and analyzing deceptive behavior."""
    
    config_class = DeceptionConfig
    
    def __init__(self, config: DeceptionConfig):
        super().__init__(config)
        self.config = config
        
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model)
        
        base_config = self.base_model.config
        self.hidden_size = base_config.hidden_size
        
        self.role_embeddings = nn.Embedding(5, config.role_embedding_dim)
        self.player_embeddings = nn.Embedding(config.max_players, config.role_embedding_dim)
        
        self.context_encoder = nn.LSTM(
            self.hidden_size + config.role_embedding_dim * 2,
            self.hidden_size,
            num_layers=2,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.deception_head = DeceptionHead(config)
        
        self.hidden_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        player_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        deception_labels: Optional[torch.Tensor] = None,
        role_labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = base_outputs.hidden_states[-1]
        
        if role_ids is not None and player_ids is not None:
            role_emb = self.role_embeddings(role_ids)
            player_emb = self.player_embeddings(player_ids)
            
            role_emb = role_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            player_emb = player_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            
            enhanced_hidden = torch.cat([hidden_states, role_emb, player_emb], dim=-1)
            
            context_output, _ = self.context_encoder(enhanced_hidden)
            hidden_states = self.hidden_projection(context_output)
        
        pooled_hidden = hidden_states.mean(dim=1)
        
        deception_outputs = self.deception_head(pooled_hidden)
        
        outputs = {
            'loss': base_outputs.loss if hasattr(base_outputs, 'loss') else None,
            'logits': base_outputs.logits,
            'deception_score': deception_outputs['deception_score'],
            'role_logits': deception_outputs['role_logits'],
            'intent_logits': deception_outputs['intent_logits'],
        }
        
        if deception_labels is not None:
            deception_loss = F.binary_cross_entropy(
                deception_outputs['deception_score'].squeeze(-1),
                deception_labels.float()
            )
            outputs['deception_loss'] = deception_loss
            
            if outputs['loss'] is not None:
                outputs['loss'] = outputs['loss'] + deception_loss
            else:
                outputs['loss'] = deception_loss
        
        if role_labels is not None:
            role_loss = F.cross_entropy(
                deception_outputs['role_logits'],
                role_labels
            )
            outputs['role_loss'] = role_loss
            
            if outputs['loss'] is not None:
                outputs['loss'] = outputs['loss'] + role_loss
        
        if return_hidden_states:
            outputs['hidden_states'] = hidden_states
        
        return outputs
    
    def generate_with_deception(
        self,
        input_ids: torch.Tensor,
        deception_level: float = 0.5,
        role_id: Optional[int] = None,
        **generate_kwargs
    ) -> torch.Tensor:
        """Generate text with controlled deception level."""
        if role_id is not None:
            role_emb = self.role_embeddings(torch.tensor([role_id], device=input_ids.device))
            self.base_model.transformer.wpe.weight.data[:role_emb.size(1)] += role_emb.squeeze(0) * deception_level
        
        outputs = self.base_model.generate(
            input_ids=input_ids,
            **generate_kwargs
        )
        
        if role_id is not None:
            self.base_model.transformer.wpe.weight.data[:role_emb.size(1)] -= role_emb.squeeze(0) * deception_level
        
        return outputs


class SmallDeceptionModel(nn.Module):
    """Smaller model for initial experiments and faster iteration."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = nn.Embedding(1024, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.deception_probe = nn.Linear(hidden_dim, 1)
        self.role_probe = nn.Linear(hidden_dim, 5)
        self.intent_probe = nn.Linear(hidden_dim, 7)
        
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        deception_labels: Optional[torch.Tensor] = None,
        role_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.positional_encoding(positions)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        hidden_states = self.transformer(
            x, src_key_padding_mask=~attention_mask if attention_mask is not None else None
        )

        pooled = hidden_states.mean(dim=1)
        logits = self.lm_head(hidden_states)
        deception_score = torch.sigmoid(self.deception_probe(pooled))
        role_logits = self.role_probe(pooled)

        outputs = {
            'logits': logits,
            'deception_score': deception_score,
            'role_logits': role_logits,
            'intent_logits': self.intent_probe(pooled),
            'hidden_states': hidden_states,
            'loss': None,
        }

        # Compute LM loss
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            outputs['loss'] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Compute deception loss
        if deception_labels is not None:
            deception_loss = F.binary_cross_entropy(
                deception_score.squeeze(-1),
                deception_labels.float(),
            )
            outputs['deception_loss'] = deception_loss
            if outputs['loss'] is not None:
                outputs['loss'] = outputs['loss'] + deception_loss
            else:
                outputs['loss'] = deception_loss

        # Compute role loss
        if role_labels is not None:
            role_loss = F.cross_entropy(role_logits, role_labels)
            outputs['role_loss'] = role_loss
            if outputs['loss'] is not None:
                outputs['loss'] = outputs['loss'] + role_loss
            else:
                outputs['loss'] = role_loss

        return outputs