"""Probing classifiers for detecting deception-related features."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import einops


class LinearProbe(nn.Module):
    """Linear probe for feature detection."""
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        device: str = 'cpu'
    ) -> Dict[str, List[float]]:
        """Train the probe."""
        self.to(device)
        X = X.to(device)
        y = y.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            
            if outputs.size(-1) == 1:
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(-1), y.float())
            else:
                loss = F.cross_entropy(outputs, y)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        return {'losses': losses}


class MLPProbe(nn.Module):
    """MLP probe for non-linear feature detection."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        val_X: Optional[torch.Tensor] = None,
        val_y: Optional[torch.Tensor] = None,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cpu',
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Train the probe with validation and early stopping."""
        self.to(device)
        X = X.to(device)
        y = y.to(device)
        
        if val_X is not None:
            val_X = val_X.to(device)
            val_y = val_y.to(device)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X)
            
            if outputs.size(-1) == 1:
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(-1), y.float())
            else:
                loss = F.cross_entropy(outputs, y)
            
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            if val_X is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self(val_X)
                    if val_outputs.size(-1) == 1:
                        val_loss = F.binary_cross_entropy_with_logits(
                            val_outputs.squeeze(-1), val_y.float()
                        )
                    else:
                        val_loss = F.cross_entropy(val_outputs, val_y)
                    
                    val_losses.append(val_loss.item())
                    scheduler.step(val_loss)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        break
        
        return {'train_losses': train_losses, 'val_losses': val_losses}


class ProbeAnalyzer:
    """Analyze model representations using various probes."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.probes = {}
        self.results = {}
    
    def extract_representations(
        self,
        input_ids: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[int, torch.Tensor]:
        """Extract representations from specified layers."""
        self.model.eval()
        representations = {}
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            else:
                hidden_states = outputs['hidden_states']
            
            if layer_indices is None:
                layer_indices = list(range(len(hidden_states)))
            
            for idx in layer_indices:
                representations[idx] = hidden_states[idx].mean(dim=1)
        
        return representations
    
    def train_probe(
        self,
        probe_name: str,
        X: torch.Tensor,
        y: torch.Tensor,
        probe_type: str = 'linear',
        **kwargs
    ) -> Dict[str, any]:
        """Train a probe on given representations."""
        input_dim = X.size(-1)
        output_dim = len(torch.unique(y)) if y.dim() > 0 else 1

        # Separate constructor kwargs from fit kwargs
        mlp_constructor_keys = {'hidden_dims', 'dropout'}
        constructor_kwargs = {k: v for k, v in kwargs.items() if k in mlp_constructor_keys}
        fit_kwargs = {k: v for k, v in kwargs.items() if k not in mlp_constructor_keys}

        if probe_type == 'linear':
            probe = LinearProbe(input_dim, output_dim)
        elif probe_type == 'mlp':
            probe = MLPProbe(input_dim, output_dim=output_dim, **constructor_kwargs)
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

        train_results = probe.fit(X, y, device=self.device, **fit_kwargs)

        self.probes[probe_name] = probe
        self.results[probe_name] = train_results

        return train_results
    
    def evaluate_probe(
        self,
        probe_name: str,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate a trained probe."""
        if probe_name not in self.probes:
            raise ValueError(f"Probe {probe_name} not found")
        
        probe = self.probes[probe_name]
        probe.eval()
        
        with torch.no_grad():
            outputs = probe(X.to(self.device))
            
            if outputs.size(-1) == 1:
                preds = (torch.sigmoid(outputs) > 0.5).float().squeeze(-1)
                probs = torch.sigmoid(outputs).squeeze(-1)
            else:
                preds = outputs.argmax(dim=-1)
                probs = F.softmax(outputs, dim=-1)
        
        y_cpu = y.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        
        metrics = {
            'accuracy': accuracy_score(y_cpu, preds_cpu),
            'f1': f1_score(y_cpu, preds_cpu, average='macro'),
        }
        
        if outputs.size(-1) == 1:
            metrics['auc'] = roc_auc_score(y_cpu, probs.cpu().numpy())
        
        return metrics
    
    def probe_all_layers(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        probe_type: str = 'linear',
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[int, Dict[str, float]]:
        """Train and evaluate probes on all layers."""
        representations = self.extract_representations(input_ids, attention_mask=attention_mask)
        
        layer_results = {}
        
        for layer_idx, layer_reps in representations.items():
            split_idx = int(0.8 * len(layer_reps))
            
            train_X = layer_reps[:split_idx]
            train_y = labels[:split_idx]
            test_X = layer_reps[split_idx:]
            test_y = labels[split_idx:]
            
            probe_name = f"layer_{layer_idx}_{probe_type}"
            
            self.train_probe(probe_name, train_X, train_y, probe_type=probe_type, **kwargs)
            
            metrics = self.evaluate_probe(probe_name, test_X, test_y)
            layer_results[layer_idx] = metrics
        
        return layer_results


class DeceptionProbeKit:
    """Specialized probes for deception-related features."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.probes = {}
    
    def create_deception_probe(self, input_dim: int) -> LinearProbe:
        """Create probe for detecting deceptive intent."""
        return LinearProbe(input_dim, 1)
    
    def create_role_probe(self, input_dim: int, num_roles: int = 5) -> LinearProbe:
        """Create probe for predicting player roles."""
        return LinearProbe(input_dim, num_roles)
    
    def create_suspicion_probe(self, input_dim: int) -> MLPProbe:
        """Create probe for detecting suspicion levels."""
        return MLPProbe(input_dim, hidden_dims=[64, 32], output_dim=1)
    
    def create_strategy_probe(self, input_dim: int, num_strategies: int = 7) -> MLPProbe:
        """Create probe for identifying strategic intents."""
        return MLPProbe(input_dim, hidden_dims=[128, 64], output_dim=num_strategies)
    
    def train_all_probes(
        self,
        representations: torch.Tensor,
        labels: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, Dict[str, any]]:
        """Train all deception-related probes."""
        results = {}
        input_dim = representations.size(-1)
        
        if 'deception' in labels:
            self.probes['deception'] = self.create_deception_probe(input_dim)
            results['deception'] = self.probes['deception'].fit(
                representations, labels['deception'], device=self.device, **kwargs
            )
        
        if 'role' in labels:
            num_roles = len(torch.unique(labels['role']))
            self.probes['role'] = self.create_role_probe(input_dim, num_roles)
            results['role'] = self.probes['role'].fit(
                representations, labels['role'], device=self.device, **kwargs
            )
        
        if 'suspicion' in labels:
            self.probes['suspicion'] = self.create_suspicion_probe(input_dim)
            results['suspicion'] = self.probes['suspicion'].fit(
                representations, labels['suspicion'], device=self.device, **kwargs
            )
        
        if 'strategy' in labels:
            num_strategies = len(torch.unique(labels['strategy']))
            self.probes['strategy'] = self.create_strategy_probe(input_dim, num_strategies)
            results['strategy'] = self.probes['strategy'].fit(
                representations, labels['strategy'], device=self.device, **kwargs
            )
        
        return results