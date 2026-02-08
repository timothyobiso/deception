"""Sparse Autoencoders (SAEs) and Variational SAEs (vSAEs) for feature discovery."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import einops
import numpy as np
from dataclasses import dataclass


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder."""
    input_dim: int
    hidden_dim: int
    sparsity_coefficient: float = 1e-3
    l1_coefficient: float = 1e-4
    use_bias: bool = True
    tied_weights: bool = False
    activation: str = 'relu'


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for discovering interpretable features."""
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=config.use_bias)
        
        if config.tied_weights:
            self.decoder = None
        else:
            self.decoder = nn.Linear(config.hidden_dim, config.input_dim, bias=config.use_bias)
        
        if config.activation == 'relu':
            self.activation = nn.ReLU()
        elif config.activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation."""
        return self.activation(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation back to input space."""
        if self.config.tied_weights:
            # F.linear computes z @ weight.T + bias, so pass encoder.weight
            # directly to get z @ encoder.weight.T (the transpose of the encoder)
            return F.linear(z, self.encoder.weight,
                          self.encoder.bias if self.config.use_bias else None)
        else:
            return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through autoencoder."""
        z = self.encode(x)
        x_hat = self.decode(z)
        
        return {
            'reconstructed': x_hat,
            'latent': z,
            'sparsity': (z > 0).float().mean()
        }
    
    def loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate loss components."""
        reconstruction_loss = F.mse_loss(outputs['reconstructed'], x)
        
        l1_loss = outputs['latent'].abs().mean()
        
        target_sparsity = 0.05
        # KL divergence for Bernoulli: p*log(p/q) + (1-p)*log((1-p)/(1-q))
        p = outputs['sparsity'].clamp(1e-7, 1 - 1e-7)
        q = torch.tensor(target_sparsity, device=x.device)
        sparsity_loss = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        
        total_loss = (
            reconstruction_loss +
            self.config.l1_coefficient * l1_loss +
            self.config.sparsity_coefficient * sparsity_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': reconstruction_loss,
            'l1': l1_loss,
            'sparsity': sparsity_loss
        }


class VariationalSAE(nn.Module):
    """Variational Sparse Autoencoder for probabilistic feature discovery."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        beta: float = 1.0,
        sparsity_prior: float = 0.05
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.sparsity_prior = sparsity_prior
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.spike_slab = nn.Parameter(torch.ones(latent_dim) * 0.1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        spike_prob = torch.sigmoid(self.spike_slab)
        mask = torch.bernoulli(spike_prob).to(z.device)
        z = z * mask
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        
        return {
            'reconstructed': x_hat,
            'mu': mu,
            'logvar': logvar,
            'latent': z,
            'sparsity': (z != 0).float().mean()
        }
    
    def loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculate ELBO loss."""
        reconstruction_loss = F.mse_loss(outputs['reconstructed'], x, reduction='sum')
        
        kl_loss = -0.5 * torch.sum(
            1 + outputs['logvar'] - outputs['mu'].pow(2) - outputs['logvar'].exp()
        )
        
        # KL divergence for Bernoulli: p*log(p/q) + (1-p)*log((1-p)/(1-q))
        p = outputs['sparsity'].clamp(1e-7, 1 - 1e-7)
        q = torch.tensor(self.sparsity_prior, device=x.device)
        sparsity_kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        
        total_loss = reconstruction_loss + self.beta * kl_loss + sparsity_kl
        
        return {
            'total': total_loss / x.size(0),
            'reconstruction': reconstruction_loss / x.size(0),
            'kl': kl_loss / x.size(0),
            'sparsity': sparsity_kl
        }


class TopKSAE(nn.Module):
    """SAE with Top-K activation for guaranteed sparsity."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 10,
        normalize: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.normalize = normalize
        
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        if normalize:
            nn.init.xavier_normal_(self.encoder.weight)
            nn.init.xavier_normal_(self.decoder.weight)
    
    def topk_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply top-k activation."""
        values, indices = torch.topk(x, self.k, dim=-1)
        
        out = torch.zeros_like(x)
        out.scatter_(-1, indices, values)
        
        return out
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with top-k sparsity."""
        z = self.encoder(x)
        z = F.relu(z)
        z_sparse = self.topk_activation(z)
        x_hat = self.decoder(z_sparse)
        
        return {
            'reconstructed': x_hat,
            'latent': z_sparse,
            'pre_sparse': z,
            'active_features': (z_sparse > 0).sum(dim=-1).float().mean()
        }
    
    def loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate reconstruction loss."""
        return F.mse_loss(outputs['reconstructed'], x)


class SAEAnalyzer:
    """Tools for analyzing SAE learned features."""
    
    def __init__(self, sae: nn.Module, device: str = 'cpu'):
        self.sae = sae
        self.device = device
        self.feature_activations = {}
        self.feature_importance = {}
    
    def collect_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None
    ) -> Dict[int, List[float]]:
        """Collect activation statistics for each feature."""
        self.sae.eval()
        hidden_dim = getattr(self.sae, 'hidden_dim', None) or self.sae.config.hidden_dim
        activations = {i: [] for i in range(hidden_dim)}
        
        sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if max_samples and sample_count >= max_samples:
                    break
                
                if isinstance(batch, tuple):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                outputs = self.sae(x)
                latent = outputs['latent']
                
                for i in range(latent.size(-1)):
                    activations[i].extend(latent[:, i].cpu().numpy().tolist())
                
                sample_count += x.size(0)
        
        self.feature_activations = activations
        return activations
    
    def compute_feature_importance(self) -> Dict[int, float]:
        """Compute importance scores for each feature."""
        importance = {}
        
        for feature_idx, acts in self.feature_activations.items():
            acts = np.array(acts)
            
            activation_freq = (acts > 0).mean()
            
            mean_activation = acts[acts > 0].mean() if any(acts > 0) else 0
            
            std_activation = acts.std()
            
            importance[feature_idx] = {
                'frequency': activation_freq,
                'mean_when_active': mean_activation,
                'std': std_activation,
                'importance_score': activation_freq * mean_activation
            }
        
        self.feature_importance = importance
        return importance
    
    def find_deception_features(
        self,
        deception_labels: torch.Tensor,
        threshold: float = 0.7
    ) -> List[int]:
        """Identify features correlated with deception."""
        deception_features = []
        
        for feature_idx, acts in self.feature_activations.items():
            acts = np.array(acts[:len(deception_labels)])
            labels = deception_labels.numpy()
            
            active_mask = acts > 0
            if active_mask.sum() < 10:
                continue
            
            deceptive_activation_rate = labels[active_mask].mean()
            baseline_deception_rate = labels.mean()
            
            if deceptive_activation_rate > threshold and \
               deceptive_activation_rate > baseline_deception_rate * 1.5:
                deception_features.append(feature_idx)
        
        return deception_features
    
    def reconstruct_with_features(
        self,
        x: torch.Tensor,
        feature_indices: List[int],
        amplification: float = 1.0
    ) -> torch.Tensor:
        """Reconstruct input using only specified features."""
        self.sae.eval()
        
        with torch.no_grad():
            z = self.sae.encode(x)
            
            z_masked = torch.zeros_like(z)
            for idx in feature_indices:
                z_masked[:, idx] = z[:, idx] * amplification
            
            x_reconstructed = self.sae.decode(z_masked)
        
        return x_reconstructed