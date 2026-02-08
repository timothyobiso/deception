"""Ablation studies for understanding deception mechanisms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import itertools
from tqdm import tqdm


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    ablation_type: str = 'zero'  # 'zero', 'mean', 'random', 'learned'
    components: List[str] = None
    granularity: str = 'feature'  # 'feature', 'neuron', 'head', 'layer'
    measure_interaction: bool = True


class ComponentAblation:
    """Ablate specific components to understand their role in deception."""
    
    def __init__(
        self,
        model: nn.Module,
        config: AblationConfig,
        device: str = 'cpu'
    ):
        self.model = model
        self.config = config
        self.device = device
        self.baseline_performance = {}
        self.ablation_values = {}
    
    def ablate_attention_heads(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        head_indices: List[int],
        replacement: str = 'zero'
    ) -> torch.Tensor:
        """Ablate specific attention heads."""
        def ablation_hook(module, input, output):
            if hasattr(module, 'self_attn'):
                attn_module = module.self_attn
            elif hasattr(module, 'attn'):
                attn_module = module.attn
            else:
                return output
            
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output
            
            batch_size, seq_len, hidden_dim = attn_output.shape
            num_heads = attn_module.num_heads if hasattr(attn_module, 'num_heads') else 12
            head_dim = hidden_dim // num_heads
            
            attn_output_reshaped = attn_output.view(batch_size, seq_len, num_heads, head_dim)
            
            for head_idx in head_indices:
                if replacement == 'zero':
                    attn_output_reshaped[:, :, head_idx, :] = 0
                elif replacement == 'mean':
                    mean_val = attn_output_reshaped[:, :, head_idx, :].mean()
                    attn_output_reshaped[:, :, head_idx, :] = mean_val
                elif replacement == 'random':
                    attn_output_reshaped[:, :, head_idx, :] = torch.randn_like(
                        attn_output_reshaped[:, :, head_idx, :]
                    ) * 0.01
            
            attn_output = attn_output_reshaped.view(batch_size, seq_len, hidden_dim)
            
            if isinstance(output, tuple):
                return (attn_output,) + output[1:]
            return attn_output
        
        if hasattr(self.model, 'transformer'):
            target_layer = self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'layers'):
            target_layer = self.model.layers[layer_idx]
        else:
            raise ValueError("Cannot identify model layers")
        
        hook = target_layer.register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            output = self.model(input_ids)
        
        hook.remove()
        
        return output
    
    def ablate_mlp_neurons(
        self,
        input_ids: torch.Tensor,
        layer_idx: int,
        neuron_indices: List[int],
        replacement: str = 'zero'
    ) -> torch.Tensor:
        """Ablate specific MLP neurons."""
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            if hasattr(module, 'mlp'):
                for neuron_idx in neuron_indices:
                    if neuron_idx < hidden.size(-1):
                        if replacement == 'zero':
                            hidden[:, :, neuron_idx] = 0
                        elif replacement == 'mean':
                            hidden[:, :, neuron_idx] = hidden[:, :, neuron_idx].mean()
                        elif replacement == 'random':
                            hidden[:, :, neuron_idx] = torch.randn_like(hidden[:, :, neuron_idx]) * 0.01
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        if hasattr(self.model, 'transformer'):
            target_layer = self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'layers'):
            target_layer = self.model.layers[layer_idx]
        else:
            raise ValueError("Cannot identify model layers")
        
        hook = target_layer.register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            output = self.model(input_ids)
        
        hook.remove()
        
        return output
    
    def ablate_features(
        self,
        input_ids: torch.Tensor,
        feature_indices: List[int],
        sae: nn.Module,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """Ablate specific SAE features."""
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            batch_size, seq_len = hidden.shape[:2]
            hidden_flat = hidden.view(-1, hidden.size(-1))
            
            with torch.no_grad():
                sae_output = sae(hidden_flat)
                latent = sae_output['latent']
                
                for feat_idx in feature_indices:
                    if self.config.ablation_type == 'zero':
                        latent[:, feat_idx] = 0
                    elif self.config.ablation_type == 'mean':
                        latent[:, feat_idx] = latent[:, feat_idx].mean()
                    elif self.config.ablation_type == 'random':
                        latent[:, feat_idx] = torch.randn_like(latent[:, feat_idx]) * 0.01
                
                reconstructed = sae.decode(latent)
                hidden_flat = reconstructed
            
            hidden = hidden_flat.view(batch_size, seq_len, -1)
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        if hasattr(self.model, 'transformer'):
            if layer_idx == -1:
                layer_idx = len(self.model.transformer.h) - 1
            target_layer = self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'layers'):
            if layer_idx == -1:
                layer_idx = len(self.model.layers) - 1
            target_layer = self.model.layers[layer_idx]
        else:
            raise ValueError("Cannot identify model layers")
        
        hook = target_layer.register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            output = self.model(input_ids)
        
        hook.remove()
        
        return output
    
    def measure_component_importance(
        self,
        input_ids: torch.Tensor,
        components: Dict[str, List[int]],
        metric_fn: callable,
        ablation_fn: callable
    ) -> Dict[str, float]:
        """Measure importance of different components."""
        with torch.no_grad():
            baseline = self.model(input_ids)
            baseline_metric = metric_fn(baseline)
        
        importance_scores = {}
        
        for component_name, component_indices in components.items():
            ablated_output = ablation_fn(input_ids, component_indices)
            ablated_metric = metric_fn(ablated_output)
            
            importance = abs(baseline_metric - ablated_metric)
            importance_scores[component_name] = importance
        
        normalized_scores = {}
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            for name, score in importance_scores.items():
                normalized_scores[name] = score / total_importance
        else:
            normalized_scores = importance_scores
        
        return normalized_scores


class InteractionAnalysis:
    """Analyze interactions between components for deception."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
    
    def analyze_pairwise_interactions(
        self,
        input_ids: torch.Tensor,
        components: Dict[str, List[int]],
        ablation_fn: callable,
        metric_fn: callable
    ) -> Dict[Tuple[str, str], float]:
        """Analyze pairwise interactions between components."""
        with torch.no_grad():
            baseline = self.model(input_ids)
            baseline_metric = metric_fn(baseline)
        
        single_effects = {}
        for name, indices in components.items():
            ablated = ablation_fn(input_ids, {name: indices})
            single_effects[name] = baseline_metric - metric_fn(ablated)
        
        interaction_effects = {}
        
        for (name1, indices1), (name2, indices2) in itertools.combinations(components.items(), 2):
            both_ablated = ablation_fn(input_ids, {name1: indices1, name2: indices2})
            both_effect = baseline_metric - metric_fn(both_ablated)
            
            expected_effect = single_effects[name1] + single_effects[name2]
            
            interaction = both_effect - expected_effect
            
            interaction_effects[(name1, name2)] = interaction
        
        return interaction_effects
    
    def find_minimal_circuit(
        self,
        input_ids: torch.Tensor,
        all_components: Dict[str, List[int]],
        ablation_fn: callable,
        metric_fn: callable,
        threshold: float = 0.9
    ) -> List[str]:
        """Find minimal set of components needed for behavior."""
        with torch.no_grad():
            baseline = self.model(input_ids)
            baseline_metric = metric_fn(baseline)
        
        component_names = list(all_components.keys())
        minimal_circuit = []
        
        for size in range(1, len(component_names) + 1):
            for subset in itertools.combinations(component_names, size):
                components_to_keep = {name: all_components[name] for name in subset}
                components_to_ablate = {
                    name: indices 
                    for name, indices in all_components.items() 
                    if name not in subset
                }
                
                ablated = ablation_fn(input_ids, components_to_ablate)
                preserved_metric = metric_fn(ablated)
                
                if preserved_metric >= threshold * baseline_metric:
                    minimal_circuit = list(subset)
                    break
            
            if minimal_circuit:
                break
        
        return minimal_circuit


class PathwayAnalysis:
    """Analyze information flow pathways for deception."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.activation_cache = {}
    
    def trace_information_flow(
        self,
        input_ids: torch.Tensor,
        source_layer: int,
        target_layer: int,
        feature_idx: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """Trace how information flows between layers."""
        flow_map = {}
        
        def create_trace_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                
                if feature_idx is not None:
                    self.activation_cache[layer_idx] = hidden[:, :, feature_idx].detach()
                else:
                    self.activation_cache[layer_idx] = hidden.detach()
            
            return hook_fn
        
        hooks = []
        for layer_idx in range(source_layer, target_layer + 1):
            if hasattr(self.model, 'transformer'):
                layer = self.model.transformer.h[layer_idx]
            elif hasattr(self.model, 'layers'):
                layer = self.model.layers[layer_idx]
            else:
                continue
            
            hook = layer.register_forward_hook(create_trace_hook(layer_idx))
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        for hook in hooks:
            hook.remove()
        
        for layer_idx in range(source_layer + 1, target_layer + 1):
            prev_activation = self.activation_cache[layer_idx - 1]
            curr_activation = self.activation_cache[layer_idx]
            
            if feature_idx is not None:
                correlation = torch.corrcoef(
                    torch.stack([
                        prev_activation.flatten(),
                        curr_activation.flatten()
                    ])
                )[0, 1]
            else:
                correlation = F.cosine_similarity(
                    prev_activation.mean(dim=1),
                    curr_activation.mean(dim=1),
                    dim=-1
                ).mean()
            
            flow_map[layer_idx] = correlation.item()
        
        self.activation_cache.clear()
        
        return flow_map
    
    def identify_critical_paths(
        self,
        input_ids: torch.Tensor,
        deception_features: List[int],
        threshold: float = 0.5
    ) -> List[Tuple[int, int]]:
        """Identify critical paths for deception signal propagation."""
        critical_paths = []
        
        num_layers = len(self.model.transformer.h) if hasattr(self.model, 'transformer') else len(self.model.layers)
        
        for feat_idx in deception_features:
            for start_layer in range(num_layers - 1):
                flow = self.trace_information_flow(
                    input_ids,
                    start_layer,
                    min(start_layer + 3, num_layers - 1),
                    feat_idx
                )
                
                for layer_idx, correlation in flow.items():
                    if abs(correlation) > threshold:
                        critical_paths.append((start_layer, layer_idx))
        
        return list(set(critical_paths))