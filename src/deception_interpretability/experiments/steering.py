"""Steering experiments for controlling deceptive behavior."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import einops
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SteeringConfig:
    """Configuration for steering experiments."""
    method: str = 'activation_addition'
    layer_indices: List[int] = None
    steering_strength: float = 1.0
    target_features: List[int] = None
    intervention_type: str = 'add'


class ActivationSteering:
    """Methods for steering model behavior via activation intervention."""
    
    def __init__(
        self,
        model: nn.Module,
        config: SteeringConfig,
        device: str = 'cpu'
    ):
        self.model = model
        self.config = config
        self.device = device
        self.hooks = []
        self.steering_vectors = {}

    def _get_layers(self):
        base = self.model
        seen = set()
        while hasattr(base, 'base_model'):
            if id(base) in seen:
                break
            seen.add(id(base))
            base = base.base_model
        seen = set()
        while hasattr(base, 'model') and base is not base.model:
            if id(base) in seen:
                break
            seen.add(id(base))
            base = base.model
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return base.model.layers
        if hasattr(base, 'layers'):
            return base.layers
        if hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
            return base.transformer.h
        raise ValueError("Cannot identify transformer layers")
    
    def compute_steering_vector(
        self,
        positive_examples: torch.Tensor,
        negative_examples: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Compute steering vector from contrasting examples."""
        self.model.eval()
        
        def extract_activations(examples):
            activations = []
            
            def hook_fn(module, input, output):
                activations.append(output[0].detach())
            
            target_layer = self._get_layers()[layer_idx]

            hook = target_layer.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                _ = self.model(examples)
            
            hook.remove()
            
            return torch.cat(activations, dim=0)
        
        pos_acts = extract_activations(positive_examples)
        neg_acts = extract_activations(negative_examples)
        
        steering_vector = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        
        steering_vector = F.normalize(steering_vector, p=2, dim=-1)
        
        return steering_vector
    
    def apply_steering(
        self,
        input_ids: torch.Tensor,
        steering_vectors: Dict[int, torch.Tensor],
        strength: Optional[float] = None
    ) -> torch.Tensor:
        """Apply steering during model forward pass."""
        if strength is None:
            strength = self.config.steering_strength
        
        self.clear_hooks()
        
        def create_steering_hook(layer_idx, vector):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                vec = vector.unsqueeze(0).to(hidden_states.device)
                if self.config.intervention_type == 'add':
                    hidden_states = hidden_states + strength * vec
                elif self.config.intervention_type == 'multiply':
                    hidden_states = hidden_states * (1 + strength * vec)
                elif self.config.intervention_type == 'replace':
                    hidden_states[:, :, :vector.size(-1)] = strength * vec
                
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                return hidden_states
            
            return hook_fn
        
        layers = self._get_layers()
        for layer_idx, vector in steering_vectors.items():
            target_layer = layers[layer_idx]

            hook = target_layer.register_forward_hook(
                create_steering_hook(layer_idx, vector)
            )
            self.hooks.append(hook)
        
        output = self.model(input_ids)
        
        self.clear_hooks()
        
        return output
    
    def clear_hooks(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def measure_steering_effect(
        self,
        input_ids: torch.Tensor,
        steering_vectors: Dict[int, torch.Tensor],
        baseline_fn: callable,
        metric_fn: callable,
        strength_range: List[float] = None
    ) -> Dict[float, float]:
        """Measure the effect of steering at different strengths."""
        if strength_range is None:
            strength_range = [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0]
        
        results = {}
        
        with torch.no_grad():
            baseline = baseline_fn(self.model(input_ids))
        
        for strength in strength_range:
            with torch.no_grad():
                steered_output = self.apply_steering(
                    input_ids, steering_vectors, strength
                )
                metric_value = metric_fn(steered_output, baseline)
                results[strength] = metric_value
        
        return results


class FeatureSteering:
    """Steering based on identified features from SAE or probes."""
    
    def __init__(
        self,
        model: nn.Module,
        sae: Optional[nn.Module] = None,
        probe: Optional[nn.Module] = None,
        device: str = 'cpu'
    ):
        self.model = model
        self.sae = sae
        self.probe = probe
        self.device = device

    def _get_layers(self):
        base = self.model
        seen = set()
        while hasattr(base, 'base_model'):
            if id(base) in seen:
                break
            seen.add(id(base))
            base = base.base_model
        seen = set()
        while hasattr(base, 'model') and base is not base.model:
            if id(base) in seen:
                break
            seen.add(id(base))
            base = base.model
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return base.model.layers
        if hasattr(base, 'layers'):
            return base.layers
        if hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
            return base.transformer.h
        raise ValueError("Cannot identify transformer layers")
    
    def steer_sae_features(
        self,
        input_ids: torch.Tensor,
        feature_indices: List[int],
        amplification: Union[float, List[float]] = 1.0,
        layer_idx: int = -1
    ) -> Dict[str, torch.Tensor]:
        """Steer specific SAE features."""
        if self.sae is None:
            raise ValueError("SAE not provided")
        
        def intervention_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            batch_size, seq_len = hidden.shape[:2]
            hidden_flat = hidden.view(-1, hidden.size(-1))
            
            with torch.no_grad():
                sae_output = self.sae(hidden_flat)
                latent = sae_output['latent']
                
                if isinstance(amplification, float):
                    amp_factors = [amplification] * len(feature_indices)
                else:
                    amp_factors = amplification
                
                for feat_idx, amp in zip(feature_indices, amp_factors):
                    latent[:, feat_idx] *= amp
                
                reconstructed = self.sae.decode(latent)
                hidden_flat = reconstructed
            
            hidden = hidden_flat.view(batch_size, seq_len, -1)
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        layers = self._get_layers()
        if layer_idx == -1:
            layer_idx = len(layers) - 1
        target_layer = layers[layer_idx]

        hook = target_layer.register_forward_hook(intervention_hook)

        with torch.no_grad():
            output = self.model(input_ids)

        hook.remove()

        return output

    def steer_probe_direction(
        self,
        input_ids: torch.Tensor,
        target_value: float,
        layer_idx: int = -1,
        strength: float = 1.0
    ) -> torch.Tensor:
        """Steer in the direction identified by a probe."""
        if self.probe is None:
            raise ValueError("Probe not provided")
        
        def intervention_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            with torch.no_grad():
                if hasattr(self.probe, 'linear'):
                    direction = self.probe.linear.weight.squeeze(0)
                else:
                    direction = self.probe.mlp[0].weight.mean(dim=0)
                
                direction = F.normalize(direction, p=2, dim=-1)
                
                current_projection = (hidden * direction.unsqueeze(0)).sum(dim=-1, keepdim=True)
                target_projection = torch.full_like(current_projection, target_value)
                
                adjustment = (target_projection - current_projection) * direction.unsqueeze(0)
                hidden = hidden + strength * adjustment
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        layers = self._get_layers()
        if layer_idx == -1:
            layer_idx = len(layers) - 1
        target_layer = layers[layer_idx]

        hook = target_layer.register_forward_hook(intervention_hook)

        with torch.no_grad():
            output = self.model(input_ids)

        hook.remove()

        return output


class SteeringEvaluator:
    """Evaluate the effectiveness of steering interventions."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = 'cpu'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_deception_steering(
        self,
        test_prompts: List[str],
        steering_fn: callable,
        deception_detector: callable,
        strength_range: List[float] = None
    ) -> Dict[str, Dict[float, float]]:
        """Evaluate deception steering across different prompts and strengths."""
        if strength_range is None:
            strength_range = [-2.0, -1.0, 0, 1.0, 2.0]
        
        results = {}
        
        for prompt in test_prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            prompt_results = {}
            
            for strength in strength_range:
                output = steering_fn(input_ids, strength=strength)
                
                if hasattr(output, 'logits'):
                    generated = self.model.generate(
                        input_ids,
                        max_length=input_ids.size(1) + 50,
                        do_sample=True,
                        temperature=0.7
                    )
                    generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                else:
                    generated_text = ""
                
                deception_score = deception_detector(output)
                
                prompt_results[strength] = {
                    'deception_score': deception_score,
                    'generated_text': generated_text
                }
            
            results[prompt[:50]] = prompt_results
        
        return results
    
    def compare_steering_methods(
        self,
        input_ids: torch.Tensor,
        methods: Dict[str, callable],
        metric_fns: Dict[str, callable]
    ) -> Dict[str, Dict[str, float]]:
        """Compare different steering methods."""
        results = {}
        
        with torch.no_grad():
            baseline_output = self.model(input_ids)
        
        for method_name, steering_fn in methods.items():
            with torch.no_grad():
                steered_output = steering_fn(input_ids)
            
            method_results = {}
            for metric_name, metric_fn in metric_fns.items():
                method_results[metric_name] = metric_fn(steered_output, baseline_output)
            
            results[method_name] = method_results
        
        return results