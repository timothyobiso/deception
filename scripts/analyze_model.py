#!/usr/bin/env python3
"""Post-training analysis: Probe for deception circuits and analyze model behavior."""

import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pandas as pd

from deception_interpretability.models.llama_deception_model import LlamaDeceptionModel
from deception_interpretability.interpretability.probes import ProbeAnalyzer, DeceptionProbeKit, LinearProbe, MLPProbe
from deception_interpretability.interpretability.sae import SparseAutoencoder, SAEConfig, SAEAnalyzer
from deception_interpretability.experiments.steering import ActivationSteering, FeatureSteering, SteeringConfig
from deception_interpretability.experiments.ablation import ComponentAblation, AblationConfig
from deception_interpretability.data.hf_dataset_loaders import UnifiedHFDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze trained deception model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='llama',
                       choices=['llama', 'gpt2', 'small'],
                       help='Type of model')
    parser.add_argument('--analysis_type', type=str, default='all',
                       choices=['all', 'probes', 'sae', 'circuits', 'steering', 'ablation', 'probes+steering'],
                       help='What analysis to run')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for analysis')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to analyze')
    return parser.parse_args()


class DeceptionAnalyzer:
    """Main analyzer for post-training mechanistic interpretability."""
    
    def __init__(self, model_path: str, model_type: str, device: str = 'cuda'):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self.results = {}

    @property
    def num_layers(self) -> int:
        """Get number of transformer layers regardless of model architecture."""
        config = getattr(self.model, 'config', None)
        if config is None:
            config = self.model.base_model.config
        return getattr(config, 'num_hidden_layers', 32)
        
    def load_model(self):
        """Load trained model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        if self.model_type == 'llama':
            # Load Llama with LoRA adapters
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.1-8B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        else:
            # Load other models
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"Model loaded successfully!")
    
    def prepare_test_data(self, num_samples: int = 1000):
        """Prepare test dataset for analysis."""
        print(f"Preparing {num_samples} test samples...")
        
        # Load test data
        dataset = UnifiedHFDataset(self.tokenizer)
        dataset_dict = dataset.load_all_datasets()
        
        # Get test split, falling back through available splits
        test_data = (
            dataset_dict.get('test')
            or dataset_dict.get('validation')
            or dataset_dict.get('train')
        )

        if test_data is None:
            raise RuntimeError(
                f"No usable split found in dataset. Available: {list(dataset_dict.keys())}"
            )

        # Sample subset
        if len(test_data) > num_samples:
            indices = np.random.choice(len(test_data), num_samples, replace=False)
            test_data = test_data.select(indices)

        test_data.set_format("torch", columns=["input_ids", "attention_mask", "deception_labels", "role_labels"])
        return test_data
    
    def run_probe_analysis(self, test_data):
        """Run probing analysis: linear, MLP, and control (shuffled) probes per layer."""
        print("\n" + "="*50)
        print("PROBE ANALYSIS")
        print("="*50)

        analyzer = ProbeAnalyzer(self.model, self.device)

        # Extract representations from all layers
        print("Extracting representations from all layers...")
        all_representations = {}
        all_labels = {'deception': [], 'role': []}

        for batch in tqdm(test_data):
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)

            representations = analyzer.extract_representations(
                input_ids, attention_mask=attention_mask
            )
            for layer_idx, reps in representations.items():
                if layer_idx not in all_representations:
                    all_representations[layer_idx] = []
                all_representations[layer_idx].append(reps.cpu())

            all_labels['deception'].append(batch.get('deception_labels', 0))
            all_labels['role'].append(batch.get('role_labels', 0))

        for layer_idx in all_representations:
            all_representations[layer_idx] = torch.cat(
                all_representations[layer_idx], dim=0
            ).float()

        all_labels['deception'] = torch.tensor(all_labels['deception'])
        all_labels['role'] = torch.tensor(all_labels['role'])

        # Shuffle and split
        n = len(all_labels['deception'])
        perm = torch.randperm(n)
        split_idx = int(0.8 * n)
        train_idx, test_idx = perm[:split_idx], perm[split_idx:]

        # Shuffled labels for control baseline
        control_labels = all_labels['deception'][torch.randperm(n)]

        def eval_probe(probe, X, y):
            probe.eval()
            with torch.no_grad():
                logits = probe(X.to(self.device)).squeeze(-1)
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == y.float().to(self.device)).float().mean()
            return acc.item()

        # Train probes on each layer
        print("\nTraining probes on each layer (linear / MLP / control)...")
        layer_results = {}
        best_acc = 0
        best_layer = 0
        best_probe = None

        for layer_idx in tqdm(sorted(all_representations.keys()), desc="Layers"):
            reps = all_representations[layer_idx]
            train_X, test_X = reps[train_idx], reps[test_idx]
            train_y = all_labels['deception'][train_idx]
            test_y = all_labels['deception'][test_idx]
            input_dim = train_X.shape[-1]

            # Linear probe
            lp = LinearProbe(input_dim, 1)
            lp.fit(train_X, train_y, device=self.device)
            linear_acc = eval_probe(lp, test_X, test_y)

            # MLP probe
            mp = MLPProbe(input_dim, hidden_dims=[128, 64], output_dim=1)
            mp.fit(train_X, train_y, device=self.device, epochs=200)
            mlp_acc = eval_probe(mp, test_X, test_y)

            # Control probe (shuffled labels)
            cp = LinearProbe(input_dim, 1)
            cp.fit(train_X, control_labels[train_idx], device=self.device)
            control_acc = eval_probe(cp, test_X, control_labels[test_idx])

            layer_results[layer_idx] = {
                'linear_accuracy': linear_acc,
                'mlp_accuracy': mlp_acc,
                'control_accuracy': control_acc,
            }

            if linear_acc > best_acc:
                best_acc = linear_acc
                best_layer = layer_idx
                best_probe = lp

        # Store best probe for steering
        self.best_probe = best_probe
        self.best_probe_layer = best_layer

        # Report
        ranked = sorted(
            layer_results.items(),
            key=lambda x: x[1]['linear_accuracy'],
            reverse=True,
        )[:5]

        print(f"\n  {'Layer':<8} {'Linear':<10} {'MLP':<10} {'Control':<10}")
        for layer_idx, m in ranked:
            print(
                f"  {layer_idx:<8} {m['linear_accuracy']:.3f}     "
                f"{m['mlp_accuracy']:.3f}     {m['control_accuracy']:.3f}"
            )

        self.results['probe_analysis'] = {
            'layer_results': layer_results,
            'best_layers': ranked,
        }

        return best_layer
    
    def run_sae_analysis(self, test_data, target_layer: int = -1):
        """Train SAE to find sparse deception features."""
        print("\n" + "="*50)
        print("SPARSE AUTOENCODER ANALYSIS")
        print("="*50)
        
        # Extract activations from target layer
        print(f"Extracting activations from layer {target_layer}...")
        activations = []
        deception_labels = []
        
        sae_data = test_data.select(range(min(500, len(test_data))))
        for batch in tqdm(sae_data):
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[target_layer]

                # Mean pool over sequence
                pooled = hidden.mean(dim=1).float()
                activations.append(pooled.cpu())
                deception_labels.append(batch.get('deception_labels', 0))
        
        activations = torch.cat(activations, dim=0)
        deception_labels = torch.tensor(deception_labels)
        
        # Train SAE
        print("Training Sparse Autoencoder...")
        sae_config = SAEConfig(
            input_dim=activations.shape[-1],
            hidden_dim=activations.shape[-1] * 2,
            sparsity_coefficient=1e-3,
            l1_coefficient=1e-4
        )
        
        sae = SparseAutoencoder(sae_config).to(self.device)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
        
        # Training loop
        for epoch in range(50):
            total_loss = 0
            for i in range(0, len(activations), 32):
                batch = activations[i:i+32].to(self.device)
                
                outputs = sae(batch)
                losses = sae.loss(batch, outputs)
                
                optimizer.zero_grad()
                losses['total'].backward()
                optimizer.step()
                
                total_loss += losses['total'].item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}")
        
        # Analyze learned features
        print("\nAnalyzing learned features...")
        sae_analyzer = SAEAnalyzer(sae, self.device)
        
        # Find features correlated with deception
        with torch.no_grad():
            all_features = sae.encode(activations.to(self.device)).cpu()
        
        # Compute correlation with deception labels
        deception_correlations = []
        for feat_idx in range(all_features.shape[-1]):
            feat_activations = all_features[:, feat_idx].float().numpy()
            labels_np = deception_labels.float().numpy()
            if feat_activations.std() == 0 or labels_np.std() == 0:
                deception_correlations.append((feat_idx, 0.0))
            else:
                correlation = np.corrcoef(feat_activations, labels_np)[0, 1]
                deception_correlations.append((feat_idx, abs(correlation)))
        
        # Get top deception features
        top_features = sorted(deception_correlations, key=lambda x: x[1], reverse=True)[:10]
        
        print("\nTop 10 features correlated with deception:")
        for feat_idx, correlation in top_features:
            print(f"  Feature {feat_idx}: {correlation:.3f} correlation")
        
        self.results['sae_analysis'] = {
            'top_deception_features': top_features,
            'reconstruction_loss': total_loss,
            'sparsity': outputs['sparsity'].mean().item()
        }
        
        return sae, [f[0] for f in top_features[:5]]  # Return SAE and top feature indices
    
    def run_circuit_discovery(self, test_data, target_layer: int):
        """Discover circuits involved in deception."""
        print("\n" + "="*50)
        print("CIRCUIT DISCOVERY")
        print("="*50)
        
        from deception_interpretability.experiments.ablation import PathwayAnalysis
        
        pathway_analyzer = PathwayAnalysis(self.model, self.device)
        
        # Test on deceptive vs honest examples
        deceptive_idx = [i for i in range(len(test_data)) if test_data[i]['deception_labels'] == 1][:10]
        honest_idx = [i for i in range(len(test_data)) if test_data[i]['deception_labels'] == 0][:10]
        deceptive_examples = [test_data[i] for i in deceptive_idx]
        honest_examples = [test_data[i] for i in honest_idx]
        
        print("Tracing information flow for deceptive examples...")
        deceptive_paths = []
        
        for ex in tqdm(deceptive_examples):
            input_ids = ex['input_ids'].unsqueeze(0).to(self.device)
            
            # Trace from middle to final layers
            start_layer = max(0, target_layer - 5)
            end_layer = min(target_layer + 5, self.num_layers - 1)
            
            flow = pathway_analyzer.trace_information_flow(
                input_ids, start_layer, end_layer
            )
            deceptive_paths.append(flow)
        
        print("Tracing information flow for honest examples...")
        honest_paths = []
        
        for ex in tqdm(honest_examples):
            input_ids = ex['input_ids'].unsqueeze(0).to(self.device)
            
            flow = pathway_analyzer.trace_information_flow(
                input_ids, start_layer, end_layer
            )
            honest_paths.append(flow)
        
        # Compare paths
        print("\nComparing deceptive vs honest circuits...")
        
        # Average flow patterns
        avg_deceptive_flow = {}
        avg_honest_flow = {}
        
        for layer in range(start_layer + 1, end_layer + 1):
            avg_deceptive_flow[layer] = np.mean([p[layer] for p in deceptive_paths if layer in p])
            avg_honest_flow[layer] = np.mean([p[layer] for p in honest_paths if layer in p])
        
        # Find layers with biggest differences
        flow_differences = {}
        for layer in avg_deceptive_flow:
            diff = abs(avg_deceptive_flow[layer] - avg_honest_flow[layer])
            flow_differences[layer] = diff
        
        critical_layers = sorted(flow_differences.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print("\nCritical layers for deception:")
        for layer, diff in critical_layers:
            print(f"  Layer {layer}: {diff:.3f} flow difference")
        
        self.results['circuit_discovery'] = {
            'critical_layers': critical_layers,
            'deceptive_flow': avg_deceptive_flow,
            'honest_flow': avg_honest_flow
        }
    
    def run_steering_experiments(self, test_data):
        """Contrastive activation steering and probe-direction steering with strength sweeps."""
        print("\n" + "="*50)
        print("STEERING EXPERIMENTS")
        print("="*50)

        best_layer = getattr(self, 'best_probe_layer', self.num_layers // 2)
        best_probe = getattr(self, 'best_probe', None)

        # Split examples by label
        deceptive_idx = [
            i for i in range(len(test_data)) if test_data[i]['deception_labels'] == 1
        ]
        honest_idx = [
            i for i in range(len(test_data)) if test_data[i]['deception_labels'] == 0
        ]

        if len(deceptive_idx) < 5 or len(honest_idx) < 5:
            print("Not enough deceptive/honest examples for steering. Skipping.")
            return

        config = SteeringConfig(
            method='activation_addition',
            layer_indices=[best_layer],
            intervention_type='add',
        )
        act_steering = ActivationSteering(self.model, config, self.device)
        strengths = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

        # --- 1. Contrastive Activation Steering ---
        print(f"\n--- Contrastive Steering (layer {best_layer}) ---")

        n_contrast = min(20, len(deceptive_idx), len(honest_idx))

        # Compute contrastive vector in small batches to avoid OOM
        print("Computing contrastive vector...")
        layers = act_steering._get_layers()
        batch_size = 4
        pos_acts = []
        neg_acts = []
        for start in tqdm(range(0, n_contrast, batch_size), desc="Contrastive batches"):
            end = min(start + batch_size, n_contrast)
            pos_batch = torch.stack(
                [test_data[i]['input_ids'] for i in deceptive_idx[start:end]]
            ).to(self.device)
            neg_batch = torch.stack(
                [test_data[i]['input_ids'] for i in honest_idx[start:end]]
            ).to(self.device)

            def _extract_hook(batch_input, storage):
                acts = []
                def hook_fn(module, inp, output):
                    acts.append(output[0].detach())
                hook = layers[best_layer].register_forward_hook(hook_fn)
                with torch.no_grad():
                    self.model(batch_input)
                hook.remove()
                storage.append(acts[0].mean(dim=(0, 1)))  # mean over batch and seq

            _extract_hook(pos_batch, pos_acts)
            _extract_hook(neg_batch, neg_acts)

        contrastive_vector = torch.stack(pos_acts).mean(0) - torch.stack(neg_acts).mean(0)
        contrastive_vector = contrastive_vector / contrastive_vector.norm()

        # Eval on held-out examples
        n_eval = 10
        eval_idx = deceptive_idx[n_contrast:n_contrast + n_eval] + honest_idx[n_contrast:n_contrast + n_eval]
        if len(eval_idx) < 10:
            eval_idx = deceptive_idx[:n_eval] + honest_idx[:n_eval]

        contrastive_results = self._steering_sweep(
            test_data, eval_idx, best_layer, contrastive_vector, strengths,
            act_steering, probe=best_probe,
        )

        header = f"  {'Strength':<10} {'Logit Diff':<12}"
        if best_probe is not None:
            header += f" {'Probe Score':<12}"
        print(header)
        for s in strengths:
            r = contrastive_results[s]
            line = f"  {s:<+10.1f} {r['logit_diff']:<12.4f}"
            if r.get('probe_score') is not None:
                line += f" {r['probe_score']:<12.3f}"
            print(line)

        # --- 2. Probe-Direction Steering ---
        probe_steer_results = None
        if best_probe is not None:
            print(f"\n--- Probe-Direction Steering (layer {best_layer}) ---")

            probe_direction = best_probe.linear.weight.squeeze(0).detach()
            probe_direction = probe_direction / probe_direction.norm()

            probe_steer_results = self._steering_sweep(
                test_data, eval_idx, best_layer, probe_direction, strengths,
                act_steering, probe=None,
            )

            print(f"  {'Strength':<10} {'Logit Diff':<12}")
            for s in strengths:
                r = probe_steer_results[s]
                print(f"  {s:<+10.1f} {r['logit_diff']:<12.4f}")
        else:
            print("\nNo probe available, skipping probe-direction steering.")

        # --- 3. Qualitative examples ---
        print("\n--- Qualitative Steering Examples ---")
        sample_idx = honest_idx[:3]
        for idx_num, i in enumerate(sample_idx):
            input_ids = test_data[i]['input_ids'].unsqueeze(0).to(self.device)
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[:100]
            print(f"\n  Example {idx_num+1}/3: {text}...")

            for s in [-2.0, 0.0, 2.0]:
                output = act_steering.apply_steering(
                    input_ids, {best_layer: contrastive_vector}, s
                )
                top_tokens = output.logits[0, -1].topk(5)
                top_words = [self.tokenizer.decode(t) for t in top_tokens.indices]
                print(f"    strength={s:+.1f}: next tokens = {top_words}")

        self.results['steering_experiments'] = {
            'contrastive_sweep': {str(k): v for k, v in contrastive_results.items()},
            'probe_direction_sweep': (
                {str(k): v for k, v in probe_steer_results.items()}
                if probe_steer_results else None
            ),
            'steering_layer': best_layer,
        }

    def _steering_sweep(self, test_data, eval_idx, layer_idx, vector, strengths,
                        act_steering, probe=None):
        """Sweep steering strength, measuring logit divergence and optional probe score."""
        batch_size = 4
        layers = act_steering._get_layers()

        # Compute baseline logits in batches
        print("  Computing baseline logits...")
        baseline_logits_list = []
        for start in range(0, len(eval_idx), batch_size):
            end = min(start + batch_size, len(eval_idx))
            batch = torch.stack(
                [test_data[i]['input_ids'] for i in eval_idx[start:end]]
            ).to(self.device)
            with torch.no_grad():
                baseline_logits_list.append(self.model(batch).logits.cpu())
        baseline_logits = torch.cat(baseline_logits_list, dim=0)

        results = {}

        for strength in tqdm(strengths, desc="Steering strengths"):
            all_steered_logits = []
            all_hidden = []

            def _make_hook(s, vec, cap_list):
                def hook_fn(module, inp, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    steered = hidden + s * vec.unsqueeze(0)
                    cap_list.append(steered.detach().cpu())
                    if isinstance(output, tuple):
                        return (steered,) + output[1:]
                    return steered
                return hook_fn

            for start in range(0, len(eval_idx), batch_size):
                end = min(start + batch_size, len(eval_idx))
                batch = torch.stack(
                    [test_data[i]['input_ids'] for i in eval_idx[start:end]]
                ).to(self.device)

                captured_hidden = []
                hook = layers[layer_idx].register_forward_hook(
                    _make_hook(strength, vector, captured_hidden)
                )
                with torch.no_grad():
                    steered_out = self.model(batch).logits.cpu()
                hook.remove()

                all_steered_logits.append(steered_out)
                if captured_hidden:
                    all_hidden.append(captured_hidden[-1])

            steered_logits = torch.cat(all_steered_logits, dim=0)
            logit_diff = (steered_logits - baseline_logits).abs().mean().item()
            entry = {'logit_diff': logit_diff, 'probe_score': None}

            if probe is not None and all_hidden:
                pooled = torch.cat(all_hidden, dim=0).mean(dim=1).float()
                with torch.no_grad():
                    score = torch.sigmoid(
                        probe(pooled.to(next(probe.parameters()).device))
                    ).mean().item()
                entry['probe_score'] = score

            results[strength] = entry

        return results
    
    def run_ablation_studies(self, test_data, target_layer: int):
        """Run ablation studies to verify importance of components."""
        print("\n" + "="*50)
        print("ABLATION STUDIES")
        print("="*50)
        
        ablation_config = AblationConfig(
            ablation_type='zero',
            granularity='head'
        )
        
        ablator = ComponentAblation(self.model, ablation_config, self.device)
        
        # Test ablating different attention heads
        print("Testing attention head ablations...")
        
        config = getattr(self.model, 'config', self.model.base_model.config)
        num_heads = getattr(config, 'num_attention_heads', 32)
        head_importance = {}
        
        for head_idx in tqdm(range(num_heads), desc="Heads"):
            total_effect = 0
            
            for ex in test_data[:20]:  # Test on subset
                input_ids = ex['input_ids'].unsqueeze(0).to(self.device)
                
                # Normal output
                with torch.no_grad():
                    normal_output = self.model(input_ids)
                    normal_logits = normal_output.logits
                
                # Ablated output
                ablated_output = ablator.ablate_attention_heads(
                    input_ids,
                    target_layer,
                    [head_idx],
                    replacement='zero'
                )
                
                # Measure change
                if hasattr(ablated_output, 'logits'):
                    change = (normal_logits - ablated_output.logits).abs().mean().item()
                    total_effect += change
            
            head_importance[head_idx] = total_effect / 20
        
        # Find most important heads
        important_heads = sorted(head_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("\nMost important attention heads:")
        for head_idx, importance in important_heads:
            print(f"  Head {head_idx}: {importance:.3f} impact")
        
        self.results['ablation_studies'] = {
            'important_heads': important_heads,
            'head_importance': head_importance
        }
    
    def save_results(self, output_dir: str):
        """Save all analysis results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        print(f"\nResults saved to {output_dir}")
    
    def create_visualizations(self, output_dir: str):
        """Create visualizations of analysis results."""
        import matplotlib.pyplot as plt

        # 1. Layer-wise probe accuracy (linear / MLP / control)
        if 'probe_analysis' in self.results:
            lr = self.results['probe_analysis']['layer_results']
            layers = sorted(lr.keys())

            # Support both old and new result formats
            has_multi = 'linear_accuracy' in lr[layers[0]]

            if has_multi:
                linear_acc = [lr[l]['linear_accuracy'] for l in layers]
                mlp_acc = [lr[l]['mlp_accuracy'] for l in layers]
                ctrl_acc = [lr[l]['control_accuracy'] for l in layers]

                plt.figure(figsize=(12, 6))
                plt.plot(layers, linear_acc, 'b-o', markersize=3, label='Linear probe')
                plt.plot(layers, mlp_acc, 'g-s', markersize=3, label='MLP probe')
                plt.plot(layers, ctrl_acc, 'r--x', markersize=3, label='Control (shuffled)')
                plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
                plt.xlabel('Layer')
                plt.ylabel('Accuracy')
                plt.title('Deception Probe Accuracy by Layer')
                plt.legend()
                plt.tight_layout()
            else:
                accuracies = [lr[l]['deception_accuracy'] for l in layers]
                plt.figure(figsize=(10, 6))
                plt.bar(layers, accuracies)
                plt.xlabel('Layer')
                plt.ylabel('Deception Detection Accuracy')
                plt.title('Probe Accuracy by Layer')

            plt.savefig(os.path.join(output_dir, 'probe_accuracy.png'), dpi=150)
            plt.close()

        # 2. Steering strength sweep
        if 'steering_experiments' in self.results:
            steer = self.results['steering_experiments']

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Contrastive sweep
            if steer.get('contrastive_sweep'):
                cs = steer['contrastive_sweep']
                strengths = sorted(cs.keys(), key=float)
                logit_diffs = [cs[s]['logit_diff'] for s in strengths]
                xs = [float(s) for s in strengths]

                axes[0].plot(xs, logit_diffs, 'b-o', label='Logit diff')
                if cs[strengths[0]].get('probe_score') is not None:
                    probe_scores = [cs[s]['probe_score'] for s in strengths]
                    ax2 = axes[0].twinx()
                    ax2.plot(xs, probe_scores, 'r-s', label='Probe score')
                    ax2.set_ylabel('Probe Score', color='r')
                    ax2.legend(loc='upper left')
                axes[0].set_xlabel('Steering Strength')
                axes[0].set_ylabel('Mean |Logit Diff|', color='b')
                axes[0].set_title('Contrastive Steering')
                axes[0].legend(loc='upper right')

            # Probe-direction sweep
            if steer.get('probe_direction_sweep'):
                ps = steer['probe_direction_sweep']
                strengths = sorted(ps.keys(), key=float)
                logit_diffs = [ps[s]['logit_diff'] for s in strengths]
                xs = [float(s) for s in strengths]

                axes[1].plot(xs, logit_diffs, 'g-o')
                axes[1].set_xlabel('Steering Strength')
                axes[1].set_ylabel('Mean |Logit Diff|')
                axes[1].set_title('Probe-Direction Steering')
            else:
                axes[1].text(0.5, 0.5, 'No probe-direction data',
                             ha='center', va='center', transform=axes[1].transAxes)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'steering_sweep.png'), dpi=150)
            plt.close()

        # 3. Information flow comparison (unchanged)
        if 'circuit_discovery' in self.results:
            deceptive_flow = self.results['circuit_discovery']['deceptive_flow']
            honest_flow = self.results['circuit_discovery']['honest_flow']

            layers = sorted(deceptive_flow.keys())
            dec_values = [deceptive_flow[l] for l in layers]
            hon_values = [honest_flow[l] for l in layers]

            plt.figure(figsize=(10, 6))
            plt.plot(layers, dec_values, 'r-', label='Deceptive')
            plt.plot(layers, hon_values, 'b-', label='Honest')
            plt.xlabel('Layer')
            plt.ylabel('Information Flow')
            plt.title('Information Flow: Deceptive vs Honest')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'information_flow.png'), dpi=150)
            plt.close()

        print("Visualizations created!")


def main():
    args = parse_args()
    
    # Create analyzer
    analyzer = DeceptionAnalyzer(args.model_path, args.model_type, args.device)
    
    # Load model
    analyzer.load_model()
    
    # Prepare test data
    test_data = analyzer.prepare_test_data(args.num_samples)
    
    # Run analyses based on selection
    if args.analysis_type in ['all', 'probes', 'probes+steering']:
        best_layer = analyzer.run_probe_analysis(test_data)
    else:
        best_layer = -2  # Default to second-to-last layer

    if args.analysis_type in ['all', 'sae']:
        sae, top_features = analyzer.run_sae_analysis(test_data, best_layer)
    else:
        sae, top_features = None, None

    if args.analysis_type in ['all', 'circuits']:
        analyzer.run_circuit_discovery(test_data, best_layer)

    if args.analysis_type in ['all', 'steering', 'probes+steering']:
        analyzer.run_steering_experiments(test_data)

    if args.analysis_type in ['all', 'ablation']:
        analyzer.run_ablation_studies(test_data, best_layer)
    
    # Save results
    analyzer.save_results(args.output_dir)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print(f"Results saved to: {args.output_dir}")
    print("\nKey findings:")
    
    if 'probe_analysis' in analyzer.results:
        best = analyzer.results['probe_analysis']['best_layers'][0]
        m = best[1]
        if 'linear_accuracy' in m:
            print(f"  - Best layer for deception: Layer {best[0]}")
            print(f"    Linear: {m['linear_accuracy']:.1%}  MLP: {m['mlp_accuracy']:.1%}  Control: {m['control_accuracy']:.1%}")
        else:
            print(f"  - Best layer for deception: Layer {best[0]} ({m['deception_accuracy']:.1%} accuracy)")

    if 'steering_experiments' in analyzer.results:
        sl = analyzer.results['steering_experiments'].get('steering_layer', '?')
        print(f"  - Steering experiments run at layer {sl}")

    if 'sae_analysis' in analyzer.results:
        n_features = len(analyzer.results['sae_analysis']['top_deception_features'])
        print(f"  - Found {n_features} features correlated with deception")

    if 'circuit_discovery' in analyzer.results:
        n_critical = len(analyzer.results['circuit_discovery']['critical_layers'])
        print(f"  - Identified {n_critical} critical layers in deception circuit")


if __name__ == '__main__':
    main()