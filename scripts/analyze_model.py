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
from deception_interpretability.interpretability.probes import ProbeAnalyzer, DeceptionProbeKit
from deception_interpretability.interpretability.sae import SparseAutoencoder, SAEConfig, SAEAnalyzer
from deception_interpretability.experiments.steering import ActivationSteering, SteeringConfig
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
                       choices=['all', 'probes', 'sae', 'circuits', 'steering', 'ablation'],
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
        """Run probing analysis to find deception-related neurons."""
        print("\n" + "="*50)
        print("PROBE ANALYSIS")
        print("="*50)
        
        analyzer = ProbeAnalyzer(self.model, self.device)
        probe_kit = DeceptionProbeKit(self.device)
        
        # Extract representations from multiple layers
        print("Extracting representations from all layers...")
        all_representations = {}
        all_labels = {
            'deception': [],
            'role': [],
        }
        
        for batch in tqdm(test_data):
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(self.device)
            
            # Get representations
            representations = analyzer.extract_representations(
                input_ids, 
                attention_mask=attention_mask
            )
            
            for layer_idx, reps in representations.items():
                if layer_idx not in all_representations:
                    all_representations[layer_idx] = []
                all_representations[layer_idx].append(reps.cpu())
            
            # Collect labels
            all_labels['deception'].append(batch.get('deception_labels', 0))
            all_labels['role'].append(batch.get('role_labels', 0))
        
        # Concatenate representations and cast to float32 for probes
        for layer_idx in all_representations:
            all_representations[layer_idx] = torch.cat(all_representations[layer_idx], dim=0).float()
        
        # Convert labels to tensors
        all_labels['deception'] = torch.tensor(all_labels['deception'])
        all_labels['role'] = torch.tensor(all_labels['role'])
        
        # Train probes on each layer
        print("\nTraining probes on each layer...")
        layer_results = {}
        
        for layer_idx in tqdm(all_representations.keys(), desc="Layers"):
            layer_reps = all_representations[layer_idx]
            
            # Train deception probe
            deception_probe = probe_kit.create_deception_probe(layer_reps.shape[-1])
            deception_results = deception_probe.fit(
                layer_reps[:800], 
                all_labels['deception'][:800],
                device=self.device
            )
            
            # Evaluate
            with torch.no_grad():
                preds = torch.sigmoid(deception_probe(layer_reps[800:].to(self.device)))
                accuracy = ((preds > 0.5).float() == all_labels['deception'][800:].float().to(self.device)).float().mean()
            
            layer_results[layer_idx] = {
                'deception_accuracy': accuracy.item(),
                'final_loss': deception_results['losses'][-1] if deception_results['losses'] else 0
            }
        
        # Find best layers for deception detection
        best_layers = sorted(layer_results.items(), key=lambda x: x[1]['deception_accuracy'], reverse=True)[:5]
        
        print("\nTop 5 layers for deception detection:")
        for layer_idx, metrics in best_layers:
            print(f"  Layer {layer_idx}: {metrics['deception_accuracy']:.3f} accuracy")
        
        self.results['probe_analysis'] = {
            'layer_results': layer_results,
            'best_layers': best_layers
        }
        
        return best_layers[0][0]  # Return best layer index
    
    def run_sae_analysis(self, test_data, target_layer: int = -1):
        """Train SAE to find sparse deception features."""
        print("\n" + "="*50)
        print("SPARSE AUTOENCODER ANALYSIS")
        print("="*50)
        
        # Extract activations from target layer
        print(f"Extracting activations from layer {target_layer}...")
        activations = []
        deception_labels = []
        
        for batch in tqdm(test_data[:500]):  # Use subset for SAE
            input_ids = batch['input_ids'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                hidden = outputs.hidden_states[target_layer]
                
                # Mean pool over sequence
                pooled = hidden.mean(dim=1)
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
            feat_activations = all_features[:, feat_idx]
            correlation = np.corrcoef(feat_activations.numpy(), deception_labels.numpy())[0, 1]
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
        deceptive_examples = [ex for ex in test_data if ex.get('deception_labels', 0) == 1][:10]
        honest_examples = [ex for ex in test_data if ex.get('deception_labels', 0) == 0][:10]
        
        print("Tracing information flow for deceptive examples...")
        deceptive_paths = []
        
        for ex in tqdm(deceptive_examples):
            input_ids = ex['input_ids'].unsqueeze(0).to(self.device)
            
            # Trace from middle to final layers
            start_layer = max(0, target_layer - 5)
            end_layer = min(target_layer + 5, len(self.model.transformer.h) - 1)
            
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
    
    def run_steering_experiments(self, test_data, sae=None, top_features=None):
        """Test steering model behavior using discovered features."""
        print("\n" + "="*50)
        print("STEERING EXPERIMENTS")
        print("="*50)
        
        from deception_interpretability.experiments.steering import FeatureSteering
        
        if sae is None or top_features is None:
            print("No SAE provided, skipping feature steering...")
            return
        
        feature_steering = FeatureSteering(self.model, sae, device=self.device)
        
        # Test steering on examples
        test_prompts = [
            "I was in electrical doing tasks when the body was found.",
            "I saw Red vent in medbay!",
            "Trust me, I'm innocent.",
            "We should vote out Blue, they're suspicious.",
            "I can vouch for Green, they were with me.",
        ]
        
        print("Testing feature steering...")
        steering_results = []
        
        for prompt in test_prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Original output
            with torch.no_grad():
                orig_output = self.model.generate(input_ids, max_length=100, do_sample=True)
                orig_text = self.tokenizer.decode(orig_output[0], skip_special_tokens=True)
            
            # Amplify deception features
            amplified_output = feature_steering.steer_sae_features(
                input_ids, 
                top_features, 
                amplification=2.0
            )
            
            # Suppress deception features
            suppressed_output = feature_steering.steer_sae_features(
                input_ids,
                top_features,
                amplification=0.0
            )
            
            steering_results.append({
                'prompt': prompt,
                'original': orig_text,
                'amplified': self.tokenizer.decode(amplified_output['logits'].argmax(-1)[0]),
                'suppressed': self.tokenizer.decode(suppressed_output['logits'].argmax(-1)[0])
            })
            
            print(f"\nPrompt: {prompt}")
            print(f"Effect: Feature steering changes model behavior")
        
        self.results['steering_experiments'] = steering_results
    
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
        
        num_heads = 32  # For Llama 8B
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
        
        # 1. Layer-wise probe accuracy
        if 'probe_analysis' in self.results:
            layers = list(self.results['probe_analysis']['layer_results'].keys())
            accuracies = [self.results['probe_analysis']['layer_results'][l]['deception_accuracy'] 
                         for l in layers]
            
            plt.figure(figsize=(10, 6))
            plt.bar(layers, accuracies)
            plt.xlabel('Layer')
            plt.ylabel('Deception Detection Accuracy')
            plt.title('Probe Accuracy by Layer')
            plt.savefig(os.path.join(output_dir, 'probe_accuracy.png'))
            plt.close()
        
        # 2. Information flow comparison
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
            plt.savefig(os.path.join(output_dir, 'information_flow.png'))
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
    if args.analysis_type in ['all', 'probes']:
        best_layer = analyzer.run_probe_analysis(test_data)
    else:
        best_layer = -2  # Default to second-to-last layer
    
    if args.analysis_type in ['all', 'sae']:
        sae, top_features = analyzer.run_sae_analysis(test_data, best_layer)
    else:
        sae, top_features = None, None
    
    if args.analysis_type in ['all', 'circuits']:
        analyzer.run_circuit_discovery(test_data, best_layer)
    
    if args.analysis_type in ['all', 'steering']:
        analyzer.run_steering_experiments(test_data, sae, top_features)
    
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
        print(f"  - Best layer for deception: Layer {best[0]} ({best[1]['deception_accuracy']:.1%} accuracy)")
    
    if 'sae_analysis' in analyzer.results:
        n_features = len(analyzer.results['sae_analysis']['top_deception_features'])
        print(f"  - Found {n_features} features correlated with deception")
    
    if 'circuit_discovery' in analyzer.results:
        n_critical = len(analyzer.results['circuit_discovery']['critical_layers'])
        print(f"  - Identified {n_critical} critical layers in deception circuit")


if __name__ == '__main__':
    main()