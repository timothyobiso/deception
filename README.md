# Mechanistic Interpretability of Deception in Language Models

Investigating whether language models fine-tuned on social deduction game transcripts develop interpretable internal representations of deception, and whether those representations can be causally steered.

## Key Results

- **Linear probes** achieve 94-97% deception detection accuracy across all transformer layers (vs. 50-61% shuffled-label control)
- **Deception features** are nonlinearly encoded at the embedding layer but become linearly separable by layer 5
- **Contrastive steering** monotonically shifts probe predictions from 0% to 100% deception probability
- **Probe-direction steering** confirms the learned classifier aligns with a causally relevant subspace

## Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Datasets

The project uses two HuggingFace datasets:

| Dataset | Source | Deceptive Roles | Labeling |
|---------|--------|-----------------|----------|
| [Werewolf Among Us](https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us) | One Night Ultimate Werewolf games | Werewolf, Minion | Evil role + strategic speech act |
| [SocialMaze](https://huggingface.co/datasets/MBZUAI/SocialMaze) | Multi-round social deduction | Criminal, Lunatic, Rumormonger | Role-based claim analysis |

Download datasets:

```bash
python scripts/download_hf_datasets.py --output_dir ./data --datasets socialmaze,werewolf-amongus
```

## Pipeline

### 1. Fine-tune Llama 3.1 8B

Joint causal LM + binary deception classification with QLoRA:

```bash
python scripts/train_llama.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --use_qlora \
    --num_epochs 3 \
    --batch_size 4
```

### 2. Run Analysis

Probe all layers and run activation steering experiments:

```bash
# Probes + steering (recommended)
python scripts/analyze_model.py \
    --checkpoint_dir ./checkpoints/checkpoint-400 \
    --analysis_type probes+steering \
    --output_dir ./results/400

# Probes only
python scripts/analyze_model.py \
    --checkpoint_dir ./checkpoints/checkpoint-400 \
    --analysis_type probes \
    --output_dir ./results/400
```

### 3. Generate Figures

```bash
# Probe accuracy across checkpoints
python scripts/plot_probes_across_checkpoints.py

# Training curves
python scripts/plot_training_curves.py
```

## Project Structure

```
deception_interpretability/
├── src/deception_interpretability/
│   ├── data/                    # Dataset loaders (HF + local)
│   ├── models/                  # LlamaDeceptionModel, SmallDeceptionModel
│   ├── interpretability/        # Probes, SAEs
│   └── experiments/             # Steering, ablation, transfer
├── scripts/
│   ├── train_llama.py           # QLoRA fine-tuning
│   ├── analyze_model.py         # Probing + steering pipeline
│   ├── plot_probes_across_checkpoints.py
│   └── plot_training_curves.py
├── writeup/                     # LaTeX paper
├── results/*/          # Results at each checkpoint
└── CLAUDE.md                    # Dev instructions
```

## Methods

**Probing.** For each of the 33 transformer layers, train three probes on mean-pooled hidden states:
- Linear probe (logistic regression)
- MLP probe (two hidden layers, 128 → 64)
- Control probe (shuffled labels, establishes spurious correlation baseline)

**Contrastive steering.** Compute a steering vector v = mean(deceptive) - mean(honest) at the best probe layer. Intervene on the residual stream at inference: h' = h + α·v for α in {-3, ..., +3}.

**Probe-direction steering.** Use the trained linear probe's weight vector as the steering direction, testing whether the probe identifies a causally relevant feature.

## Remote Training

```bash
# Deploy code to GPU server
./scripts/sync_to_remote.sh

# Pull checkpoints back
./scripts/sync_from_remote.sh -c

# Pull logs
./scripts/sync_from_remote.sh -l
```

## Citation

```bibtex
@article{obiso2025deception,
  title={Mechanistic Interpretability of Deception in Language Models Trained on Social Deduction Games},
  author={Obiso, Timothy},
  year={2025}
}
```

## License

MIT
