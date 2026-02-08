# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mechanistic interpretability research studying deceptive behavior in language models trained on social deduction game transcripts (Mafia, Among Us, Secret Hitler). The pipeline: train models on game data → probe for deception circuits → steer/ablate features → test transfer to other domains.

## Common Commands

```bash
# Install dependencies
uv sync

# Lint and format
ruff check src/
black src/

# Run tests
pytest test_datasets.py
pytest test_socialmaze.py

# Download datasets
python scripts/download_hf_datasets.py --output_dir ./data --datasets socialmaze,werewolf-amongus

# Train small model (fast iteration)
python scripts/train.py --model_type small --dataset unified-hf --num_epochs 10 --batch_size 16 --device cuda

# Train full model with W&B
python scripts/train.py --model_type full --base_model gpt2 --dataset unified-hf --wandb_project deception-interpretability

# Train Llama 3.1 8B with QLoRA
python scripts/train_llama.py --model_name meta-llama/Llama-3.1-8B-Instruct --use_qlora --num_epochs 3 --batch_size 4

# Evaluate model
python scripts/evaluate_and_report.py --checkpoint_path ./checkpoints/best_model.pt --output_dir ./results

# Sync code to/from remote training server
./scripts/sync_to_remote.sh        # deploy code
./scripts/sync_from_remote.sh -c   # pull checkpoints
```

## Architecture

All source modules live under `src/deception_interpretability/` as subpackages. Import as `from deception_interpretability.data.hf_dataset_loaders import ...` etc.

### Data Pipeline (`src/deception_interpretability/data/`)
- `SocialDeductionDataset` base class with game-specific subclasses (Mafia, AmongUs, SecretHitler, SocialMaze, WerewolfAmongUs)
- `GameTranscript` dataclass: game_id, players, roles (player->role mapping), messages, outcome
- Automatic deception label generation from game transcripts
- Supports local files and HuggingFace Hub datasets (`hf_dataset_loaders.py` vs `real_dataset_loaders.py`)

### Models (`src/deception_interpretability/models/`)
- **DeceptionModel**: GPT-2 base + role embeddings + BiLSTM context encoder + multi-task `DeceptionHead` (deception detection, role prediction, suspicion scoring, intent classification)
- **SmallDeceptionModel**: Lightweight transformer encoder (4 layers, 8 heads, 256 dims, ~13M params) with three probe heads
- **LlamaDeceptionModel**: Llama 3.1 8B fine-tuning with QLoRA/LoRA via PEFT

### Interpretability (`src/deception_interpretability/interpretability/`)
- **Probes** (`probes.py`): `LinearProbe`, `MLPProbe`, `ProbeAnalyzer` (layer-wise probing), `DeceptionProbeKit`
- **SAEs** (`sae.py`): Three variants -- `SparseAutoencoder` (L1), `VariationalSAE` (KL + spike-and-slab), `TopKSAE` (fixed sparsity). `SAEAnalyzer` for feature importance and deception-correlated feature discovery.

### Experiments (`src/deception_interpretability/experiments/`)
- **Steering** (`steering.py`): `ActivationSteering` uses PyTorch hooks for interventions (add/multiply/replace) at specified layers. `FeatureSteering` operates via SAE features or probe directions. `SteeringEvaluator` measures effectiveness.
- **Transfer** (`transfer.py`): Tests generalization to math reasoning, creative writing, sycophancy detection, negotiation, factual accuracy, and cross-game transfer.
- **Ablation** (`ablation.py`): Attention head, MLP neuron, and SAE feature ablation with multiple replacement strategies (zero/mean/random/learned). Includes interaction analysis and information flow tracing.

### Training (`scripts/train.py`)
- Multi-task loss: LM loss + BCE (deception) + CE (role) + weighted combination
- W&B experiment tracking, LR scheduling, checkpoint management via `src/deception_interpretability/utils/training_utils.py`

## Code Style

- Line length: 100 (both ruff and black)
- Target: Python 3.10+
- Small commit messages
