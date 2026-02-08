#!/bin/bash

# Setup script for Llama 3.1 8B training

echo "=========================================="
echo "Llama 3.1 8B Deception Training Setup"
echo "=========================================="

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ No GPU detected. Training will be slow!"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -U transformers peft bitsandbytes accelerate datasets wandb

# Login to HuggingFace (for Llama access)
echo ""
echo "You need access to meta-llama/Llama-3.1-8B-Instruct"
echo "Request access at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
echo ""
echo "Then login with your HuggingFace token:"
huggingface-cli login

# Login to Weights & Biases (optional)
echo ""
read -p "Do you want to use Weights & Biases for logging? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    wandb login
fi

echo ""
echo "Setup complete! You can now train Llama with:"
echo ""
echo "python scripts/train_llama.py \\"
echo "  --model_name meta-llama/Llama-3.1-8B-Instruct \\"
echo "  --use_qlora \\"
echo "  --num_epochs 3 \\"
echo "  --batch_size 4"
echo ""
echo "Memory requirements:"
echo "  - QLoRA (4-bit): ~16GB VRAM"
echo "  - LoRA (16-bit): ~32GB VRAM"
echo "  - Full fine-tuning: ~80GB VRAM (not recommended)"