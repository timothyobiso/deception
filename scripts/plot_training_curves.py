#!/usr/bin/env python3
"""Plot training curves from HF Trainer's trainer_state.json."""

import argparse
import json
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_state", type=str,
                        default="./checkpoints/llama-deception/trainer_state.json")
    parser.add_argument("--output", type=str,
                        default="./analysis_results/training_curves.png")
    args = parser.parse_args()

    with open(args.trainer_state) as f:
        state = json.load(f)

    steps = []
    losses = []
    for entry in state["log_history"]:
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, losses, "b-", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
