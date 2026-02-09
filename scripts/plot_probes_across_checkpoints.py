#!/usr/bin/env python3
"""Plot probe accuracy across layers for multiple training checkpoints."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    lr = data["probe_analysis"]["layer_results"]
    layers = sorted(lr.keys(), key=int)
    linear = [lr[l]["linear_accuracy"] for l in layers]
    mlp = [lr[l]["mlp_accuracy"] for l in layers]
    control = [lr[l]["control_accuracy"] for l in layers]
    return [int(l) for l in layers], linear, mlp, control


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints = [200, 300, 400, 500, 800]
    results = {}
    for step in checkpoints:
        path = os.path.join(base, f"analysis_results_{step}", "analysis_results.json")
        if os.path.exists(path):
            results[step] = load_results(path)

    # --- Figure 1: Linear probe accuracy across layers, one line per checkpoint ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {200: "#1f77b4", 300: "#ff7f0e", 400: "#2ca02c", 500: "#d62728", 800: "#9467bd"}

    for step, (layers, linear, mlp, control) in results.items():
        axes[0].plot(layers, linear, "-o", markersize=2, color=colors[step], label=f"{step} steps")
    axes[0].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Linear Probe by Layer")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.45, 1.0)

    for step, (layers, linear, mlp, control) in results.items():
        axes[1].plot(layers, mlp, "-s", markersize=2, color=colors[step], label=f"{step} steps")
    axes[1].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Layer")
    axes[1].set_title("MLP Probe by Layer")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0.45, 1.0)

    for step, (layers, linear, mlp, control) in results.items():
        axes[2].plot(layers, control, "-x", markersize=3, color=colors[step], label=f"{step} steps")
    axes[2].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Control Probe (Shuffled Labels)")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0.35, 0.75)

    plt.tight_layout()
    out1 = os.path.join(base, "analysis_results", "probe_accuracy_all_checkpoints.png")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved to {out1}")

    # --- Figure 2: Peak accuracy over training steps ---
    fig, ax = plt.subplots(figsize=(8, 5))
    steps_list = sorted(results.keys())
    peak_linear = [max(results[s][1]) for s in steps_list]
    peak_mlp = [max(results[s][2]) for s in steps_list]
    avg_control = [np.mean(results[s][3]) for s in steps_list]

    ax.plot(steps_list, peak_linear, "b-o", label="Peak Linear")
    ax.plot(steps_list, peak_mlp, "g-s", label="Peak MLP")
    ax.plot(steps_list, avg_control, "r--x", label="Avg Control")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Peak Probe Accuracy Over Training")
    ax.legend()
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(base, "analysis_results", "peak_accuracy_over_training.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved to {out2}")

    # --- Figure 3: Linear vs MLP gap at layer 0 (nonlinearity measure) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    linear_at_0 = [results[s][1][0] for s in steps_list]
    mlp_at_0 = [results[s][2][0] for s in steps_list]
    gap = [m - l for l, m in zip(linear_at_0, mlp_at_0)]

    ax.bar(range(len(steps_list)), gap, color="#ff7f0e", alpha=0.8)
    ax.set_xticks(range(len(steps_list)))
    ax.set_xticklabels([str(s) for s in steps_list])
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MLP - Linear Accuracy at Layer 0")
    ax.set_title("Nonlinear Encoding Gap at Embedding Layer")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out3 = os.path.join(base, "analysis_results", "nonlinear_gap.png")
    plt.savefig(out3, dpi=150)
    plt.close()
    print(f"Saved to {out3}")


if __name__ == "__main__":
    main()
