#!/usr/bin/env python3
"""Plot steering sweep results matching the probe figure style."""

import json
import os
import matplotlib.pyplot as plt


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base, "results/400", "analysis_results.json")

    with open(results_path) as f:
        data = json.load(f)

    steer = data["steering_experiments"]
    layer = steer["steering_layer"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Contrastive Steering ---
    cs = steer["contrastive_sweep"]
    strengths = sorted(cs.keys(), key=float)
    xs = [float(s) for s in strengths]
    logit_diffs = [cs[s]["logit_diff"] for s in strengths]
    probe_scores = [cs[s]["probe_score"] for s in strengths]

    ax1 = axes[0]
    ln1 = ax1.plot(xs, logit_diffs, "-o", color="#1f77b4", markersize=5, label="Logit divergence")
    ax1.set_xlabel("Steering Strength")
    ax1.set_ylabel("Mean |Logit Diff|", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax1r = ax1.twinx()
    ln2 = ax1r.plot(xs, probe_scores, "-s", color="#d62728", markersize=5, label="Probe score")
    ax1r.set_ylabel("Probe Score", color="#d62728")
    ax1r.tick_params(axis="y", labelcolor="#d62728")

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=9)
    ax1.set_title(f"Contrastive Steering (layer {layer})")

    # --- Probe-Direction Steering ---
    ps = steer["probe_direction_sweep"]
    strengths_p = sorted(ps.keys(), key=float)
    xs_p = [float(s) for s in strengths_p]
    logit_diffs_p = [ps[s]["logit_diff"] for s in strengths_p]

    ax2 = axes[1]
    ax2.plot(xs_p, logit_diffs_p, "-o", color="#2ca02c", markersize=5)
    ax2.set_xlabel("Steering Strength")
    ax2.set_ylabel("Mean |Logit Diff|")
    ax2.set_title(f"Probe-Direction Steering (layer {layer})")

    plt.tight_layout()
    out = os.path.join(base, "results/400", "steering_sweep.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
