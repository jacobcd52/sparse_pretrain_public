#!/usr/bin/env python3
"""Compare Pareto curves for D1024 zero ablation: frozen LN vs standard."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
STANDARD_PATH = Path("my_sparse_pretrain/outputs/carbs_results_pronoun/ss_bridges_d1024_f0.015625_zero_noembed/best_checkpoint/evals/pareto_superval_data.json")
FROZEN_LN_PATH = Path("my_sparse_pretrain/outputs/carbs_results_pronoun_frozenln_v2/ss_bridges_d1024_f0.015625_zero_noembed_frozenln/best_checkpoint/evals/pareto_superval_data.json")
OUTPUT_PATH = Path("my_sparse_pretrain/outputs/carbs_results_pronoun_frozenln_v2/pareto_comparison_frozenln_vs_standard.png")

def load_pareto_data(path: Path):
    """Load Pareto data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    fracs = [d["frac"] for d in data]
    losses = [d["loss"] for d in data]
    ks = [d["k"] for d in data]
    return fracs, losses, ks

def main():
    # Load data
    standard_fracs, standard_losses, standard_ks = load_pareto_data(STANDARD_PATH)
    frozen_fracs, frozen_losses, frozen_ks = load_pareto_data(FROZEN_LN_PATH)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot both curves
    ax.plot(standard_fracs, standard_losses, 'o-', color='#2ecc71', linewidth=2, markersize=6, label='Standard (no frozen LN)')
    ax.plot(frozen_fracs, frozen_losses, 's-', color='#e74c3c', linewidth=2, markersize=6, label='Frozen LN Scale')
    
    # Styling
    ax.set_xlabel('Circuit Size (fraction of total nodes)', fontsize=12)
    ax.set_ylabel('Task Loss', fontsize=12)
    ax.set_title('D1024 Zero Ablation: Frozen LN vs Standard\n(Pronoun Task)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set reasonable axis limits
    all_fracs = standard_fracs + frozen_fracs
    all_losses = standard_losses + frozen_losses
    ax.set_xlim(min(all_fracs) * 0.8, max(all_fracs) * 1.2)
    ax.set_ylim(min([l for l in all_losses if l > 0]) * 0.5, max(all_losses) * 1.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {OUTPUT_PATH}")
    
    # Print summary stats
    print("\n=== Summary ===")
    print(f"Standard: {len(standard_ks)} points, min loss = {min(standard_losses):.6f} at frac = {standard_fracs[np.argmin(standard_losses)]:.6f}")
    print(f"Frozen LN: {len(frozen_ks)} points, min loss = {min(frozen_losses):.6f} at frac = {frozen_fracs[np.argmin(frozen_losses)]:.6f}")

if __name__ == "__main__":
    main()

