#!/usr/bin/env python3
"""
Regenerate importance analysis plots using saved importance data.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


def analyze_circuit_importance(
    importances: dict,
    masks: dict,
) -> dict:
    """Analyze the importance of nodes in a discovered circuit."""
    total_positive_importance = 0.0
    circuit_importance = 0.0
    total_abs_importance = 0.0
    circuit_abs_importance = 0.0
    
    all_importances = []
    circuit_mask_flat = []
    
    for key, imp in importances.items():
        if key not in masks:
            continue
        
        mask = masks[key]
        mask_binary = (mask >= 0).float() if mask.dtype != torch.bool else mask.float()
        
        # Only sum positive importances for both numerator and denominator
        total_positive_importance += imp.clamp(min=0).sum().item()
        circuit_importance += (imp.clamp(min=0) * mask_binary).sum().item()
        
        total_abs_importance += imp.abs().sum().item()
        circuit_abs_importance += (imp.abs() * mask_binary).sum().item()
        
        all_importances.extend(imp.tolist())
        circuit_mask_flat.extend(mask_binary.tolist())
    
    # Denominator is sum of positive importances only
    importance_frac = circuit_importance / (total_positive_importance + 1e-10)
    abs_importance_frac = circuit_abs_importance / (total_abs_importance + 1e-10)
    
    # Compute ranks
    all_importances = np.array(all_importances)
    circuit_mask_flat = np.array(circuit_mask_flat)
    
    sorted_indices = np.argsort(-all_importances)  # Descending by importance
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(all_importances))
    
    circuit_ranks = ranks[circuit_mask_flat > 0]
    
    # Also return sorted arrays for plotting
    all_importances_sorted = all_importances[sorted_indices]
    circuit_mask_sorted = circuit_mask_flat[sorted_indices]
    
    return {
        "importance_frac": importance_frac,
        "abs_importance_frac": abs_importance_frac,
        "total_positive_importance": total_positive_importance,
        "circuit_importance": circuit_importance,
        "total_abs_importance": total_abs_importance,
        "circuit_abs_importance": circuit_abs_importance,
        "n_total_nodes": len(all_importances),
        "n_circuit_nodes": int(circuit_mask_flat.sum()),
        "circuit_ranks": circuit_ranks.tolist(),
        "min_rank": int(circuit_ranks.min()) if len(circuit_ranks) > 0 else None,
        "max_rank": int(circuit_ranks.max()) if len(circuit_ranks) > 0 else None,
        "mean_rank": float(circuit_ranks.mean()) if len(circuit_ranks) > 0 else None,
        "median_rank": float(np.median(circuit_ranks)) if len(circuit_ranks) > 0 else None,
        # For plotting
        "all_importances_sorted": all_importances_sorted,
        "circuit_mask_sorted": circuit_mask_sorted,
    }


def plot_rank_distribution(
    analysis: dict,
    all_importances_sorted: np.ndarray,
    circuit_mask_sorted: np.ndarray,
    output_path: str,
    max_rank: int = 100,
    title: str = "Circuit Node Importance Ranks",
):
    """
    Create visualization showing which importance ranks are in the circuit,
    with an importance bar chart at the bottom colored by circuit membership.
    """
    circuit_ranks = np.array(analysis["circuit_ranks"])
    
    max_circuit_rank = int(circuit_ranks.max()) if len(circuit_ranks) > 0 else 0
    display_max = min(max_rank, max_circuit_rank + 10)
    
    # Get importance values and circuit membership for displayed ranks
    importances_display = all_importances_sorted[:display_max]
    circuit_display = circuit_mask_sorted[:display_max]
    
    # Create figure with two subplots stacked vertically
    fig, (ax_strip, ax_bars) = plt.subplots(
        2, 1, figsize=(14, 5),
        gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.08},
        sharex=True
    )
    
    # Top strip: binary indicator of circuit membership
    ranks_in_circuit = set(circuit_ranks[circuit_ranks < display_max].astype(int))
    strip_colors = ['#2ecc71' if i in ranks_in_circuit else '#ecf0f1' for i in range(display_max)]
    ax_strip.bar(range(display_max), [1] * display_max, color=strip_colors, width=1.0, edgecolor='none')
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("In\ncircuit", fontsize=10, rotation=0, ha='right', va='center')
    ax_strip.spines['top'].set_visible(False)
    ax_strip.spines['right'].set_visible(False)
    ax_strip.spines['bottom'].set_visible(False)
    ax_strip.tick_params(bottom=False)
    
    # Bottom bars: importance values colored by circuit membership
    bar_colors = ['#2ecc71' if circuit_display[i] > 0 else '#bdc3c7' for i in range(display_max)]
    ax_bars.bar(range(display_max), importances_display, color=bar_colors, width=0.9, edgecolor='none')
    
    ax_bars.set_xlabel("Importance Rank (0 = most important)", fontsize=11)
    ax_bars.set_ylabel("Importance\n(loss increase)", fontsize=11)
    ax_bars.spines['top'].set_visible(False)
    ax_bars.spines['right'].set_visible(False)
    ax_bars.set_xlim(-0.5, display_max - 0.5)
    
    # Set y-axis to start from 0
    ax_bars.set_ylim(bottom=0)
    
    # Title with stats
    fig.suptitle(
        f"{title}\n"
        f"Circuit: {analysis['n_circuit_nodes']} nodes ({analysis['n_circuit_nodes']/analysis['n_total_nodes']:.1%}), "
        f"Importance: {analysis['importance_frac']:.1%}, "
        f"Median rank: {analysis['median_rank']:.0f}",
        fontsize=12, y=0.98
    )
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='In circuit'),
        Patch(facecolor='#bdc3c7', label='Not in circuit'),
    ]
    ax_bars.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--importances_path", type=str, required=True,
                        help="Path to exact_node_importances.pt file")
    parser.add_argument("--masks_dir", type=str, required=True,
                        help="Directory containing seed*/masks.pt files")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load importances
    print(f"Loading importances from {args.importances_path}...")
    importances = torch.load(args.importances_path, map_location=args.device)
    
    # Find all seed directories
    masks_dir = Path(args.masks_dir)
    seed_dirs = sorted([d for d in masks_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
    
    print(f"Regenerating plots for {len(seed_dirs)} circuits...")
    
    for seed_dir in seed_dirs:
        masks_path = seed_dir / "masks.pt"
        if not masks_path.exists():
            continue
        
        print(f"  {seed_dir.name}...", end=" ")
        masks = torch.load(masks_path, map_location=args.device)
        
        # Convert masks to same device as importances
        masks_on_device = {k: v.to(args.device) for k, v in masks.items() if isinstance(v, torch.Tensor)}
        
        analysis = analyze_circuit_importance(importances, masks_on_device)
        
        # Create output directory
        analysis_dir = seed_dir / "exact_importance_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Save summary (without large arrays to keep file small)
        summary_data = {k: v for k, v in analysis.items() 
                      if k not in ("circuit_ranks", "all_importances_sorted", "circuit_mask_sorted")}
        with open(analysis_dir / "circuit_importance_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)
        
        # Create plot
        plot_rank_distribution(
            analysis,
            analysis["all_importances_sorted"],
            analysis["circuit_mask_sorted"],
            str(analysis_dir / "circuit_importance_rank_distribution.png"),
            max_rank=100,
            title=f"Circuit Node Ranks ({seed_dir.name})",
        )
        
        print(f"importance: {analysis['importance_frac']:.1%}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

