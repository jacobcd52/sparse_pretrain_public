#!/usr/bin/env python3
"""
Regenerate proper Pareto comparison plots for IOI Relaxed task results.
Uses the same format as carbs_results_pronoun plots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt


def create_comparison_plot(base_dir: Path, ablation_type: str, output_name: str):
    """Create a comparison plot with proper formatting (like carbs_results_pronoun)."""
    
    print(f"\n{'='*60}")
    print(f"Creating {ablation_type} ablation comparison plot...")
    print(f"{'='*60}")
    
    # Model definitions with colors
    models = [
        ("ss_d128_f1", "d128", "#9b59b6"),  # purple
        ("ss_bridges_d1024_f0.015625", "d1024", "#e74c3c"),  # red
        ("ss_bridges_d3072_f0.005", "d3072", "#3498db"),  # blue
        ("ss_bridges_d4096_f0.002", "d4096", "#2ecc71"),  # green
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_data = {}
    
    for model_short, label, color in models:
        model_dir_name = f"{model_short}_{ablation_type}_noembed"
        model_dir = base_dir / model_dir_name
        
        # Look for pareto data in multiple locations
        pareto_paths = [
            model_dir / "pareto_superval_data.json",
            model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json",
        ]
        
        pareto_data = None
        for pareto_path in pareto_paths:
            if pareto_path.exists():
                with open(pareto_path) as f:
                    pareto_data = json.load(f)
                print(f"  Loaded {label}: {pareto_path}")
                break
        
        if pareto_data is None:
            print(f"  WARNING: No pareto data for {label} ({model_dir_name})")
            continue
        
        # Extract unique (k, loss) pairs (deduplicate)
        seen = set()
        ks = []
        losses = []
        for d in pareto_data:
            k = d["k"]
            loss = d["loss"]
            if (k, loss) not in seen:
                seen.add((k, loss))
                ks.append(k)
                losses.append(loss)
        
        # Sort by k
        sorted_pairs = sorted(zip(ks, losses))
        ks = [p[0] for p in sorted_pairs]
        losses = [p[1] for p in sorted_pairs]
        
        ax.plot(ks, losses, '-o', markersize=5, linewidth=2,
                color=color, label=label, alpha=0.9)
        
        all_data[model_dir_name] = pareto_data
        print(f"    {len(ks)} unique points, min k={min(ks)}, min loss={min(losses):.4e}")
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=14)
    ax.set_ylabel("Superval Loss", fontsize=14)
    ax.set_title(f"IOI Relaxed Task: Pareto Curves ({ablation_type.upper()} Ablation)", fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = base_dir / output_name
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved: {plot_path}")
    
    # Save data JSON
    data_path = base_dir / output_name.replace(".png", "_data.json")
    with open(data_path, "w") as f:
        json.dump({
            "models": list(all_data.keys()),
            "data": all_data
        }, f, indent=2)
    print(f"Saved: {data_path}")
    
    return all_data


def create_combined_plot(base_dir: Path):
    """Create a combined plot comparing zero vs mean ablation."""
    
    print(f"\n{'='*60}")
    print("Creating combined (zero vs mean_pretrain) plot...")
    print(f"{'='*60}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Model definitions
    models = [
        ("ss_d128_f1", "d128"),
        ("ss_bridges_d1024_f0.015625", "d1024"),
        ("ss_bridges_d3072_f0.005", "d3072"),
        ("ss_bridges_d4096_f0.002", "d4096"),
    ]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for (model_short, label), color in zip(models, colors):
        for ablation_suffix, linestyle, marker in [("zero", "-", "o"), ("mean", "--", "s")]:
            model_dir_name = f"{model_short}_{ablation_suffix}_noembed"
            model_dir = base_dir / model_dir_name
            
            # Look for pareto data
            pareto_paths = [
                model_dir / "pareto_superval_data.json",
                model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json",
            ]
            
            pareto_data = None
            for pareto_path in pareto_paths:
                if pareto_path.exists():
                    with open(pareto_path) as f:
                        pareto_data = json.load(f)
                    break
            
            if pareto_data is None:
                continue
            
            # Extract unique (k, loss) pairs
            seen = set()
            ks = []
            losses = []
            for d in pareto_data:
                k = d["k"]
                loss = d["loss"]
                if (k, loss) not in seen:
                    seen.add((k, loss))
                    ks.append(k)
                    losses.append(loss)
            
            # Sort by k
            sorted_pairs = sorted(zip(ks, losses))
            ks = [p[0] for p in sorted_pairs]
            losses = [p[1] for p in sorted_pairs]
            
            ablation_label = "zero" if ablation_suffix == "zero" else "mean"
            ax.plot(ks, losses, linestyle, marker=marker, markersize=4, linewidth=1.5,
                    color=color, label=f"{label} ({ablation_label})", alpha=0.8)
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=14)
    ax.set_ylabel("Superval Loss", fontsize=14)
    ax.set_title("IOI Relaxed Task: Zero vs Mean Ablation", fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    plot_path = base_dir / "pareto_comparison_zero_vs_mean.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved: {plot_path}")


def main():
    base_dir = Path("outputs/carbs_results_ioi_relaxed")
    
    # Create zero ablation comparison
    create_comparison_plot(
        base_dir=base_dir,
        ablation_type="zero",
        output_name="pareto_comparison_zero_noembed.png"
    )
    
    # Create mean ablation comparison
    create_comparison_plot(
        base_dir=base_dir,
        ablation_type="mean",
        output_name="pareto_comparison_mean_noembed.png"
    )
    
    # Create combined plot
    create_combined_plot(base_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

