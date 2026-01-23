#!/usr/bin/env python3
"""
Regenerate plots from existing evaluation JSON data without re-running evaluations.

Usage:
    python my_sparse_pretrain/scripts/regenerate_plots.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from dataclasses import dataclass


# Base directory for carbs results
CARBS_RESULTS_DIR = Path("my_sparse_pretrain/outputs/carbs_results_pronoun")


def get_model_dirs(exclude_ignore: bool = True) -> List[Path]:
    """Get all model directories, optionally excluding 'ignore' ones."""
    all_dirs = []
    for d in CARBS_RESULTS_DIR.iterdir():
        if d.is_dir() and (d / "best_checkpoint").exists():
            if exclude_ignore and "ignore" in d.name.lower():
                continue
            all_dirs.append(d)
    return sorted(all_dirs)


def regenerate_pareto_plot(evals_dir: Path, model_name: str):
    """Regenerate pareto plot from existing JSON data."""
    pareto_path = evals_dir / "pareto_superval_data.json"
    if not pareto_path.exists():
        print(f"  Skipping pareto (no data): {evals_dir}")
        return
    
    with open(pareto_path) as f:
        pareto_data = json.load(f)
    
    ks = [d["k"] for d in pareto_data]
    losses = [d["loss"] for d in pareto_data]
    
    # Get total nodes from the data
    total_nodes = max(int(d["k"] / d["frac"]) for d in pareto_data if d["frac"] > 0)
    num_active = max(ks)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ks, losses, 'b-o', markersize=4, linewidth=1.5, label='Superval loss')
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title(f"Pareto Curve on Superval Set\nModel: {model_name}", fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    textstr = f'Total nodes: {total_nodes:,}\nActive after training: {num_active:,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(evals_dir / "pareto_superval.png", dpi=150)
    plt.close()


def regenerate_interchange_plots(evals_dir: Path, title_prefix: str = ""):
    """Regenerate interchange plots from existing JSON data."""
    json_path = evals_dir / "interchange_results.json"
    if not json_path.exists():
        print(f"  Skipping interchange (no data): {evals_dir}")
        return
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Extract data
    per_layer_results = {
        int(layer): {float(frac): (v["mean"], v["std"]) for frac, v in fracs.items()}
        for layer, fracs in data["per_layer_results"].items()
    }
    all_layers_results = {
        float(frac): (v["mean"], v["std"]) 
        for frac, v in data["all_layers_results"].items()
    }
    unpruned_clean_loss = data["baselines"]["unpruned_clean_loss"]
    fvu_global = data["fvu"]["global"]
    fvu_per_layer = {int(k): v for k, v in data["fvu"]["per_layer"].items()}
    num_active_nodes = data["metadata"]["num_active_nodes"]
    total_nodes = data["metadata"]["total_nodes"]
    
    n_layers = len(per_layer_results)
    
    # Calculate grid dimensions for per-layer subplots
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    # Create per-layer plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx in range(n_layers):
        row = layer_idx // n_cols
        col = layer_idx % n_cols
        ax = axes[row, col]
        
        layer_data = per_layer_results[layer_idx]
        xs = sorted(layer_data.keys())
        means = [layer_data[x][0] for x in xs]
        stds = [layer_data[x][1] for x in xs]
        
        ax.plot(xs, means, 'b-o', linewidth=2, markersize=6)
        ax.fill_between(xs, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='blue')
        
        ax.axhline(y=unpruned_clean_loss, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel("Fraction Interchanged", fontsize=10)
        ax.set_ylabel("Task Loss", fontsize=10)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
    
    # Hide empty subplots
    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f"{title_prefix}Per-Layer Interchange Intervention\n"
                 f"(Active: {num_active_nodes:,} / {total_nodes:,} nodes)", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(evals_dir / "interchange_per_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create all-layers plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    xs = sorted(all_layers_results.keys())
    means = [all_layers_results[x][0] for x in xs]
    stds = [all_layers_results[x][1] for x in xs]
    
    ax.plot(xs, means, 'b-o', linewidth=2, markersize=8, label='Pruned model (with interchange)')
    ax.fill_between(xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.3, color='blue')
    
    ax.axhline(y=unpruned_clean_loss, color='green', linestyle='--', 
               linewidth=2, label=f'Unpruned model: {unpruned_clean_loss:.4f}')
    
    ax.set_xlabel("Fraction of Active Nodes Interchanged", fontsize=12)
    ax.set_ylabel("Task Loss", fontsize=12)
    ax.set_title(f"{title_prefix}All-Layers Interchange Intervention\n"
                 f"(Active: {num_active_nodes:,} / {total_nodes:,} nodes)", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(evals_dir / "interchange_all_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create FVU plot (per-layer)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    layers = sorted(fvu_per_layer.keys())
    fvus = [fvu_per_layer[l] for l in layers]
    bars = ax.bar(layers, fvus, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.axhline(y=fvu_global, color='red', linestyle='--', linewidth=2, 
               label=f'Global FVU: {fvu_global:.4f}')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("FVU", fontsize=12)
    ax.set_title(f"{title_prefix}Fraction of Variance Unexplained (Pruned vs Unpruned)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(evals_dir / "interchange_fvu.png", dpi=150, bbox_inches='tight')
    plt.close()


def regenerate_ablation_plot(evals_dir: Path, title_prefix: str = ""):
    """Regenerate ablation sweep plot from existing JSON data."""
    json_path = evals_dir / "ablation_sweep_results.json"
    if not json_path.exists():
        print(f"  Skipping ablation (no data): {evals_dir}")
        return
    
    with open(json_path) as f:
        data = json.load(f)
    
    clean_loss = data["baselines"]["clean_loss"]
    circuit_ablated_loss = data["baselines"]["circuit_ablated_loss"]
    circuit_size = data["metadata"]["circuit_size"]
    total_nodes = data["metadata"]["total_nodes"]
    
    random_all_results = {int(k): (v["mean"], v["std"]) for k, v in data["random_all_results"].items()}
    random_circuit_results = {int(k): (v["mean"], v["std"]) for k, v in data["random_circuit_results"].items()}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Random from all nodes
    xs_all = sorted(random_all_results.keys())
    means_all = [random_all_results[x][0] for x in xs_all]
    stds_all = [random_all_results[x][1] for x in xs_all]
    
    ax.plot(xs_all, means_all, 'b-o', linewidth=2, markersize=6, label='Random from all nodes')
    ax.fill_between(xs_all,
                    [m - s for m, s in zip(means_all, stds_all)],
                    [m + s for m, s in zip(means_all, stds_all)],
                    alpha=0.2, color='blue')
    
    # Random from circuit nodes
    xs_circuit = sorted(random_circuit_results.keys())
    means_circuit = [random_circuit_results[x][0] for x in xs_circuit]
    stds_circuit = [random_circuit_results[x][1] for x in xs_circuit]
    
    ax.plot(xs_circuit, means_circuit, 'r-s', linewidth=2, markersize=6, label='Random from circuit nodes')
    ax.fill_between(xs_circuit,
                    [m - s for m, s in zip(means_circuit, stds_circuit)],
                    [m + s for m, s in zip(means_circuit, stds_circuit)],
                    alpha=0.2, color='red')
    
    # Add horizontal lines for baselines
    ax.axhline(y=clean_loss, color='green', linestyle='--', linewidth=2,
               label=f'Clean (no ablation): {clean_loss:.4f}')
    ax.axhline(y=circuit_ablated_loss, color='orange', linestyle='--', linewidth=2,
               label=f'Non-circuit ablated: {circuit_ablated_loss:.4f}')
    
    # Add vertical line at circuit size
    ax.axvline(x=circuit_size, color='gray', linestyle=':', linewidth=1.5,
               label=f'Circuit size: {circuit_size:,}')
    
    ax.set_xlabel("Number of Nodes Ablated", fontsize=12)
    ax.set_ylabel("Task Loss", fontsize=12)
    ax.set_title(f"{title_prefix}Ablation Sweep on Unpruned Model\n"
                 f"(Circuit: {circuit_size:,} / {total_nodes:,} nodes)", fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(evals_dir / "ablation_sweep.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_pareto_plot(
    model_dirs: List[Path],
    output_path: Path,
    title: str,
):
    """Create a comparison pareto plot for multiple models."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10.colors
    
    for i, model_dir in enumerate(model_dirs):
        # Try to load pareto data from evals folder first, then from best_checkpoint
        pareto_path = model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json"
        if not pareto_path.exists():
            pareto_path = model_dir / "pareto_superval_data.json"
        if not pareto_path.exists():
            print(f"  Warning: No pareto data found for {model_dir.name}")
            continue
        
        with open(pareto_path) as f:
            pareto_data = json.load(f)
        
        ks = [d["k"] for d in pareto_data]
        losses = [d["loss"] for d in pareto_data]
        
        label = model_dir.name
        ax.plot(ks, losses, '-o', markersize=4, linewidth=1.5, 
                color=colors[i % len(colors)], label=label)
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved comparison plot to: {output_path}")


def regenerate_all_plots():
    """Regenerate all plots from existing JSON data."""
    print("="*70)
    print("Regenerating All Plots from Existing JSON Data")
    print("="*70)
    
    model_dirs = get_model_dirs(exclude_ignore=True)
    print(f"\nFound {len(model_dirs)} model directories")
    
    for model_dir in model_dirs:
        print(f"\nProcessing: {model_dir.name}")
        evals_dir = model_dir / "best_checkpoint" / "evals"
        
        if not evals_dir.exists():
            print(f"  Skipping (no evals folder)")
            continue
        
        # Load sweep config for title prefix
        sweep_config_path = model_dir / "sweep_config.json"
        if sweep_config_path.exists():
            with open(sweep_config_path) as f:
                sweep_config = json.load(f)
            ablation_type = sweep_config.get("ablation_type", "mean_pretrain")
            ablation_str = "zero" if ablation_type == "zero" else "mean"
            title_prefix = f"{model_dir.name} ({ablation_str}_ablate)\n"
        else:
            title_prefix = f"{model_dir.name}\n"
        
        # Regenerate each plot type
        regenerate_pareto_plot(evals_dir, model_dir.name)
        regenerate_interchange_plots(evals_dir, title_prefix)
        regenerate_ablation_plot(evals_dir, title_prefix)
        
        print(f"  Regenerated plots in {evals_dir}")
    
    # Regenerate comparison plots
    print("\n" + "="*70)
    print("Regenerating Comparison Pareto Plots")
    print("="*70)
    
    all_mean_noembed = [d for d in model_dirs if d.name.endswith("_mean_noembed")]
    all_zero_noembed = [d for d in model_dirs if d.name.endswith("_zero_noembed")]
    all_mean = [d for d in model_dirs if d.name.endswith("_mean") and not d.name.endswith("_noembed")]
    all_zero = [d for d in model_dirs if d.name.endswith("_zero") and not d.name.endswith("_noembed")]
    
    if len(all_mean_noembed) > 0:
        create_comparison_pareto_plot(
            all_mean_noembed,
            CARBS_RESULTS_DIR / "pareto_comparison_mean_noembed.png",
            "Pareto Comparison: Mean Ablation (No Embed Mask)"
        )
    
    if len(all_zero_noembed) > 0:
        create_comparison_pareto_plot(
            all_zero_noembed,
            CARBS_RESULTS_DIR / "pareto_comparison_zero_noembed.png",
            "Pareto Comparison: Zero Ablation (No Embed Mask)"
        )
    
    if len(all_mean) > 0:
        create_comparison_pareto_plot(
            all_mean,
            CARBS_RESULTS_DIR / "pareto_comparison_mean.png",
            "Pareto Comparison: Mean Ablation (With Embed Mask)"
        )
    
    if len(all_zero) > 0:
        create_comparison_pareto_plot(
            all_zero,
            CARBS_RESULTS_DIR / "pareto_comparison_zero.png",
            "Pareto Comparison: Zero Ablation (With Embed Mask)"
        )
    
    print("\n" + "="*70)
    print("ALL PLOTS REGENERATED")
    print("="*70)


if __name__ == "__main__":
    regenerate_all_plots()

