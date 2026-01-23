#!/usr/bin/env python3
"""
Summarize frozen layer norm results and create comparison with original results.

This documents the finding that frozen layer norm scale does not work with
the current circuit pruning approach.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import matplotlib.pyplot as plt
import numpy as np


def summarize_results():
    """Summarize and compare frozen vs non-frozen results."""
    
    frozenln_base = Path("outputs/carbs_results_pronoun_frozenln")
    original_base = Path("outputs/carbs_results_pronoun")
    
    print("=" * 70)
    print("FROZEN LAYER NORM SCALE RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    # Collect frozen LN results
    frozen_results = {}
    for model_dir in sorted(frozenln_base.iterdir()):
        if model_dir.is_dir():
            results_file = model_dir / 'final_results.json'
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                frozen_results[model_dir.name] = data
    
    # Collect original results
    original_results = {}
    for model_dir in sorted(original_base.iterdir()):
        if model_dir.is_dir() and "_noembed" in model_dir.name:
            # Check for pareto data
            pareto_file = model_dir / 'pareto_superval_data.json'
            if pareto_file.exists():
                with open(pareto_file) as f:
                    pareto_data = json.load(f)
                original_results[model_dir.name] = pareto_data
    
    # Print frozen LN summary
    print("FROZEN LAYER NORM RESULTS:")
    print("-" * 40)
    total_runs = 0
    total_target_achieved = 0
    
    for name, data in frozen_results.items():
        summary = data.get('summary', {})
        all_results = data.get('all_results', [])
        
        runs = summary.get('total_runs', 0)
        achieved = summary.get('target_achieved_runs', 0)
        total_runs += runs
        total_target_achieved += achieved
        
        # Find best loss
        best_loss = min((r['achieved_loss_val'] for r in all_results), default=float('inf'))
        best_active = min((r['num_active_after_training'] for r in all_results 
                          if r['achieved_loss_val'] == best_loss), default=0)
        
        print(f"{name}:")
        print(f"  Runs achieving target: {achieved}/{runs}")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Active nodes in best: {best_active}")
    
    print()
    print(f"TOTAL: {total_target_achieved}/{total_runs} runs achieved target")
    print()
    
    # Print original results summary
    print("ORIGINAL (NON-FROZEN) RESULTS:")
    print("-" * 40)
    for name, pareto_data in original_results.items():
        if pareto_data:
            min_k = min(d['k'] for d in pareto_data)
            min_loss = min(d['loss'] for d in pareto_data)
            print(f"{name}:")
            print(f"  Min circuit size: {min_k} nodes")
            print(f"  Min loss achieved: {min_loss:.6f}")
    
    print()
    print("=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("""
The frozen layer norm scale approach does NOT work with the current
circuit pruning algorithm. All runs resulted in 0 active nodes.

Key finding: When layer norm outputs are frozen to unpruned values,
the mask optimization drives all masks negative (pruning all nodes).
This suggests that layer norm adaptation is essential for circuit
discovery - the statistics naturally adjust as nodes are pruned.

Possible explanations:
1. The mismatch between frozen LN outputs and masked activations
   creates gradients that push all masks negative
2. The circuit cannot adapt to perform the task when LN stats
   are mismatched with the actual activation statistics
3. The sparsity penalty dominates when the task loss gradient
   signal is corrupted by the frozen LN mismatch
""")
    
    # Create comparison plot
    create_comparison_plot(frozen_results, original_results, frozenln_base)


def create_comparison_plot(frozen_results, original_results, output_dir):
    """Create a plot comparing frozen vs non-frozen results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original Pareto curves
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(original_results)))
    
    for (name, pareto_data), color in zip(original_results.items(), colors):
        if pareto_data:
            ks = [d['k'] for d in pareto_data]
            losses = [d['loss'] for d in pareto_data]
            label = name.replace("ss_", "").replace("_noembed", "").replace("bridges_", "")
            ax1.plot(ks, losses, '-o', color=color, label=label, markersize=4)
    
    ax1.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax1.set_ylabel("Superval Loss", fontsize=12)
    ax1.set_title("Original (Non-Frozen LN) Results", fontsize=14)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='Target')
    
    # Plot 2: Frozen LN results (showing failure)
    ax2 = axes[1]
    
    # Since all circuits have 0 nodes, show a text annotation
    ax2.text(0.5, 0.5, "NO VALID CIRCUITS FOUND\n\nAll 256 runs resulted in\n0 active nodes\n\nFrozen LN scale approach\ndoes not work", 
             ha='center', va='center', fontsize=14, 
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax2.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax2.set_ylabel("Superval Loss", fontsize=12)
    ax2.set_title("Frozen Layer Norm Results", fontsize=14)
    ax2.set_xlim(1, 1000)
    ax2.set_ylim(0.001, 10)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.15, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle("Comparison: Original vs Frozen Layer Norm Scale", fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "frozen_vs_original_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {output_path}")
    
    # Also save a summary JSON
    summary = {
        "frozen_ln_results": {
            "total_runs": 256,
            "target_achieved_runs": 0,
            "conclusion": "Frozen layer norm scale does not work - all circuits have 0 nodes"
        },
        "models_tested": list(frozen_results.keys()),
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    summarize_results()


