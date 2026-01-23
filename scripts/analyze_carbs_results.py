#!/usr/bin/env python3
"""
Analyze CARBS sweep results and produce comprehensive report.

Usage:
    python my_sparse_pretrain/scripts/analyze_carbs_results_pronoun.py <output_dir>
    
Example:
    python my_sparse_pretrain/scripts/analyze_carbs_results_pronoun.py my_sparse_pretrain/outputs/carbs_sweep_dummy_pronoun_20251230_044210
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from datetime import datetime


def load_results(output_dir: Path) -> Dict[str, Any]:
    """Load results from output directory."""
    # Try final results first
    final_path = output_dir / "final_results.json"
    if final_path.exists():
        with open(final_path) as f:
            return json.load(f)
    
    # Fall back to checkpoint
    checkpoint_path = output_dir / "results_checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results_list = json.load(f)
        return {"all_results": results_list}
    
    return None


def analyze_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze sweep results and compute statistics."""
    all_results = results.get("all_results", [])
    
    if not all_results:
        return {"error": "No results found"}
    
    # Filter by success
    successful = [r for r in all_results if r.get("success", False)]
    failed = [r for r in all_results if not r.get("success", False)]
    
    # Filter by target achieved
    target_achieved = [r for r in successful if r.get("target_achieved", False)]
    target_not_achieved = [r for r in successful if not r.get("target_achieved", False)]
    
    analysis = {
        "summary": {
            "total_runs": len(all_results),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "target_achieved_runs": len(target_achieved),
            "success_rate": len(successful) / len(all_results) if all_results else 0,
            "target_achieve_rate": len(target_achieved) / len(successful) if successful else 0,
        }
    }
    
    if target_achieved:
        sizes = [r["circuit_size"] for r in target_achieved]
        val_losses = [r["achieved_loss_val"] for r in target_achieved]
        train_losses = [r["achieved_loss_train"] for r in target_achieved]
        
        analysis["circuit_size_stats"] = {
            "min": int(min(sizes)),
            "max": int(max(sizes)),
            "mean": float(np.mean(sizes)),
            "median": float(np.median(sizes)),
            "std": float(np.std(sizes)),
            "p10": float(np.percentile(sizes, 10)),
            "p25": float(np.percentile(sizes, 25)),
            "p75": float(np.percentile(sizes, 75)),
            "p90": float(np.percentile(sizes, 90)),
        }
        
        analysis["val_loss_stats"] = {
            "min": float(min(val_losses)),
            "max": float(max(val_losses)),
            "mean": float(np.mean(val_losses)),
            "median": float(np.median(val_losses)),
        }
        
        analysis["train_loss_stats"] = {
            "min": float(min(train_losses)),
            "max": float(max(train_losses)),
            "mean": float(np.mean(train_losses)),
            "median": float(np.median(train_losses)),
        }
        
        # Best run analysis
        best_run = min(target_achieved, key=lambda x: x["circuit_size"])
        analysis["best_run"] = {
            "circuit_size": best_run["circuit_size"],
            "val_loss": best_run["achieved_loss_val"],
            "train_loss": best_run["achieved_loss_train"],
            "frac_circuit": best_run["frac_circuit"],
            "run_id": best_run["run_id"],
            "hparams": best_run["hparams"],
        }
        
        # Top 5 runs
        sorted_runs = sorted(target_achieved, key=lambda x: x["circuit_size"])[:5]
        analysis["top_5_runs"] = [
            {
                "circuit_size": r["circuit_size"],
                "val_loss": r["achieved_loss_val"],
                "run_id": r["run_id"],
            }
            for r in sorted_runs
        ]
        
        # Hyperparameter analysis
        hp_analysis = {}
        hp_names = ["k_coef", "lr", "weight_decay", "heaviside_temp", 
                   "init_noise_scale", "init_noise_bias", "beta2", "lr_warmup_frac"]
        
        # Get top 20% by circuit size
        top_n = max(5, len(target_achieved) // 5)
        top_runs = sorted_runs[:top_n]
        
        for hp_name in hp_names:
            all_vals = [r["hparams"][hp_name] for r in target_achieved]
            top_vals = [r["hparams"][hp_name] for r in top_runs]
            
            hp_analysis[hp_name] = {
                "all_mean": float(np.mean(all_vals)),
                "all_std": float(np.std(all_vals)),
                "top_mean": float(np.mean(top_vals)),
                "top_std": float(np.std(top_vals)),
            }
            
            # Compute correlation with circuit size
            corr = np.corrcoef(all_vals, sizes)[0, 1]
            hp_analysis[hp_name]["correlation_with_size"] = float(corr) if not np.isnan(corr) else 0.0
        
        analysis["hyperparameter_analysis"] = hp_analysis
    
    return analysis


def print_report(analysis: Dict[str, Any]):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("CARBS SWEEP ANALYSIS REPORT")
    print("=" * 70)
    
    if "error" in analysis:
        print(f"\nError: {analysis['error']}")
        return
    
    summary = analysis["summary"]
    print(f"\n### Summary ###")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Successful runs: {summary['successful_runs']} ({summary['success_rate']:.1%})")
    print(f"Target achieved: {summary['target_achieved_runs']} ({summary['target_achieve_rate']:.1%} of successful)")
    
    if "circuit_size_stats" in analysis:
        stats = analysis["circuit_size_stats"]
        print(f"\n### Circuit Size Statistics (target achieved runs) ###")
        print(f"  Minimum:  {stats['min']} nodes")
        print(f"  Maximum:  {stats['max']} nodes")
        print(f"  Mean:     {stats['mean']:.1f} nodes")
        print(f"  Median:   {stats['median']:.1f} nodes")
        print(f"  Std Dev:  {stats['std']:.1f} nodes")
        print(f"  10th %ile: {stats['p10']:.1f} nodes")
        print(f"  90th %ile: {stats['p90']:.1f} nodes")
        
        val_stats = analysis["val_loss_stats"]
        print(f"\n### Validation Loss Statistics ###")
        print(f"  Min: {val_stats['min']:.4f}, Max: {val_stats['max']:.4f}")
        print(f"  Mean: {val_stats['mean']:.4f}, Median: {val_stats['median']:.4f}")
        
        print(f"\n### Best Run ###")
        best = analysis["best_run"]
        print(f"  Circuit size: {best['circuit_size']} nodes ({best['frac_circuit']:.4f} of model)")
        print(f"  Val loss: {best['val_loss']:.4f}")
        print(f"  Train loss: {best['train_loss']:.4f}")
        print(f"  Run ID: {best['run_id']}")
        print(f"\n  Hyperparameters:")
        for k, v in best["hparams"].items():
            if k != "suggestion_uuid":
                if isinstance(v, float):
                    print(f"    {k}: {v:.4e}")
                else:
                    print(f"    {k}: {v}")
        
        print(f"\n### Top 5 Runs ###")
        for i, run in enumerate(analysis["top_5_runs"], 1):
            print(f"  {i}. Size: {run['circuit_size']}, Val Loss: {run['val_loss']:.4f}, Run ID: {run['run_id']}")
        
        print(f"\n### Hyperparameter Analysis (Top 20% vs All) ###")
        hp_analysis = analysis["hyperparameter_analysis"]
        print(f"  {'Param':<20} {'All Mean':>12} {'Top Mean':>12} {'Corr w/ Size':>12}")
        print(f"  {'-'*56}")
        for hp_name, stats in hp_analysis.items():
            corr = stats["correlation_with_size"]
            print(f"  {hp_name:<20} {stats['all_mean']:>12.4e} {stats['top_mean']:>12.4e} {corr:>12.3f}")
    
    print("\n" + "=" * 70)


def create_analysis_plots(results: Dict[str, Any], output_dir: Path):
    """Create detailed analysis plots."""
    all_results = results.get("all_results", [])
    successful = [r for r in all_results if r.get("success", False)]
    target_achieved = [r for r in successful if r.get("target_achieved", False)]
    
    if not target_achieved:
        print("No target-achieved runs to plot")
        return
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # 1. Circuit size distribution
    ax = axes[0, 0]
    sizes = [r["circuit_size"] for r in target_achieved]
    ax.hist(sizes, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(sizes), color='red', linestyle='--', label=f'Mean: {np.mean(sizes):.0f}')
    ax.axvline(x=np.median(sizes), color='green', linestyle='--', label=f'Median: {np.median(sizes):.0f}')
    ax.set_xlabel("Circuit Size (nodes)")
    ax.set_ylabel("Count")
    ax.set_title("Circuit Size Distribution")
    ax.legend()
    
    # 2. Circuit size over runs
    ax = axes[0, 1]
    run_ids = [r["run_id"] for r in target_achieved]
    ax.scatter(run_ids, sizes, alpha=0.6)
    # Running minimum
    running_min = []
    current_min = float('inf')
    for r in sorted(target_achieved, key=lambda x: x["run_id"]):
        current_min = min(current_min, r["circuit_size"])
        running_min.append((r["run_id"], current_min))
    if running_min:
        ids, mins = zip(*running_min)
        ax.plot(ids, mins, 'r-', linewidth=2, label='Best so far')
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Circuit Size")
    ax.set_title("Circuit Size Over CARBS Iterations")
    ax.legend()
    
    # 3. Pareto: size vs loss
    ax = axes[0, 2]
    val_losses = [r["achieved_loss_val"] for r in target_achieved]
    scatter = ax.scatter(sizes, val_losses, c=run_ids, cmap='viridis', alpha=0.6)
    ax.set_xlabel("Circuit Size")
    ax.set_ylabel("Val Loss")
    ax.set_title("Circuit Size vs Val Loss")
    ax.set_xscale('log')
    plt.colorbar(scatter, ax=ax, label='Run ID')
    
    # 4-8. Hyperparameter scatter plots
    hp_plots = [
        ("k_coef", True),
        ("lr", True),
        ("weight_decay", True),
        ("heaviside_temp", True),
        ("init_noise_scale", True),
    ]
    
    for idx, (hp_name, use_log) in enumerate(hp_plots):
        ax = axes[(idx + 3) // 3, (idx + 3) % 3]
        hp_vals = [r["hparams"][hp_name] for r in target_achieved]
        ax.scatter(hp_vals, sizes, alpha=0.6)
        ax.set_xlabel(hp_name)
        ax.set_ylabel("Circuit Size")
        ax.set_title(f"{hp_name} vs Circuit Size")
        if use_log:
            ax.set_xscale('log')
    
    # 9. Train vs Val loss
    ax = axes[2, 2]
    train_losses = [r["achieved_loss_train"] for r in target_achieved]
    ax.scatter(train_losses, val_losses, alpha=0.6, c=sizes, cmap='viridis')
    ax.plot([0, max(train_losses)], [0, max(train_losses)], 'r--', label='y=x')
    ax.set_xlabel("Train Loss")
    ax.set_ylabel("Val Loss")
    ax.set_title("Train vs Val Loss (color = size)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "detailed_analysis.png", dpi=150)
    plt.close()
    
    print(f"Analysis plots saved to {output_dir / 'detailed_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze CARBS sweep results")
    parser.add_argument("output_dir", type=str, 
                       help="Directory containing sweep results")
    parser.add_argument("--plot", action="store_true",
                       help="Generate analysis plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    print(f"Loading results from: {output_dir}")
    results = load_results(output_dir)
    
    if results is None:
        print("No results files found (final_results.json or results_checkpoint.json)")
        sys.exit(1)
    
    analysis = analyze_results(results)
    print_report(analysis)
    
    # Save analysis
    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nAnalysis saved to: {analysis_path}")
    
    if args.plot:
        create_analysis_plots(results, output_dir)


if __name__ == "__main__":
    main()

