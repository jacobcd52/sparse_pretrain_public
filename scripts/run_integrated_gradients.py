#!/usr/bin/env python3
"""
Run integrated gradients analysis for node importance estimation.

This script:
1. Loads a weight-sparse model and task
2. Computes integrated gradients for all nodes
3. Validates by comparing with exact ablation for a sample of nodes
4. Saves importance scores and creates visualizations
5. Optionally analyzes discovered circuits

Usage:
    python scripts/run_integrated_gradients.py \
        --model_path jacobcd52/ss_bridges_d1024_f0.015625 \
        --task_name ioi_relaxed \
        --output_dir outputs/carbs_results_ioi_relaxed/ss_bridges_d1024_f0.015625_zero_noembed \
        --use_binary_loss
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sparse_pretrain.src.pruning.run_pruning import load_model
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.integrated_gradients import (
    IntegratedGradientsComputer,
    IGConfig,
    analyze_circuit_importance,
    plot_validation_scatter,
    plot_rank_distribution,
    plot_importance_histogram,
)


def main():
    parser = argparse.ArgumentParser(description="Run integrated gradients analysis")
    
    # Model and task
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint or HuggingFace repo ID")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name (e.g., ioi_relaxed, pronoun_distractor)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name (defaults to model's tokenizer)")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    
    # IG config
    parser.add_argument("--n_steps", type=int, default=50,
                        help="Number of interpolation steps for IG")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of batches to average over")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for task examples")
    
    # Validation config
    parser.add_argument("--n_nodes_per_location", type=int, default=10,
                        help="Number of nodes to sample per location for validation")
    parser.add_argument("--n_ablation_batches", type=int, default=10,
                        help="Number of batches for ablation validation")
    
    # Loss type
    parser.add_argument("--use_binary_loss", action="store_true",
                        help="Use binary CE loss (for IOI-style tasks)")
    parser.add_argument("--use_full_ce_loss", action="store_true",
                        help="Use full vocabulary CE loss (for pronoun tasks)")
    
    # Circuit analysis
    parser.add_argument("--masks_path", type=str, default=None,
                        help="Path to masks.pt file for circuit analysis")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip_validation", action="store_true",
                        help="Skip validation (only compute importances)")
    
    args = parser.parse_args()
    
    # Determine loss type
    if args.use_full_ce_loss:
        use_binary_loss = False
    elif args.use_binary_loss:
        use_binary_loss = True
    else:
        # Default based on task name
        if "ioi" in args.task_name.lower():
            use_binary_loss = True
        elif "pronoun" in args.task_name.lower() and "distractor" not in args.task_name.lower():
            use_binary_loss = False
        else:
            use_binary_loss = True  # Default to binary
    
    print(f"Using binary loss: {use_binary_loss}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, model_config_dict = load_model(args.model_path, args.device)
    model.eval()
    print(f"Model loaded: {model.config.n_layer} layers, d_model={model.config.d_model}")
    
    # Get tokenizer
    tokenizer_name = args.tokenizer
    if tokenizer_name is None:
        tokenizer_name = model_config_dict.get("training_config", {}).get("tokenizer_name")
    
    if tokenizer_name is None:
        raise ValueError("No tokenizer found. Specify --tokenizer argument.")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create task
    print(f"Creating task: {args.task_name}")
    task = get_task(args.task_name, tokenizer, seed=args.seed)
    
    # Create IG config
    ig_config = IGConfig(
        n_steps=args.n_steps,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )
    
    # Create IG computer
    print("Initializing Integrated Gradients computer...")
    ig_computer = IntegratedGradientsComputer(
        model=model,
        task=task,
        config=ig_config,
        use_binary_loss=use_binary_loss,
        device=args.device,
    )
    
    # Run validation (computes both IG and ablation for sample of nodes)
    if not args.skip_validation:
        print(f"\nRunning validation with {args.n_nodes_per_location} nodes per location...")
        validation_results, importances = ig_computer.validate_ig(
            n_nodes_per_location=args.n_nodes_per_location,
            n_ablation_batches=args.n_ablation_batches,
            show_progress=True,
        )
        
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        print(f"N samples: {validation_results['n_samples']}")
        print(f"Pearson r: {validation_results['pearson_r']:.4f}")
        print(f"R²: {validation_results['r_squared']:.4f}")
        print(f"Slope: {validation_results['slope']:.4f}")
        print(f"Intercept: {validation_results['intercept']:.6f}")
        print(f"p-value: {validation_results['p_value']:.2e}")
        print(f"Baseline loss: {validation_results['baseline_loss']:.4f}")
        
        # Save validation results
        val_path = output_dir / "ig_validation_results.json"
        with open(val_path, "w") as f:
            json.dump(validation_results, f, indent=2)
        print(f"\nValidation results saved to: {val_path}")
        
        # Create validation scatter plot
        scatter_path = output_dir / "ig_validation_scatter.png"
        plot_validation_scatter(
            validation_results,
            str(scatter_path),
            title=f"IG Validation - {args.task_name}",
        )
        print(f"Scatter plot saved to: {scatter_path}")
        
        # Create zoomed scatter plot
        scatter_zoomed_path = output_dir / "ig_validation_scatter_zoomed.png"
        # Filter to smaller range for zoomed view
        nodes = validation_results["nodes"]
        ig_vals = np.array([d["ig_imp"] for d in nodes])
        abl_vals = np.array([d["ablation_imp"] for d in nodes])
        
        # Zoom to central 90% of data
        ig_q5, ig_q95 = np.percentile(ig_vals, [5, 95])
        abl_q5, abl_q95 = np.percentile(abl_vals, [5, 95])
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(ig_vals, abl_vals, alpha=0.5, s=20)
        
        slope = validation_results["slope"]
        intercept = validation_results["intercept"]
        x_range = np.array([ig_q5, ig_q95])
        ax.plot(x_range, slope * x_range + intercept, 'r-', 
                label=f'Fit: y = {slope:.3f}x + {intercept:.4f}')
        ax.plot([ig_q5, ig_q95], [ig_q5, ig_q95], 'k--', alpha=0.3, label='y = x')
        
        ax.set_xlim(ig_q5, ig_q95)
        ax.set_ylim(abl_q5, abl_q95)
        ax.set_xlabel("Integrated Gradients Importance", fontsize=12)
        ax.set_ylabel("Ablation Importance (loss increase)", fontsize=12)
        ax.set_title(f"IG Validation (Zoomed) - {args.task_name}\n"
                     f"R² = {validation_results['r_squared']:.3f}, "
                     f"r = {validation_results['pearson_r']:.3f}", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(scatter_zoomed_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Zoomed scatter plot saved to: {scatter_zoomed_path}")
    else:
        # Just compute importances without validation
        print("\nComputing integrated gradients for all nodes...")
        importances = ig_computer.compute_all_importances(show_progress=True)
    
    # Save importances
    importances_path = output_dir / "node_importances.pt"
    torch.save({k: v.cpu() for k, v in importances.items()}, importances_path)
    print(f"\nNode importances saved to: {importances_path}")
    
    # Circuit analysis if masks provided
    if args.masks_path is not None:
        print(f"\n{'='*60}")
        print("CIRCUIT ANALYSIS")
        print(f"{'='*60}")
        
        masks = torch.load(args.masks_path, map_location=args.device)
        
        # Ensure masks are binary
        masks_binary = {}
        for key, mask in masks.items():
            if isinstance(mask, torch.Tensor):
                masks_binary[key] = (mask >= 0).float()  # tau >= 0 means active
        
        analysis = analyze_circuit_importance(importances, masks_binary)
        
        print(f"Circuit nodes: {analysis['n_circuit_nodes']} / {analysis['n_total_nodes']}")
        print(f"Importance fraction: {analysis['importance_frac']:.2%}")
        print(f"Abs importance fraction: {analysis['abs_importance_frac']:.2%}")
        print(f"Rank range: {analysis['min_rank']} - {analysis['max_rank']}")
        print(f"Mean rank: {analysis['mean_rank']:.1f}")
        print(f"Median rank: {analysis['median_rank']:.1f}")
        
        # Save analysis
        analysis_dir = Path(args.masks_path).parent / "importance_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Save summary
        summary_path = analysis_dir / "circuit_importance_summary.json"
        # Remove circuit_ranks from saved version (too large)
        analysis_summary = {k: v for k, v in analysis.items() if k != "circuit_ranks"}
        with open(summary_path, "w") as f:
            json.dump(analysis_summary, f, indent=2)
        
        # Create rank distribution plot
        rank_plot_path = analysis_dir / "circuit_importance_rank_distribution.png"
        plot_rank_distribution(
            analysis,
            str(rank_plot_path),
            max_rank=100,
            title=f"Circuit Node Ranks - {args.task_name}",
        )
        print(f"Rank distribution plot saved to: {rank_plot_path}")
        
        # Create importance histogram
        hist_path = analysis_dir / "circuit_importance_histogram.png"
        plot_importance_histogram(
            analysis,
            importances,
            masks_binary,
            str(hist_path),
            title=f"Node Importance Distribution - {args.task_name}",
        )
        print(f"Importance histogram saved to: {hist_path}")
    
    print(f"\n{'='*60}")
    print("INTEGRATED GRADIENTS ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

