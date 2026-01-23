#!/usr/bin/env python
"""
Run CARBS sweep + evals + HTML for D3072 model on IOI Relaxed task with frozen layer norm.

Uses binary cross-entropy loss (softmax over [correct, incorrect] tokens only).
Both zero and mean ablation.

Results saved to carbs_results_ioi_relaxed_frozenln/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep
from sparse_pretrain.scripts.run_all_evals import run_all_evals_for_model
from sparse_pretrain.scripts.generate_circuit_htmls import generate_circuit_html_for_model


MODEL_PATH = "jacobcd52/ss_bridges_d3072_f0.005"
MODEL_NAME = "d3072"
OUTPUT_BASE_DIR = "outputs/carbs_results_ioi_relaxed_frozenln"


def run_sweep(ablation_type: str):
    """Run CARBS + evals + HTML for a single ablation type."""
    
    print("\n" + "#" * 70)
    print(f"# MODEL: {MODEL_NAME} ({MODEL_PATH})")
    print(f"# ABLATION: {ablation_type}")
    print(f"# FROZEN LN: True")
    print("#" * 70 + "\n")
    
    torch.cuda.empty_cache()
    
    # Create config - matching IOI relaxed settings but with frozen LN and 2000 steps
    config = CleanSweepConfig(
        model_path=MODEL_PATH,
        tokenizer_name="SimpleStories/SimpleStories-1.25M",
        task_name="ioi_relaxed",
        num_runs=32,
        parallel_suggestions=1,
        num_steps=2000,  # 2000 steps per run as requested
        batch_size=64,
        task_max_length=0,
        mean_cache_seq_length=256,
        init_noise_scale=0.01,
        init_noise_bias=0.1,
        lr_warmup_frac=0.0,
        target_loss=0.15,
        use_autocast=True,
        autocast_dtype="bfloat16",
        mean_cache_batches=100,
        bisection_eval_batches=5,
        bisection_max_iters=15,
        loss_penalty_scale=50000.0,
        output_base_dir=OUTPUT_BASE_DIR,
        device="cuda",
        use_wandb=False,
        wandb_project="circuit-pruning-carbs",
        ablation_type=ablation_type,
        mask_token_embeds=False,
        use_binary_loss=True,  # Binary CE loss for IOI relaxed
        k_coef_center=1e-3,
        lr_center=1e-2,
        freeze_layernorm_scale=True,  # FROZEN LN
    )
    
    # Compute output dir
    model_short = config.get_model_short_name()
    ablation_suffix = f"_{ablation_type}"
    embed_suffix = "_noembed"
    output_dir = Path(config.output_base_dir) / f"{model_short}{ablation_suffix}{embed_suffix}_frozenln"
    
    print(f"Output directory: {output_dir}")
    print(f"Starting at {datetime.now()}")
    
    # ===== CARBS sweep =====
    print("\n" + "=" * 70)
    print("STARTING CARBS SWEEP")
    print("=" * 70 + "\n")
    
    try:
        run_carbs_sweep(config)
        print("\nCARBS sweep completed successfully!")
    except Exception as e:
        print(f"\nCARBS sweep failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== Run evaluations =====
    print("\n" + "=" * 70)
    print("RUNNING ALL EVALUATIONS")
    print("=" * 70 + "\n")
    
    torch.cuda.empty_cache()
    
    try:
        results = run_all_evals_for_model(output_dir, device="cuda")
        print("\nEvaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\nEvaluations failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== Generate HTML circuit =====
    print("\n" + "=" * 70)
    print("GENERATING HTML CIRCUIT VISUALIZATION")
    print("=" * 70 + "\n")
    
    torch.cuda.empty_cache()
    
    try:
        html_path = generate_circuit_html_for_model(
            output_dir,
            device='cuda',
            max_nodes=500,
            n_dashboard_samples=200,
            use_bisection=True,
        )
        print(f"HTML saved to: {html_path}")
    except Exception as e:
        print(f"\nHTML generation failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{ablation_type} ablation DONE at {datetime.now()}")
    return True


def create_pareto_comparison_plot():
    """Create a comparison plot of Pareto curves for zero vs mean ablation."""
    
    print("\n" + "=" * 70)
    print("CREATING PARETO COMPARISON PLOT")
    print("=" * 70 + "\n")
    
    base_dir = Path(OUTPUT_BASE_DIR)
    
    models = [
        ("ss_bridges_d3072_f0.005_zero_noembed_frozenln", "Zero Ablation", "#e74c3c"),
        ("ss_bridges_d3072_f0.005_mean_noembed_frozenln", "Mean Ablation", "#3498db"),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_dir_name, label, color in models:
        model_dir = base_dir / model_dir_name
        pareto_file = model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json"
        
        if not pareto_file.exists():
            print(f"  WARNING: No pareto data for {label}")
            continue
        
        with open(pareto_file) as f:
            data = json.load(f)
        
        # Extract data
        fracs = [d["frac"] for d in data]
        losses = [d["loss"] for d in data]
        
        if fracs and losses:
            ax.plot(fracs, losses, 'o-', label=label, color=color, markersize=5, linewidth=2)
            print(f"  {label}: {len(fracs)} points, min loss = {min(losses):.6f}")
    
    ax.set_xlabel("Circuit Size (fraction of total nodes)", fontsize=12)
    ax.set_ylabel("Task Loss (binary CE)", fontsize=12)
    ax.set_title("IOI Relaxed Task: D3072 Frozen LN - Zero vs Mean Ablation", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    # Save plot
    plot_path = base_dir / "pareto_comparison_frozenln.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\nPareto comparison saved to: {plot_path}")


def create_comparison_with_standard():
    """Create a comparison plot of frozen LN vs standard for both ablation types."""
    
    print("\n" + "=" * 70)
    print("CREATING FROZEN LN vs STANDARD COMPARISON PLOT")
    print("=" * 70 + "\n")
    
    frozenln_dir = Path(OUTPUT_BASE_DIR)
    standard_dir = Path("outputs/carbs_results_ioi_relaxed")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, ablation_type in zip(axes, ["zero", "mean"]):
        # Standard
        standard_pareto = standard_dir / f"ss_bridges_d3072_f0.005_{ablation_type}_noembed" / "best_checkpoint" / "evals" / "pareto_superval_data.json"
        # Frozen LN
        frozenln_pareto = frozenln_dir / f"ss_bridges_d3072_f0.005_{ablation_type}_noembed_frozenln" / "best_checkpoint" / "evals" / "pareto_superval_data.json"
        
        for pareto_file, label, color in [
            (standard_pareto, "Standard", "#2ecc71"),
            (frozenln_pareto, "Frozen LN", "#e74c3c"),
        ]:
            if not pareto_file.exists():
                print(f"  WARNING: No pareto data for {label} ({ablation_type})")
                continue
            
            with open(pareto_file) as f:
                data = json.load(f)
            
            fracs = [d["frac"] for d in data]
            losses = [d["loss"] for d in data]
            
            if fracs and losses:
                ax.plot(fracs, losses, 'o-', label=label, color=color, markersize=5, linewidth=2)
                print(f"  {ablation_type} - {label}: {len(fracs)} points, min loss = {min(losses):.6f}")
        
        ax.set_xlabel("Circuit Size (fraction)", fontsize=11)
        ax.set_ylabel("Task Loss (binary CE)", fontsize=11)
        ax.set_title(f"D3072 {ablation_type.title()} Ablation: Frozen LN vs Standard", fontsize=12)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    
    plt.tight_layout()
    plot_path = frozenln_dir / "pareto_comparison_frozenln_vs_standard.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\nFrozen LN vs Standard comparison saved to: {plot_path}")


def main():
    print("#" * 70)
    print("# IOI RELAXED TASK: D3072 WITH FROZEN LAYER NORM")
    print(f"# Started at: {datetime.now()}")
    print("#" * 70 + "\n")
    
    # Run both ablation types
    for ablation_type in ["zero", "mean"]:
        success = run_sweep(ablation_type)
        if not success:
            print(f"\n[WARNING] {ablation_type} ablation failed, continuing...")
        torch.cuda.empty_cache()
    
    # Create comparison plots
    create_pareto_comparison_plot()
    create_comparison_with_standard()
    
    print("\n" + "#" * 70)
    print(f"# ALL SWEEPS COMPLETED at {datetime.now()}")
    print("#" * 70)


if __name__ == "__main__":
    main()

