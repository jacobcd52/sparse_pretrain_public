#!/usr/bin/env python
"""
Run CARBS sweep + evals + HTML for ALL bridges models on the IOI Relaxed task.

Models:
- ss_d128_f1
- ss_bridges_d1024_f0.015625 (already done for zero)
- ss_bridges_d3072_f0.005 (already running for zero)
- ss_bridges_d4096_f0.002 (already running for zero)

Runs BOTH zero ablation (for d128 only, others already running) 
AND mean ablation (for all 4 models).

Uses binary cross-entropy loss (softmax over [correct, incorrect] tokens only).

Results saved to carbs_results_ioi_relaxed/
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


# All models
ALL_MODELS = [
    ("jacobcd52/ss_d128_f1", "d128"),
    ("jacobcd52/ss_bridges_d1024_f0.015625", "d1024"),
    ("jacobcd52/ss_bridges_d3072_f0.005", "d3072"),
    ("jacobcd52/ss_bridges_d4096_f0.002", "d4096"),
]

OUTPUT_BASE_DIR = "outputs/carbs_results_ioi_relaxed"


def run_model(model_path: str, model_name: str, ablation_type: str):
    """Run CARBS + evals + HTML for a single model."""
    
    print("\n" + "#" * 70)
    print(f"# MODEL: {model_name} ({model_path})")
    print(f"# ABLATION: {ablation_type}")
    print("#" * 70 + "\n")
    
    torch.cuda.empty_cache()
    
    # Create config
    config = CleanSweepConfig(
        model_path=model_path,
        tokenizer_name="SimpleStories/SimpleStories-1.25M",
        task_name="ioi_relaxed",
        num_runs=32,
        parallel_suggestions=1,
        num_steps=1000,  # 1000 steps per run
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
        use_wandb=True,
        ablation_type=ablation_type,
        mask_token_embeds=False,
        use_binary_loss=True,  # Binary CE loss
        k_coef_center=1e-3,
        lr_center=1e-2,
    )
    
    # Compute output dir
    model_short = config.get_model_short_name()
    ablation_suffix = f"_{ablation_type}"
    embed_suffix = "_noembed"
    output_dir = Path(config.output_base_dir) / f"{model_short}{ablation_suffix}{embed_suffix}"
    
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
    
    print(f"\n{model_name} ({ablation_type}) DONE at {datetime.now()}")
    return True


def create_pareto_comparison_plot(ablation_type: str):
    """Create a comparison plot of Pareto curves for all models with a given ablation type."""
    
    print("\n" + "=" * 70)
    print(f"CREATING PARETO COMPARISON PLOT ({ablation_type} ablation)")
    print("=" * 70 + "\n")
    
    base_dir = Path(OUTPUT_BASE_DIR)
    
    # All models
    all_models = [
        ("ss_d128_f1", "d128", "#9b59b6"),
        ("ss_bridges_d1024_f0.015625", "d1024", "#e74c3c"),
        ("ss_bridges_d3072_f0.005", "d3072", "#3498db"),
        ("ss_bridges_d4096_f0.002", "d4096", "#2ecc71"),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_short, label, color in all_models:
        model_dir_name = f"{model_short}_{ablation_type}_noembed"
        model_dir = base_dir / model_dir_name
        pareto_file = model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json"
        
        if not pareto_file.exists():
            # Try alternate location
            pareto_file = model_dir / "pareto_superval_data.json"
        
        if not pareto_file.exists():
            print(f"  WARNING: No pareto data for {label} ({ablation_type})")
            continue
        
        with open(pareto_file) as f:
            data = json.load(f)
        
        # Extract pareto frontier
        if "pareto_frontier" in data:
            frontier = data["pareto_frontier"]
            sizes = [p["circuit_size"] for p in frontier]
            losses = [p["loss"] for p in frontier]
        else:
            # Old format
            sizes = data.get("circuit_sizes", [])
            losses = data.get("losses", [])
        
        if sizes and losses:
            ax.plot(sizes, losses, 'o-', label=label, color=color, markersize=5, linewidth=2)
            print(f"  {label}: {len(sizes)} points")
    
    ax.set_xlabel("Circuit Size (# nodes)", fontsize=12)
    ax.set_ylabel("Task Loss (superval)", fontsize=12)
    ax.set_title(f"IOI Relaxed Task: Pareto Curves ({ablation_type.upper()} ablation)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    # Save plot
    plot_path = base_dir / f"pareto_comparison_{ablation_type}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\nPareto comparison saved to: {plot_path}")


def create_combined_pareto_plot():
    """Create a combined plot comparing zero vs mean ablation for all models."""
    
    print("\n" + "=" * 70)
    print("CREATING COMBINED PARETO COMPARISON PLOT (zero vs mean)")
    print("=" * 70 + "\n")
    
    base_dir = Path(OUTPUT_BASE_DIR)
    
    all_models = [
        ("ss_d128_f1", "d128"),
        ("ss_bridges_d1024_f0.015625", "d1024"),
        ("ss_bridges_d3072_f0.005", "d3072"),
        ("ss_bridges_d4096_f0.002", "d4096"),
    ]
    
    colors = {
        "d128": "#9b59b6",
        "d1024": "#e74c3c",
        "d3072": "#3498db",
        "d4096": "#2ecc71",
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_short, label in all_models:
        for ablation_type, linestyle in [("zero", "-"), ("mean", "--")]:
            model_dir_name = f"{model_short}_{ablation_type}_noembed"
            model_dir = base_dir / model_dir_name
            pareto_file = model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json"
            
            if not pareto_file.exists():
                pareto_file = model_dir / "pareto_superval_data.json"
            
            if not pareto_file.exists():
                continue
            
            with open(pareto_file) as f:
                data = json.load(f)
            
            if "pareto_frontier" in data:
                frontier = data["pareto_frontier"]
                sizes = [p["circuit_size"] for p in frontier]
                losses = [p["loss"] for p in frontier]
            else:
                sizes = data.get("circuit_sizes", [])
                losses = data.get("losses", [])
            
            if sizes and losses:
                plot_label = f"{label} ({ablation_type})"
                ax.plot(sizes, losses, linestyle, label=plot_label, 
                       color=colors[label], markersize=4, linewidth=2, marker='o')
    
    ax.set_xlabel("Circuit Size (# nodes)", fontsize=12)
    ax.set_ylabel("Task Loss (superval)", fontsize=12)
    ax.set_title("IOI Relaxed Task: Zero vs Mean Ablation", fontsize=14)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    plot_path = base_dir / "pareto_comparison_zero_vs_mean.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\nCombined pareto comparison saved to: {plot_path}")


def main():
    print("#" * 70)
    print("# IOI RELAXED TASK: ALL BRIDGES MODELS (ZERO + MEAN)")
    print(f"# Started at: {datetime.now()}")
    print("#" * 70 + "\n")
    
    # ===== ZERO ABLATION: d128 only (others already running) =====
    print("\n" + "=" * 70)
    print("PHASE 1: ZERO ABLATION - d128 model")
    print("=" * 70 + "\n")
    
    model_path, model_name = ("jacobcd52/ss_d128_f1", "d128")
    success = run_model(model_path, model_name, "zero")
    if not success:
        print(f"\n[WARNING] {model_name} (zero) failed, continuing...")
    torch.cuda.empty_cache()
    
    # Create zero ablation comparison plot (will include d128 + any already completed)
    create_pareto_comparison_plot("zero")
    
    # ===== MEAN ABLATION: ALL 4 models =====
    print("\n" + "=" * 70)
    print("PHASE 2: MEAN ABLATION - ALL MODELS")
    print("=" * 70 + "\n")
    
    for model_path, model_name in ALL_MODELS:
        success = run_model(model_path, model_name, "mean")
        if not success:
            print(f"\n[WARNING] {model_name} (mean) failed, continuing with next model...")
        torch.cuda.empty_cache()
    
    # Create mean ablation comparison plot
    create_pareto_comparison_plot("mean")
    
    # Create combined comparison plot
    create_combined_pareto_plot()
    
    print("\n" + "#" * 70)
    print(f"# ALL MODELS COMPLETED at {datetime.now()}")
    print("#" * 70)


if __name__ == "__main__":
    main()










