#!/usr/bin/env python
"""
Run CARBS sweep + evals + HTML for ALL bridges models on the IOI Relaxed task.

Models:
- ss_bridges_d1024_f0.015625 (already done separately)
- ss_bridges_d3072_f0.005
- ss_bridges_d4096_f0.002

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

from my_sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep
from my_sparse_pretrain.scripts.run_all_evals import run_all_evals_for_model
from my_sparse_pretrain.scripts.generate_circuit_htmls import generate_circuit_html_for_model


# Models to run (excluding d1024 which is already done)
MODELS = [
    ("jacobcd52/ss_bridges_d3072_f0.005", "d3072"),
    ("jacobcd52/ss_bridges_d4096_f0.002", "d4096"),
]

OUTPUT_BASE_DIR = "my_sparse_pretrain/outputs/carbs_results_ioi_relaxed"


def run_model(model_path: str, model_name: str):
    """Run CARBS + evals + HTML for a single model."""
    
    print("\n" + "#" * 70)
    print(f"# MODEL: {model_name} ({model_path})")
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
        ablation_type="zero",
        mask_token_embeds=False,
        use_binary_loss=True,  # Binary CE loss
        k_coef_center=1e-3,
        lr_center=1e-2,
    )
    
    # Compute output dir
    model_short = config.get_model_short_name()
    ablation_suffix = "_zero"
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
    
    print(f"\n{model_name} DONE at {datetime.now()}")
    return True


def create_pareto_comparison_plot():
    """Create a comparison plot of Pareto curves for all models."""
    
    print("\n" + "=" * 70)
    print("CREATING PARETO COMPARISON PLOT")
    print("=" * 70 + "\n")
    
    base_dir = Path(OUTPUT_BASE_DIR)
    
    # All models including d1024
    all_models = [
        ("ss_bridges_d1024_f0.015625_zero_noembed", "d1024", "#e74c3c"),
        ("ss_bridges_d3072_f0.005_zero_noembed", "d3072", "#3498db"),
        ("ss_bridges_d4096_f0.002_zero_noembed", "d4096", "#2ecc71"),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_dir_name, label, color in all_models:
        model_dir = base_dir / model_dir_name
        pareto_file = model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json"
        
        if not pareto_file.exists():
            # Try alternate location
            pareto_file = model_dir / "pareto_superval_data.json"
        
        if not pareto_file.exists():
            print(f"  WARNING: No pareto data for {label}")
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
    ax.set_title("IOI Relaxed Task: Pareto Curves by Model Size", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    # Save plot
    plot_path = base_dir / "pareto_comparison_all_models.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"\nPareto comparison saved to: {plot_path}")


def main():
    print("#" * 70)
    print("# IOI RELAXED TASK: ALL BRIDGES MODELS")
    print(f"# Started at: {datetime.now()}")
    print("#" * 70 + "\n")
    
    # Run each model
    for model_path, model_name in MODELS:
        success = run_model(model_path, model_name)
        if not success:
            print(f"\n[WARNING] {model_name} failed, continuing with next model...")
        torch.cuda.empty_cache()
    
    # Create comparison plot
    create_pareto_comparison_plot()
    
    print("\n" + "#" * 70)
    print(f"# ALL MODELS COMPLETED at {datetime.now()}")
    print("#" * 70)


if __name__ == "__main__":
    main()










