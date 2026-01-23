#!/usr/bin/env python
"""
Run CARBS sweep + evals + HTML for ALL bridges models on the IOI Relaxed task
with MEAN ablation.

Uses binary cross-entropy loss (softmax over [correct, incorrect] tokens only).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime

from sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep
from sparse_pretrain.scripts.run_all_evals import run_all_evals_for_model
from sparse_pretrain.scripts.generate_circuit_htmls import generate_circuit_html_for_model


ALL_MODELS = [
    ("jacobcd52/ss_d128_f1", "d128"),
    ("jacobcd52/ss_bridges_d1024_f0.015625", "d1024"),
    ("jacobcd52/ss_bridges_d3072_f0.005", "d3072"),
    ("jacobcd52/ss_bridges_d4096_f0.002", "d4096"),
]

OUTPUT_BASE_DIR = "outputs/carbs_results_ioi_relaxed"


def run_model(model_path: str, model_name: str):
    """Run CARBS + evals + HTML for a single model with mean ablation."""
    
    print("\n" + "#" * 70)
    print(f"# MODEL: {model_name} ({model_path})")
    print(f"# ABLATION: mean")
    print("#" * 70 + "\n")
    
    torch.cuda.empty_cache()
    
    config = CleanSweepConfig(
        model_path=model_path,
        tokenizer_name="SimpleStories/SimpleStories-1.25M",
        task_name="ioi_relaxed",
        num_runs=32,
        parallel_suggestions=1,
        num_steps=1000,
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
        ablation_type="mean_pretrain",  # MEAN ablation (uses SimpleStories mean)
        mask_token_embeds=False,
        use_binary_loss=True,
        k_coef_center=1e-3,
        lr_center=1e-2,
    )
    
    model_short = config.get_model_short_name()
    output_dir = Path(config.output_base_dir) / f"{model_short}_mean_pretrain_noembed"
    
    print(f"Output directory: {output_dir}")
    print(f"Starting at {datetime.now()}")
    
    # CARBS sweep
    print("\n" + "=" * 70)
    print("STARTING CARBS SWEEP")
    print("=" * 70 + "\n")
    
    try:
        run_carbs_sweep(config)
        print("\nCARBS sweep completed successfully!")
    except Exception as e:
        print(f"\nCARBS sweep failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Evaluations
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
        print(f"\nEvaluations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # HTML circuit
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
        print(f"\nHTML generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{model_name} (mean) DONE at {datetime.now()}")
    return True


def create_pareto_plots():
    """Create comparison plots."""
    
    base_dir = Path(OUTPUT_BASE_DIR)
    
    models = [
        ("ss_d128_f1", "d128", "#9b59b6"),
        ("ss_bridges_d1024_f0.015625", "d1024", "#e74c3c"),
        ("ss_bridges_d3072_f0.005", "d3072", "#3498db"),
        ("ss_bridges_d4096_f0.002", "d4096", "#2ecc71"),
    ]
    
    # Mean ablation plot
    print("\n" + "=" * 70)
    print("CREATING PARETO COMPARISON PLOT (mean ablation)")
    print("=" * 70 + "\n")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_short, label, color in models:
        model_dir = base_dir / f"{model_short}_mean_pretrain_noembed"
        pareto_file = model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json"
        
        if not pareto_file.exists():
            print(f"  Missing: {label}")
            continue
        
        with open(pareto_file) as f:
            data = json.load(f)
        
        sizes = [p["k"] for p in data]
        losses = [p["loss"] for p in data]
        
        ax.plot(sizes, losses, 'o-', label=label, color=color, markersize=4, linewidth=2)
        print(f"  {label}: {len(sizes)} points, min_loss={min(losses):.4f}")
    
    ax.set_xlabel("Circuit Size (# nodes)", fontsize=12)
    ax.set_ylabel("Task Loss (binary CE)", fontsize=12)
    ax.set_title("IOI Relaxed Task: Pareto Curves (Mean Ablation)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    plot_path = base_dir / "pareto_comparison_mean.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved: {plot_path}")
    
    # Combined plot: zero vs mean
    print("\n" + "=" * 70)
    print("CREATING COMBINED PLOT (zero vs mean)")
    print("=" * 70 + "\n")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_short, label, color in models:
        for ablation_type, linestyle in [("zero", "-"), ("mean_pretrain", "--")]:
            model_dir = base_dir / f"{model_short}_{ablation_type}_noembed"
            pareto_file = model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json"
            
            if not pareto_file.exists():
                continue
            
            with open(pareto_file) as f:
                data = json.load(f)
            
            sizes = [p["k"] for p in data]
            losses = [p["loss"] for p in data]
            
            plot_label = f"{label} ({ablation_type})"
            ax.plot(sizes, losses, linestyle, label=plot_label, 
                   color=color, markersize=4, linewidth=2, marker='o')
    
    ax.set_xlabel("Circuit Size (# nodes)", fontsize=12)
    ax.set_ylabel("Task Loss (binary CE)", fontsize=12)
    ax.set_title("IOI Relaxed Task: Zero vs Mean Ablation", fontsize=14)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    plot_path = base_dir / "pareto_comparison_zero_vs_mean.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved: {plot_path}")


def main():
    print("#" * 70)
    print("# IOI RELAXED TASK: ALL MODELS - MEAN ABLATION")
    print(f"# Started at: {datetime.now()}")
    print("#" * 70 + "\n")
    
    for model_path, model_name in ALL_MODELS:
        success = run_model(model_path, model_name)
        if not success:
            print(f"\n[WARNING] {model_name} (mean) failed, continuing...")
        torch.cuda.empty_cache()
    
    create_pareto_plots()
    
    print("\n" + "#" * 70)
    print(f"# ALL MODELS COMPLETED at {datetime.now()}")
    print("#" * 70)


if __name__ == "__main__":
    main()

