#!/usr/bin/env python
"""
Run CARBS sweep + evals + HTML for d1024 model on the IOI Relaxed task.

Uses binary cross-entropy loss (softmax over [correct, incorrect] tokens only).

Results saved to carbs_results_ioi_relaxed/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import torch
from datetime import datetime

from sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep
from sparse_pretrain.scripts.run_all_evals import run_all_evals_for_model
from sparse_pretrain.scripts.generate_circuit_htmls import generate_circuit_html_for_model


def main():
    task_name = "ioi_relaxed"
    output_base_dir = "outputs/carbs_results_ioi_relaxed"
    
    print("=" * 70)
    print(f"CARBS SWEEP: ss_bridges_d1024_f0.015625")
    print(f"Task: {task_name} (Indirect Object Identification - Relaxed)")
    print(f"Loss: Binary Cross-Entropy")
    print(f"Steps: 1000 per run")
    print(f"Ablation: zero, No embed masking")
    print("=" * 70)
    
    # Clear GPU
    torch.cuda.empty_cache()
    
    # Create config
    config = CleanSweepConfig(
        model_path="jacobcd52/ss_bridges_d1024_f0.015625",
        tokenizer_name="SimpleStories/SimpleStories-1.25M",
        task_name=task_name,
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
        output_base_dir=output_base_dir,
        device="cuda",
        use_wandb=True,
        ablation_type="zero",
        mask_token_embeds=False,
        use_binary_loss=True,  # Binary CE loss
        k_coef_center=1e-3,    # Default k_coef
        lr_center=1e-2,
    )
    
    # Compute output dir
    model_short = config.get_model_short_name()
    ablation_suffix = "_zero" if config.ablation_type == "zero" else "_mean"
    embed_suffix = "" if config.mask_token_embeds else "_noembed"
    output_dir = Path(config.output_base_dir) / f"{model_short}{ablation_suffix}{embed_suffix}"
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Starting at {datetime.now()}")
    print()
    
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
        return
    
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
    
    print("\n" + "=" * 70)
    print(f"ALL DONE at {datetime.now()}")
    print("=" * 70)


if __name__ == "__main__":
    main()










