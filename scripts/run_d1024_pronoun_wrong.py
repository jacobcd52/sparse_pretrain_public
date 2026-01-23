#!/usr/bin/env python
"""
Run CARBS sweep for d1024 model on the "wrong pronoun" task.
Target: predict the WRONG pronoun (e.g., "he" for Rita).

Results saved to carbs_results_pronoun_wrong/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import torch
from datetime import datetime

from sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep
from sparse_pretrain.scripts.run_all_evals import run_all_evals_for_model


def main():
    print("=" * 70)
    print("CARBS SWEEP: ss_bridges_d1024_f0.015625")
    print("Task: dummy_pronoun_wrong (INVERTED - predict wrong pronoun)")
    print("Ablation: zero, No embed masking")
    print("=" * 70)
    
    # Clear GPU
    torch.cuda.empty_cache()
    
    # Create config for the wrong pronoun task
    config = CleanSweepConfig(
        model_path="jacobcd52/ss_bridges_d1024_f0.015625",
        tokenizer_name="SimpleStories/SimpleStories-1.25M",
        task_name="dummy_pronoun_wrong",  # The new inverted task
        num_runs=32,
        parallel_suggestions=1,
        num_steps=2000,
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
        output_base_dir="outputs/carbs_results_pronoun_wrong",  # New output folder
        device="cuda",
        use_wandb=True,  # Enable wandb logging
        ablation_type="zero",
        mask_token_embeds=False,
        use_binary_loss=False,  # Standard CE loss
        k_coef_center=0.0002,
    )
    
    # The output dir will be: carbs_results_pronoun_wrong/ss_bridges_d1024_f0.015625_zero_noembed
    model_short = config.get_model_short_name()
    ablation_suffix = "_zero" if config.ablation_type == "zero" else "_mean"
    embed_suffix = "" if config.mask_token_embeds else "_noembed"
    output_dir = Path(config.output_base_dir) / f"{model_short}{ablation_suffix}{embed_suffix}"
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Starting CARBS sweep at {datetime.now()}")
    print()
    
    # Run CARBS sweep
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
    
    # Run all evaluations
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
    
    print("\n" + "=" * 70)
    print(f"ALL DONE at {datetime.now()}")
    print("=" * 70)


if __name__ == "__main__":
    main()

