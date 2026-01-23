#!/usr/bin/env python3
"""
Run CARBS sweep on a randomly re-initialized d128_f1 model with 2000 training steps,
then run all evaluations on the best checkpoint.

This script:
1. Creates a randomly initialized version of jacobcd52/ss_d128_f1 (preserving sparsity pattern)
2. Runs CARBS hyperparameter sweep with 2000 steps per run
3. Runs all evaluations on the best checkpoint

Output: outputs/carbs_results_pronoun/ss_d128_f1_randinit_zero_noembed/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
from datetime import datetime

from sparse_pretrain.scripts.run_carbs_random_init import create_random_init_model
from sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep
from sparse_pretrain.scripts.run_all_evals import run_all_evals_for_model


def main():
    print("=" * 70)
    print("CARBS SWEEP: Random Init d128_f1 Model with 2000 Steps")
    print("=" * 70)
    
    # Configuration
    MODEL = "jacobcd52/ss_d128_f1"
    OUTPUT_BASE_DIR = "outputs/carbs_results_pronoun"
    NUM_STEPS = 2000  # As requested
    NUM_RUNS = 32
    TASK = "dummy_pronoun"
    ABLATION = "zero"
    RANDOM_SEED = 42
    
    model_name = MODEL.split("/")[-1]
    random_init_dir = f"outputs/random_init_models/{model_name}_seed{RANDOM_SEED}"
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL}")
    print(f"  Task: {TASK}")
    print(f"  Ablation: {ABLATION}")
    print(f"  Num training steps: {NUM_STEPS}")
    print(f"  Num CARBS runs: {NUM_RUNS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Output base dir: {OUTPUT_BASE_DIR}")
    print(f"  Random init model dir: {random_init_dir}")
    
    # Step 1: Create random-init model
    print(f"\n{'='*70}")
    print("STEP 1: Creating Random-Init Model")
    print(f"{'='*70}")
    
    random_model_path = create_random_init_model(
        model_path=MODEL,
        output_dir=random_init_dir,
        seed=RANDOM_SEED,
    )
    
    # Step 2: Run CARBS sweep
    print(f"\n{'='*70}")
    print("STEP 2: Running CARBS Sweep")
    print(f"{'='*70}")
    print(f"Starting at {datetime.now()}")
    
    # Create config
    config = CleanSweepConfig(
        model_path=random_model_path,  # Use local random-init model
        task_name=TASK,
        num_runs=NUM_RUNS,
        parallel_suggestions=1,
        num_steps=NUM_STEPS,
        target_loss=0.15,
        init_noise_scale=0.01,
        init_noise_bias=0.1,
        lr_warmup_frac=0.0,
        use_wandb=True,
        ablation_type=ABLATION,
        mask_token_embeds=False,  # noembed
        k_coef_center=1e-3,
        output_base_dir=OUTPUT_BASE_DIR,
    )
    
    # Override model name for output directory naming
    # The sweep will use the random_model_path, but we want the output dir 
    # to be named based on the original model + randinit suffix
    ablation_suffix = "_zero" if ABLATION == "zero" else "_mean"
    embed_suffix = "_noembed"  # mask_token_embeds=False
    full_suffix = f"_randinit{ablation_suffix}{embed_suffix}"
    
    # Create output directory
    output_dir = Path(OUTPUT_BASE_DIR) / f"{model_name}{full_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Run the sweep
    result = run_carbs_sweep(config)
    
    print(f"\nCARBS sweep completed at {datetime.now()}")
    
    if result["best_result"]:
        print(f"\nBest circuit size: {result['best_result']['circuit_size']}")
        print(f"Best val loss: {result['best_result']['achieved_loss_val']:.4f}")
        print(f"Runs achieving target: {result['summary']['target_achieved_runs']}")
    else:
        print("\nNo runs achieved target loss!")
        return
    
    # Step 3: Run evaluations
    print(f"\n{'='*70}")
    print("STEP 3: Running Evaluations on Best Checkpoint")
    print(f"{'='*70}")
    
    # Find the output directory that was created
    # Since run_carbs_sweep modifies the output dir based on config, we need to find it
    possible_dirs = [
        output_dir,  # Expected location
        Path(OUTPUT_BASE_DIR) / f"{Path(random_model_path).name}_zero_noembed",  # Alternative
    ]
    
    model_dir = None
    for d in possible_dirs:
        if d.exists() and (d / "best_checkpoint").exists():
            model_dir = d
            break
    
    if model_dir is None:
        # Search in output base dir - look for either randinit or seed42 naming
        for d in Path(OUTPUT_BASE_DIR).iterdir():
            if d.is_dir() and "d128_f1" in d.name:
                if ("randinit" in d.name or "seed42" in d.name):
                    if (d / "best_checkpoint").exists():
                        model_dir = d
                        break
    
    if model_dir is None:
        print("ERROR: Could not find output directory with best_checkpoint!")
        print(f"Searched in: {OUTPUT_BASE_DIR}")
        return
    
    print(f"Running evaluations for: {model_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_results = run_all_evals_for_model(model_dir, device)
    
    print(f"\n{'='*70}")
    print("ALL COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {model_dir}")
    print(f"Evaluations saved to: {model_dir / 'best_checkpoint' / 'evals'}")


if __name__ == "__main__":
    main()

