#!/usr/bin/env python3
"""
Run 4 CARBS sweeps for D3072 model:
1. IOI Relaxed, zero ablation
2. IOI Relaxed, mean ablation
3. IOI Mixed, zero ablation
4. IOI Mixed, mean ablation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import os
import torch

from my_sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep
from my_sparse_pretrain.scripts.run_all_evals import run_all_evals_for_model
from my_sparse_pretrain.scripts.generate_circuit_htmls import generate_circuit_html_for_model

# Model config
MODEL_PATH = "jacobcd52/ss_bridges_d3072_f0.005"
TOKENIZER = "SimpleStories/SimpleStories-1.25M"

# Sweep configs - output_base_dir is the TASK folder, model name is appended by run_carbs_sweep
SWEEPS = [
    # 1. IOI Relaxed, zero ablation
    {
        "task": "ioi_relaxed",
        "ablation": "zero",
        "output_base_dir": "my_sparse_pretrain/outputs/carbs_results_ioi_relaxed",
    },
    # 2. IOI Relaxed, mean ablation
    {
        "task": "ioi_relaxed",
        "ablation": "mean_pretrain",
        "output_base_dir": "my_sparse_pretrain/outputs/carbs_results_ioi_relaxed",
    },
    # 3. IOI Mixed, zero ablation
    {
        "task": "ioi_mixed",
        "ablation": "zero",
        "output_base_dir": "my_sparse_pretrain/outputs/carbs_results_ioi_mixed",
    },
    # 4. IOI Mixed, mean ablation
    {
        "task": "ioi_mixed",
        "ablation": "mean_pretrain",
        "output_base_dir": "my_sparse_pretrain/outputs/carbs_results_ioi_mixed",
    },
]


def main():
    for i, sweep in enumerate(SWEEPS, 1):
        print("\n" + "=" * 80)
        print(f"SWEEP {i}/4: {sweep['task']} with {sweep['ablation']} ablation")
        print("=" * 80 + "\n")
        
        # Clear GPU
        torch.cuda.empty_cache()
        
        config = CleanSweepConfig(
            model_path=MODEL_PATH,
            tokenizer_name=TOKENIZER,
            task_name=sweep["task"],
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
            ablation_type=sweep["ablation"],
            mask_token_embeds=False,
            use_wandb=True,
            wandb_project="circuit-pruning-carbs",
            use_binary_loss=True,  # Binary loss for IOI tasks
            k_coef_center=1e-3,
            lr_center=1e-2,
            output_base_dir=sweep["output_base_dir"],  # Task folder only!
        )
        
        # The sweep will create: output_base_dir/ss_bridges_d3072_f0.005_{ablation}_noembed/
        ablation_suffix = "_zero" if sweep["ablation"] == "zero" else "_mean"
        expected_output_dir = Path(sweep["output_base_dir"]) / f"ss_bridges_d3072_f0.005{ablation_suffix}_noembed"
        print(f"Expected output directory: {expected_output_dir}")
        
        # Run CARBS sweep
        try:
            result = run_carbs_sweep(config)
            print(f"\n✓ CARBS sweep completed!")
            
            # Run all evaluations
            print("\nRunning evaluations...")
            run_all_evals_for_model(expected_output_dir, device="cuda")
            print("✓ Evaluations completed!")
            
            # Generate HTML circuit visualization
            print("\nGenerating circuit HTML...")
            try:
                generate_circuit_html_for_model(expected_output_dir)
                print(f"✓ Circuit HTML saved!")
            except Exception as e:
                print(f"✗ HTML generation failed: {e}")
                
        except Exception as e:
            print(f"✗ Sweep {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("ALL SWEEPS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
