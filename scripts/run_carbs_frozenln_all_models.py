#!/usr/bin/env python3
"""
Run CARBS hyperparameter sweep for multiple models WITH frozen layer norm scale.

This script runs CARBS sweeps for all 4 models in both zero and mean ablation modes
with the freeze_layernorm_scale option enabled.

Models:
- jacobcd52/ss_bridges_d3072_f0.005
- jacobcd52/ss_bridges_d4096_f0.002
- jacobcd52/ss_bridges_d1024_f0.015625
- jacobcd52/ss_d128_f1

Usage:
    nohup python scripts/run_carbs_frozenln_all_models.py > carbs_frozenln_all_models.log 2>&1 &
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datetime import datetime
import torch

from sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep

# Models to run
MODELS = [
    "jacobcd52/ss_bridges_d1024_f0.015625",  # Start with smaller model
    "jacobcd52/ss_d128_f1",
    "jacobcd52/ss_bridges_d3072_f0.005",
    "jacobcd52/ss_bridges_d4096_f0.002",
]


def main():
    parser = argparse.ArgumentParser(description="Run CARBS sweep with frozen LN for multiple models")
    parser.add_argument("--num-runs", type=int, default=32,
                       help="Total CARBS runs per model (default: 32)")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Parallel suggestions per batch (default: 1)")
    parser.add_argument("--steps", type=int, default=1000,
                       help="Training steps per run (default: 1000)")
    parser.add_argument("--target-loss", type=float, default=0.15,
                       help="Target task loss (default: 0.15)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--models", type=str, nargs="+", default=MODELS,
                       help="Models to run (default: all 4)")
    parser.add_argument("--ablation", type=str, default=None,
                       choices=["zero", "mean_pretrain"],
                       help="Single ablation type (default: run both)")
    parser.add_argument("--k-coef-center", type=float, default=1e-3,
                       help="K coefficient search center (default: 1e-3)")
    parser.add_argument("--wandb-project", type=str, default="circuit-pruning-carbs-frozenln",
                       help="Wandb project name (default: circuit-pruning-carbs-frozenln)")
    
    args = parser.parse_args()
    
    # Run both ablation types unless specified
    if args.ablation is not None:
        ablation_types = [args.ablation]
    else:
        ablation_types = ["zero", "mean_pretrain"]
    
    print("=" * 70)
    print("CARBS SWEEP WITH FROZEN LAYER NORM SCALE")
    print("=" * 70)
    print(f"Models to process: {len(args.models)}")
    for m in args.models:
        print(f"  - {m}")
    print(f"Ablation types: {ablation_types}")
    print(f"Runs per model: {args.num_runs}")
    print(f"Steps per run: {args.steps}")
    print(f"Target loss: {args.target_loss}")
    print(f"K coef search center: {args.k_coef_center}")
    print(f"Wandb logging: {not args.no_wandb}")
    print(f"Wandb project: {args.wandb_project}")
    print("Freeze layer norm scale: True")
    print("Mask token embeds: False (noembed)")
    print("=" * 70)
    
    results = {}
    
    # Run all ablation types
    for ablation_type in ablation_types:
        ablation_suffix = "_zero" if ablation_type == "zero" else "_mean"
        full_suffix = f"{ablation_suffix}_noembed_frozenln"
        print(f"\n{'#'*70}")
        print(f"# ABLATION TYPE: {ablation_type.upper()} (no embed masking, FROZEN LN)")
        print(f"{'#'*70}")
        
        for i, model_path in enumerate(args.models):
            model_name = model_path.split("/")[-1]
            run_key = f"{model_name}{full_suffix}"
            
            print(f"\n{'='*70}")
            print(f"MODEL {i+1}/{len(args.models)}: {model_path} ({ablation_type})")
            print(f"{'='*70}")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Output directory for frozen LN results
            output_base = "outputs/carbs_results_pronoun_frozenln"
            
            config = CleanSweepConfig(
                model_path=model_path,
                task_name="dummy_pronoun",
                num_runs=args.num_runs,
                parallel_suggestions=args.parallel,
                num_steps=args.steps,
                target_loss=args.target_loss,
                init_noise_scale=0.01,
                init_noise_bias=0.1,
                lr_warmup_frac=0.0,
                use_wandb=not args.no_wandb,
                wandb_project=args.wandb_project,
                ablation_type=ablation_type,
                mask_token_embeds=False,  # noembed only
                k_coef_center=args.k_coef_center,
                freeze_layernorm_scale=True,  # Enable frozen LN
                output_base_dir=output_base,
            )
            
            try:
                result = run_carbs_sweep(config)
                results[run_key] = {
                    "status": "success",
                    "ablation_type": ablation_type,
                    "best_circuit_size": result["best_result"]["circuit_size"] if result["best_result"] else None,
                    "target_achieved_runs": result["summary"]["target_achieved_runs"],
                }
            except Exception as e:
                print(f"ERROR processing {model_path}: {e}")
                import traceback
                traceback.print_exc()
                results[run_key] = {"status": "failed", "ablation_type": ablation_type, "error": str(e)}
            
            # Clear GPU memory between models
            torch.cuda.empty_cache()
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL MODELS COMPLETE - SUMMARY")
    print("=" * 70)
    
    for run_key, result in results.items():
        if result["status"] == "success":
            print(f"{run_key}:")
            print(f"  Best circuit size: {result['best_circuit_size']}")
            print(f"  Runs achieving target: {result['target_achieved_runs']}")
        else:
            print(f"{run_key}: FAILED - {result['error']}")
    
    print("\nResults saved to: outputs/carbs_results_pronoun_frozenln/")


if __name__ == "__main__":
    main()

