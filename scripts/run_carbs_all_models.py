#!/usr/bin/env python3
"""
Run CARBS hyperparameter sweep for multiple models.

Models:
- jacobcd52/ss_bridges_d3072_f0.005
- jacobcd52/ss_bridges_d4096_f0.002
- jacobcd52/ss_bridges_d1024_f0.015625
- jacobcd52/ss_d128_f1

Usage:
    nohup python my_sparse_pretrain/scripts/run_carbs_all_models.py > carbs_all_models.log 2>&1 &
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datetime import datetime
import torch

from my_sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep

# Models to run
MODELS = [
    "jacobcd52/ss_bridges_d1024_f0.015625",  # Start with smaller model
    "jacobcd52/ss_d128_f1",
    "jacobcd52/ss_bridges_d3072_f0.005",
    "jacobcd52/ss_bridges_d4096_f0.002",
]


def main():
    parser = argparse.ArgumentParser(description="Run CARBS sweep for multiple models")
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
    parser.add_argument("--ablation", type=str, default="mean_pretrain",
                       choices=["zero", "mean_pretrain", "mean_task"],
                       help="Ablation type (default: mean_pretrain)")
    parser.add_argument("--both-ablations", action="store_true",
                       help="Run all models with zero ablation first, then mean_pretrain")
    parser.add_argument("--mask-token-embeds", action="store_true",
                       help="Enable token embedding masking/pruning")
    parser.add_argument("--k-coef-center", type=float, default=1e-3,
                       help="K coefficient search center (default: 1e-3)")
    parser.add_argument("--freeze-layernorm-scale", action="store_true",
                       help="Freeze layer norm outputs to unpruned values during forward pass")
    
    args = parser.parse_args()
    
    # If both-ablations flag is set, run with zero first, then mean_pretrain
    if args.both_ablations:
        ablation_types = ["zero", "mean_pretrain"]
    else:
        ablation_types = [args.ablation]
    
    print("=" * 70)
    print("CARBS SWEEP FOR MULTIPLE MODELS")
    print("=" * 70)
    print(f"Models to process: {len(args.models)}")
    for m in args.models:
        print(f"  - {m}")
    print(f"Ablation types: {ablation_types}")
    print(f"Runs per model: {args.num_runs}")
    print(f"Steps per run: {args.steps}")
    print(f"Target loss: {args.target_loss}")
    print(f"Mask token embeds: {args.mask_token_embeds}")
    print(f"K coef search center: {args.k_coef_center}")
    print(f"Freeze layer norm scale: {args.freeze_layernorm_scale}")
    print("=" * 70)
    
    results = {}
    
    # Run all ablation types
    embed_suffix = "" if args.mask_token_embeds else "_noembed"
    for ablation_type in ablation_types:
        ablation_suffix = "_zero" if ablation_type == "zero" else "_mean"
        full_suffix = f"{ablation_suffix}{embed_suffix}"
        print(f"\n{'#'*70}")
        print(f"# ABLATION TYPE: {ablation_type.upper()}{' (no embed masking)' if embed_suffix else ''}")
        print(f"{'#'*70}")
        
        for i, model_path in enumerate(args.models):
            model_name = model_path.split("/")[-1]
            run_key = f"{model_name}{full_suffix}"
            
            print(f"\n{'='*70}")
            print(f"MODEL {i+1}/{len(args.models)}: {model_path} ({ablation_type})")
            print(f"{'='*70}")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Use different output directory if freeze_layernorm_scale is enabled
            output_base = "my_sparse_pretrain/outputs/carbs_results_pronoun_frozenln" if args.freeze_layernorm_scale else "my_sparse_pretrain/outputs/carbs_results_pronoun"
            
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
                ablation_type=ablation_type,
                mask_token_embeds=args.mask_token_embeds,
                k_coef_center=args.k_coef_center,
                freeze_layernorm_scale=args.freeze_layernorm_scale,
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
    
    frozen_suffix = "_frozenln" if args.freeze_layernorm_scale else ""
    print(f"\nResults saved to: my_sparse_pretrain/outputs/carbs_results_pronoun{frozen_suffix}/")


if __name__ == "__main__":
    main()

