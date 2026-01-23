#!/usr/bin/env python3
"""
Run 100 seeds for any CARBS checkpoint, saving masks for runs that achieve target loss.

Usage:
    python my_sparse_pretrain/scripts/run_100_seeds_generic.py --checkpoint-dir PATH
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import numpy as np
from typing import Dict, Any, Optional
from contextlib import nullcontext
from tqdm import tqdm

from transformers import AutoTokenizer

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.trainer import PruningTrainer
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model
from my_sparse_pretrain.src.pruning.discretize import evaluate_at_k


def run_single_seed_quick(
    seed: int,
    hparams: Dict[str, float],
    model,
    tokenizer,
    mean_cache: Optional[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a single pruning experiment with a specific seed (minimal output)."""
    device = config["device"]
    
    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create pruning config
    pruning_config = PruningConfig(
        k_coef=hparams["k_coef"],
        init_noise_scale=config["init_noise_scale"],
        init_noise_bias=config["init_noise_bias"],
        weight_decay=hparams["weight_decay"],
        lr=hparams["lr"],
        beta2=hparams["beta2"],
        lr_warmup_frac=config["lr_warmup_frac"],
        heaviside_temp=hparams["heaviside_temp"],
        num_steps=config["num_steps"],
        batch_size=config["batch_size"],
        seq_length=config.get("task_max_length", 0),
        device=device,
        log_every=500,  # Less logging
        target_loss=config["target_loss"],
        ablation_type=config["ablation_type"],
        mask_token_embeds=config["mask_token_embeds"],
        freeze_layernorm_scale=config.get("freeze_layernorm_scale", False),
    )
    
    # Recreate task with this seed
    train_task = get_task(config["task_name"], tokenizer, seed=seed, split="train")
    val_task = get_task(config["task_name"], tokenizer, seed=seed, split="val")
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if config["ablation_type"] != "zero" and mean_cache is not None:
        masked_model.set_means_from_dict(mean_cache)
    
    # Create trainer
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        val_task=val_task,
        config=pruning_config,
        use_wandb=False,
    )
    
    # Autocast context
    if config.get("use_autocast", True):
        dtype = torch.bfloat16 if config.get("autocast_dtype", "bfloat16") == "bfloat16" else torch.float16
        autocast_ctx = torch.autocast('cuda', dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    
    with autocast_ctx:
        trainer.train(
            num_steps=config["num_steps"],
            show_progress=False,
            histogram_every=0,
            pareto_probe_every=0,
        )
    
    # Get active nodes
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    
    # Save mask state
    original_state = masked_model.get_mask_state()
    
    # Evaluate at full capacity
    eval_batches = config.get("bisection_eval_batches", 5)
    with autocast_ctx:
        loss_at_all = evaluate_at_k(masked_model, val_task, num_active, pruning_config, num_batches=eval_batches)
    
    target_achieved = loss_at_all <= config["target_loss"]
    
    # Only save if target achieved
    if target_achieved:
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(original_state, seed_dir / "masks.pt")
        
        # Save summary
        summary = {
            "seed": seed,
            "loss_at_all_active": float(loss_at_all),
            "num_active": num_active,
            "total_nodes": total_nodes,
            "target_achieved": target_achieved,
        }
        with open(seed_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    return {
        "seed": seed,
        "loss_at_all_active": float(loss_at_all),
        "num_active": num_active,
        "total_nodes": total_nodes,
        "target_achieved": target_achieved,
        "mask_state": original_state if target_achieved else None,
    }


def run_checkpoint(checkpoint_dir: Path, device: str = "cuda", num_seeds: int = 100):
    """Run 100 seeds for a single checkpoint directory."""
    
    # Load sweep config
    with open(checkpoint_dir / "sweep_config.json") as f:
        config = json.load(f)
    
    # Load best hparams
    with open(checkpoint_dir / "best_checkpoint" / "hparams.json") as f:
        hparams = json.load(f)
    hparams = {k: v for k, v in hparams.items() if k != "suggestion_uuid"}
    
    config["device"] = device
    
    print("=" * 70)
    print(f"Running {num_seeds} Seeds")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Model: {config['model_path']}")
    print(f"Task: {config['task_name']}")
    print(f"Target loss: {config['target_loss']}")
    print(f"Freeze LayerNorm Scale: {config.get('freeze_layernorm_scale', False)}")
    print(f"Num steps: {config['num_steps']}")
    print(f"Hyperparameters:")
    for k, v in hparams.items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 70)
    
    # Output directory
    output_dir = checkpoint_dir / "repetitions_100"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model, _ = load_model(config["model_path"], device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load mean cache if needed
    mean_cache = None
    if config["ablation_type"] != "zero":
        mean_cache_path = checkpoint_dir / "mean_cache.pt"
        if mean_cache_path.exists():
            mean_cache = torch.load(mean_cache_path, map_location=device, weights_only=True)
            mean_cache = {k: v.to(device) for k, v in mean_cache.items()}
    
    # Run all seeds
    print(f"\nRunning {num_seeds} seeds...")
    all_results = []
    successful_seeds = []
    failed_seeds = []
    
    for seed in tqdm(range(num_seeds), desc="Seeds"):
        try:
            result = run_single_seed_quick(
                seed=seed,
                hparams=hparams,
                model=model,
                tokenizer=tokenizer,
                mean_cache=mean_cache,
                config=config,
                output_dir=output_dir,
            )
            all_results.append(result)
            
            if result["target_achieved"]:
                successful_seeds.append(seed)
                print(f"  Seed {seed}: PASS (loss={result['loss_at_all_active']:.4f}, nodes={result['num_active']})")
            else:
                failed_seeds.append(seed)
                print(f"  Seed {seed}: FAIL (loss={result['loss_at_all_active']:.4f})")
                
        except Exception as e:
            print(f"  Seed {seed}: ERROR - {e}")
            failed_seeds.append(seed)
        
        # Clear cache periodically
        if seed % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print(f"Complete: {len(successful_seeds)}/{num_seeds} seeds achieved target loss")
    print(f"{'='*70}")
    
    # Save summary
    phase1_summary = {
        "total_seeds": num_seeds,
        "successful_seeds": successful_seeds,
        "failed_seeds": failed_seeds,
        "success_rate": len(successful_seeds) / num_seeds,
        "config": config,
        "hparams": hparams,
    }
    with open(output_dir / "phase1_summary.json", "w") as f:
        json.dump(phase1_summary, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return len(successful_seeds), num_seeds


def main():
    parser = argparse.ArgumentParser(description="Run 100 seeds for CARBS checkpoint(s)")
    parser.add_argument("--checkpoint-dir", type=str, action="append", required=True,
                       help="Path to CARBS checkpoint directory (can specify multiple)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (default: cuda)")
    parser.add_argument("--num-seeds", type=int, default=100,
                       help="Number of seeds to run (default: 100)")
    
    args = parser.parse_args()
    
    results = []
    for checkpoint_dir in args.checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"ERROR: Checkpoint directory not found: {checkpoint_path}")
            continue
        
        successful, total = run_checkpoint(checkpoint_path, args.device, args.num_seeds)
        results.append((checkpoint_dir, successful, total))
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for checkpoint_dir, successful, total in results:
        print(f"{checkpoint_dir}: {successful}/{total} ({100*successful/total:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()




