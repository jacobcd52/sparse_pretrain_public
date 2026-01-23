#!/usr/bin/env python3
"""
Run pruning with wandb logging, then sweep over target losses.

Usage:
    python scripts/run_loss_sweep.py [--seed SEED] [--steps STEPS]

Customize hyperparameters by editing the config below.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from transformers import AutoTokenizer
from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.trainer import PruningTrainer
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator
from sparse_pretrain.src.pruning.discretize import evaluate_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--no-bisection", action="store_true", help="Skip bisection sweep at end")
    parser.add_argument("--model", type=str, default="jacobcd52/ss_bridges_d1024_f0.015625", help="Model path")
    args = parser.parse_args()
    # ========== CONFIGURATION ==========
    model_path = args.model
    tokenizer_name = "SimpleStories/SimpleStories-1.25M"
    task_name = "dummy_pronoun"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training steps
    num_steps = args.steps
    seed = args.seed
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Best hyperparameters from CARBS sweep (run_id=33, circuit_size=82)
    config = PruningConfig(
        k_coef=0.008204493422240241,
        init_noise_scale=0.08106951670803242,
        init_noise_bias=0.10150472819805145,
        heaviside_temp=2.3153621381887484,
        lr=0.014619242159397475,
        weight_decay=0.0010410156219549463,
        beta2=0.9648776770245959,
        lr_warmup_frac=0.0,  # No warmup (from CARBS search)
        num_steps=num_steps,
        batch_size=64,
        seq_length=512,
        device=device,
        log_every=50,
    )
    
    print(f"Seed: {seed}, Steps: {num_steps}")
    
    # Loss sweep settings
    target_losses = np.logspace(np.log10(0.001), np.log10(0.5), 8).tolist()

    # ===================================

    # Load model
    print(f"Loading model: {model_path}")
    model, _ = load_model(model_path, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create train and val tasks (different templates, same names)
    train_task = get_task(task_name, tokenizer, seed=seed, split="train")
    val_task = get_task(task_name, tokenizer, seed=seed, split="val")
    print(f"Train task: {train_task.name} ({len(train_task.templates)} templates)")
    print(f"Val task: {val_task.name} ({len(val_task.templates)} templates)")
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)

    # Output - use model short name in directory
    model_short = model_path.split("/")[-1] if "/" in model_path else model_path
    output_dir = Path(f"outputs/{model_short}_{train_task.name}_seed{seed}_steps{num_steps}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute mean cache
    print("Computing mean cache...")
    data_iter = create_data_iterator(
        tokenizer_name=tokenizer_name,
        batch_size=64,
        seq_length=512,
        num_batches=100,
        seed=seed,
    )
    mean_cache = masked_model.compute_mean_cache(data_iter, num_batches=100, show_progress=True)
    masked_model.set_means_from_dict(mean_cache)
    torch.save(mean_cache, output_dir / "mean_cache.pt")
    
    # Create trainer with wandb (including val_task for validation metrics)
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        val_task=val_task,
        config=config,
        use_wandb=True,
        wandb_project="circuit-pruning",
        wandb_run_name=f"{model_short}_{train_task.name}_seed{seed}_steps{num_steps}",
    )
    
    # Train with comprehensive logging
    print(f"\nTraining for {num_steps} steps...")
    trainer.train(
        num_steps=num_steps, 
        show_progress=True, 
        histogram_every=100,
        detailed_log_every=50,  # Log detailed stats every 50 steps
        pareto_probe_every=100,  # Evaluate Pareto curve every 100 steps
    )
    
    # Save checkpoint
    trainer.save_checkpoint(output_dir / "checkpoint.pt")
    print(f"Checkpoint saved to {output_dir / 'checkpoint.pt'}")
    
    import wandb
    
    if not args.no_bisection:
        # Sweep over target losses (for both train and val)
        print("\n" + "="*60)
        print("SWEEPING OVER TARGET LOSSES (TRAIN)")
        print("="*60)
        
        sweep_results_train = []
        num_active = masked_model.masks.get_total_active_nodes()
        print(f"Active nodes after training: {num_active}")
        
        # Save original state for restoration between sweeps
        original_state = masked_model.get_mask_state()
        
        for target_loss in target_losses:
            masked_model.load_mask_state(original_state)
            low, high = 1, num_active
            best_k, best_loss = high, float('inf')
            
            for _ in range(30):
                mid = (low + high) // 2
                loss = evaluate_at_k(masked_model, train_task, mid, config)
                
                if loss <= target_loss:
                    best_k, best_loss = mid, loss
                    high = mid - 1
                else:
                    low = mid + 1
                
                if low > high:
                    break
            
            sweep_results_train.append({"target_loss": target_loss, "achieved_loss": best_loss, "circuit_size": best_k})
            print(f"  Target: {target_loss:.4f}, Achieved: {best_loss:.4f}, Size: {best_k}")
        
        # Sweep on validation task
        print("\n" + "="*60)
        print("SWEEPING OVER TARGET LOSSES (VAL)")
        print("="*60)
        
        sweep_results_val = []
        for target_loss in target_losses:
            masked_model.load_mask_state(original_state)
            low, high = 1, num_active
            best_k, best_loss = high, float('inf')
            
            for _ in range(30):
                mid = (low + high) // 2
                loss = evaluate_at_k(masked_model, val_task, mid, config)
                
                if loss <= target_loss:
                    best_k, best_loss = mid, loss
                    high = mid - 1
                else:
                    low = mid + 1
                
                if low > high:
                    break
            
            sweep_results_val.append({"target_loss": target_loss, "achieved_loss": best_loss, "circuit_size": best_k})
            print(f"  Target: {target_loss:.4f}, Achieved: {best_loss:.4f}, Size: {best_k}")
        
        # Restore original state
        masked_model.load_mask_state(original_state)
        
        # Create wandb Tables for train and val Pareto frontiers
        train_table = wandb.Table(columns=["target_loss", "achieved_loss", "circuit_size", "circuit_frac"])
        for r in sweep_results_train:
            train_table.add_data(
                r["target_loss"], 
                r["achieved_loss"], 
                r["circuit_size"],
                r["circuit_size"] / masked_model.masks.get_total_nodes()
            )
        
        val_table = wandb.Table(columns=["target_loss", "achieved_loss", "circuit_size", "circuit_frac"])
        for r in sweep_results_val:
            val_table.add_data(
                r["target_loss"], 
                r["achieved_loss"], 
                r["circuit_size"],
                r["circuit_size"] / masked_model.masks.get_total_nodes()
            )
        
        wandb.log({
            "sweep/train_pareto_table": train_table,
            "sweep/val_pareto_table": val_table,
        })
        
        # Log summary statistics
        train_sizes = [r["circuit_size"] for r in sweep_results_train if r["achieved_loss"] < float('inf')]
        val_sizes = [r["circuit_size"] for r in sweep_results_val if r["achieved_loss"] < float('inf')]
        
        if train_sizes:
            wandb.run.summary["sweep/train_min_circuit_size"] = min(train_sizes)
            wandb.run.summary["sweep/train_max_circuit_size"] = max(train_sizes)
            wandb.run.summary["sweep/train_circuit_size_range"] = max(train_sizes) - min(train_sizes)
        
        if val_sizes:
            wandb.run.summary["sweep/val_min_circuit_size"] = min(val_sizes)
            wandb.run.summary["sweep/val_max_circuit_size"] = max(val_sizes)
            wandb.run.summary["sweep/val_circuit_size_range"] = max(val_sizes) - min(val_sizes)
        
        # Save results
        with open(output_dir / "results.json", "w") as f:
            json.dump({
                "config": config.to_dict(), 
                "sweep_results_train": sweep_results_train,
                "sweep_results_val": sweep_results_val,
            }, f, indent=2)
        
        # Plot Pareto curves (train and val side by side)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Train
        ax = axes[0]
        ax.plot([r["circuit_size"] for r in sweep_results_train], 
                [r["achieved_loss"] for r in sweep_results_train], 'o-', markersize=8, linewidth=2, color='tab:blue')
        ax.set_xlabel("Circuit Size (nodes)")
        ax.set_ylabel("Task Loss")
        ax.set_title(f"Train Pareto Curve ({num_steps} steps)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Val
        ax = axes[1]
        ax.plot([r["circuit_size"] for r in sweep_results_val], 
                [r["achieved_loss"] for r in sweep_results_val], 'o-', markersize=8, linewidth=2, color='tab:orange')
        ax.set_xlabel("Circuit Size (nodes)")
        ax.set_ylabel("Task Loss")
        ax.set_title(f"Val Pareto Curve ({num_steps} steps)")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "pareto_curve.png", dpi=150)
        
        # Plot tau histogram
        all_taus = masked_model.masks.get_all_tau_values().detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_taus, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', label='Threshold')
        ax.set_xlabel("Tau Value")
        ax.set_ylabel("Count")
        ax.set_title("Tau Distribution")
        ax.legend()
        plt.savefig(output_dir / "tau_histogram.png", dpi=150)
    
    wandb.finish()
    
    print(f"\nResults saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

