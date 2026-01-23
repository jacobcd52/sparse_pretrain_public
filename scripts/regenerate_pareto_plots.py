#!/usr/bin/env python3
"""
Regenerate Pareto plots for existing CARBS results using fixed batches.

This script loads the best checkpoint from each ss_ model and regenerates
the pareto_superval.png with fixed batches to ensure monotonicity.
Also creates a comparison plot with all Pareto curves on the same axes.

Usage:
    python my_sparse_pretrain/scripts/regenerate_pareto_plots.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
from tqdm import tqdm
from typing import Dict, List

from transformers import AutoTokenizer

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model
from my_sparse_pretrain.src.pruning.discretize import evaluate_at_k_fixed_batches


def regenerate_pareto_for_model(
    result_dir: Path,
    device: str = "cuda",
) -> Dict:
    """Regenerate Pareto plot for a single model using fixed batches."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {result_dir.name}")
    print(f"{'='*60}")
    
    # Load sweep config
    with open(result_dir / "sweep_config.json") as f:
        sweep_config = json.load(f)
    
    # Load best checkpoint info
    best_checkpoint_dir = result_dir / "best_checkpoint"
    with open(best_checkpoint_dir / "summary.json") as f:
        summary = json.load(f)
    with open(best_checkpoint_dir / "hparams.json") as f:
        hparams = json.load(f)
    
    # Load model
    model_path = sweep_config["model_path"]
    tokenizer_name = sweep_config["tokenizer_name"]
    task_name = sweep_config["task_name"]
    
    print(f"Loading model: {model_path}")
    model, model_config = load_model(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load mean cache if exists
    mean_cache_path = result_dir / "mean_cache.pt"
    if mean_cache_path.exists():
        mean_cache = torch.load(mean_cache_path, map_location=device, weights_only=True)
    else:
        mean_cache = {}
    
    # Load superval task
    superval_task = get_task(task_name, tokenizer, seed=42, split="superval")
    print(f"Superval task: {superval_task.name} ({len(superval_task.templates)} templates)")
    
    # Create pruning config
    pruning_config = PruningConfig(
        k_coef=hparams["k_coef"],
        init_noise_scale=sweep_config.get("init_noise_scale", 0.01),
        init_noise_bias=sweep_config.get("init_noise_bias", 0.1),
        weight_decay=hparams["weight_decay"],
        lr=hparams["lr"],
        beta2=hparams["beta2"],
        lr_warmup_frac=sweep_config.get("lr_warmup_frac", 0.0),
        heaviside_temp=hparams["heaviside_temp"],
        num_steps=sweep_config.get("num_steps", 1000),
        batch_size=sweep_config.get("batch_size", 64),
        seq_length=sweep_config.get("task_max_length", 0),
        device=device,
        ablation_type=sweep_config.get("ablation_type", "zero"),
        mask_token_embeds=sweep_config.get("mask_token_embeds", True),
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if sweep_config.get("ablation_type", "zero") != "zero" and mean_cache:
        masked_model.set_means_from_dict(mean_cache)
    
    # Load masks
    masks_path = best_checkpoint_dir / "masks.pt"
    mask_state = torch.load(masks_path, map_location=device, weights_only=True)
    masked_model.load_mask_state(mask_state)
    
    # Get node counts
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    
    print(f"Total nodes: {total_nodes:,}")
    print(f"Active after training: {num_active:,}")
    
    # Autocast context
    use_autocast = sweep_config.get("use_autocast", True)
    autocast_dtype = sweep_config.get("autocast_dtype", "bfloat16")
    if use_autocast:
        dtype = torch.bfloat16 if autocast_dtype == "bfloat16" else torch.float16
        autocast_ctx = torch.autocast('cuda', dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    
    # Pre-generate fixed batches
    num_batches = sweep_config.get("bisection_eval_batches", 5)
    task_max_length = sweep_config.get("task_max_length", 0)
    batch_size = sweep_config.get("batch_size", 64)
    
    print(f"Pre-generating {num_batches} fixed evaluation batches...")
    fixed_batches = []
    for _ in range(num_batches):
        batch = superval_task.generate_batch(
            batch_size=batch_size,
            max_length=task_max_length,
        )
        fixed_batches.append(batch)
    
    # Save original state
    original_state = masked_model.get_mask_state()
    
    # Define 16 loss targets log-spaced between 0.001 and 0.5
    loss_targets = np.geomspace(0.001, 0.5, num=16)
    print(f"Finding minimum k for {len(loss_targets)} loss targets: {[f'{t:.4f}' for t in loss_targets]}")
    
    pareto_data = []
    
    for target_loss_val in tqdm(loss_targets, desc="Pareto sweep"):
        # Bisection to find smallest k that achieves <= target_loss
        low, high = 1, num_active
        best_k = None
        best_loss = None
        
        while low <= high:
            mid = (low + high) // 2
            with autocast_ctx:
                loss = evaluate_at_k_fixed_batches(masked_model, fixed_batches, mid, device=device)
            masked_model.load_mask_state(original_state)
            
            if loss <= target_loss_val:
                # This k works, try smaller
                best_k = mid
                best_loss = loss
                high = mid - 1
            else:
                # This k doesn't work, need more nodes
                low = mid + 1
        
        if best_k is not None:
            pareto_data.append({
                "k": int(best_k),
                "loss": float(best_loss),
                "frac": best_k / total_nodes,
                "target_loss": float(target_loss_val),
            })
        else:
            # Even with all active nodes, can't achieve this target
            with autocast_ctx:
                loss = evaluate_at_k_fixed_batches(masked_model, fixed_batches, num_active, device=device)
            masked_model.load_mask_state(original_state)
            print(f"  Target {target_loss_val:.4f} not achievable (best loss with all {num_active} nodes: {loss:.4f})")
    
    # Save Pareto data
    with open(result_dir / "pareto_superval_data.json", "w") as f:
        json.dump(pareto_data, f, indent=2)
    
    # Create individual plot
    target_loss = sweep_config.get("target_loss", 0.15)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ks = [d["k"] for d in pareto_data]
    losses = [d["loss"] for d in pareto_data]
    
    ax.plot(ks, losses, 'b-o', markersize=4, linewidth=1.5, label='Superval loss')
    ax.axhline(y=target_loss, color='r', linestyle='--', linewidth=2, label=f'Target: {target_loss}')
    
    # Mark the circuit size that achieves target
    target_achieved_ks = [k for k, l in zip(ks, losses) if l <= target_loss]
    if target_achieved_ks:
        min_k = min(target_achieved_ks)
        min_loss = losses[ks.index(min_k)]
        ax.scatter([min_k], [min_loss], c='green', s=150, zorder=5, marker='*', 
                   label=f'Min circuit @ target: {min_k} nodes')
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title(f"Pareto Curve on Superval Set\nModel: {result_dir.name}", fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add text box with stats
    textstr = f'Total nodes: {total_nodes:,}\nActive after training: {num_active:,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(result_dir / "pareto_superval.png", dpi=150)
    plt.close()
    
    print(f"Pareto plot saved to {result_dir / 'pareto_superval.png'}")
    
    # Clean up
    del masked_model, model
    torch.cuda.empty_cache()
    
    return {
        "name": result_dir.name,
        "pareto_data": pareto_data,
        "total_nodes": total_nodes,
        "num_active": num_active,
        "model_path": model_path,
    }


def create_comparison_plot(
    all_results: List[Dict],
    output_path: Path,
):
    """Create comparison plot with all Pareto curves on same axes."""
    
    print(f"\n{'='*60}")
    print("Creating comparison plot...")
    print(f"{'='*60}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for result, color in zip(all_results, colors):
        name = result["name"]
        pareto_data = result["pareto_data"]
        
        ks = [d["k"] for d in pareto_data]
        losses = [d["loss"] for d in pareto_data]
        
        # Make the line label shorter/cleaner
        label = name.replace("ss_", "").replace("bridges_", "br_")
        
        ax.plot(ks, losses, '-o', markersize=3, linewidth=1.5, 
                color=color, label=label, alpha=0.8)
    
    # Add target loss line
    ax.axhline(y=0.15, color='red', linestyle='--', linewidth=2, 
               label='Target: 0.15', alpha=0.7)
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=14)
    ax.set_ylabel("Superval Loss", fontsize=14)
    ax.set_title("Pareto Curves Comparison (Superval Set)\nAll SS Models", fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 0.5)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Comparison plot saved to {output_path}")


def main():
    results_base = Path("my_sparse_pretrain/outputs/carbs_results_pronoun")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find all ss_ directories
    ss_dirs = sorted([d for d in results_base.iterdir() 
                      if d.is_dir() and d.name.startswith("ss_")])
    
    print(f"Found {len(ss_dirs)} ss_ model directories:")
    for d in ss_dirs:
        print(f"  - {d.name}")
    
    # Process each model
    all_results = []
    for result_dir in ss_dirs:
        try:
            result = regenerate_pareto_for_model(result_dir, device=device)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing {result_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison plot
    if all_results:
        comparison_path = results_base / "pareto_comparison_ss_models.png"
        create_comparison_plot(all_results, comparison_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

