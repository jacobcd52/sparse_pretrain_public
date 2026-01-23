#!/usr/bin/env python3
"""
Run a single pruning experiment with randomly initialized weights.

This script:
1. Loads the model from HuggingFace
2. Measures the weight sparsity fraction from the original weights
3. Randomly reinitializes all weights
4. Applies the same sparsity pattern (top-k by magnitude with min per neuron)
5. Runs the pruning experiment

Usage:
    python my_sparse_pretrain/scripts/run_single_pruning_random_init.py \
        --model jacobcd52/ss_bridges_d1024_f0.015625 \
        --ablation zero \
        --steps 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import math
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List

from transformers import AutoTokenizer

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.trainer import PruningTrainer
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator


def compute_weight_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Compute the weight sparsity statistics from a model.
    
    Returns:
        Dict with:
        - l0_fraction: Fraction of nonzero weights
        - total_params: Total number of sparse parameters
        - total_nonzero: Total number of nonzero parameters
    """
    total_params = 0
    total_nonzero = 0
    
    for name, param in model.named_parameters():
        # Only consider 2D+ parameters (not biases, not norm weights)
        if len(param.shape) < 2:
            continue
        
        # Skip bigram table
        if "bigram" in name.lower():
            continue
        
        n_params = param.numel()
        n_nonzero = (param != 0).sum().item()
        total_params += n_params
        total_nonzero += n_nonzero
    
    l0_fraction = total_nonzero / total_params if total_params > 0 else 1.0
    
    return {
        "l0_fraction": l0_fraction,
        "total_params": total_params,
        "total_nonzero": total_nonzero,
    }


@torch.no_grad()
def sparsify_parameter_simple(param: torch.Tensor, k: int):
    """Simple top-k sparsification by magnitude."""
    flat = param.view(-1)
    
    if k >= flat.numel():
        return
    
    # Find threshold for top-k
    _, topk_indices = torch.topk(flat.abs(), k, sorted=False)
    
    # Create mask and apply
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(0, topk_indices, True)
    
    flat.mul_(mask.float())


@torch.no_grad()
def sparsify_parameter_with_min_per_neuron(
    param: torch.Tensor,
    k: int,
    min_weights_per_neuron: int = 4,
):
    """
    Sparsify while ensuring minimum weights per neuron.
    
    Paper Section 2.3: "Never zero out values that would cause a neuron or 
    attention channel to have fewer than j nonzero values"
    """
    if len(param.shape) != 2:
        sparsify_parameter_simple(param, k)
        return
    
    n_neurons, n_inputs = param.shape
    j = min_weights_per_neuron
    
    # First, identify the j largest weights per neuron (these are protected)
    abs_param = param.abs()
    
    if j >= n_inputs:
        # All weights are protected
        return
    
    # Get indices of top-j weights per neuron
    _, topj_indices = torch.topk(abs_param, j, dim=1, sorted=False)
    
    # Create mask for protected weights
    protected_mask = torch.zeros_like(param, dtype=torch.bool)
    protected_mask.scatter_(1, topj_indices, True)
    
    # Count protected weights
    n_protected = protected_mask.sum().item()
    
    # Calculate remaining budget
    remaining_k = max(0, k - n_protected)
    
    if remaining_k <= 0:
        # Only keep protected weights
        param.mul_(protected_mask.float())
        return
    
    # For non-protected weights, select top-k by magnitude
    unprotected_abs = abs_param.clone()
    unprotected_abs[protected_mask] = -1  # Exclude protected from consideration
    
    flat_unprotected = unprotected_abs.view(-1)
    
    if remaining_k < (flat_unprotected >= 0).sum():
        # Find top remaining_k among non-protected
        _, topk_indices = torch.topk(flat_unprotected, remaining_k, sorted=False)
        
        additional_mask = torch.zeros_like(flat_unprotected, dtype=torch.bool)
        additional_mask.scatter_(0, topk_indices, True)
        additional_mask = additional_mask.view_as(param)
    else:
        # Keep all non-protected
        additional_mask = ~protected_mask
    
    # Combine masks
    final_mask = protected_mask | additional_mask
    param.mul_(final_mask.float())


@torch.no_grad()
def randomly_reinitialize_and_sparsify(
    model: nn.Module,
    target_l0_fraction: float,
    min_weights_per_neuron: int = 4,
    init_std: float = 0.02,
):
    """
    Randomly reinitialize all weights and apply sparsification.
    
    This matches the initialization + sparsification approach used during
    sparse pretraining:
    1. Initialize weights from N(0, init_std)
    2. Apply top-k sparsification with min per neuron constraint
    
    Args:
        model: The model to reinitialize
        target_l0_fraction: Target fraction of nonzero weights
        min_weights_per_neuron: Minimum nonzero weights per neuron (paper: j=4)
        init_std: Standard deviation for weight initialization
    """
    n_layers = model.config.n_layer
    
    for name, param in model.named_parameters():
        # Only reinitialize 2D+ parameters
        if len(param.shape) < 2:
            continue
        
        # Skip bigram table
        if "bigram" in name.lower():
            continue
        
        # Reinitialize with normal distribution
        if "c_proj" in name and "weight" in name:
            # Special scaled init for residual projections (per GPT-2)
            std = init_std / math.sqrt(2 * n_layers)
        else:
            std = init_std
        
        nn.init.normal_(param, mean=0.0, std=std)
        
        # Apply sparsification
        total_elements = param.numel()
        k = max(1, int(total_elements * target_l0_fraction))
        
        if min_weights_per_neuron > 0 and len(param.shape) == 2:
            sparsify_parameter_with_min_per_neuron(param, k, min_weights_per_neuron)
        else:
            sparsify_parameter_simple(param, k)


def compute_mean_cache_from_task(masked_model, task, num_batches=100, device="cuda"):
    """
    Compute mean activation cache from task data.
    
    Uses the positive examples from the task (same data used for training).
    """
    from tqdm import tqdm
    
    # Generate batches from the task
    def task_data_iterator():
        for _ in range(num_batches):
            positive_ids, _, _, _, _ = task.generate_batch(batch_size=64, max_length=0)
            yield positive_ids
    
    data_iter = task_data_iterator()
    mean_cache = masked_model.compute_mean_cache(data_iter, num_batches=num_batches, show_progress=True)
    return mean_cache


def main():
    parser = argparse.ArgumentParser(description="Run single pruning experiment with random init")
    parser.add_argument("--model", type=str, default="jacobcd52/ss_bridges_d1024_f0.015625")
    parser.add_argument("--task", type=str, default="dummy_pronoun")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--k-coef", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--noise-bias", type=float, default=0.1)
    parser.add_argument("--heaviside-temp", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--target-loss", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None,
                       help="Custom wandb run name (default: auto-generated)")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    parser.add_argument("--max-html-nodes", type=int, default=500, 
                       help="Max nodes to include in HTML (for size reduction)")
    parser.add_argument("--pareto-every", type=int, default=100,
                       help="Log Pareto curve every N steps (0 to disable)")
    parser.add_argument("--ablation", type=str, default="zero",
                       choices=["zero", "mean_pretrain", "mean_task"],
                       help="Ablation type: zero, mean_pretrain (SimpleStories), or mean_task")
    parser.add_argument("--mask-token-embeds", action="store_true",
                       help="Also learn a mask over vocabulary (token embeddings)")
    parser.add_argument("--min-weights-per-neuron", type=int, default=4,
                       help="Minimum nonzero weights per neuron for sparsification")
    parser.add_argument("--init-std", type=float, default=0.02,
                       help="Standard deviation for weight initialization")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for weight initialization")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    device = args.device
    model_name = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"my_sparse_pretrain/outputs/random_init_run/{model_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Single Pruning Run with Random Weight Initialization")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Steps: {args.steps}")
    print(f"k_coef: {args.k_coef}")
    print(f"lr: {args.lr}")
    print(f"Ablation: {args.ablation}")
    print(f"Mask token embeds: {args.mask_token_embeds}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, _ = load_model(args.model, device)
    
    # Step 1: Compute original weight sparsity
    print("\nStep 1: Computing weight sparsity from original model...")
    sparsity_stats = compute_weight_sparsity(model)
    original_l0 = sparsity_stats["l0_fraction"]
    print(f"  Original L0 fraction: {original_l0:.6f}")
    print(f"  Original nonzero weights: {sparsity_stats['total_nonzero']:,} / {sparsity_stats['total_params']:,}")
    
    # Step 2: Randomly reinitialize and apply same sparsity
    print(f"\nStep 2: Randomly reinitializing weights and applying L0={original_l0:.6f}...")
    randomly_reinitialize_and_sparsify(
        model,
        target_l0_fraction=original_l0,
        min_weights_per_neuron=args.min_weights_per_neuron,
        init_std=args.init_std,
    )
    
    # Verify sparsity after reinitialization
    new_sparsity_stats = compute_weight_sparsity(model)
    print(f"  New L0 fraction: {new_sparsity_stats['l0_fraction']:.6f}")
    print(f"  New nonzero weights: {new_sparsity_stats['total_nonzero']:,} / {new_sparsity_stats['total_params']:,}")
    
    # Save config
    config_dict = vars(args)
    config_dict["original_l0_fraction"] = original_l0
    config_dict["new_l0_fraction"] = new_sparsity_stats["l0_fraction"]
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "SimpleStories/SimpleStories-1.25M", 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create tasks
    train_task = get_task(args.task, tokenizer, seed=42, split="train")
    val_task = get_task(args.task, tokenizer, seed=42, split="val")
    print(f"\nTrain task: {train_task.name} ({len(train_task.templates)} templates)")
    print(f"Val task: {val_task.name} ({len(val_task.templates)} templates)")
    
    # Create pruning config
    config = PruningConfig(
        k_coef=args.k_coef,
        init_noise_scale=args.noise_scale,
        init_noise_bias=args.noise_bias,
        weight_decay=args.weight_decay,
        lr=args.lr,
        beta2=args.beta2,
        lr_warmup_frac=0.0,
        heaviside_temp=args.heaviside_temp,
        num_steps=args.steps,
        batch_size=64,
        seq_length=0,  # Dynamic padding
        device=device,
        log_every=50,
        target_loss=args.target_loss,
        ablation_type=args.ablation,
        mask_token_embeds=args.mask_token_embeds,
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)
    
    # Handle ablation type
    if config.ablation_type == "zero":
        print("\nUsing ZERO ablation (masked nodes → 0)")
    elif config.ablation_type == "mean_pretrain":
        print("\nComputing mean activation cache from SimpleStories...")
        data_iter = create_data_iterator(
            tokenizer_name="SimpleStories/SimpleStories-1.25M",
            batch_size=64,
            seq_length=256,
            num_batches=100,
            seed=42,
        )
        mean_cache = masked_model.compute_mean_cache(data_iter, num_batches=100, show_progress=True)
        masked_model.set_means_from_dict(mean_cache)
        print("Using MEAN_PRETRAIN ablation (masked nodes → mean over SimpleStories)")
    elif config.ablation_type == "mean_task":
        print("\nComputing mean activation cache from task data...")
        mean_cache = compute_mean_cache_from_task(masked_model, train_task, num_batches=100, device=device)
        masked_model.set_means_from_dict(mean_cache)
        print("Using MEAN_TASK ablation (masked nodes → mean over task data)")
    else:
        raise ValueError(f"Unknown ablation_type: {config.ablation_type}")
    
    # Create trainer
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        val_task=val_task,
        config=config,
        use_wandb=not args.no_wandb,
        wandb_run_name=args.wandb_name or f"random_init_{model_name}_{timestamp}",
    )
    
    # Train
    print(f"\nTraining for {args.steps} steps...")
    with torch.autocast('cuda', dtype=torch.bfloat16):
        trainer.train(
            num_steps=args.steps,
            show_progress=True,
            histogram_every=100,
            pareto_probe_every=args.pareto_every,
        )
    
    # Get results
    num_active_non_embed = masked_model.masks.get_total_active_nodes()
    total_non_embed = masked_model.masks.get_total_nodes()
    num_active_embed = 0
    total_embed = 0
    if masked_model.token_mask is not None:
        num_active_embed = masked_model.token_mask.get_num_active()
        total_embed = masked_model.vocab_size
    num_active_total = num_active_non_embed + num_active_embed
    total_nodes = total_non_embed + total_embed
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Active nodes (total): {num_active_total:,} / {total_nodes:,} ({100*num_active_total/total_nodes:.2f}%)")
    print(f"  Non-embed nodes: {num_active_non_embed:,} / {total_non_embed:,} ({100*num_active_non_embed/total_non_embed:.2f}%)")
    if masked_model.token_mask is not None:
        print(f"  Token embed nodes: {num_active_embed:,} / {total_embed:,} ({100*num_active_embed/total_embed:.2f}%)")
    
    # Save checkpoint
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save masks
    mask_state = masked_model.get_mask_state()
    torch.save(mask_state, checkpoint_dir / "masks.pt")
    
    # Save mean cache (only if computed)
    if config.ablation_type != "zero":
        torch.save(mean_cache, checkpoint_dir / "mean_cache.pt")
    
    # Save config
    with open(checkpoint_dir / "pruning_config.json", "w") as f:
        json.dump({
            "model_path": args.model,
            "task": args.task,
            "k_coef": args.k_coef,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "noise_scale": args.noise_scale,
            "noise_bias": args.noise_bias,
            "heaviside_temp": args.heaviside_temp,
            "beta2": args.beta2,
            "steps": args.steps,
            "num_active_total": num_active_total,
            "num_active_non_embed": num_active_non_embed,
            "num_active_embed": num_active_embed,
            "total_nodes": total_nodes,
            "total_non_embed": total_non_embed,
            "total_embed": total_embed,
            "original_l0_fraction": original_l0,
            "new_l0_fraction": new_sparsity_stats["l0_fraction"],
            "random_init": True,
        }, f, indent=2)
    
    print(f"\nCheckpoint saved to: {checkpoint_dir}")
    print(f"\nDone! Output directory: {output_dir}")


if __name__ == "__main__":
    main()

