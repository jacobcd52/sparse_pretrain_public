#!/usr/bin/env python3
"""
Run pruning multiple times with different seeds using fixed hyperparameters.

This script takes a CARBS best checkpoint and runs pruning N times with
different random seeds to assess the stability/reproducibility of the circuit.

Usage:
    python my_sparse_pretrain/scripts/run_seed_repetitions.py \
        --checkpoint-dir my_sparse_pretrain/outputs/carbs_results_pronoun/ss_bridges_d1024_f0.015625_zero_noembed \
        --num-seeds 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import nullcontext
from tqdm import tqdm
from itertools import combinations

from transformers import AutoTokenizer

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.trainer import PruningTrainer
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator
from my_sparse_pretrain.src.pruning.discretize import evaluate_at_k, evaluate_at_k_fixed_batches


def run_single_seed(
    seed: int,
    hparams: Dict[str, float],
    model,
    tokenizer,
    train_task,
    val_task,
    mean_cache: Optional[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a single pruning experiment with a specific seed."""
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
        log_every=100,
        target_loss=config["target_loss"],
        ablation_type=config["ablation_type"],
        mask_token_embeds=config["mask_token_embeds"],
    )
    
    # Recreate task with this seed for reproducibility
    train_task_seeded = get_task(config["task_name"], tokenizer, seed=seed, split="train")
    val_task_seeded = get_task(config["task_name"], tokenizer, seed=seed, split="val")
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if config["ablation_type"] != "zero" and mean_cache is not None:
        masked_model.set_means_from_dict(mean_cache)
    
    # Create trainer
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task_seeded,
        val_task=val_task_seeded,
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
            show_progress=True,
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
        loss_at_all = evaluate_at_k(masked_model, val_task_seeded, num_active, pruning_config, num_batches=eval_batches)
    masked_model.load_mask_state(original_state)
    
    # Bisection search for smallest circuit
    target_loss = config["target_loss"]
    best_k, best_loss = num_active, loss_at_all
    
    if loss_at_all <= target_loss:
        low, high = 1, num_active
        max_iters = config.get("bisection_max_iters", 15)
        for _ in range(max_iters):
            if low > high:
                break
            mid = (low + high) // 2
            with autocast_ctx:
                loss = evaluate_at_k(masked_model, val_task_seeded, mid, pruning_config, num_batches=eval_batches)
            masked_model.load_mask_state(original_state)
            
            if loss <= target_loss:
                best_k, best_loss = mid, loss
                high = mid - 1
            else:
                low = mid + 1
    
    # Get the binary circuit mask (which nodes are active at best_k)
    masked_model.masks.keep_top_k(best_k)
    circuit_mask = masked_model.get_circuit_mask()
    
    # Restore to original state for Pareto evaluation
    masked_model.load_mask_state(original_state)
    
    # Save checkpoint
    seed_dir = output_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(original_state, seed_dir / "masks.pt")
    torch.save(circuit_mask, seed_dir / "circuit_mask.pt")
    
    result = {
        "seed": seed,
        "circuit_size": best_k,
        "achieved_loss_val": best_loss,
        "loss_at_all_active": loss_at_all,
        "num_active_after_training": num_active,
        "total_nodes": total_nodes,
        "frac_active": num_active / total_nodes,
        "frac_circuit": best_k / total_nodes,
        "target_achieved": loss_at_all <= target_loss,
        "mask_state": original_state,
        "circuit_mask": circuit_mask,
    }
    
    # Save summary
    summary = {k: v for k, v in result.items() if k not in ["mask_state", "circuit_mask"]}
    with open(seed_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return result


def get_circuit_nodes_set(circuit_mask: Dict[str, torch.Tensor]) -> set:
    """
    Convert a circuit mask dictionary to a set of active node identifiers.
    
    Returns:
        Set of tuples (location_name, node_index) for all active nodes
    """
    active_nodes = set()
    for location, mask in circuit_mask.items():
        # mask is a boolean tensor, True = active
        active_indices = torch.where(mask)[0].cpu().tolist()
        for idx in active_indices:
            active_nodes.add((location, idx))
    return active_nodes


def compute_pairwise_overlap(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute the pairwise overlap between all pairs of circuits.
    
    For each pair (A, B), computes:
    - Intersection: nodes in both A and B
    - Union: nodes in A or B
    - Proportion of A in B: |A ∩ B| / |A|
    - Proportion of B in A: |A ∩ B| / |B|
    - Jaccard similarity: |A ∩ B| / |A ∪ B|
    
    Returns:
        Dictionary with overlap statistics
    """
    n_seeds = len(results)
    
    # Convert all circuit masks to sets
    circuit_sets = []
    for r in results:
        nodes = get_circuit_nodes_set(r["circuit_mask"])
        circuit_sets.append(nodes)
    
    # Compute pairwise overlaps
    prop_a_in_b = []  # Proportion of A's nodes that are in B
    prop_b_in_a = []  # Proportion of B's nodes that are in A  
    jaccard = []
    intersection_sizes = []
    union_sizes = []
    
    for i, j in combinations(range(n_seeds), 2):
        set_a = circuit_sets[i]
        set_b = circuit_sets[j]
        
        intersection = set_a & set_b
        union = set_a | set_b
        
        intersection_sizes.append(len(intersection))
        union_sizes.append(len(union))
        
        # Proportion of A in B
        if len(set_a) > 0:
            prop_a_in_b.append(len(intersection) / len(set_a))
        # Proportion of B in A  
        if len(set_b) > 0:
            prop_b_in_a.append(len(intersection) / len(set_b))
        
        # Jaccard
        if len(union) > 0:
            jaccard.append(len(intersection) / len(union))
    
    # Combine prop_a_in_b and prop_b_in_a for the "proportion in other" metric
    all_proportions = prop_a_in_b + prop_b_in_a
    
    return {
        "n_pairs": len(list(combinations(range(n_seeds), 2))),
        "circuit_sizes": [len(s) for s in circuit_sets],
        "mean_circuit_size": np.mean([len(s) for s in circuit_sets]),
        "std_circuit_size": np.std([len(s) for s in circuit_sets]),
        
        # Intersection statistics  
        "mean_intersection_size": np.mean(intersection_sizes),
        "std_intersection_size": np.std(intersection_sizes),
        
        # Proportion of one circuit in the other (the metric requested)
        "mean_prop_in_other": np.mean(all_proportions),
        "std_prop_in_other": np.std(all_proportions),
        "min_prop_in_other": np.min(all_proportions),
        "max_prop_in_other": np.max(all_proportions),
        
        # Jaccard similarity
        "mean_jaccard": np.mean(jaccard),
        "std_jaccard": np.std(jaccard),
        "min_jaccard": np.min(jaccard),
        "max_jaccard": np.max(jaccard),
        
        # Raw data
        "all_proportions": all_proportions,
        "all_jaccard": jaccard,
    }


def create_pareto_plot(
    results: List[Dict],
    model,
    tokenizer,
    mean_cache: Optional[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    output_dir: Path,
):
    """Create Pareto plot showing all seeds on the superval set."""
    print("\nCreating Pareto plot on superval set for all seeds...")
    
    device = config["device"]
    
    # Load superval task
    superval_task = get_task(config["task_name"], tokenizer, seed=42, split="superval")
    superval_templates = len(superval_task.templates) if hasattr(superval_task, 'templates') else 'N/A'
    print(f"Superval task: {superval_task.name} ({superval_templates} templates)")
    
    # Autocast context
    if config.get("use_autocast", True):
        dtype = torch.bfloat16 if config.get("autocast_dtype", "bfloat16") == "bfloat16" else torch.float16
        autocast_ctx = torch.autocast('cuda', dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    
    # Pre-generate fixed batches for consistent evaluation
    num_eval_batches = max(config.get("bisection_eval_batches", 5), 20)
    print(f"Pre-generating {num_eval_batches} fixed evaluation batches...")
    
    # Create a temporary config for evaluation
    pruning_config = PruningConfig(
        batch_size=config["batch_size"],
        seq_length=config.get("task_max_length", 0),
        device=device,
        ablation_type=config["ablation_type"],
        mask_token_embeds=config["mask_token_embeds"],
    )
    
    fixed_batches = []
    for _ in range(num_eval_batches):
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = superval_task.generate_batch(
            batch_size=pruning_config.batch_size,
            max_length=pruning_config.seq_length,
        )
        # Store as tuple (the format expected by evaluate_at_k_fixed_batches)
        fixed_batches.append((
            positive_ids.to(device),
            negative_ids.to(device),
            correct_tokens.to(device),
            incorrect_tokens.to(device),
            eval_positions.to(device),
        ))
    
    # Collect Pareto data for each seed
    all_pareto_data = {}
    
    for result in results:
        seed = result["seed"]
        print(f"\nEvaluating seed {seed}...")
        
        # Create masked model and load state
        masked_model = MaskedSparseGPT(model, pruning_config)
        masked_model.to(device)
        if config["ablation_type"] != "zero" and mean_cache is not None:
            masked_model.set_means_from_dict(mean_cache)
        masked_model.load_mask_state(result["mask_state"])
        
        # Get max nodes
        num_active = masked_model.masks.get_total_active_nodes()
        total_nodes = masked_model.masks.get_total_nodes()
        
        original_state = masked_model.get_mask_state()
        
        # Sample k values logarithmically
        k_values = np.unique(np.geomspace(1, num_active, num=25).astype(int))
        k_values = sorted(set(k_values) | {1, num_active})
        
        pareto_data = []
        for k in tqdm(k_values, desc=f"Seed {seed} Pareto"):
            with autocast_ctx:
                loss = evaluate_at_k_fixed_batches(masked_model, fixed_batches, k, device=device)
            masked_model.load_mask_state(original_state)
            
            pareto_data.append({
                "k": int(k),
                "loss": float(loss),
                "frac": k / total_nodes,
            })
        
        all_pareto_data[seed] = {
            "pareto_curve": pareto_data,
            "num_active": num_active,
            "total_nodes": total_nodes,
        }
    
    # Save Pareto data
    with open(output_dir / "pareto_all_seeds.json", "w") as f:
        json.dump(all_pareto_data, f, indent=2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))
    
    for i, (seed, data) in enumerate(all_pareto_data.items()):
        ks = [d["k"] for d in data["pareto_curve"]]
        losses = [d["loss"] for d in data["pareto_curve"]]
        
        ax.plot(ks, losses, '-o', markersize=3, linewidth=1.5, 
                color=colors[i], alpha=0.7, label=f'Seed {seed}')
    
    ax.axhline(y=config["target_loss"], color='r', linestyle='--', 
               linewidth=2, label=f'Target: {config["target_loss"]}')
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title(f"Pareto Curves for {len(results)} Seeds\nModel: {config['model_path'].split('/')[-1]}", fontsize=14)
    ax.set_xscale('log')
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_all_seeds.png", dpi=150)
    plt.close()
    
    print(f"Pareto plot saved to {output_dir / 'pareto_all_seeds.png'}")
    
    return all_pareto_data


def main():
    parser = argparse.ArgumentParser(description="Run pruning with multiple seeds")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Path to CARBS best checkpoint directory")
    parser.add_argument("--num-seeds", type=int, default=10,
                       help="Number of seeds to run (default: 10)")
    parser.add_argument("--start-seed", type=int, default=0,
                       help="Starting seed number (default: 0)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (default: cuda)")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Load sweep config
    sweep_config_path = checkpoint_dir / "sweep_config.json"
    with open(sweep_config_path) as f:
        config = json.load(f)
    
    # Load best hparams
    hparams_path = checkpoint_dir / "best_checkpoint" / "hparams.json"
    with open(hparams_path) as f:
        hparams = json.load(f)
    
    # Remove suggestion_uuid if present
    hparams = {k: v for k, v in hparams.items() if k != "suggestion_uuid"}
    
    config["device"] = args.device
    
    print("=" * 70)
    print("Running Seed Repetitions")
    print("=" * 70)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Model: {config['model_path']}")
    print(f"Task: {config['task_name']}")
    print(f"Ablation type: {config['ablation_type']}")
    print(f"Mask token embeds: {config['mask_token_embeds']}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Hyperparameters:")
    for k, v in hparams.items():
        print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 70)
    
    device = config["device"]
    
    # Output directory
    output_dir = checkpoint_dir / "repetitions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model, _ = load_model(config["model_path"], device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create tasks
    train_task = get_task(config["task_name"], tokenizer, seed=42, split="train")
    val_task = get_task(config["task_name"], tokenizer, seed=42, split="val")
    train_templates = len(train_task.templates) if hasattr(train_task, 'templates') else 'N/A'
    val_templates = len(val_task.templates) if hasattr(val_task, 'templates') else 'N/A'
    print(f"Train task: {train_task.name} ({train_templates} templates)")
    print(f"Val task: {val_task.name} ({val_templates} templates)")
    
    # Load or compute mean cache
    mean_cache = None
    if config["ablation_type"] != "zero":
        mean_cache_path = checkpoint_dir / "mean_cache.pt"
        if mean_cache_path.exists():
            print(f"Loading mean cache from {mean_cache_path}...")
            mean_cache = torch.load(mean_cache_path, map_location=device, weights_only=True)
            mean_cache = {k: v.to(device) for k, v in mean_cache.items()}
        else:
            print("Mean cache not found, computing...")
            temp_config = PruningConfig(device=device, mask_token_embeds=config["mask_token_embeds"])
            temp_masked = MaskedSparseGPT(model, temp_config)
            temp_masked.to(device)
            
            data_iter = create_data_iterator(
                tokenizer_name=config["tokenizer_name"],
                batch_size=64,
                seq_length=config.get("mean_cache_seq_length", 256),
                num_batches=config.get("mean_cache_batches", 100),
                seed=42,
            )
            mean_cache = temp_masked.compute_mean_cache(
                data_iter, 
                num_batches=config.get("mean_cache_batches", 100), 
                show_progress=True
            )
            del temp_masked
            torch.save(mean_cache, output_dir / "mean_cache.pt")
    
    torch.cuda.empty_cache()
    
    # Run all seeds
    results = []
    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")
        
        result = run_single_seed(
            seed=seed,
            hparams=hparams,
            model=model,
            tokenizer=tokenizer,
            train_task=train_task,
            val_task=val_task,
            mean_cache=mean_cache,
            config=config,
            output_dir=output_dir,
        )
        
        results.append(result)
        
        print(f"Seed {seed}: circuit_size={result['circuit_size']}, "
              f"val_loss={result['achieved_loss_val']:.4f}, "
              f"target_achieved={result['target_achieved']}")
    
    # Create Pareto plot
    pareto_data = create_pareto_plot(
        results=results,
        model=model,
        tokenizer=tokenizer,
        mean_cache=mean_cache,
        config=config,
        output_dir=output_dir,
    )
    
    # Compute overlap statistics
    print("\n" + "=" * 70)
    print("Computing pairwise circuit overlap...")
    print("=" * 70)
    
    overlap_stats = compute_pairwise_overlap(results)
    
    print(f"\nNumber of pairs: {overlap_stats['n_pairs']}")
    print(f"\nCircuit size statistics:")
    print(f"  Mean: {overlap_stats['mean_circuit_size']:.1f}")
    print(f"  Std: {overlap_stats['std_circuit_size']:.1f}")
    print(f"  Sizes: {overlap_stats['circuit_sizes']}")
    
    print(f"\nProportion of nodes in one circuit that are also in the other:")
    print(f"  Mean: {overlap_stats['mean_prop_in_other']:.4f} ({overlap_stats['mean_prop_in_other']*100:.1f}%)")
    print(f"  Std: {overlap_stats['std_prop_in_other']:.4f}")
    print(f"  Min: {overlap_stats['min_prop_in_other']:.4f}")
    print(f"  Max: {overlap_stats['max_prop_in_other']:.4f}")
    
    print(f"\nJaccard similarity (intersection / union):")
    print(f"  Mean: {overlap_stats['mean_jaccard']:.4f}")
    print(f"  Std: {overlap_stats['std_jaccard']:.4f}")
    print(f"  Min: {overlap_stats['min_jaccard']:.4f}")
    print(f"  Max: {overlap_stats['max_jaccard']:.4f}")
    
    # Save overlap stats (without raw data for cleaner JSON)
    overlap_summary = {k: v for k, v in overlap_stats.items() 
                       if k not in ["all_proportions", "all_jaccard"]}
    with open(output_dir / "overlap_stats.json", "w") as f:
        json.dump(overlap_summary, f, indent=2)
    
    # Save full results summary
    full_summary = {
        "config": config,
        "hparams": hparams,
        "num_seeds": args.num_seeds,
        "seeds_run": list(range(args.start_seed, args.start_seed + args.num_seeds)),
        "results": [{k: v for k, v in r.items() if k not in ["mask_state", "circuit_mask"]} 
                    for r in results],
        "overlap_stats": overlap_summary,
    }
    with open(output_dir / "full_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)
    
    # Create overlap histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of proportion in other
    ax1 = axes[0]
    ax1.hist(overlap_stats['all_proportions'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(x=overlap_stats['mean_prop_in_other'], color='red', linestyle='--', 
                label=f'Mean: {overlap_stats["mean_prop_in_other"]:.3f}')
    ax1.set_xlabel("Proportion of nodes in common", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of Pairwise Node Overlap", fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    # Histogram of Jaccard
    ax2 = axes[1]
    ax2.hist(overlap_stats['all_jaccard'], bins=20, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(x=overlap_stats['mean_jaccard'], color='red', linestyle='--',
                label=f'Mean: {overlap_stats["mean_jaccard"]:.3f}')
    ax2.set_xlabel("Jaccard Similarity", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Jaccard Similarity", fontsize=14)
    ax2.legend()
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "overlap_histograms.png", dpi=150)
    plt.close()
    
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

