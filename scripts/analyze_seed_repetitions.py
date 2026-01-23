#!/usr/bin/env python3
"""
Analyze saved seed repetitions - create Pareto plot and compute overlap statistics.

Usage:
    python scripts/analyze_seed_repetitions.py \
        --repetitions-dir outputs/carbs_results_pronoun/ss_bridges_d1024_f0.015625_zero_noembed/repetitions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from contextlib import nullcontext
from tqdm import tqdm
from itertools import combinations

from transformers import AutoTokenizer

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model
from sparse_pretrain.src.pruning.discretize import evaluate_at_k_fixed_batches


def get_circuit_nodes_set(circuit_mask: Dict[str, torch.Tensor]) -> set:
    """
    Convert a circuit mask dictionary to a set of active node identifiers.
    """
    active_nodes = set()
    for location, mask in circuit_mask.items():
        active_indices = torch.where(mask)[0].cpu().tolist()
        for idx in active_indices:
            active_nodes.add((location, idx))
    return active_nodes


def compute_pairwise_overlap(circuit_masks: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Compute pairwise overlap between all pairs of circuits."""
    n_seeds = len(circuit_masks)
    
    # Convert all circuit masks to sets
    circuit_sets = []
    for cm in circuit_masks:
        nodes = get_circuit_nodes_set(cm)
        circuit_sets.append(nodes)
    
    # Compute pairwise overlaps
    prop_a_in_b = []
    prop_b_in_a = []
    jaccard = []
    intersection_sizes = []
    union_sizes = []
    
    # New metric: proportion of smaller circuit in larger
    prop_smaller_in_larger = []
    
    for i, j in combinations(range(n_seeds), 2):
        set_a = circuit_sets[i]
        set_b = circuit_sets[j]
        
        intersection = set_a & set_b
        union = set_a | set_b
        
        intersection_sizes.append(len(intersection))
        union_sizes.append(len(union))
        
        if len(set_a) > 0:
            prop_a_in_b.append(len(intersection) / len(set_a))
        if len(set_b) > 0:
            prop_b_in_a.append(len(intersection) / len(set_b))
        
        if len(union) > 0:
            jaccard.append(len(intersection) / len(union))
        
        # Proportion of smaller circuit that is in larger circuit
        smaller_set = set_a if len(set_a) <= len(set_b) else set_b
        if len(smaller_set) > 0:
            prop_smaller_in_larger.append(len(intersection) / len(smaller_set))
    
    all_proportions = prop_a_in_b + prop_b_in_a
    
    return {
        "n_pairs": len(list(combinations(range(n_seeds), 2))),
        "circuit_sizes": [len(s) for s in circuit_sets],
        "mean_circuit_size": np.mean([len(s) for s in circuit_sets]),
        "std_circuit_size": np.std([len(s) for s in circuit_sets]),
        "mean_intersection_size": np.mean(intersection_sizes),
        "std_intersection_size": np.std(intersection_sizes),
        "mean_prop_in_other": np.mean(all_proportions),
        "std_prop_in_other": np.std(all_proportions),
        "min_prop_in_other": np.min(all_proportions),
        "max_prop_in_other": np.max(all_proportions),
        "mean_jaccard": np.mean(jaccard),
        "std_jaccard": np.std(jaccard),
        "min_jaccard": np.min(jaccard),
        "max_jaccard": np.max(jaccard),
        # New metric: proportion of smaller in larger
        "mean_prop_smaller_in_larger": np.mean(prop_smaller_in_larger),
        "std_prop_smaller_in_larger": np.std(prop_smaller_in_larger),
        "min_prop_smaller_in_larger": np.min(prop_smaller_in_larger),
        "max_prop_smaller_in_larger": np.max(prop_smaller_in_larger),
        "all_proportions": all_proportions,
        "all_jaccard": jaccard,
        "all_prop_smaller_in_larger": prop_smaller_in_larger,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze seed repetitions")
    parser.add_argument("--repetitions-dir", type=str, required=True,
                       help="Path to repetitions directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (default: cuda)")
    
    args = parser.parse_args()
    
    repetitions_dir = Path(args.repetitions_dir)
    checkpoint_dir = repetitions_dir.parent
    
    # Load sweep config
    with open(checkpoint_dir / "sweep_config.json") as f:
        config = json.load(f)
    
    config["device"] = args.device
    device = args.device
    
    print("=" * 70)
    print("Analyzing Seed Repetitions")
    print("=" * 70)
    
    # Find all seed directories
    seed_dirs = sorted([d for d in repetitions_dir.iterdir() if d.is_dir() and d.name.startswith("seed")])
    print(f"Found {len(seed_dirs)} seed directories")
    
    # Load all masks and circuit masks
    mask_states = []
    circuit_masks = []
    summaries = []
    
    for seed_dir in seed_dirs:
        mask_state = torch.load(seed_dir / "masks.pt", map_location=device, weights_only=True)
        circuit_mask = torch.load(seed_dir / "circuit_mask.pt", map_location=device, weights_only=True)
        with open(seed_dir / "summary.json") as f:
            summary = json.load(f)
        
        mask_states.append(mask_state)
        circuit_masks.append(circuit_mask)
        summaries.append(summary)
        
        print(f"  {seed_dir.name}: circuit_size={summary['circuit_size']}, val_loss={summary['achieved_loss_val']:.4f}")
    
    # Compute overlap statistics
    print("\n" + "=" * 70)
    print("Computing pairwise circuit overlap...")
    print("=" * 70)
    
    overlap_stats = compute_pairwise_overlap(circuit_masks)
    
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
    
    print(f"\nProportion of SMALLER circuit's nodes that are in the LARGER circuit:")
    print(f"  Mean: {overlap_stats['mean_prop_smaller_in_larger']:.4f} ({overlap_stats['mean_prop_smaller_in_larger']*100:.1f}%)")
    print(f"  Std: {overlap_stats['std_prop_smaller_in_larger']:.4f}")
    print(f"  Min: {overlap_stats['min_prop_smaller_in_larger']:.4f}")
    print(f"  Max: {overlap_stats['max_prop_smaller_in_larger']:.4f}")
    
    # Save overlap stats
    overlap_summary = {k: v for k, v in overlap_stats.items() 
                       if k not in ["all_proportions", "all_jaccard", "all_prop_smaller_in_larger"]}
    with open(repetitions_dir / "overlap_stats.json", "w") as f:
        json.dump(overlap_summary, f, indent=2)
    
    # Create overlap histogram - now with 3 plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1 = axes[0]
    ax1.hist(overlap_stats['all_proportions'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(x=overlap_stats['mean_prop_in_other'], color='red', linestyle='--', 
                label=f'Mean: {overlap_stats["mean_prop_in_other"]:.3f}')
    ax1.set_xlabel("Proportion of nodes in common", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Distribution of Pairwise Node Overlap\n(both directions)", fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    ax2 = axes[1]
    ax2.hist(overlap_stats['all_jaccard'], bins=20, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(x=overlap_stats['mean_jaccard'], color='red', linestyle='--',
                label=f'Mean: {overlap_stats["mean_jaccard"]:.3f}')
    ax2.set_xlabel("Jaccard Similarity", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Jaccard Similarity\n(intersection / union)", fontsize=14)
    ax2.legend()
    ax2.set_xlim(0, 1)
    
    ax3 = axes[2]
    ax3.hist(overlap_stats['all_prop_smaller_in_larger'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax3.axvline(x=overlap_stats['mean_prop_smaller_in_larger'], color='red', linestyle='--',
                label=f'Mean: {overlap_stats["mean_prop_smaller_in_larger"]:.3f}')
    ax3.set_xlabel("Proportion of smaller circuit in larger", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("Prop. of Smaller Circuit's Nodes\nthat are in Larger Circuit", fontsize=14)
    ax3.legend()
    ax3.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(repetitions_dir / "overlap_histograms.png", dpi=150)
    plt.close()
    print(f"\nOverlap histograms saved to {repetitions_dir / 'overlap_histograms.png'}")
    
    # Now create Pareto plot
    print("\n" + "=" * 70)
    print("Creating Pareto plot on superval set for all seeds...")
    print("=" * 70)
    
    # Load model and tokenizer
    print("\nLoading model...")
    model, _ = load_model(config["model_path"], device)
    
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load superval task
    superval_task = get_task(config["task_name"], tokenizer, seed=42, split="superval")
    superval_templates = len(superval_task.templates) if hasattr(superval_task, 'templates') else 'N/A'
    print(f"Superval task: {superval_task.name} ({superval_templates} templates)")
    
    # Create config
    pruning_config = PruningConfig(
        batch_size=config["batch_size"],
        seq_length=config.get("task_max_length", 0),
        device=device,
        ablation_type=config["ablation_type"],
        mask_token_embeds=config["mask_token_embeds"],
    )
    
    # Pre-generate fixed batches
    num_eval_batches = 20
    print(f"Pre-generating {num_eval_batches} fixed evaluation batches...")
    
    fixed_batches = []
    for _ in range(num_eval_batches):
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = superval_task.generate_batch(
            batch_size=pruning_config.batch_size,
            max_length=pruning_config.seq_length,
        )
        fixed_batches.append((
            positive_ids.to(device),
            negative_ids.to(device),
            correct_tokens.to(device),
            incorrect_tokens.to(device),
            eval_positions.to(device),
        ))
    
    # Autocast
    dtype = torch.bfloat16 if config.get("autocast_dtype", "bfloat16") == "bfloat16" else torch.float16
    autocast_ctx = torch.autocast('cuda', dtype=dtype)
    
    # Collect Pareto data for each seed
    all_pareto_data = {}
    
    for i, (seed_dir, mask_state, summary) in enumerate(zip(seed_dirs, mask_states, summaries)):
        seed = summary["seed"]
        print(f"\nEvaluating seed {seed}...")
        
        # Create masked model and load state
        masked_model = MaskedSparseGPT(model, pruning_config)
        masked_model.to(device)
        masked_model.load_mask_state(mask_state)
        
        num_active = masked_model.masks.get_total_active_nodes()
        total_nodes = masked_model.masks.get_total_nodes()
        
        original_state = masked_model.get_mask_state()
        
        # Sample k values
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
    with open(repetitions_dir / "pareto_all_seeds.json", "w") as f:
        json.dump(all_pareto_data, f, indent=2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(seed_dirs)))
    
    for i, (seed, data) in enumerate(all_pareto_data.items()):
        ks = [d["k"] for d in data["pareto_curve"]]
        losses = [d["loss"] for d in data["pareto_curve"]]
        
        ax.plot(ks, losses, '-o', markersize=3, linewidth=1.5, 
                color=colors[i], alpha=0.7, label=f'Seed {seed}')
    
    ax.axhline(y=config["target_loss"], color='r', linestyle='--', 
               linewidth=2, label=f'Target: {config["target_loss"]}')
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title(f"Pareto Curves for {len(seed_dirs)} Seeds\nModel: {config['model_path'].split('/')[-1]}", fontsize=14)
    ax.set_xscale('log')
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(repetitions_dir / "pareto_all_seeds.png", dpi=150)
    plt.close()
    
    print(f"\nPareto plot saved to {repetitions_dir / 'pareto_all_seeds.png'}")
    
    # Save full summary
    full_summary = {
        "config": config,
        "num_seeds": len(seed_dirs),
        "results": summaries,
        "overlap_stats": overlap_summary,
    }
    with open(repetitions_dir / "full_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)
    
    print(f"\nFull summary saved to {repetitions_dir / 'full_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

