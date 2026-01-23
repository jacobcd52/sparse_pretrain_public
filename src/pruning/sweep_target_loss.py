"""
Sweep over target losses to generate a loss vs circuit size curve.

This script loads a trained pruning checkpoint and runs discretization
at multiple target loss values to find the Pareto frontier of
loss vs circuit size.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.discretize import evaluate_at_k


def sweep_target_losses(
    masked_model: MaskedSparseGPT,
    task,
    config: PruningConfig,
    target_losses: List[float],
    num_eval_batches: int = 20,
    show_progress: bool = True,
) -> List[Tuple[float, int, float]]:
    """
    Sweep over target losses and find circuit size for each.
    
    Args:
        masked_model: Masked model with trained masks
        task: Binary task for evaluation
        config: Pruning config
        target_losses: List of target loss values to sweep
        num_eval_batches: Batches per evaluation
        show_progress: Whether to print progress
        
    Returns:
        List of (target_loss, circuit_size, achieved_loss) tuples
    """
    results = []
    
    # Count active nodes (tau >= 0)
    num_active = 0
    for key, mask in masked_model.masks.masks.items():
        num_active += (mask.tau >= 0).sum().item()
    num_active = int(num_active)
    
    total_nodes = sum(mask.num_nodes for mask in masked_model.masks.masks.values())
    
    if show_progress:
        print(f"Total nodes in model: {total_nodes}")
        print(f"Active nodes after training: {num_active}")
        print(f"Sweeping over {len(target_losses)} target loss values...")
        print()
    
    # Save original mask state
    original_state = masked_model.get_mask_state()
    
    # Evaluate at all active nodes first (best possible loss)
    loss_at_all = evaluate_at_k(masked_model, task, num_active, config, num_eval_batches)
    if show_progress:
        print(f"Loss with all {num_active} active nodes: {loss_at_all:.4f}")
    
    # Evaluate at 1 node (worst loss in our range)
    loss_at_one = evaluate_at_k(masked_model, task, 1, config, num_eval_batches)
    if show_progress:
        print(f"Loss with 1 node: {loss_at_one:.4f}")
        print()
    
    for i, target_loss in enumerate(sorted(target_losses, reverse=True)):
        if show_progress:
            print(f"[{i+1}/{len(target_losses)}] Target loss: {target_loss:.4f}")
        
        # Binary search for minimum k that achieves target loss
        k_low = 1
        k_high = num_active
        best_k = num_active
        best_loss = loss_at_all
        
        # Check if target is achievable
        if loss_at_all > target_loss:
            if show_progress:
                print(f"  Cannot achieve target (best possible: {loss_at_all:.4f})")
            results.append((target_loss, num_active, loss_at_all))
            continue
        
        # Check if we need more than 1 node
        if loss_at_one <= target_loss:
            if show_progress:
                print(f"  Achieved with 1 node (loss: {loss_at_one:.4f})")
            results.append((target_loss, 1, loss_at_one))
            continue
        
        # Binary search
        while k_high - k_low > 1:
            k_mid = (k_low + k_high) // 2
            
            loss_mid = evaluate_at_k(masked_model, task, k_mid, config, num_eval_batches)
            
            if loss_mid <= target_loss:
                # Can achieve target, try fewer nodes
                k_high = k_mid
                best_k = k_mid
                best_loss = loss_mid
            else:
                # Need more nodes
                k_low = k_mid
        
        # Final check at k_high
        final_loss = evaluate_at_k(masked_model, task, k_high, config, num_eval_batches)
        if final_loss <= target_loss:
            best_k = k_high
            best_loss = final_loss
        
        if show_progress:
            print(f"  k={best_k}, achieved_loss={best_loss:.4f}")
        
        results.append((target_loss, best_k, best_loss))
        
        # Restore original state for next iteration
        masked_model.load_mask_state(original_state)
    
    return results


def plot_loss_vs_circuit_size(
    results: List[Tuple[float, int, float]],
    total_nodes: int,
    output_path: str,
    title: str = "Loss vs Circuit Size",
):
    """
    Plot and save loss vs circuit size curve.
    
    Args:
        results: List of (target_loss, circuit_size, achieved_loss) tuples
        total_nodes: Total number of nodes in model
        output_path: Path to save the plot
        title: Plot title
    """
    # Sort by circuit size
    results_sorted = sorted(results, key=lambda x: x[1])
    
    circuit_sizes = [r[1] for r in results_sorted]
    achieved_losses = [r[2] for r in results_sorted]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Linear scale
    ax1.plot(circuit_sizes, achieved_losses, 'b-o', markersize=8, linewidth=2)
    ax1.set_xlabel('Circuit Size (# active nodes)', fontsize=12)
    ax1.set_ylabel('Task Loss', fontsize=12)
    ax1.set_title(f'{title} (Linear Scale)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Add percentage annotation
    for i, (cs, loss) in enumerate(zip(circuit_sizes, achieved_losses)):
        if i % max(1, len(circuit_sizes) // 5) == 0:  # Label every ~5 points
            pct = 100 * cs / total_nodes
            ax1.annotate(f'{pct:.1f}%', (cs, loss), textcoords="offset points", 
                        xytext=(5, 5), fontsize=8)
    
    # Plot 2: Log-log scale
    ax2.loglog(circuit_sizes, achieved_losses, 'b-o', markersize=8, linewidth=2)
    ax2.set_xlabel('Circuit Size (# active nodes)', fontsize=12)
    ax2.set_ylabel('Task Loss', fontsize=12)
    ax2.set_title(f'{title} (Log-Log Scale)', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train masks and sweep over target losses to generate loss vs circuit size curve"
    )
    
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (HF repo ID or local path)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name (defaults to model config)")
    
    # Task
    parser.add_argument("--task", type=str, default="dummy_pronoun",
                        help="Task name")
    
    # Mask training parameters
    parser.add_argument("--num_training_steps", type=int, default=2000,
                        help="Number of mask training steps")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate for mask training")
    parser.add_argument("--k_coef", type=float, default=1e-4,
                        help="Sparsity penalty coefficient")
    
    # Sweep parameters
    parser.add_argument("--min_loss", type=float, default=0.002,
                        help="Minimum target loss")
    parser.add_argument("--max_loss", type=float, default=0.5,
                        help="Maximum target loss")
    parser.add_argument("--num_sweep_points", type=int, default=8,
                        help="Number of log-spaced target loss values")
    
    # Evaluation
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--num_eval_batches", type=int, default=20,
                        help="Number of batches per evaluation")
    
    # Mean cache
    parser.add_argument("--skip_mean_cache", action="store_true",
                        help="Skip mean cache (use zero ablation)")
    parser.add_argument("--dataset", type=str, default="SimpleStories/SimpleStories",
                        help="Dataset for mean cache computation")
    parser.add_argument("--text_column", type=str, default="story",
                        help="Text column in dataset")
    parser.add_argument("--mean_cache_batches", type=int, default=100,
                        help="Batches for mean cache computation")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for all results")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, model_config_dict = load_model(args.model_path, args.device)
    print(f"Model loaded: {model.config.n_layer} layers, d_model={model.config.d_model}")
    
    # Get tokenizer
    tokenizer_name = args.tokenizer
    if tokenizer_name is None:
        tokenizer_name = model_config_dict.get("training_config", {}).get("tokenizer_name")
    
    if tokenizer_name is None:
        raise ValueError(
            "No tokenizer found. Either:\n"
            "  1. Specify --tokenizer argument\n"
            "  2. Ensure the model config.json has training_config.tokenizer_name"
        )
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create task
    print(f"Creating task: {args.task}")
    task = get_task(args.task, tokenizer, seed=args.seed)
    
    # Create config with training parameters
    config = PruningConfig(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_steps=args.num_training_steps,
        lr=args.lr,
        k_coef=args.k_coef,
        device=args.device,
    )
    
    # Create masked model
    print("Creating masked model...")
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(args.device)  # Ensure all components are on the right device
    
    # Compute mean cache if not skipping
    mean_cache_path = output_dir / "mean_cache.pt"
    
    if not args.skip_mean_cache:
        if mean_cache_path.exists():
            print(f"Loading mean cache from {mean_cache_path}...")
            mean_cache = torch.load(mean_cache_path, map_location=args.device)
            mean_cache = {k: v.to(args.device) for k, v in mean_cache.items()}
            masked_model.set_means_from_dict(mean_cache)
            print("Mean cache loaded.")
        else:
            print("Computing mean cache...")
            from my_sparse_pretrain.src.pruning.run_pruning import create_data_iterator
            data_iter = create_data_iterator(
                tokenizer_name=tokenizer_name,
                dataset_name=args.dataset,
                text_column=args.text_column,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                num_batches=args.mean_cache_batches,
                seed=args.seed,
            )
            mean_cache = masked_model.compute_mean_cache(
                data_iter,
                num_batches=args.mean_cache_batches,
                show_progress=True,
            )
            masked_model.set_means_from_dict(mean_cache)
            torch.save(mean_cache, mean_cache_path)
            print(f"Mean cache saved to {mean_cache_path}")
    else:
        print("Skipping mean cache (using zero ablation)")
    
    # ==========================================================================
    # STEP 1: Train masks
    # ==========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: Training masks")
    print(f"{'='*60}")
    print(f"Training for {args.num_training_steps} steps...")
    
    from my_sparse_pretrain.src.pruning.trainer import PruningTrainer
    
    trainer = PruningTrainer(masked_model, task, config)
    trainer.train()
    
    # Save checkpoint
    checkpoint_path = output_dir / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # ==========================================================================
    # STEP 2: Sweep over target losses
    # ==========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Sweeping over target losses")
    print(f"{'='*60}")
    
    # Generate target losses (log-spaced)
    target_losses = np.logspace(
        np.log10(args.min_loss),
        np.log10(args.max_loss),
        args.num_sweep_points
    ).tolist()
    
    print(f"Target losses: {[f'{t:.4f}' for t in target_losses]}")
    print()
    
    # Run sweep
    results = sweep_target_losses(
        masked_model=masked_model,
        task=task,
        config=config,
        target_losses=target_losses,
        num_eval_batches=args.num_eval_batches,
        show_progress=True,
    )
    
    # Get total nodes
    total_nodes = sum(mask.num_nodes for mask in masked_model.masks.masks.values())
    
    # Save results as JSON
    results_data = {
        "model_path": args.model_path,
        "task": args.task,
        "total_nodes": total_nodes,
        "sweep_results": [
            {
                "target_loss": r[0],
                "circuit_size": r[1],
                "achieved_loss": r[2],
                "circuit_fraction": r[1] / total_nodes,
            }
            for r in results
        ],
    }
    
    results_path = output_dir / "loss_sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Plot
    plot_path = output_dir / "loss_vs_circuit_size.png"
    plot_loss_vs_circuit_size(
        results=results,
        total_nodes=total_nodes,
        output_path=str(plot_path),
        title=f"Loss vs Circuit Size ({args.task})",
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"{'Target Loss':>12} | {'Circuit Size':>12} | {'Achieved Loss':>13} | {'% of Model':>10}")
    print("-" * 60)
    for target, size, achieved in sorted(results, key=lambda x: x[1]):
        pct = 100 * size / total_nodes
        print(f"{target:>12.4f} | {size:>12} | {achieved:>13.4f} | {pct:>9.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

