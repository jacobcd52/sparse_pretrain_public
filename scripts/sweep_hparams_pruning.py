"""
Hyperparameter sweep for circuit pruning.

Runs multiple pruning experiments with different hyperparameters and
generates a combined pareto plot showing loss vs circuit size for each setting.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.trainer import PruningTrainer
from sparse_pretrain.src.pruning.discretize import evaluate_at_k


@dataclass
class SweepResult:
    """Result from a single hyperparameter setting."""
    hparams: Dict[str, Any]
    pareto_points: List[Tuple[float, int, float]]  # (target_loss, circuit_size, achieved_loss)
    total_nodes: int
    run_name: str


def sweep_target_losses(
    masked_model: MaskedSparseGPT,
    task,
    config: PruningConfig,
    target_losses: List[float],
    num_eval_batches: int = 10,
) -> List[Tuple[float, int, float]]:
    """
    Sweep over target losses to find minimum circuit size for each.
    
    Returns:
        List of (target_loss, circuit_size, achieved_loss) tuples
    """
    results = []
    
    # Count active nodes (tau >= 0)
    num_active = 0
    for key, mask in masked_model.masks.masks.items():
        num_active += (mask.tau >= 0).sum().item()
    num_active = int(num_active)
    
    if num_active == 0:
        return [(t, 0, float('inf')) for t in target_losses]
    
    # Save original mask state
    original_state = masked_model.get_mask_state()
    
    # Evaluate at all active nodes first (best possible loss)
    loss_at_all = evaluate_at_k(masked_model, task, num_active, config, num_eval_batches)
    
    # Evaluate at 1 node (worst loss in our range)
    loss_at_one = evaluate_at_k(masked_model, task, 1, config, num_eval_batches)
    
    print(f"  Active nodes: {num_active}, loss@1: {loss_at_one:.4f}, loss@all: {loss_at_all:.4f}")
    
    for target_loss in sorted(target_losses, reverse=True):
        # Check if target is achievable
        if loss_at_all > target_loss:
            results.append((target_loss, num_active, loss_at_all))
            continue
        
        # Check if we need more than 1 node
        if loss_at_one <= target_loss:
            results.append((target_loss, 1, loss_at_one))
            continue
        
        # Binary search for minimum k
        k_low = 1
        k_high = num_active
        best_k = num_active
        best_loss = loss_at_all
        
        while k_high - k_low > 1:
            k_mid = (k_low + k_high) // 2
            loss_mid = evaluate_at_k(masked_model, task, k_mid, config, num_eval_batches)
            
            if loss_mid <= target_loss:
                k_high = k_mid
                best_k = k_mid
                best_loss = loss_mid
            else:
                k_low = k_mid
        
        # Final check at k_high
        final_loss = evaluate_at_k(masked_model, task, k_high, config, num_eval_batches)
        if final_loss <= target_loss:
            best_k = k_high
            best_loss = final_loss
        
        results.append((target_loss, best_k, best_loss))
        
        # Restore original state for next iteration
        masked_model.load_mask_state(original_state)
    
    return sorted(results, key=lambda x: x[0])  # Sort by target loss


def run_single_sweep(
    model_path: str,
    task_name: str,
    hparams: Dict[str, Any],
    target_losses: List[float],
    device: str = "cuda",
    num_steps: int = 250,
    batch_size: int = 64,
    seq_length: int = 256,
    mean_cache_batches: int = 50,
    seed: int = 42,
    run_name: str = "run",
    mean_cache: Dict[str, torch.Tensor] = None,
    tokenizer_name: str = None,
) -> SweepResult:
    """Run a single pruning experiment with given hyperparameters."""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"Hyperparameters: {hparams}")
    print(f"{'='*60}")
    
    model, model_config_dict = load_model(model_path, device)
    
    # Get tokenizer
    if tokenizer_name is None:
        tokenizer_name = model_config_dict.get("training_config", {}).get("tokenizer_name")
    if tokenizer_name is None:
        raise ValueError("No tokenizer found in model config")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create task
    task = get_task(task_name, tokenizer, seed=seed)
    
    # Create config with hyperparameters
    config = PruningConfig(
        batch_size=batch_size,
        seq_length=seq_length,
        num_steps=num_steps,
        device=device,
        **hparams,
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)
    
    # Use provided mean cache or compute new one
    if mean_cache is not None:
        masked_model.set_means_from_dict({k: v.to(device) for k, v in mean_cache.items()})
    else:
        print("Computing mean cache...")
        data_iter = create_data_iterator(
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            seq_length=seq_length,
            num_batches=mean_cache_batches,
            seed=seed,
        )
        mean_cache = masked_model.compute_mean_cache(
            data_iter,
            num_batches=mean_cache_batches,
            show_progress=True,
        )
        masked_model.set_means_from_dict(mean_cache)
    
    # Train masks
    print(f"Training masks for {num_steps} steps...")
    trainer = PruningTrainer(masked_model, task, config)
    final_metrics = trainer.train(show_progress=True)
    
    print(f"Training complete. Final: loss={final_metrics['task_loss']:.4f}, "
          f"active_nodes={final_metrics['num_active_nodes']}, acc={final_metrics['accuracy']:.2%}")
    
    # Sweep over target losses
    print("Sweeping target losses...")
    pareto_points = sweep_target_losses(masked_model, task, config, target_losses)
    
    total_nodes = masked_model.masks.get_total_nodes()
    
    # Clean up GPU memory
    del masked_model, model, trainer
    torch.cuda.empty_cache()
    
    return SweepResult(
        hparams=hparams,
        pareto_points=pareto_points,
        total_nodes=total_nodes,
        run_name=run_name,
    ), mean_cache, tokenizer_name


def plot_pareto_curves(
    results: List[SweepResult],
    output_path: str,
    title: str = "Pareto Curves: Circuit Size vs Target Loss",
):
    """Plot all pareto curves with different markers for lr and colors for k_coef."""
    
    # Define marker shapes for different lr values
    lr_markers = {
        0.001: 'o',   # circle
        0.01: 's',    # square
        0.1: '^',     # triangle
        0.003: 'D',   # diamond
    }
    
    # Define colors for different k_coef values
    k_colors = {
        0.001: '#1f77b4',   # blue
        0.01: '#ff7f0e',    # orange
        0.1: '#2ca02c',     # green
        0.0001: '#d62728',  # red
        1e-05: '#9467bd',   # purple
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for result in results:
        k_coef = result.hparams['k_coef']
        lr = result.hparams['lr']
        
        # Get marker and color
        marker = lr_markers.get(lr, 'x')
        color = k_colors.get(k_coef, 'gray')
        
        # Extract data
        target_losses = [p[0] for p in result.pareto_points]
        circuit_sizes = [p[1] for p in result.pareto_points]
        
        label = f"k={k_coef:.0e}, lr={lr:.0e}"
        
        # Linear scale plot (circuit size vs target loss)
        ax1.plot(target_losses, circuit_sizes, marker=marker, color=color, 
                label=label, markersize=8, linewidth=2, alpha=0.8)
        
        # Log-log scale plot
        ax2.loglog(target_losses, circuit_sizes, marker=marker, color=color,
                  label=label, markersize=8, linewidth=2, alpha=0.8)
    
    # Configure linear plot
    ax1.set_xlabel('Target Loss', fontsize=12)
    ax1.set_ylabel('Circuit Size (# nodes)', fontsize=12)
    ax1.set_title(f'{title} (Linear Scale)', fontsize=14)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # Configure log-log plot
    ax2.set_xlabel('Target Loss', fontsize=12)
    ax2.set_ylabel('Circuit Size (# nodes)', fontsize=12)
    ax2.set_title(f'{title} (Log-Log Scale)', fontsize=14)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add legend explanation
    fig.text(0.5, 0.02, 'Markers: ○=lr=1e-3, □=lr=1e-2  |  Colors: blue=k=1e-3, orange=k=1e-2, green=k=1e-1', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for circuit pruning")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model (HF repo ID or local path)")
    parser.add_argument("--task", type=str, default="dummy_pronoun",
                        help="Task name")
    parser.add_argument("--output_dir", type=str, default="outputs/pruning_sweep",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_steps", type=int, default=250,
                        help="Training steps per run")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    
    # Target loss sweep parameters
    parser.add_argument("--min_target_loss", type=float, default=0.001,
                        help="Minimum target loss")
    parser.add_argument("--max_target_loss", type=float, default=0.5,
                        help="Maximum target loss")
    parser.add_argument("--num_target_losses", type=int, default=8,
                        help="Number of log-spaced target losses")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate target losses (log-spaced)
    target_losses = np.logspace(
        np.log10(args.min_target_loss),
        np.log10(args.max_target_loss),
        args.num_target_losses
    ).tolist()
    
    print(f"Target losses: {[f'{t:.4f}' for t in target_losses]}")
    
    # Define hyperparameter grid
    hparam_grid = {
        'k_coef': [1e-3, 1e-2, 1e-1],
        'lr': [1e-3, 1e-2],
    }
    
    # Generate all combinations
    all_combos = list(product(*hparam_grid.values()))
    param_names = list(hparam_grid.keys())
    
    print(f"\nRunning {len(all_combos)} experiments:")
    for i, combo in enumerate(all_combos):
        hparams = dict(zip(param_names, combo))
        print(f"  {i+1}. {hparams}")
    
    # Run experiments
    results = []
    mean_cache = None
    tokenizer_name = None
    
    for i, combo in enumerate(all_combos):
        hparams = dict(zip(param_names, combo))
        run_name = f"k={hparams['k_coef']:.0e}, lr={hparams['lr']:.0e}"
        
        try:
            result, mean_cache, tokenizer_name = run_single_sweep(
                model_path=args.model_path,
                task_name=args.task,
                hparams=hparams,
                target_losses=target_losses,
                device=args.device,
                num_steps=args.num_steps,
                batch_size=args.batch_size,
                seed=args.seed + i,
                run_name=run_name,
                mean_cache=mean_cache,
                tokenizer_name=tokenizer_name,
            )
            results.append(result)
            
            # Save intermediate results
            intermediate_path = output_dir / "intermediate_results.json"
            with open(intermediate_path, 'w') as f:
                json.dump([{
                    'run_name': r.run_name,
                    'hparams': r.hparams,
                    'pareto_points': r.pareto_points,
                    'total_nodes': r.total_nodes,
                } for r in results], f, indent=2)
                
        except Exception as e:
            print(f"ERROR in run {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("No successful runs. Exiting.")
        return
    
    # Save final results
    final_results_path = output_dir / "sweep_results.json"
    with open(final_results_path, 'w') as f:
        json.dump({
            'model_path': args.model_path,
            'task': args.task,
            'num_steps': args.num_steps,
            'target_losses': target_losses,
            'results': [{
                'run_name': r.run_name,
                'hparams': r.hparams,
                'pareto_points': r.pareto_points,
                'total_nodes': r.total_nodes,
            } for r in results]
        }, f, indent=2)
    print(f"\nResults saved to: {final_results_path}")
    
    # Generate combined plot
    plot_path = output_dir / "pareto_curves.png"
    plot_pareto_curves(
        results=results,
        output_path=str(plot_path),
        title=f"Circuit Size vs Target Loss ({args.task})",
    )
    
    # Print summary table
    print("\n" + "="*80)
    print("SWEEP COMPLETE")
    print("="*80)
    print(f"\nTarget losses: {[f'{t:.4f}' for t in target_losses]}")
    print(f"\n{'Run':<20} | " + " | ".join([f"L≤{t:.3f}" for t in target_losses]))
    print("-" * 80)
    
    for result in results:
        row = f"{result.run_name:<20} |"
        for target, size, achieved in result.pareto_points:
            row += f" {size:>6} |"
        print(row)
    
    print("="*80)
    print(f"\nPlot saved to: {plot_path}")
    print(f"Results saved to: {final_results_path}")


if __name__ == "__main__":
    main()
