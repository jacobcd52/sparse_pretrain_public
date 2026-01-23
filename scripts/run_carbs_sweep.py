#!/usr/bin/env python3
"""
CARBS hyperparameter sweep for circuit pruning.

Based on Appendix A.5 of Gao et al. (2025):
- Uses CARBS (Cost Aware pareto-Region Bayesian Search) for hyperparameter optimization
- 32 iterations of CARBS with 8 parallel pruning jobs per iteration (256 total)
- Search centers from Table 2 of the paper

Run from the global_circuits directory:
    python scripts/run_carbs_sweep.py [--task TASK] [--iterations N] [--parallel N]

Results saved to outputs/carbs_sweep_<task>/
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
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import traceback
import pickle
from tqdm import tqdm

from transformers import AutoTokenizer

from carbs import CARBS, CARBSParams, Param, LogSpace, LogitSpace, LinearSpace, ObservationInParam

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.trainer import PruningTrainer
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator
from sparse_pretrain.src.pruning.discretize import evaluate_at_k

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class SweepConfig:
    """Configuration for the CARBS sweep."""
    # Task
    task_name: str = "dummy_pronoun"
    
    # Model
    model_path: str = "jacobcd52/ss_bridges_d1024_f0.015625"
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M"
    
    # CARBS settings (from paper: 32 iterations, 8 parallel)
    carbs_iterations: int = 32  # Number of CARBS iterations
    parallel_suggestions: int = 8  # Suggestions per iteration
    # Total = carbs_iterations * parallel_suggestions = 256
    
    # Training settings
    num_steps: int = 2000  # Steps per pruning run (not specified in paper)
    batch_size: int = 64  # Paper: 64 task datapoints (128 sequences)
    task_max_length: int = 0  # 0 = dynamic padding (pad to max in each batch) - most efficient
    mean_cache_seq_length: int = 256  # Paper: "up to 256 tokens" for mean activation computation
    
    # Target loss for evaluation
    target_loss: float = 0.15
    
    # Discretization evaluation
    num_eval_batches: int = 20
    
    # Speed optimizations
    use_autocast: bool = True  # Use mixed precision for faster training
    autocast_dtype: str = "bfloat16"  # "bfloat16" (stable) or "float16" (faster but less stable)
    mean_cache_batches: int = 100  # Batches for mean activation estimation
    bisection_eval_batches: int = 5  # Batches per bisection evaluation
    bisection_max_iters: int = 30  # Max bisection iterations
    
    # CARBS penalty for missing target (soft penalty instead of harsh cliff)
    # output = circuit_size + loss_penalty_scale * max(0, loss - target)
    # This gives CARBS gradient info about how far from target, not just pass/fail
    loss_penalty_scale: float = 50000.0  # Scale factor for loss overshoot penalty
    
    # Output
    output_dir: str = "outputs/carbs_sweep"
    
    # Device
    device: str = "cuda"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "circuit-pruning-carbs"
    
    # Ablation type: "zero", "mean_pretrain", "mean_task"
    ablation_type: str = "mean_pretrain"
    
    # Token embedding mask
    mask_token_embeds: bool = False
    
    # K_coef search center (override default)
    k_coef_center: float = 1e-3
    
    def to_dict(self) -> Dict:
        return asdict(self)


def get_carbs_param_spaces(k_coef_center: float = 1e-3) -> List[Param]:
    """
    Define CARBS parameter spaces based on empirical tuning.
    
    Search centers:
    - k_coef: configurable (default 1e-3)
    - wd (weight_decay): 1e-3
    - lr: 1e-2
    - inv_beta2: 5e-2 (so beta2 = 0.95)
    - heaviside_temp: 1.0
    
    Fixed (not searched):
    - init_noise_scale: 0.01
    - init_noise_bias: 0.1
    - lr_warmup_frac: 0.0 (no warmup)
    """
    return [
        # Sparsity penalty coefficient (log scale, very wide range)
        Param(
            name="k_coef",
            space=LogSpace(scale=2.0, min=1e-8, max=1e-1),
            search_center=k_coef_center,
        ),
        # Weight decay (log scale)
        Param(
            name="weight_decay",
            space=LogSpace(scale=1.0, min=1e-6, max=1e-1),
            search_center=1e-3,
        ),
        # Learning rate (log scale)
        Param(
            name="lr",
            space=LogSpace(scale=1.0, min=1e-5, max=1e-1),
            search_center=1e-2,
        ),
        # Beta2 (logit space for values near 1)
        Param(
            name="beta2",
            space=LogitSpace(),  # Maps to (0, 1)
            search_center=0.95,  # inv_beta2 = 0.05 -> beta2 = 0.95
        ),
        # Heaviside temperature (log scale)
        Param(
            name="heaviside_temp",
            space=LogSpace(scale=1.0, min=0.1, max=10.0),
            search_center=1.0,
        ),
    ]


def detect_task_max_length(task, n_samples: int = 100, buffer: int = 8) -> int:
    """
    Auto-detect the max sequence length needed for a task.
    
    Args:
        task: BinaryTask instance
        n_samples: Number of examples to sample
        buffer: Extra tokens to add as safety margin
        
    Returns:
        Recommended max_length (max observed + buffer, rounded up to nearest 8)
    """
    max_len = 0
    for _ in range(n_samples):
        ex = task.generate_example()
        max_len = max(max_len, len(ex.positive_ids))
    
    # Add buffer and round up to nearest 8 for efficient tensor ops
    recommended = max_len + buffer
    recommended = ((recommended + 7) // 8) * 8
    
    return recommended


def run_single_pruning(
    hparams: Dict[str, float],
    model,
    tokenizer,
    train_task,
    val_task,
    mean_cache: Dict[str, torch.Tensor],
    sweep_config: SweepConfig,
    run_id: int,
) -> Dict[str, Any]:
    """
    Run a single pruning experiment with given hyperparameters.
    
    Returns:
        Dictionary with results including circuit_size, achieved_loss, etc.
    """
    device = sweep_config.device
    
    # Create pruning config from hparams
    # noise_scale and noise_bias are fixed at values that give 100% active nodes at init
    config = PruningConfig(
        k_coef=hparams["k_coef"],
        init_noise_scale=0.01,  # Fixed - not searched
        init_noise_bias=0.1,    # Fixed - not searched (gives 100% active at init)
        weight_decay=hparams["weight_decay"],
        lr=hparams["lr"],
        beta2=hparams["beta2"],
        lr_warmup_frac=0.0,  # No warmup - fixed, not searched
        heaviside_temp=hparams["heaviside_temp"],
        num_steps=sweep_config.num_steps,
        batch_size=sweep_config.batch_size,
        seq_length=sweep_config.task_max_length,  # Task prompts are short (~10-15 tokens)
        device=device,
        log_every=100,
        target_loss=sweep_config.target_loss,
        ablation_type=sweep_config.ablation_type,
        mask_token_embeds=sweep_config.mask_token_embeds,
    )
    
    # Create masked model (fresh for each run)
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)
    
    # Set mean cache (only if not using zero ablation)
    if sweep_config.ablation_type != "zero":
        masked_model.set_means_from_dict(mean_cache)
    
    # Create trainer (no wandb for individual runs to avoid clutter)
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        val_task=val_task,
        config=config,
        use_wandb=False,
    )
    
    # Train with optional autocast for speed
    try:
        # Use autocast context for faster mixed-precision training
        if sweep_config.use_autocast:
            dtype = torch.bfloat16 if sweep_config.autocast_dtype == "bfloat16" else torch.float16
            autocast_ctx = torch.autocast('cuda', dtype=dtype)
        else:
            autocast_ctx = nullcontext()
        
        with autocast_ctx:
            trainer.train(
                num_steps=sweep_config.num_steps,
                show_progress=False,
                histogram_every=0,
                pareto_probe_every=0,
            )
        
        # Get active nodes count (including token mask if enabled)
        num_active_non_embed = masked_model.masks.get_total_active_nodes()
        total_non_embed = masked_model.masks.get_total_nodes()
        num_active_embed = 0
        total_embed = 0
        if masked_model.token_mask is not None:
            num_active_embed = masked_model.token_mask.get_num_active()
            total_embed = masked_model.vocab_size
        num_active = num_active_non_embed + num_active_embed
        total_nodes = total_non_embed + total_embed
        
        # Evaluate at target loss via bisection
        original_state = masked_model.get_mask_state()
        
        # Find smallest circuit that achieves target loss
        target_loss = sweep_config.target_loss
        low, high = 1, num_active
        best_k, best_loss = high, float('inf')
        
        # Use reduced eval batches for speed
        eval_batches = sweep_config.bisection_eval_batches
        
        # First check if we can achieve target at all
        with autocast_ctx:
            loss_at_all = evaluate_at_k(masked_model, val_task, num_active, config, num_batches=eval_batches)
        masked_model.load_mask_state(original_state)
        
        if loss_at_all <= target_loss:
            # Initialize with the known-good solution (k=num_active achieves target)
            best_k, best_loss = num_active, loss_at_all
            
            # Binary search for smallest k (reduced iterations for speed)
            for _ in range(sweep_config.bisection_max_iters):
                if low > high:
                    break
                mid = (low + high) // 2
                with autocast_ctx:
                    loss = evaluate_at_k(masked_model, val_task, mid, config, num_batches=eval_batches)
                masked_model.load_mask_state(original_state)
                
                if loss <= target_loss:
                    best_k, best_loss = mid, loss
                    high = mid - 1
                else:
                    low = mid + 1
        else:
            # Can't achieve target, use all active nodes
            best_k, best_loss = num_active, loss_at_all
        
        # Compute edge count (approximate)
        # Edge count = number of nonzero weights connected to active nodes
        # For simplicity, use node count as proxy (paper's main metric)
        circuit_size = best_k
        
        # Also evaluate on train set for comparison
        with autocast_ctx:
            train_loss = evaluate_at_k(masked_model, train_task, best_k, config, num_batches=eval_batches)
        masked_model.load_mask_state(original_state)
        
        result = {
            "success": True,
            "circuit_size": circuit_size,
            "achieved_loss_val": best_loss,           # Loss at bisected circuit_size
            "loss_at_all_active": loss_at_all,        # Loss with all active nodes (more stable)
            "achieved_loss_train": train_loss,
            "num_active_after_training": num_active,
            "total_nodes": total_nodes,
            "frac_active": num_active / total_nodes,
            "frac_circuit": circuit_size / total_nodes,
            "target_loss": target_loss,
            "target_achieved": loss_at_all <= target_loss,  # Use stable loss_at_all for gating
            "hparams": hparams,
            "run_id": run_id,
        }
        
    except Exception as e:
        result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "hparams": hparams,
            "run_id": run_id,
        }
    
    return result


def run_carbs_sweep(sweep_config: SweepConfig) -> Dict[str, Any]:
    """
    Run the full CARBS hyperparameter sweep.
    
    Following the paper's approach:
    - 32 iterations of CARBS with 8 parallel suggestions per iteration
    - Optimize for smallest circuit size that achieves target loss
    """
    print("=" * 70)
    print("CARBS Hyperparameter Sweep for Circuit Pruning")
    print("=" * 70)
    print(f"Task: {sweep_config.task_name}")
    print(f"Model: {sweep_config.model_path}")
    print(f"CARBS iterations: {sweep_config.carbs_iterations}")
    print(f"Parallel suggestions: {sweep_config.parallel_suggestions}")
    print(f"Total runs: {sweep_config.carbs_iterations * sweep_config.parallel_suggestions}")
    print(f"Training steps per run: {sweep_config.num_steps}")
    print(f"Target loss: {sweep_config.target_loss}")
    print(f"Ablation type: {sweep_config.ablation_type}")
    print(f"Mask token embeds: {sweep_config.mask_token_embeds}")
    print(f"K_coef search center: {sweep_config.k_coef_center}")
    print("=" * 70)
    
    device = sweep_config.device
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"{sweep_config.output_dir}_{sweep_config.task_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save sweep config
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config.to_dict(), f, indent=2)
    
    # Initialize wandb if requested
    if sweep_config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=sweep_config.wandb_project,
            name=f"carbs_sweep_{sweep_config.task_name}_{timestamp}",
            config=sweep_config.to_dict(),
        )
    
    # Load model once
    print("\nLoading model...")
    model, _ = load_model(sweep_config.model_path, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sweep_config.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create tasks
    train_task = get_task(sweep_config.task_name, tokenizer, seed=42, split="train")
    val_task = get_task(sweep_config.task_name, tokenizer, seed=42, split="val")
    print(f"Train task: {train_task.name} ({len(train_task.templates)} templates)")
    print(f"Val task: {val_task.name} ({len(val_task.templates)} templates)")
    
    # Report padding mode
    if sweep_config.task_max_length <= 0:
        print("Using dynamic padding (pad to max length in each batch)")
    
    # Compute mean cache once (reused across all runs)
    print(f"\nComputing mean activation cache ({sweep_config.mean_cache_batches} batches)...")
    temp_config = PruningConfig(device=device)
    temp_masked = MaskedSparseGPT(model, temp_config)
    temp_masked.to(device)
    
    data_iter = create_data_iterator(
        tokenizer_name=sweep_config.tokenizer_name,
        batch_size=64,
        seq_length=sweep_config.mean_cache_seq_length,  # Paper: 256 tokens for mean estimation
        num_batches=sweep_config.mean_cache_batches,
        seed=42,
    )
    mean_cache = temp_masked.compute_mean_cache(
        data_iter, 
        num_batches=sweep_config.mean_cache_batches, 
        show_progress=True
    )
    
    # Save mean cache
    torch.save(mean_cache, output_dir / "mean_cache.pt")
    del temp_masked
    torch.cuda.empty_cache()
    
    # Initialize CARBS
    print("\nInitializing CARBS...")
    param_spaces = get_carbs_param_spaces(k_coef_center=sweep_config.k_coef_center)
    
    carbs_params = CARBSParams(
        better_direction_sign=-1,  # Minimize circuit size
        is_wandb_logging_enabled=False,  # We handle wandb ourselves
        resample_frequency=5,
        num_random_samples=4,
        initial_search_radius=0.3,
    )
    
    carbs = CARBS(carbs_params, param_spaces)
    
    # Run sweep
    all_results = []
    best_result = None
    run_id = 0
    
    print(f"\nStarting CARBS sweep ({sweep_config.carbs_iterations} iterations)...")
    
    for iteration in tqdm(range(sweep_config.carbs_iterations), desc="CARBS iterations"):
        iteration_results = []
        suggestions = []
        
        # Get parallel suggestions
        for _ in range(sweep_config.parallel_suggestions):
            suggestion_out = carbs.suggest()
            suggestions.append(suggestion_out.suggestion)
        
        # Run all suggestions (sequentially since we're on one GPU)
        # Paper: "generate 8 suggestions, run all of them, update on those 8 results, repeat"
        for i, hparams in enumerate(suggestions):
            run_id += 1
            print(f"\n--- Iteration {iteration+1}/{sweep_config.carbs_iterations}, "
                  f"Run {i+1}/{sweep_config.parallel_suggestions} (total: {run_id}) ---")
            print(f"Hyperparameters: k_coef={hparams['k_coef']:.2e}, lr={hparams['lr']:.2e}, "
                  f"wd={hparams['weight_decay']:.2e}, temp={hparams['heaviside_temp']:.2f}")
            
            result = run_single_pruning(
                hparams=hparams,
                model=model,
                tokenizer=tokenizer,
                train_task=train_task,
                val_task=val_task,
                mean_cache=mean_cache,
                sweep_config=sweep_config,
                run_id=run_id,
            )
            
            iteration_results.append(result)
            all_results.append(result)
            
            # Print result immediately
            if result["success"]:
                print(f"Result: circuit_size={result['circuit_size']}, "
                      f"val_loss={result['achieved_loss_val']:.4f}, "
                      f"target_achieved={result['target_achieved']}")
                
                # Track best
                if result["target_achieved"]:
                    if best_result is None or result["circuit_size"] < best_result["circuit_size"]:
                        best_result = result
                        print(f"*** New best! Circuit size: {result['circuit_size']} ***")
            else:
                print(f"Run FAILED: {result['error']}")
        
        # BATCH UPDATE: Observe all 8 results at once after completing the iteration
        # This matches the paper: "update on those 8 results, and repeating"
        for i, (hparams, result) in enumerate(zip(suggestions, iteration_results)):
            if result["success"]:
                # Soft penalty: circuit_size + scale * max(0, loss - target)
                # This gives CARBS gradient info about how far from target
                circuit_size = result["circuit_size"]
                loss = result["loss_at_all_active"]  # Use stable loss measurement
                target = result["target_loss"]
                
                # Compute soft penalty for missing target
                loss_overshoot = max(0.0, loss - target)
                penalty = sweep_config.loss_penalty_scale * loss_overshoot
                output_value = circuit_size + penalty
                
                carbs.observe(ObservationInParam(
                    input=hparams,
                    output=output_value,
                    cost=1.0,
                ))
            else:
                carbs.observe(ObservationInParam(
                    input=hparams,
                    output=0,
                    cost=1.0,
                    is_failure=True,
                ))
            
            # Log to wandb
            if sweep_config.use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    "run_id": run_id,
                    "iteration": iteration,
                    "success": result["success"],
                }
                if result["success"]:
                    log_dict.update({
                        "circuit_size": result["circuit_size"],
                        "achieved_loss_val": result["achieved_loss_val"],
                        "achieved_loss_train": result["achieved_loss_train"],
                        "target_achieved": result["target_achieved"],
                        "frac_circuit": result["frac_circuit"],
                    })
                    for k, v in hparams.items():
                        log_dict[f"hparam/{k}"] = v
                    if best_result:
                        log_dict["best_circuit_size"] = best_result["circuit_size"]
                wandb.log(log_dict)
            
            # Save checkpoint periodically
            if run_id % 10 == 0:
                with open(output_dir / "results_checkpoint.json", "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
                # Save CARBS state
                with open(output_dir / "carbs_state.pkl", "wb") as f:
                    pickle.dump(carbs, f)
    
    # Final results
    print("\n" + "=" * 70)
    print("CARBS SWEEP COMPLETE")
    print("=" * 70)
    
    # Filter successful runs
    successful_runs = [r for r in all_results if r["success"]]
    target_achieved_runs = [r for r in successful_runs if r["target_achieved"]]
    
    print(f"\nTotal runs: {len(all_results)}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Runs achieving target loss: {len(target_achieved_runs)}")
    
    if best_result:
        print(f"\nBest result:")
        print(f"  Circuit size: {best_result['circuit_size']} nodes")
        print(f"  Circuit fraction: {best_result['frac_circuit']:.4f}")
        print(f"  Val loss: {best_result['achieved_loss_val']:.4f}")
        print(f"  Train loss: {best_result['achieved_loss_train']:.4f}")
        print(f"  Hyperparameters:")
        for k, v in best_result["hparams"].items():
            print(f"    {k}: {v:.4e}" if isinstance(v, float) else f"    {k}: {v}")
    
    # Statistics
    if target_achieved_runs:
        sizes = [r["circuit_size"] for r in target_achieved_runs]
        print(f"\nCircuit size statistics (runs achieving target):")
        print(f"  Min: {min(sizes)}")
        print(f"  Max: {max(sizes)}")
        print(f"  Mean: {np.mean(sizes):.1f}")
        print(f"  Median: {np.median(sizes):.1f}")
        print(f"  Std: {np.std(sizes):.1f}")
    
    # Save final results
    final_results = {
        "sweep_config": sweep_config.to_dict(),
        "all_results": all_results,
        "best_result": best_result,
        "summary": {
            "total_runs": len(all_results),
            "successful_runs": len(successful_runs),
            "target_achieved_runs": len(target_achieved_runs),
        },
    }
    
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Save CARBS final state
    with open(output_dir / "carbs_final.pkl", "wb") as f:
        pickle.dump(carbs, f)
    
    # Create visualizations
    create_sweep_visualizations(all_results, output_dir, sweep_config)
    
    # Log final summary to wandb
    if sweep_config.use_wandb and WANDB_AVAILABLE:
        wandb.run.summary["total_runs"] = len(all_results)
        wandb.run.summary["successful_runs"] = len(successful_runs)
        wandb.run.summary["target_achieved_runs"] = len(target_achieved_runs)
        if best_result:
            wandb.run.summary["best_circuit_size"] = best_result["circuit_size"]
            wandb.run.summary["best_val_loss"] = best_result["achieved_loss_val"]
        wandb.finish()
    
    print(f"\nResults saved to: {output_dir}")
    
    return final_results


def create_sweep_visualizations(
    all_results: List[Dict],
    output_dir: Path,
    sweep_config: SweepConfig,
):
    """Create visualizations of the sweep results."""
    successful = [r for r in all_results if r["success"]]
    if not successful:
        print("No successful runs to visualize")
        return
    
    # 1. Circuit size over iterations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Circuit size vs run number
    ax = axes[0, 0]
    run_ids = [r["run_id"] for r in successful]
    sizes = [r["circuit_size"] for r in successful]
    achieved = [r["target_achieved"] for r in successful]
    
    colors = ['green' if a else 'red' for a in achieved]
    ax.scatter(run_ids, sizes, c=colors, alpha=0.6)
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Circuit Size (nodes)")
    ax.set_title("Circuit Size Over Sweep\n(green=target achieved, red=not achieved)")
    ax.set_yscale('log')
    
    # Plot 2: Val loss vs circuit size (Pareto)
    ax = axes[0, 1]
    losses = [r["achieved_loss_val"] for r in successful]
    ax.scatter(sizes, losses, c=run_ids, cmap='viridis', alpha=0.6)
    ax.axhline(y=sweep_config.target_loss, color='r', linestyle='--', label=f'Target: {sweep_config.target_loss}')
    ax.set_xlabel("Circuit Size (nodes)")
    ax.set_ylabel("Val Loss")
    ax.set_title("Pareto Frontier: Circuit Size vs Loss")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Run ID')
    
    # Plot 3: Hyperparameter importance (k_coef vs circuit size)
    ax = axes[1, 0]
    k_coefs = [r["hparams"]["k_coef"] for r in successful]
    ax.scatter(k_coefs, sizes, c=achieved, cmap='RdYlGn', alpha=0.6)
    ax.set_xlabel("k_coef")
    ax.set_ylabel("Circuit Size")
    ax.set_title("k_coef vs Circuit Size")
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 4: Learning rate vs circuit size
    ax = axes[1, 1]
    lrs = [r["hparams"]["lr"] for r in successful]
    ax.scatter(lrs, sizes, c=achieved, cmap='RdYlGn', alpha=0.6)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Circuit Size")
    ax.set_title("Learning Rate vs Circuit Size")
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_overview.png", dpi=150)
    plt.close()
    
    # 2. Best run evolution
    target_runs = [r for r in successful if r["target_achieved"]]
    if target_runs:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        best_so_far = []
        current_best = float('inf')
        for r in sorted(target_runs, key=lambda x: x["run_id"]):
            if r["circuit_size"] < current_best:
                current_best = r["circuit_size"]
            best_so_far.append((r["run_id"], current_best))
        
        ids, bests = zip(*best_so_far)
        ax.plot(ids, bests, 'b-', linewidth=2, label='Best circuit size')
        ax.scatter([r["run_id"] for r in target_runs], 
                   [r["circuit_size"] for r in target_runs],
                   alpha=0.3, label='All runs')
        ax.set_xlabel("Run ID")
        ax.set_ylabel("Circuit Size (nodes)")
        ax.set_title("Best Circuit Size Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "best_evolution.png", dpi=150)
        plt.close()
    
    # 3. Hyperparameter distributions for best runs
    if len(target_runs) >= 5:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Get top 20% runs by circuit size
        sorted_runs = sorted(target_runs, key=lambda x: x["circuit_size"])
        top_runs = sorted_runs[:max(5, len(sorted_runs) // 5)]
        
        hparam_names = ["k_coef", "lr", "weight_decay", "heaviside_temp",
                       "init_noise_scale", "init_noise_bias", "beta2"]
        
        for ax, hp_name in zip(axes, hparam_names):
            all_vals = [r["hparams"][hp_name] for r in target_runs]
            top_vals = [r["hparams"][hp_name] for r in top_runs]
            
            # Use log scale for certain params
            if hp_name in ["k_coef", "lr", "weight_decay", "init_noise_scale", "heaviside_temp"]:
                all_vals = np.log10(all_vals)
                top_vals = np.log10(top_vals)
                xlabel = f"log10({hp_name})"
            else:
                xlabel = hp_name
            
            ax.hist(all_vals, bins=20, alpha=0.5, label='All', density=True)
            ax.hist(top_vals, bins=10, alpha=0.7, label='Top 20%', density=True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        
        plt.suptitle("Hyperparameter Distributions: All vs Top 20%")
        plt.tight_layout()
        plt.savefig(output_dir / "hparam_distributions.png", dpi=150)
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CARBS hyperparameter sweep for circuit pruning")
    parser.add_argument("--task", type=str, default="dummy_pronoun",
                       help="Task name (dummy_pronoun, dummy_quote, dummy_article)")
    parser.add_argument("--iterations", type=int, default=32,
                       help="Number of CARBS iterations (default: 32)")
    parser.add_argument("--parallel", type=int, default=8,
                       help="Parallel suggestions per iteration (default: 8)")
    parser.add_argument("--steps", type=int, default=2000,
                       help="Training steps per run (default: 2000)")
    parser.add_argument("--target-loss", type=float, default=0.15,
                       help="Target task loss (default: 0.15)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (default: cuda)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--ablation", type=str, default="mean_pretrain",
                       choices=["zero", "mean_pretrain", "mean_task"],
                       help="Ablation type (default: mean_pretrain)")
    parser.add_argument("--mask-token-embeds", action="store_true",
                       help="Also learn a mask over vocabulary (token embeddings)")
    parser.add_argument("--k-coef-center", type=float, default=1e-3,
                       help="Search center for k_coef (default: 1e-3)")
    
    args = parser.parse_args()
    
    sweep_config = SweepConfig(
        task_name=args.task,
        carbs_iterations=args.iterations,
        parallel_suggestions=args.parallel,
        num_steps=args.steps,
        target_loss=args.target_loss,
        device=args.device,
        use_wandb=not args.no_wandb,
        ablation_type=args.ablation,
        mask_token_embeds=args.mask_token_embeds,
        k_coef_center=args.k_coef_center,
    )
    
    run_carbs_sweep(sweep_config)


if __name__ == "__main__":
    main()

