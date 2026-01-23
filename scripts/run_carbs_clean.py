#!/usr/bin/env python3
"""
Clean CARBS hyperparameter sweep for circuit pruning.

Features:
- Configurable parallel batch size (default 1)
- Total runs = num_runs (default 32)
- Saves "best so far" checkpoints during sweep
- Keeps only the overall best checkpoint at the end
- Creates Pareto plot on superval set

Usage:
    python my_sparse_pretrain/scripts/run_carbs_clean.py --model MODEL_PATH [options]
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
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from contextlib import nullcontext
import traceback
import shutil
from tqdm import tqdm

from transformers import AutoTokenizer

from carbs import CARBS, CARBSParams, Param, LogSpace, LogitSpace, ObservationInParam

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.trainer import PruningTrainer
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator
from my_sparse_pretrain.src.pruning.discretize import evaluate_at_k, evaluate_at_k_fixed_batches

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class CleanSweepConfig:
    """Configuration for the clean CARBS sweep."""
    # Model
    model_path: str = "jacobcd52/ss_bridges_d1024_f0.015625"
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M"
    
    # Task
    task_name: str = "dummy_pronoun"
    
    # CARBS settings
    num_runs: int = 32  # Total number of runs
    parallel_suggestions: int = 1  # Suggestions per batch (default 1)
    
    # Training settings
    num_steps: int = 500
    batch_size: int = 64
    task_max_length: int = 0  # 0 = dynamic padding
    mean_cache_seq_length: int = 256
    
    # Fixed hyperparameters (not searched)
    init_noise_scale: float = 0.01
    init_noise_bias: float = 0.1
    lr_warmup_frac: float = 0.0
    
    # Target loss for evaluation
    target_loss: float = 0.15
    
    # Speed settings
    use_autocast: bool = True
    autocast_dtype: str = "bfloat16"
    mean_cache_batches: int = 100
    bisection_eval_batches: int = 5
    bisection_max_iters: int = 15  # Half of original 30
    
    # CARBS soft penalty
    loss_penalty_scale: float = 50000.0
    
    # Output
    output_base_dir: str = "my_sparse_pretrain/outputs/carbs_results_pronoun"
    
    # Device
    device: str = "cuda"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "circuit-pruning-carbs"
    
    # Ablation type: "zero", "mean_pretrain", "mean_task"
    ablation_type: str = "mean_pretrain"
    
    # Token embedding mask
    mask_token_embeds: bool = False
    
    # Binary loss (for tense task)
    # If True, compute CE over only [correct, incorrect] logits
    use_binary_loss: bool = False
    
    # K_coef search center
    k_coef_center: float = 1e-3
    
    # LR search center
    lr_center: float = 1e-2
    
    # Freeze layer norm scale
    # If True, layer norm outputs from unpruned forward are used in pruned forward
    freeze_layernorm_scale: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_model_short_name(self) -> str:
        """Extract short name from model path for output dir."""
        return self.model_path.split("/")[-1]


def get_carbs_param_spaces(k_coef_center: float = 1e-3, lr_center: float = 1e-2) -> List[Param]:
    """Define CARBS parameter spaces."""
    return [
        Param(
            name="k_coef",
            space=LogSpace(scale=2.0, min=1e-8, max=1e-1),
            search_center=k_coef_center,
        ),
        Param(
            name="weight_decay",
            space=LogSpace(scale=1.0, min=1e-6, max=1e-1),
            search_center=1e-3,
        ),
        Param(
            name="lr",
            space=LogSpace(scale=1.0, min=1e-5, max=1e-1),
            search_center=lr_center,
        ),
        Param(
            name="beta2",
            space=LogitSpace(),
            search_center=0.95,
        ),
        Param(
            name="heaviside_temp",
            space=LogSpace(scale=1.0, min=0.1, max=10.0),
            search_center=1.0,
        ),
    ]


def run_single_pruning(
    hparams: Dict[str, float],
    model,
    tokenizer,
    train_task,
    val_task,
    mean_cache: Dict[str, torch.Tensor],
    config: CleanSweepConfig,
    run_id: int,
) -> Dict[str, Any]:
    """Run a single pruning experiment."""
    device = config.device
    
    # Create pruning config
    pruning_config = PruningConfig(
        k_coef=hparams["k_coef"],
        init_noise_scale=config.init_noise_scale,
        init_noise_bias=config.init_noise_bias,
        weight_decay=hparams["weight_decay"],
        lr=hparams["lr"],
        beta2=hparams["beta2"],
        lr_warmup_frac=config.lr_warmup_frac,
        heaviside_temp=hparams["heaviside_temp"],
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        seq_length=config.task_max_length,
        device=device,
        log_every=100,
        target_loss=config.target_loss,
        ablation_type=config.ablation_type,
        mask_token_embeds=config.mask_token_embeds,
        use_binary_loss=config.use_binary_loss,
        freeze_layernorm_scale=config.freeze_layernorm_scale,
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if config.ablation_type != "zero":
        masked_model.set_means_from_dict(mean_cache)
    
    # Create trainer
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        val_task=val_task,
        config=pruning_config,
        use_wandb=False,
    )
    
    try:
        # Autocast context
        if config.use_autocast:
            dtype = torch.bfloat16 if config.autocast_dtype == "bfloat16" else torch.float16
            autocast_ctx = torch.autocast('cuda', dtype=dtype)
        else:
            autocast_ctx = nullcontext()
        
        with autocast_ctx:
            trainer.train(
                num_steps=config.num_steps,
                show_progress=False,
                histogram_every=0,
                pareto_probe_every=0,
            )
        
        # Get active nodes
        num_active = masked_model.masks.get_total_active_nodes()
        total_nodes = masked_model.masks.get_total_nodes()
        
        # Save mask state for potential checkpoint
        original_state = masked_model.get_mask_state()
        
        # Evaluate at full capacity first
        eval_batches = config.bisection_eval_batches
        with autocast_ctx:
            loss_at_all = evaluate_at_k(masked_model, val_task, num_active, pruning_config, num_batches=eval_batches)
        masked_model.load_mask_state(original_state)
        
        # Bisection search for smallest circuit
        target_loss = config.target_loss
        best_k, best_loss = num_active, loss_at_all
        
        if loss_at_all <= target_loss:
            low, high = 1, num_active
            for _ in range(config.bisection_max_iters):
                if low > high:
                    break
                mid = (low + high) // 2
                with autocast_ctx:
                    loss = evaluate_at_k(masked_model, val_task, mid, pruning_config, num_batches=eval_batches)
                masked_model.load_mask_state(original_state)
                
                if loss <= target_loss:
                    best_k, best_loss = mid, loss
                    high = mid - 1
                else:
                    low = mid + 1
        
        result = {
            "success": True,
            "circuit_size": best_k,
            "achieved_loss_val": best_loss,
            "loss_at_all_active": loss_at_all,
            "num_active_after_training": num_active,
            "total_nodes": total_nodes,
            "frac_active": num_active / total_nodes,
            "frac_circuit": best_k / total_nodes,
            "target_loss": target_loss,
            "target_achieved": loss_at_all <= target_loss,
            "hparams": hparams,
            "run_id": run_id,
            "mask_state": original_state,  # Include for checkpoint saving
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


def save_best_checkpoint(result: Dict, output_dir: Path, config: CleanSweepConfig):
    """Save checkpoint for the best result."""
    checkpoint_dir = output_dir / "best_checkpoint"
    
    # Clear previous best if exists
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mask state
    torch.save(result["mask_state"], checkpoint_dir / "masks.pt")
    
    # Save config
    config_dict = config.to_dict()
    config_dict["best_circuit_size"] = result["circuit_size"]
    config_dict["best_val_loss"] = result["achieved_loss_val"]
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save hparams
    with open(checkpoint_dir / "hparams.json", "w") as f:
        json.dump(result["hparams"], f, indent=2)
    
    # Save summary (without mask_state which is large)
    summary = {k: v for k, v in result.items() if k != "mask_state"}
    with open(checkpoint_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


def create_pareto_plot_superval(
    best_result: Dict,
    model,
    tokenizer,
    mean_cache: Dict[str, torch.Tensor],
    config: CleanSweepConfig,
    output_dir: Path,
):
    """Create Pareto plot on superval set for the best checkpoint."""
    print("\nCreating Pareto plot on superval set...")
    
    device = config.device
    
    # Load superval task
    superval_task = get_task(config.task_name, tokenizer, seed=42, split="superval")
    superval_templates = len(superval_task.templates) if hasattr(superval_task, 'templates') else 'N/A'
    print(f"Superval task: {superval_task.name} ({superval_templates} templates)")
    
    # Create masked model and load best checkpoint
    pruning_config = PruningConfig(
        k_coef=best_result["hparams"]["k_coef"],
        init_noise_scale=config.init_noise_scale,
        init_noise_bias=config.init_noise_bias,
        weight_decay=best_result["hparams"]["weight_decay"],
        lr=best_result["hparams"]["lr"],
        beta2=best_result["hparams"]["beta2"],
        lr_warmup_frac=config.lr_warmup_frac,
        heaviside_temp=best_result["hparams"]["heaviside_temp"],
        num_steps=config.num_steps,
        batch_size=config.batch_size,
        seq_length=config.task_max_length,
        device=device,
        ablation_type=config.ablation_type,
        mask_token_embeds=config.mask_token_embeds,
        use_binary_loss=config.use_binary_loss,
        freeze_layernorm_scale=config.freeze_layernorm_scale,
    )
    
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if config.ablation_type != "zero":
        masked_model.set_means_from_dict(mean_cache)
    masked_model.load_mask_state(best_result["mask_state"])
    
    # Get max nodes
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    
    # Autocast context
    if config.use_autocast:
        dtype = torch.bfloat16 if config.autocast_dtype == "bfloat16" else torch.float16
        autocast_ctx = torch.autocast('cuda', dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    
    # Pre-generate fixed batches for consistent evaluation across all k values
    # This ensures the Pareto curve is monotonically decreasing
    print(f"Pre-generating {config.bisection_eval_batches} fixed evaluation batches...")
    fixed_batches = []
    for _ in range(config.bisection_eval_batches):
        batch = superval_task.generate_batch(
            batch_size=config.batch_size,
            max_length=config.task_max_length,
        )
        fixed_batches.append(batch)
    
    # Evaluate at different k values for Pareto curve
    original_state = masked_model.get_mask_state()
    
    # Pre-generate fixed batches so all evaluations use the same data
    num_eval_batches = max(config.bisection_eval_batches, 20)  # Use at least 20 for Pareto
    print(f"Pre-generating {num_eval_batches} fixed evaluation batches...")
    fixed_batches = []
    for _ in range(num_eval_batches):
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = superval_task.generate_batch(
            batch_size=pruning_config.batch_size,
            max_length=pruning_config.seq_length,
        )
        fixed_batches.append({
            "positive_ids": positive_ids.to(config.device),
            "negative_ids": negative_ids.to(config.device),
            "correct_tokens": correct_tokens.to(config.device),
            "incorrect_tokens": incorrect_tokens.to(config.device),
            "eval_positions": eval_positions.to(config.device),
        })
    
    # Define 16 loss targets log-spaced between 0.001 and 0.5
    loss_targets = np.geomspace(0.001, 0.5, num=16)
    print(f"Finding minimum k for {len(loss_targets)} loss targets: {[f'{t:.4f}' for t in loss_targets]}")
    
    pareto_data = []
    
    for target_loss in tqdm(loss_targets, desc="Pareto sweep"):
        # Bisection to find smallest k that achieves <= target_loss
        low, high = 1, num_active
        best_k = None
        best_loss = None
        
        while low <= high:
            mid = (low + high) // 2
            with autocast_ctx:
                loss = evaluate_at_k_fixed_batches(masked_model, fixed_batches, mid, device=device)
            masked_model.load_mask_state(original_state)
            
            if loss <= target_loss:
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
                "target_loss": float(target_loss),
            })
        else:
            # Even with all active nodes, can't achieve this target
            # Evaluate at num_active to show the best we can do
            with autocast_ctx:
                loss = evaluate_at_k_fixed_batches(masked_model, fixed_batches, num_active, device=device)
            masked_model.load_mask_state(original_state)
            print(f"  Target {target_loss:.4f} not achievable (best loss with all {num_active} nodes: {loss:.4f})")
    
    # Save Pareto data
    with open(output_dir / "pareto_superval_data.json", "w") as f:
        json.dump(pareto_data, f, indent=2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ks = [d["k"] for d in pareto_data]
    losses = [d["loss"] for d in pareto_data]
    
    ax.plot(ks, losses, 'b-o', markersize=4, linewidth=1.5, label='Superval loss')
    ax.axhline(y=config.target_loss, color='r', linestyle='--', linewidth=2, label=f'Target: {config.target_loss}')
    
    # Mark the circuit size that achieves target
    target_achieved_ks = [k for k, l in zip(ks, losses) if l <= config.target_loss]
    if target_achieved_ks:
        min_k = min(target_achieved_ks)
        min_loss = losses[ks.index(min_k)]
        ax.scatter([min_k], [min_loss], c='green', s=150, zorder=5, marker='*', 
                   label=f'Min circuit @ target: {min_k} nodes')
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title(f"Pareto Curve on Superval Set\nModel: {config.get_model_short_name()}", fontsize=14)
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
    plt.savefig(output_dir / "pareto_superval.png", dpi=150)
    plt.close()
    
    print(f"Pareto plot saved to {output_dir / 'pareto_superval.png'}")
    
    return pareto_data


def run_carbs_sweep(config: CleanSweepConfig) -> Dict[str, Any]:
    """Run the full CARBS sweep."""
    print("=" * 70)
    print("CARBS Hyperparameter Sweep for Circuit Pruning")
    print("=" * 70)
    print(f"Model: {config.model_path}")
    print(f"Task: {config.task_name}")
    print(f"Total runs: {config.num_runs}")
    print(f"Parallel suggestions: {config.parallel_suggestions}")
    print(f"Training steps per run: {config.num_steps}")
    print(f"Target loss: {config.target_loss}")
    print(f"Ablation type: {config.ablation_type}")
    print(f"Mask token embeds: {config.mask_token_embeds}")
    print(f"K_coef search center: {config.k_coef_center}, LR search center: {config.lr_center}")
    print(f"Fixed: noise_scale={config.init_noise_scale}, noise_bias={config.init_noise_bias}, warmup={config.lr_warmup_frac}")
    print(f"Freeze layer norm scale: {config.freeze_layernorm_scale}")
    print("=" * 70)
    
    device = config.device
    
    # Setup output directory - include ablation type, embed suffix, and frozen LN suffix
    model_name = config.get_model_short_name()
    ablation_suffix = "_zero" if config.ablation_type == "zero" else "_mean"
    embed_suffix = "" if config.mask_token_embeds else "_noembed"
    frozen_ln_suffix = "_frozenln" if config.freeze_layernorm_scale else ""
    full_suffix = f"{ablation_suffix}{embed_suffix}{frozen_ln_suffix}"
    output_dir = Path(config.output_base_dir) / f"{model_name}{full_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / "sweep_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Initialize wandb with informative name
    if config.use_wandb and WANDB_AVAILABLE:
        wandb_name = f"{config.task_name}_{model_name}{full_suffix}_{datetime.now().strftime('%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            name=wandb_name,
            config=config.to_dict(),
            reinit=True,  # Allow multiple init() calls
        )
    
    # Load model
    print("\nLoading model...")
    model, _ = load_model(config.model_path, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create tasks
    train_task = get_task(config.task_name, tokenizer, seed=42, split="train")
    val_task = get_task(config.task_name, tokenizer, seed=42, split="val")
    train_templates = len(train_task.templates) if hasattr(train_task, 'templates') else 'N/A'
    val_templates = len(val_task.templates) if hasattr(val_task, 'templates') else 'N/A'
    print(f"Train task: {train_task.name} ({train_templates} templates)")
    print(f"Val task: {val_task.name} ({val_templates} templates)")
    
    # Compute mean cache based on ablation type
    mean_cache = None
    if config.ablation_type == "zero":
        print("\nUsing ZERO ablation (masked nodes → 0)")
    elif config.ablation_type == "mean_pretrain":
        print(f"\nComputing mean activation cache from SimpleStories ({config.mean_cache_batches} batches)...")
        temp_config = PruningConfig(device=device, mask_token_embeds=config.mask_token_embeds)
        temp_masked = MaskedSparseGPT(model, temp_config)
        temp_masked.to(device)
        
        data_iter = create_data_iterator(
            tokenizer_name=config.tokenizer_name,
            batch_size=64,
            seq_length=config.mean_cache_seq_length,
            num_batches=config.mean_cache_batches,
            seed=42,
        )
        mean_cache = temp_masked.compute_mean_cache(data_iter, num_batches=config.mean_cache_batches, show_progress=True)
        del temp_masked
        print("Using MEAN_PRETRAIN ablation (masked nodes → mean over SimpleStories)")
    elif config.ablation_type == "mean_task":
        print(f"\nComputing mean activation cache from task data ({config.mean_cache_batches} batches)...")
        temp_config = PruningConfig(device=device, mask_token_embeds=config.mask_token_embeds)
        temp_masked = MaskedSparseGPT(model, temp_config)
        temp_masked.to(device)
        
        def task_data_iterator():
            for _ in range(config.mean_cache_batches):
                positive_ids, _, _, _, _ = train_task.generate_batch(batch_size=64, max_length=0)
                yield positive_ids
        
        mean_cache = temp_masked.compute_mean_cache(task_data_iterator(), num_batches=config.mean_cache_batches, show_progress=True)
        del temp_masked
        print("Using MEAN_TASK ablation (masked nodes → mean over task data)")
    
    if mean_cache is not None:
        torch.save(mean_cache, output_dir / "mean_cache.pt")
    torch.cuda.empty_cache()
    
    # Initialize CARBS
    print("\nInitializing CARBS...")
    param_spaces = get_carbs_param_spaces(k_coef_center=config.k_coef_center, lr_center=config.lr_center)
    
    carbs_params = CARBSParams(
        better_direction_sign=-1,
        is_wandb_logging_enabled=False,
        resample_frequency=5,
        num_random_samples=4,
        initial_search_radius=0.3,
    )
    
    carbs = CARBS(carbs_params, param_spaces)
    
    # Run sweep
    all_results = []
    best_result = None
    run_id = 0
    
    # Calculate iterations needed
    num_iterations = (config.num_runs + config.parallel_suggestions - 1) // config.parallel_suggestions
    
    print(f"\nStarting CARBS sweep ({num_iterations} iterations, {config.parallel_suggestions} per batch)...")
    
    for iteration in tqdm(range(num_iterations), desc="CARBS iterations"):
        # How many runs this iteration
        runs_this_iter = min(config.parallel_suggestions, config.num_runs - run_id)
        if runs_this_iter <= 0:
            break
        
        suggestions = []
        for _ in range(runs_this_iter):
            suggestion_out = carbs.suggest()
            suggestions.append(suggestion_out.suggestion)
        
        iteration_results = []
        for i, hparams in enumerate(suggestions):
            run_id += 1
            print(f"\n--- Run {run_id}/{config.num_runs} ---")
            print(f"k_coef={hparams['k_coef']:.2e}, lr={hparams['lr']:.2e}, "
                  f"wd={hparams['weight_decay']:.2e}, temp={hparams['heaviside_temp']:.2f}")
            
            result = run_single_pruning(
                hparams=hparams,
                model=model,
                tokenizer=tokenizer,
                train_task=train_task,
                val_task=val_task,
                mean_cache=mean_cache,
                config=config,
                run_id=run_id,
            )
            
            iteration_results.append(result)
            
            if result["success"]:
                print(f"Result: circuit_size={result['circuit_size']}, "
                      f"active_after_training={result['num_active_after_training']}, "
                      f"val_loss={result['achieved_loss_val']:.4f}, "
                      f"target_achieved={result['target_achieved']}")
                
                # Update best and save checkpoint
                if result["target_achieved"]:
                    if best_result is None or result["circuit_size"] < best_result["circuit_size"]:
                        best_result = result.copy()  # Copy to preserve mask_state
                        print(f"*** New best! Circuit size: {result['circuit_size']} ***")
                        save_best_checkpoint(best_result, output_dir, config)
            else:
                print(f"Run FAILED: {result['error']}")
            
            # Store result without mask_state for all_results (saves memory)
            result_no_mask = {k: v for k, v in result.items() if k != "mask_state"}
            all_results.append(result_no_mask)
        
        # Batch update CARBS
        for hparams, result in zip(suggestions, iteration_results):
            if result["success"]:
                circuit_size = result["circuit_size"]
                loss = result["loss_at_all_active"]
                target = result["target_loss"]
                
                loss_overshoot = max(0.0, loss - target)
                penalty = config.loss_penalty_scale * loss_overshoot
                output_value = circuit_size + penalty
                
                carbs.observe(ObservationInParam(input=hparams, output=output_value, cost=1.0))
            else:
                carbs.observe(ObservationInParam(input=hparams, output=0, cost=1.0, is_failure=True))
            
            # Log to wandb
            if config.use_wandb and WANDB_AVAILABLE and result["success"]:
                log_dict = {
                    "run_id": result["run_id"],
                    "circuit_size": result["circuit_size"],
                    "achieved_loss_val": result["achieved_loss_val"],
                    "target_achieved": result["target_achieved"],
                }
                for k, v in hparams.items():
                    log_dict[f"hparam/{k}"] = v
                if best_result:
                    log_dict["best_circuit_size"] = best_result["circuit_size"]
                wandb.log(log_dict)
    
    # Final summary
    print("\n" + "=" * 70)
    print("CARBS SWEEP COMPLETE")
    print("=" * 70)
    
    successful_runs = [r for r in all_results if r["success"]]
    target_achieved_runs = [r for r in successful_runs if r["target_achieved"]]
    
    print(f"\nTotal runs: {len(all_results)}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Runs achieving target: {len(target_achieved_runs)}")
    
    if best_result:
        print(f"\nBest result:")
        print(f"  Circuit size: {best_result['circuit_size']} nodes")
        print(f"  Circuit fraction: {best_result['frac_circuit']:.4f}")
        print(f"  Val loss: {best_result['achieved_loss_val']:.4f}")
        print(f"  Hyperparameters:")
        for k, v in best_result["hparams"].items():
            print(f"    {k}: {v:.4e}" if isinstance(v, float) else f"    {k}: {v}")
        
        # Create Pareto plot on superval
        pareto_data = create_pareto_plot_superval(
            best_result=best_result,
            model=model,
            tokenizer=tokenizer,
            mean_cache=mean_cache,
            config=config,
            output_dir=output_dir,
        )
    else:
        print("\nNo runs achieved target loss!")
        pareto_data = None
    
    # Save final results (without mask states)
    final_results = {
        "sweep_config": config.to_dict(),
        "all_results": all_results,
        "best_result": {k: v for k, v in best_result.items() if k != "mask_state"} if best_result else None,
        "pareto_superval": pareto_data,
        "summary": {
            "total_runs": len(all_results),
            "successful_runs": len(successful_runs),
            "target_achieved_runs": len(target_achieved_runs),
        },
    }
    
    with open(output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Finish wandb
    if config.use_wandb and WANDB_AVAILABLE:
        if best_result:
            wandb.run.summary["best_circuit_size"] = best_result["circuit_size"]
            wandb.run.summary["best_val_loss"] = best_result["achieved_loss_val"]
        wandb.finish()
    
    print(f"\nResults saved to: {output_dir}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Clean CARBS sweep for circuit pruning")
    parser.add_argument("--model", type=str, default="jacobcd52/ss_bridges_d1024_f0.015625",
                       help="Model path")
    parser.add_argument("--task", type=str, default="dummy_pronoun",
                       help="Task name")
    parser.add_argument("--num-runs", type=int, default=32,
                       help="Total number of CARBS runs (default: 32)")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Parallel suggestions per batch (default: 1)")
    parser.add_argument("--steps", type=int, default=500,
                       help="Training steps per run (default: 500)")
    parser.add_argument("--target-loss", type=float, default=0.15,
                       help="Target task loss (default: 0.15)")
    parser.add_argument("--noise-scale", type=float, default=0.01,
                       help="Fixed init_noise_scale (default: 0.01)")
    parser.add_argument("--noise-bias", type=float, default=0.1,
                       help="Fixed init_noise_bias (default: 0.1)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (default: cuda)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--ablation", type=str, default="mean_pretrain",
                       choices=["zero", "mean_pretrain", "mean_task"],
                       help="Ablation type: zero, mean_pretrain (SimpleStories), or mean_task")
    parser.add_argument("--mask-token-embeds", action="store_true",
                       help="Also learn a mask over vocabulary (token embeddings)")
    parser.add_argument("--binary-loss", action="store_true",
                       help="Use binary CE loss (for tense task)")
    parser.add_argument("--k-coef-center", type=float, default=1e-3,
                       help="Search center for k_coef (default: 1e-3)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output base directory (default: carbs_results_pronoun)")
    parser.add_argument("--freeze-layernorm-scale", action="store_true",
                       help="Freeze layer norm outputs to unpruned values during forward pass")
    
    args = parser.parse_args()
    
    config = CleanSweepConfig(
        model_path=args.model,
        task_name=args.task,
        num_runs=args.num_runs,
        parallel_suggestions=args.parallel,
        num_steps=args.steps,
        target_loss=args.target_loss,
        init_noise_scale=args.noise_scale,
        init_noise_bias=args.noise_bias,
        device=args.device,
        use_wandb=not args.no_wandb,
        ablation_type=args.ablation,
        mask_token_embeds=args.mask_token_embeds,
        use_binary_loss=args.binary_loss,
        k_coef_center=args.k_coef_center,
        output_base_dir=args.output_dir if args.output_dir else "my_sparse_pretrain/outputs/carbs_results_pronoun",
        freeze_layernorm_scale=args.freeze_layernorm_scale,
    )
    
    run_carbs_sweep(config)


if __name__ == "__main__":
    main()

