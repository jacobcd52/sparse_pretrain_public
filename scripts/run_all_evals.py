#!/usr/bin/env python3
"""
Run all evaluations for pruned models in carbs_results_pronoun.

This script:
1. Finds all pruned model directories (excluding 'ignore' ones)
2. For each model, runs:
   - Pareto curve on superval set
   - Ablation sweep
   - FVU between pruned and unpruned  
   - Interchange intervention
3. Saves results to an 'evals' folder within each model directory
4. Creates comparison pareto plots for model variants

Usage:
    python my_sparse_pretrain/scripts/run_all_evals.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from contextlib import nullcontext

from transformers import AutoTokenizer

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model
from my_sparse_pretrain.src.pruning.discretize import evaluate_at_k_fixed_batches
from my_sparse_pretrain.src.pruning.interchange_eval import (
    InterchangeEvalConfig,
    run_and_save_interchange_eval,
    AblationSweepConfig,
    run_and_save_ablation_sweep,
    MaskRelaxationConfig,
    run_and_save_mask_relaxation,
)


# Base directory for carbs results
CARBS_RESULTS_DIR = Path("my_sparse_pretrain/outputs/carbs_results_pronoun")


def get_model_dirs(exclude_ignore: bool = True) -> List[Path]:
    """Get all model directories, optionally excluding 'ignore' ones."""
    all_dirs = []
    for d in CARBS_RESULTS_DIR.iterdir():
        if d.is_dir() and (d / "best_checkpoint").exists():
            if exclude_ignore and "ignore" in d.name.lower():
                continue
            all_dirs.append(d)
    return sorted(all_dirs)


def load_model_checkpoint(model_dir: Path, device: str = "cuda"):
    """Load a model checkpoint from a CARBS result directory."""
    # Load sweep config
    with open(model_dir / "sweep_config.json") as f:
        sweep_config = json.load(f)
    
    best_checkpoint_dir = model_dir / "best_checkpoint"
    
    # Load hparams and summary
    with open(best_checkpoint_dir / "hparams.json") as f:
        hparams = json.load(f)
    with open(best_checkpoint_dir / "summary.json") as f:
        summary = json.load(f)
    
    # Load model
    model_path = sweep_config["model_path"]
    tokenizer_name = sweep_config["tokenizer_name"]
    
    print(f"  Loading model: {model_path}")
    model, _ = load_model(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load mean cache if exists
    mean_cache_path = model_dir / "mean_cache.pt"
    if mean_cache_path.exists():
        mean_cache = torch.load(mean_cache_path, map_location=device, weights_only=True)
    else:
        mean_cache = {}
    
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
        num_steps=sweep_config.get("num_steps", 500),
        batch_size=sweep_config.get("batch_size", 64),
        seq_length=sweep_config.get("task_max_length", 0),
        device=device,
        ablation_type=sweep_config.get("ablation_type", "mean_pretrain"),
        mask_token_embeds=sweep_config.get("mask_token_embeds", False),
        use_binary_loss=sweep_config.get("use_binary_loss", False),
        freeze_layernorm_scale=sweep_config.get("freeze_layernorm_scale", False),
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if sweep_config.get("ablation_type", "mean_pretrain") != "zero" and mean_cache:
        masked_model.set_means_from_dict(mean_cache)
    
    # Load masks
    masks_path = best_checkpoint_dir / "masks.pt"
    mask_state = torch.load(masks_path, map_location=device, weights_only=True)
    masked_model.load_mask_state(mask_state)
    
    return masked_model, model, tokenizer, sweep_config, hparams, mean_cache, pruning_config


def run_pareto_eval(
    masked_model: MaskedSparseGPT,
    tokenizer,
    sweep_config: Dict,
    output_dir: Path,
    device: str = "cuda",
    num_eval_batches: int = 20,
):
    """Run pareto curve evaluation on superval set."""
    print("  Running Pareto evaluation on superval set...")
    
    # Load superval task
    task_name = sweep_config["task_name"]
    task = get_task(task_name, tokenizer, seed=42, split="superval")
    
    # Get active nodes
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    
    # Pre-generate fixed batches
    pruning_config = masked_model.config
    fixed_batches = []
    for _ in range(num_eval_batches):
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = task.generate_batch(
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
    
    # Sample k values logarithmically
    k_values = np.unique(np.geomspace(1, num_active, num=30).astype(int))
    k_values = sorted(set(k_values) | {1, num_active})
    
    # Save original state
    original_state = masked_model.get_mask_state()
    
    pareto_data = []
    with torch.autocast(device, dtype=torch.bfloat16):
        for k in tqdm(k_values, desc="Pareto sweep", leave=False):
            loss = evaluate_at_k_fixed_batches(masked_model, fixed_batches, k, device=device)
            masked_model.load_mask_state(original_state)
            pareto_data.append({
                "k": int(k),
                "loss": float(loss),
                "frac": k / total_nodes,
            })
    
    # Save data
    with open(output_dir / "pareto_superval_data.json", "w") as f:
        json.dump(pareto_data, f, indent=2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ks = [d["k"] for d in pareto_data]
    losses = [d["loss"] for d in pareto_data]
    
    model_name = sweep_config["model_path"].split("/")[-1]
    
    ax.plot(ks, losses, 'b-o', markersize=4, linewidth=1.5, label='Superval loss')
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title(f"Pareto Curve on Superval Set\nModel: {model_name}", fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    textstr = f'Total nodes: {total_nodes:,}\nActive after training: {num_active:,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_superval.png", dpi=150)
    plt.close()
    
    print(f"    Saved pareto plot to {output_dir / 'pareto_superval.png'}")
    return pareto_data


def run_interchange_eval(
    masked_model: MaskedSparseGPT,
    base_model,
    tokenizer,
    sweep_config: Dict,
    pruning_config: PruningConfig,
    output_dir: Path,
    model_name: str,
    device: str = "cuda",
):
    """Run interchange intervention evaluation."""
    print("  Running Interchange evaluation...")
    
    task_name = sweep_config["task_name"]
    task = get_task(task_name, tokenizer, seed=42, split="superval")
    
    config = InterchangeEvalConfig(
        fractions=[0.0, 0.25, 0.5, 0.75, 1.0],  # 5 fraction points
        num_trials=3,
        num_batches=10,
        batch_size=pruning_config.batch_size,
        seq_length=pruning_config.seq_length,
        device=device,
    )
    
    ablation_type = sweep_config.get("ablation_type", "mean_pretrain")
    ablation_str = "zero" if ablation_type == "zero" else "mean"
    title_prefix = f"{model_name} ({ablation_str}_ablate)\n"
    
    with torch.autocast(device, dtype=torch.bfloat16):
        result = run_and_save_interchange_eval(
            masked_model=masked_model,
            base_model=base_model,
            task=task,
            output_dir=output_dir,
            config=config,
            pruning_config=pruning_config,
            title_prefix=title_prefix,
            show_progress=True,
        )
    
    return result


def run_ablation_sweep_eval(
    masked_model: MaskedSparseGPT,
    base_model,
    tokenizer,
    sweep_config: Dict,
    pruning_config: PruningConfig,
    mean_cache: Dict,
    output_dir: Path,
    model_name: str,
    device: str = "cuda",
):
    """Run ablation sweep evaluation."""
    print("  Running Ablation sweep evaluation...")
    
    task_name = sweep_config["task_name"]
    task = get_task(task_name, tokenizer, seed=42, split="superval")
    
    config = AblationSweepConfig(
        num_points=11,
        num_trials=10,
        num_batches=10,
        batch_size=pruning_config.batch_size,
        seq_length=pruning_config.seq_length,
        device=device,
    )
    
    ablation_type = sweep_config.get("ablation_type", "mean_pretrain")
    ablation_str = "zero" if ablation_type == "zero" else "mean"
    title_prefix = f"{model_name} ({ablation_str}_ablate)\n"
    
    with torch.autocast(device, dtype=torch.bfloat16):
        result = run_and_save_ablation_sweep(
            masked_model=masked_model,
            base_model=base_model,
            task=task,
            output_dir=output_dir,
            config=config,
            pruning_config=pruning_config,
            mean_cache=mean_cache if ablation_type != "zero" else None,
            title_prefix=title_prefix,
            show_progress=True,
        )
    
    return result


def run_mask_relaxation_eval(
    masked_model: MaskedSparseGPT,
    tokenizer,
    sweep_config: Dict,
    pruning_config: PruningConfig,
    output_dir: Path,
    model_name: str,
    device: str = "cuda",
):
    """Run mask relaxation evaluation.
    
    This evaluation relaxes the mask by activating fractions of the masked nodes,
    measuring how loss changes as more non-circuit nodes are included.
    """
    print("  Running Mask relaxation evaluation...")
    
    task_name = sweep_config["task_name"]
    task = get_task(task_name, tokenizer, seed=42, split="superval")
    
    config = MaskRelaxationConfig(
        num_points=11,
        num_trials=10,
        num_batches=10,
        batch_size=pruning_config.batch_size,
        seq_length=pruning_config.seq_length,
        device=device,
    )
    
    ablation_type = sweep_config.get("ablation_type", "mean_pretrain")
    ablation_str = "zero" if ablation_type == "zero" else "mean"
    title_prefix = f"{model_name} ({ablation_str}_ablate)\n"
    
    with torch.autocast(device, dtype=torch.bfloat16):
        result = run_and_save_mask_relaxation(
            masked_model=masked_model,
            task=task,
            output_dir=output_dir,
            config=config,
            title_prefix=title_prefix,
            show_progress=True,
        )
    
    return result


def run_all_evals_for_model(model_dir: Path, device: str = "cuda"):
    """Run all evaluations for a single model."""
    print(f"\n{'='*70}")
    print(f"Processing: {model_dir.name}")
    print(f"{'='*70}")
    
    # Create evals output directory
    evals_dir = model_dir / "best_checkpoint" / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        (
            masked_model, base_model, tokenizer,
            sweep_config, hparams, mean_cache, pruning_config
        ) = load_model_checkpoint(model_dir, device)
        
        model_name = model_dir.name
        
        # Run all evaluations
        results = {}
        
        # 1. Pareto curve
        try:
            pareto_data = run_pareto_eval(
                masked_model, tokenizer, sweep_config, evals_dir, device
            )
            results["pareto"] = "success"
        except Exception as e:
            print(f"    ERROR in pareto: {e}")
            results["pareto"] = f"error: {e}"
        
        # 2. Interchange evaluation (includes FVU)
        try:
            interchange_result = run_interchange_eval(
                masked_model, base_model, tokenizer,
                sweep_config, pruning_config, evals_dir,
                model_name, device
            )
            results["interchange"] = {
                "fvu_global": interchange_result.fvu_global,
                "fvu_per_layer": {str(k): v for k, v in interchange_result.fvu_per_layer.items()},
            }
        except Exception as e:
            print(f"    ERROR in interchange: {e}")
            import traceback
            traceback.print_exc()
            results["interchange"] = f"error: {e}"
        
        # 3. Ablation sweep
        try:
            ablation_result = run_ablation_sweep_eval(
                masked_model, base_model, tokenizer,
                sweep_config, pruning_config, mean_cache,
                evals_dir, model_name, device
            )
            results["ablation_sweep"] = {
                "clean_loss": ablation_result.clean_loss,
                "circuit_ablated_loss": ablation_result.circuit_ablated_loss,
            }
        except Exception as e:
            print(f"    ERROR in ablation sweep: {e}")
            import traceback
            traceback.print_exc()
            results["ablation_sweep"] = f"error: {e}"
        
        # 4. Mask relaxation sweep
        try:
            relaxation_result = run_mask_relaxation_eval(
                masked_model, tokenizer,
                sweep_config, pruning_config,
                evals_dir, model_name, device
            )
            results["mask_relaxation"] = {
                "circuit_only_loss": relaxation_result.circuit_only_loss,
                "all_active_loss": relaxation_result.all_active_loss,
            }
        except Exception as e:
            print(f"    ERROR in mask relaxation: {e}")
            import traceback
            traceback.print_exc()
            results["mask_relaxation"] = f"error: {e}"
        
        # Save combined results
        with open(evals_dir / "eval_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"  All evaluations saved to: {evals_dir}")
        
        # Free memory
        del masked_model, base_model
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"  FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def create_comparison_pareto_plot(
    model_dirs: List[Path],
    output_path: Path,
    title: str,
):
    """Create a comparison pareto plot for multiple models."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10.colors
    
    for i, model_dir in enumerate(model_dirs):
        # Try to load pareto data from evals folder first, then from best_checkpoint
        pareto_path = model_dir / "best_checkpoint" / "evals" / "pareto_superval_data.json"
        if not pareto_path.exists():
            pareto_path = model_dir / "pareto_superval_data.json"
        if not pareto_path.exists():
            print(f"  Warning: No pareto data found for {model_dir.name}")
            continue
        
        with open(pareto_path) as f:
            pareto_data = json.load(f)
        
        ks = [d["k"] for d in pareto_data]
        losses = [d["loss"] for d in pareto_data]
        
        label = model_dir.name
        ax.plot(ks, losses, '-o', markersize=4, linewidth=1.5, 
                color=colors[i % len(colors)], label=label)
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved comparison plot to: {output_path}")


def create_all_comparison_plots():
    """Create comparison pareto plots for each model variant type."""
    all_dirs = get_model_dirs(exclude_ignore=True)
    
    # Group models by their base model (without the ablation/embed suffix)
    base_models = {}
    for d in all_dirs:
        name = d.name
        # Extract base model name by removing suffix
        for suffix in ["_mean_noembed", "_zero_noembed", "_mean", "_zero"]:
            if name.endswith(suffix):
                base_name = name[:-len(suffix)]
                break
        else:
            base_name = name  # No suffix found
        
        if base_name not in base_models:
            base_models[base_name] = []
        base_models[base_name].append(d)
    
    # For each base model, get the four variants
    for base_name, dirs in base_models.items():
        mean_noembed = [d for d in dirs if d.name.endswith("_mean_noembed")]
        zero_noembed = [d for d in dirs if d.name.endswith("_zero_noembed")]
        mean = [d for d in dirs if d.name.endswith("_mean") and not d.name.endswith("_noembed")]
        zero = [d for d in dirs if d.name.endswith("_zero") and not d.name.endswith("_noembed")]
        
        print(f"\nBase model: {base_name}")
        print(f"  mean_noembed: {[d.name for d in mean_noembed]}")
        print(f"  zero_noembed: {[d.name for d in zero_noembed]}")
        print(f"  mean: {[d.name for d in mean]}")
        print(f"  zero: {[d.name for d in zero]}")
    
    # Now create comparison plots across base models
    # Find all mean_noembed models
    all_mean_noembed = [d for d in all_dirs if d.name.endswith("_mean_noembed")]
    all_zero_noembed = [d for d in all_dirs if d.name.endswith("_zero_noembed")]
    all_mean = [d for d in all_dirs if d.name.endswith("_mean") and not d.name.endswith("_noembed")]
    all_zero = [d for d in all_dirs if d.name.endswith("_zero") and not d.name.endswith("_noembed")]
    
    # Create comparison plots
    if len(all_mean_noembed) > 0:
        create_comparison_pareto_plot(
            all_mean_noembed,
            CARBS_RESULTS_DIR / "pareto_comparison_mean_noembed.png",
            "Pareto Comparison: Mean Ablation (No Embed Mask)"
        )
    
    if len(all_zero_noembed) > 0:
        create_comparison_pareto_plot(
            all_zero_noembed,
            CARBS_RESULTS_DIR / "pareto_comparison_zero_noembed.png",
            "Pareto Comparison: Zero Ablation (No Embed Mask)"
        )
    
    if len(all_mean) > 0:
        create_comparison_pareto_plot(
            all_mean,
            CARBS_RESULTS_DIR / "pareto_comparison_mean.png",
            "Pareto Comparison: Mean Ablation (With Embed Mask)"
        )
    
    if len(all_zero) > 0:
        create_comparison_pareto_plot(
            all_zero,
            CARBS_RESULTS_DIR / "pareto_comparison_zero.png",
            "Pareto Comparison: Zero Ablation (With Embed Mask)"
        )


def main():
    print("="*70)
    print("Running All Evaluations for Pruned Models")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Get all model directories (excluding ignore ones)
    model_dirs = get_model_dirs(exclude_ignore=True)
    print(f"\nFound {len(model_dirs)} model directories:")
    for d in model_dirs:
        print(f"  - {d.name}")
    
    # Run evaluations for each model
    all_results = {}
    for model_dir in model_dirs:
        results = run_all_evals_for_model(model_dir, device)
        all_results[model_dir.name] = results
    
    # Create comparison plots
    print("\n" + "="*70)
    print("Creating Comparison Pareto Plots")
    print("="*70)
    create_all_comparison_plots()
    
    print("\n" + "="*70)
    print("ALL EVALUATIONS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

