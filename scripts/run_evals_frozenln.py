#!/usr/bin/env python3
"""
Run all evaluations for frozen layer norm pruned models.

This script runs evaluations for models in carbs_results_pronoun_frozenln_v2.

Usage:
    python my_sparse_pretrain/scripts/run_evals_frozenln.py
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
CARBS_RESULTS_DIR = Path("my_sparse_pretrain/outputs/carbs_results_pronoun_frozenln_v2")


def get_model_dirs() -> List[Path]:
    """Get all model directories with best_checkpoint."""
    all_dirs = []
    for d in CARBS_RESULTS_DIR.iterdir():
        if d.is_dir() and (d / "best_checkpoint").exists():
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
    
    # Create pruning config with freeze_layernorm_scale=True
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
        freeze_layernorm_scale=sweep_config.get("freeze_layernorm_scale", True),  # Enable frozen LN
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
    ax.set_title(f"Pareto Curve on Superval Set (Frozen LN)\nModel: {model_name}", fontsize=14)
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
        fractions=[i/10 for i in range(11)],  # 0.0, 0.1, 0.2, ..., 1.0
        num_trials=3,
        num_batches=10,
        batch_size=pruning_config.batch_size,
        seq_length=pruning_config.seq_length,
        device=device,
    )
    
    ablation_type = sweep_config.get("ablation_type", "mean_pretrain")
    ablation_str = "zero" if ablation_type == "zero" else "mean"
    title_prefix = f"{model_name} (Frozen LN, {ablation_str}_ablate)\n"
    
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
    title_prefix = f"{model_name} (Frozen LN, {ablation_str}_ablate)\n"
    
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
    """Run mask relaxation evaluation."""
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
    title_prefix = f"{model_name} (Frozen LN, {ablation_str}_ablate)\n"
    
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


def generate_circuit_html(
    masked_model: MaskedSparseGPT,
    tokenizer,
    sweep_config: Dict,
    output_path: Path,
    device: str = "cuda",
):
    """Generate circuit HTML visualization."""
    print("  Generating circuit HTML visualization...")
    
    task_name = sweep_config["task_name"]
    task = get_task(task_name, tokenizer, seed=42, split="val")
    
    # Import the HTML generation function from run_single_pruning
    from my_sparse_pretrain.scripts.run_single_pruning import generate_html_with_dashboards
    
    html_path = generate_html_with_dashboards(
        masked_model=masked_model,
        tokenizer=tokenizer,
        output_path=output_path,
        max_nodes=500,
        n_dashboard_samples=200,
        n_top_examples=10,
        device=device,
        task=task,
    )
    
    print(f"    Saved circuit HTML to {html_path}")
    return html_path


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
            import traceback
            traceback.print_exc()
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
        
        # 5. Circuit HTML visualization
        try:
            html_path = generate_circuit_html(
                masked_model, tokenizer, sweep_config,
                model_dir / "circuit.html", device
            )
            results["circuit_html"] = "success"
        except Exception as e:
            print(f"    ERROR in circuit HTML: {e}")
            import traceback
            traceback.print_exc()
            results["circuit_html"] = f"error: {e}"
        
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


def create_comparison_pareto_plot():
    """Create comparison pareto plots for zero vs mean ablation."""
    all_dirs = get_model_dirs()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'zero': 'blue', 'mean': 'red'}
    
    for model_dir in all_dirs:
        # Try to load pareto data
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
        
        # Determine ablation type from directory name
        if "_zero_" in model_dir.name:
            color = colors['zero']
            label = "Zero Ablation (Frozen LN)"
            linestyle = '-'
        else:
            color = colors['mean']
            label = "Mean Ablation (Frozen LN)"
            linestyle = '--'
        
        ax.plot(ks, losses, linestyle + 'o', markersize=4, linewidth=1.5, 
                color=color, label=label)
    
    ax.set_xlabel("Circuit Size (nodes)", fontsize=12)
    ax.set_ylabel("Superval Loss", fontsize=12)
    ax.set_title("Pareto Comparison: D1024 Model with Frozen Layer Norm Scale\nZero vs Mean Ablation", fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_path = CARBS_RESULTS_DIR / "pareto_comparison_frozenln.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved comparison plot to: {output_path}")


def main():
    print("="*70)
    print("Running Evaluations for Frozen LN Pruned Models")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Get all model directories
    model_dirs = get_model_dirs()
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
    print("Creating Comparison Pareto Plot")
    print("="*70)
    create_comparison_pareto_plot()
    
    # Save summary
    with open(CARBS_RESULTS_DIR / "eval_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("ALL EVALUATIONS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

