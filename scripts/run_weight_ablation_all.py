#!/usr/bin/env python3
"""
Run weight ablation evaluation for all pruned models in carbs_results_pronoun.

This script runs the weight ablation evaluation for each model that already
has node ablation results, saving outputs to the same evals directory.

Usage:
    python scripts/run_weight_ablation_all.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
from tqdm import tqdm
from typing import Dict, List

from transformers import AutoTokenizer

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model
from sparse_pretrain.src.pruning.weight_ablation_eval import (
    WeightAblationConfig,
    run_and_save_weight_ablation_sweep,
)


# Base directory for carbs results
CARBS_RESULTS_DIR = Path("outputs/carbs_results_pronoun")


def get_model_dirs_with_evals() -> List[Path]:
    """Get all model directories that have existing evals."""
    model_dirs = []
    for d in CARBS_RESULTS_DIR.iterdir():
        if d.is_dir() and (d / "best_checkpoint" / "evals").exists():
            if "ignore" in d.name.lower():
                continue
            model_dirs.append(d)
    return sorted(model_dirs)


def load_model_checkpoint(model_dir: Path, device: str = "cuda"):
    """Load a model checkpoint from a CARBS result directory."""
    # Load sweep config
    with open(model_dir / "sweep_config.json") as f:
        sweep_config = json.load(f)

    best_checkpoint_dir = model_dir / "best_checkpoint"

    # Load hparams and summary
    with open(best_checkpoint_dir / "hparams.json") as f:
        hparams = json.load(f)

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

    return masked_model, model, tokenizer, sweep_config, pruning_config


def run_weight_ablation_for_model(model_dir: Path, device: str = "cuda"):
    """Run weight ablation evaluation for a single model."""
    print(f"\n{'='*70}")
    print(f"Processing: {model_dir.name}")
    print(f"{'='*70}")

    evals_dir = model_dir / "best_checkpoint" / "evals"

    # Check if weight ablation already exists
    if (evals_dir / "weight_ablation_sweep.png").exists():
        print(f"  Weight ablation already exists, skipping...")
        return {"status": "skipped"}

    try:
        # Load model
        (
            masked_model, base_model, tokenizer,
            sweep_config, pruning_config
        ) = load_model_checkpoint(model_dir, device)

        model_name = model_dir.name

        # Load task
        task_name = sweep_config["task_name"]
        task = get_task(task_name, tokenizer, seed=42, split="superval")

        # Configure weight ablation
        config = WeightAblationConfig(
            num_points=11,      # 0%, 10%, ..., 100% of circuit weight count
            num_trials=10,      # Random trials per point
            num_batches=10,     # Evaluation batches
            batch_size=pruning_config.batch_size,
            seq_length=pruning_config.seq_length,
            device=device,
        )

        ablation_type = sweep_config.get("ablation_type", "mean_pretrain")
        ablation_str = "zero" if ablation_type == "zero" else "mean"
        title_prefix = f"{model_name} ({ablation_str}_ablate)\n"

        # Run weight ablation
        print("  Running Weight Ablation evaluation...")
        with torch.autocast(device, dtype=torch.bfloat16):
            result = run_and_save_weight_ablation_sweep(
                masked_model=masked_model,
                base_model=base_model,
                task=task,
                output_dir=evals_dir,
                config=config,
                pruning_config=pruning_config,
                title_prefix=title_prefix,
                show_progress=True,
            )

        print(f"  Weight ablation saved to: {evals_dir}")

        # Free memory
        del masked_model, base_model
        torch.cuda.empty_cache()

        return {
            "status": "success",
            "circuit_weight_count": result.circuit_weight_count,
            "total_weight_count": result.total_weight_count,
            "clean_loss": result.clean_loss,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


def main():
    print("="*70)
    print("Running Weight Ablation for All Pruned Models")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Get all model directories with existing evals
    model_dirs = get_model_dirs_with_evals()
    print(f"\nFound {len(model_dirs)} models with existing evaluations:")
    for d in model_dirs:
        print(f"  - {d.name}")

    # Run weight ablation for each model
    results = {}
    for model_dir in model_dirs:
        result = run_weight_ablation_for_model(model_dir, device)
        results[model_dir.name] = result

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = [k for k, v in results.items() if v.get("status") == "success"]
    skipped = [k for k, v in results.items() if v.get("status") == "skipped"]
    errors = [k for k, v in results.items() if v.get("status") == "error"]

    print(f"\nSuccessful: {len(successful)}")
    for name in successful:
        print(f"  ✓ {name}")

    if skipped:
        print(f"\nSkipped (already exists): {len(skipped)}")
        for name in skipped:
            print(f"  - {name}")

    if errors:
        print(f"\nErrors: {len(errors)}")
        for name in errors:
            print(f"  ✗ {name}: {results[name].get('error', 'unknown')}")

    print("\n" + "="*70)
    print("WEIGHT ABLATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
