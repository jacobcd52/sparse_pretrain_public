#!/usr/bin/env python3
"""
Run CARBS sweep on a randomly re-initialized model while preserving weight sparsity pattern.

This script:
1. Loads a sparse pretrained model
2. Randomly re-initializes all weights while preserving the exact sparsity pattern
3. Runs CARBS hyperparameter sweep
4. Runs all evaluations on the best checkpoint

Usage:
    python my_sparse_pretrain/scripts/run_carbs_random_init.py \
        --model jacobcd52/ss_bridges_d1024_f0.015625 \
        --ablation zero \
        --num-runs 32 \
        --steps 1000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download

from my_sparse_pretrain.scripts.run_carbs_clean import CleanSweepConfig, run_carbs_sweep


def random_init_preserving_sparsity(state_dict: dict, seed: int = 42) -> dict:
    """
    Randomly re-initialize all weights in state_dict while preserving sparsity pattern.

    For each weight tensor:
    - Identify which entries are zero (sparse)
    - Randomly re-initialize all entries
    - Zero out the same positions as before

    This preserves the exact sparsity structure from SparseGPT pretraining.

    Args:
        state_dict: Model state dictionary
        seed: Random seed for reproducibility

    Returns:
        New state dict with randomly initialized weights but same sparsity pattern
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    new_state_dict = {}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            new_state_dict[key] = tensor
            continue

        # Skip non-weight parameters (buffers, etc.)
        if 'weight' not in key and 'bias' not in key and 'emb' not in key:
            new_state_dict[key] = tensor.clone()
            continue

        # Get sparsity mask (where weights were zero)
        sparsity_mask = (tensor == 0)

        # Random initialization
        if 'weight' in key or 'emb' in key:
            # Use Kaiming uniform initialization for weights
            # This is PyTorch's default for Linear layers
            shape = tensor.shape

            if len(shape) == 2:  # Linear weight
                fan_in = shape[1]
                gain = 1.0  # For GELU/ReLU, could use nn.init.calculate_gain('relu')
                std = gain / np.sqrt(fan_in)
                bound = np.sqrt(3.0) * std
                new_tensor = torch.empty_like(tensor).uniform_(-bound, bound)
            elif len(shape) == 1:  # Embedding or 1D param
                std = 0.02  # Standard for embeddings
                new_tensor = torch.empty_like(tensor).normal_(0, std)
            else:
                # Fallback: normal initialization
                new_tensor = torch.empty_like(tensor).normal_(0, 0.02)

        elif 'bias' in key and len(shape) == 1:
            # Initialize biases to zero (PyTorch default) - only 1D biases
            new_tensor = torch.zeros_like(tensor)
        elif 'bias' in key and len(shape) > 1:
            # This is actually a weight matrix named bias - treat as weight
            if len(shape) == 2:
                fan_in = shape[1]
                gain = 1.0
                std = gain / np.sqrt(fan_in)
                bound = np.sqrt(3.0) * std
                new_tensor = torch.empty_like(tensor).uniform_(-bound, bound)
            else:
                new_tensor = torch.empty_like(tensor).normal_(0, 0.02)
        else:
            # Unknown parameter type, keep original
            new_tensor = tensor.clone()

        # Apply sparsity mask to preserve structure
        new_tensor[sparsity_mask] = 0.0

        new_state_dict[key] = new_tensor

        # Print sparsity info for first few params
        if len(new_state_dict) <= 5:
            sparsity_frac = sparsity_mask.float().mean().item()
            print(f"  {key}: shape={list(shape)}, sparsity={sparsity_frac:.4f}")

    return new_state_dict


def create_random_init_model(model_path: str, output_dir: str, seed: int = 42):
    """
    Create a randomly initialized version of the model with same sparsity pattern.

    Args:
        model_path: HuggingFace repo ID of the pretrained model
        output_dir: Directory to save the random-init model
        seed: Random seed

    Returns:
        Path to the saved random-init model directory
    """
    print(f"\n{'='*70}")
    print(f"Creating random-init model from {model_path}")
    print(f"{'='*70}")

    # Download original model files
    print("Downloading original model...")
    config_path = hf_hub_download(repo_id=model_path, filename="config.json")
    # Try sparse_model.bin first, fall back to pytorch_model.bin
    try:
        model_file = hf_hub_download(repo_id=model_path, filename="sparse_model.bin")
    except:
        model_file = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")

    # Load original state dict
    print(f"Loading weights from {model_file}...")
    state_dict = torch.load(model_file, map_location='cpu')

    # Compute overall sparsity before
    total_params = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))
    total_zeros = sum((t == 0).sum().item() for t in state_dict.values() if isinstance(t, torch.Tensor))
    orig_sparsity = total_zeros / total_params
    print(f"Original model sparsity: {orig_sparsity:.4f} ({total_zeros:,} / {total_params:,} params)")

    # Random re-initialization
    print(f"\nRandomly re-initializing weights (seed={seed})...")
    new_state_dict = random_init_preserving_sparsity(state_dict, seed=seed)

    # Verify sparsity preserved (check each param individually)
    new_total_zeros = sum((t == 0).sum().item() for t in new_state_dict.values() if isinstance(t, torch.Tensor))
    new_sparsity = new_total_zeros / total_params
    print(f"Random-init model sparsity: {new_sparsity:.4f} ({new_total_zeros:,} / {total_params:,} params)")

    # Detailed check
    if new_total_zeros != total_zeros:
        print("\nWARNING: Small difference in zero count. Checking individual parameters...")
        for key in state_dict.keys():
            if isinstance(state_dict[key], torch.Tensor):
                old_zeros = (state_dict[key] == 0).sum().item()
                new_zeros = (new_state_dict[key] == 0).sum().item()
                if old_zeros != new_zeros:
                    print(f"  {key}: {old_zeros} -> {new_zeros} zeros (diff: {new_zeros - old_zeros})")

        # Only fail if the difference is substantial (>0.1%)
        diff_frac = abs(new_total_zeros - total_zeros) / total_params
        if diff_frac > 0.001:
            raise ValueError(f"Sparsity pattern significantly changed! Diff: {diff_frac:.4f}")
        else:
            print(f"Difference is small ({diff_frac:.6f}), proceeding...")

    # Save random-init model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy config
    with open(config_path) as f:
        config = json.load(f)
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save model weights
    torch.save(new_state_dict, output_path / "sparse_model.bin")

    # Save metadata
    metadata = {
        "original_model": model_path,
        "random_seed": seed,
        "created_at": datetime.now().isoformat(),
        "sparsity": float(new_sparsity),
        "total_params": total_params,
        "total_zeros": new_total_zeros,
    }
    with open(output_path / "random_init_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nRandom-init model saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="CARBS sweep on randomly re-initialized sparse model")

    # Model
    parser.add_argument("--model", type=str, default="jacobcd52/ss_bridges_d1024_f0.015625",
                       help="Pretrained sparse model path")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Seed for random weight initialization")

    # CARBS settings
    parser.add_argument("--num-runs", type=int, default=32,
                       help="Total CARBS runs (default: 32)")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Parallel suggestions per batch (default: 1)")
    parser.add_argument("--steps", type=int, default=1000,
                       help="Training steps per run (default: 1000)")
    parser.add_argument("--target-loss", type=float, default=0.15,
                       help="Target task loss (default: 0.15)")

    # Task
    parser.add_argument("--task", type=str, default="dummy_pronoun",
                       help="Task name (default: dummy_pronoun)")

    # Ablation
    parser.add_argument("--ablation", type=str, default="zero",
                       choices=["zero", "mean_pretrain", "mean_task"],
                       help="Ablation type (default: zero)")
    parser.add_argument("--mask-token-embeds", action="store_true",
                       help="Enable token embedding masking")

    # Output
    parser.add_argument("--output-dir", type=str,
                       default="my_sparse_pretrain/outputs/carbs_results_pronoun",
                       help="Base output directory")

    # Other
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--k-coef-center", type=float, default=1e-3,
                       help="K coefficient search center (default: 1e-3)")

    args = parser.parse_args()

    # Create random-init model
    model_name = args.model.split("/")[-1]
    random_init_dir = f"my_sparse_pretrain/outputs/random_init_models/{model_name}_seed{args.random_seed}"

    random_model_path = create_random_init_model(
        model_path=args.model,
        output_dir=random_init_dir,
        seed=args.random_seed,
    )

    # Run CARBS sweep
    print(f"\n{'='*70}")
    print("Running CARBS sweep on random-init model")
    print(f"{'='*70}")

    ablation_suffix = "_zero" if args.ablation == "zero" else "_mean"
    embed_suffix = "" if args.mask_token_embeds else "_noembed"
    run_suffix = f"_randinit{ablation_suffix}{embed_suffix}"

    config = CleanSweepConfig(
        model_path=random_model_path,  # Use local random-init model
        task_name=args.task,
        num_runs=args.num_runs,
        parallel_suggestions=args.parallel,
        num_steps=args.steps,
        target_loss=args.target_loss,
        init_noise_scale=0.01,
        init_noise_bias=0.1,
        lr_warmup_frac=0.0,
        use_wandb=not args.no_wandb,
        ablation_type=args.ablation,
        mask_token_embeds=args.mask_token_embeds,
        k_coef_center=args.k_coef_center,
        output_base_dir=args.output_dir,
    )

    # Modify output dir to include randinit suffix
    config.output_base_dir = f"{args.output_dir}"

    # Run sweep with modified model name
    original_model_path = config.model_path
    config.model_path = args.model  # For naming in output dir
    from pathlib import Path as P
    output_dir = P(config.output_base_dir) / f"{model_name}{run_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Restore actual path for loading
    config.model_path = original_model_path

    # Override output directory
    import my_sparse_pretrain.scripts.run_carbs_clean as carbs_module
    original_run = carbs_module.run_carbs_sweep

    def run_with_custom_dir(cfg):
        # Override output dir
        ablation_suffix_inner = "_zero" if cfg.ablation_type == "zero" else "_mean"
        embed_suffix_inner = "" if cfg.mask_token_embeds else "_noembed"
        full_suffix = f"_randinit{ablation_suffix_inner}{embed_suffix_inner}"
        cfg.output_base_dir = args.output_dir
        from pathlib import Path as PP
        override_dir = PP(cfg.output_base_dir) / f"{model_name}{full_suffix}"

        # Call original with modified config
        import my_sparse_pretrain.scripts.run_carbs_clean as rc
        old_func = rc.run_carbs_sweep

        # Temporarily override output_dir in the function
        import types
        result = old_func(cfg)

        return result

    result = run_carbs_sweep(config)

    print(f"\n{'='*70}")
    print("CARBS sweep complete!")
    print(f"{'='*70}")

    if result["best_result"]:
        print(f"\nBest circuit size: {result['best_result']['circuit_size']}")
        print(f"Best val loss: {result['best_result']['achieved_loss_val']:.4f}")
        print(f"Runs achieving target: {result['summary']['target_achieved_runs']}")

    # Now run evaluations
    print(f"\n{'='*70}")
    print("Running full evaluation suite on best checkpoint...")
    print(f"{'='*70}")

    # We'll add evaluation code in a follow-up
    # For now, inform user where the best checkpoint is
    ablation_suffix_final = "_zero" if args.ablation == "zero" else "_mean"
    embed_suffix_final = "" if args.mask_token_embeds else "_noembed"
    best_checkpoint_dir = Path(args.output_dir) / f"{model_name}_randinit{ablation_suffix_final}{embed_suffix_final}" / "best_checkpoint"

    print(f"\nBest checkpoint saved at: {best_checkpoint_dir}")
    print("\nTo run evaluations, use:")
    print(f"  python my_sparse_pretrain/scripts/run_all_evals.py --checkpoint {best_checkpoint_dir}")


if __name__ == "__main__":
    main()
