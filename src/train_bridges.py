"""
Training loop for bridges training.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

Bridges couple a frozen dense model to a weight-sparse model trained from scratch.
The training objective includes:
- Standard LM loss on sparse model
- KL distillation from dense to sparse
- NMSE reconstruction losses for bridges
- KL losses for hybrid forward passes
"""

import os
import math
import time
import json
from pathlib import Path
from typing import Optional, List
from dataclasses import asdict

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from .config import ModelConfig, SparsityConfig
from .config_bridges import FullBridgesConfig
from .model import SparseGPT, create_model
from .bridges import (
    BridgeSet,
    BridgeNMSEResult,
    KLTargetCache,
    compute_bridge_nmse_loss,
    compute_hybrid_kl_losses,
    kl_divergence,
    verify_model_is_dense,
)
from .sparsity import WeightSparsifier, SharkfinScheduler, normalize_grad_rms_
from .data import create_dataloader, create_validation_data


def upload_to_hub(
    sparse_model: nn.Module,
    bridge_set: BridgeSet,
    config: FullBridgesConfig,
    repo_id: str,
    checkpoint_dir: str,
    wandb_url: Optional[str] = None,
):
    """
    Upload sparse model, bridges, and config to HuggingFace Hub.
    
    Args:
        sparse_model: The trained sparse model
        bridge_set: The trained bridge modules
        config: Full bridges training configuration
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        checkpoint_dir: Local checkpoint directory
        wandb_url: Optional W&B run URL to include in README
    """
    from huggingface_hub import HfApi, create_repo
    
    print(f"\nUploading to HuggingFace Hub: {repo_id}")
    
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Note: {e}")
    
    checkpoint_path = Path(checkpoint_dir)
    
    # Save sparse model state dict
    sparse_model_path = checkpoint_path / "sparse_model.bin"
    torch.save(sparse_model.state_dict(), sparse_model_path)
    
    # Save bridges state dict
    bridges_path = checkpoint_path / "bridges.bin"
    torch.save(bridge_set.state_dict(), bridges_path)
    
    # Save config as JSON
    config_path = checkpoint_path / "config.json"
    config_dict = {
        "model_config": {
            "n_layer": config.sparse_model.n_layer,
            "d_model": config.sparse_model.d_model,
            "n_ctx": config.sparse_model.n_ctx,
            "d_head": config.sparse_model.d_head,
            "d_mlp": config.sparse_model.d_mlp,
            "vocab_size": config.sparse_model.vocab_size,
            "use_rms_norm": config.sparse_model.use_rms_norm,
            "tie_embeddings": config.sparse_model.tie_embeddings,
            "use_positional_embeddings": config.sparse_model.use_positional_embeddings,
            "use_bigram_table": config.sparse_model.use_bigram_table,
            "use_attention_sinks": config.sparse_model.use_attention_sinks,
            "activation": config.sparse_model.activation,
            "dropout": config.sparse_model.dropout,
            "use_bias": config.sparse_model.use_bias,
        },
        "sparsity_config": {
            "enable_weight_sparsity": config.sparsity.enable_weight_sparsity,
            "target_l0_fraction": config.sparsity.target_l0_fraction,
            "enable_activation_sparsity": config.sparsity.enable_activation_sparsity,
            "activation_topk_fraction": config.sparsity.activation_topk_fraction,
        },
        "bridges_config": {
            "encoder_afrac": config.bridges.encoder_afrac,
            "n_layers": config.sparse_model.n_layer,
            "d_dense": config.dense_model.repo_id or config.dense_model.local_path,
            "d_sparse": config.sparse_model.d_model,
        },
        "training_config": {
            "total_tokens": config.training.total_tokens,
            "batch_size": config.training.batch_size,
            "dataset_name": config.training.dataset_name,
            "tokenizer_name": config.training.tokenizer_name,
        },
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save the full YAML config
    yaml_path = checkpoint_path / "training_config.yaml"
    config.to_yaml(str(yaml_path))
    
    # Create a README
    readme_path = checkpoint_path / "README.md"
    
    # Build wandb section if URL is available
    wandb_section = ""
    if wandb_url:
        wandb_section = f"""
## Training Run

- **W&B Run**: [{wandb_url}]({wandb_url})
"""
    
    dense_model_source = config.dense_model.repo_id or config.dense_model.local_path
    readme_content = f"""# {repo_id.split('/')[-1]}

Weight-sparse transformer with bridges, trained with the procedure from Gao et al. (2025).

## Model Details (Sparse Model)

- **Layers**: {config.sparse_model.n_layer}
- **Model Dimension**: {config.sparse_model.d_model}
- **Context Length**: {config.sparse_model.n_ctx}
- **Head Dimension**: {config.sparse_model.d_head}
- **Vocabulary Size**: {config.sparse_model.vocab_size}

## Bridges

- **Dense Model**: {dense_model_source}
- **Encoder Activation Fraction**: {config.bridges.encoder_afrac}

## Sparsity

- **Weight Sparsity**: {config.sparsity.enable_weight_sparsity}
- **Target L0 Fraction**: {config.sparsity.target_l0_fraction}
- **Activation Sparsity**: {config.sparsity.enable_activation_sparsity}

## Training

- **Dataset**: {config.training.dataset_name}
- **Tokenizer**: {config.training.tokenizer_name}
- **Total Tokens**: {config.training.total_tokens:,}
{wandb_section}
## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model and bridges
sparse_model_path = hf_hub_download(repo_id="{repo_id}", filename="sparse_model.bin")
bridges_path = hf_hub_download(repo_id="{repo_id}", filename="bridges.bin")
config_path = hf_hub_download(repo_id="{repo_id}", filename="config.json")

# Load (requires the SparseGPT and BridgeSet classes from this repo)
sparse_state_dict = torch.load(sparse_model_path)
bridges_state_dict = torch.load(bridges_path)
```
"""
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    # Upload files
    files_to_upload = [
        ("sparse_model.bin", sparse_model_path),
        ("bridges.bin", bridges_path),
        ("config.json", config_path),
        ("training_config.yaml", yaml_path),
        ("README.md", readme_path),
    ]
    
    for filename, filepath in files_to_upload:
        if filepath.exists():
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
    
    print(f"  Done! Model available at: https://huggingface.co/{repo_id}")


def load_dense_model(config: FullBridgesConfig, device: str = "cpu") -> SparseGPT:
    """
    Load the frozen dense model.
    
    Args:
        config: Full bridges config
        device: Device to load the model on
        
    Returns:
        Loaded dense model in eval mode with requires_grad=False
    """
    if config.dense_model.local_path:
        # Load from local checkpoint
        local_path = Path(config.dense_model.local_path)
        
        # Load config
        config_path = local_path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        model_config = ModelConfig(**config_dict["model_config"])
        sparsity_config = SparsityConfig(**config_dict["sparsity_config"])
        
        # Create and load model
        model = SparseGPT(model_config, sparsity_config)
        
        model_path = local_path / "pytorch_model.bin"
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
    else:
        # Load from HuggingFace Hub
        model = SparseGPT.from_pretrained(config.dense_model.repo_id, device=device)
    
    # Freeze the model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count various parameter statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_params": total,
        "trainable_params": trainable,
    }


def compute_grad_stats(model: nn.Module) -> dict:
    """Compute gradient statistics for logging."""
    stats = {}
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms.append(grad_norm)
    
    if grad_norms:
        stats["grad/norm_mean"] = sum(grad_norms) / len(grad_norms)
        stats["grad/norm_max"] = max(grad_norms)
        stats["grad/norm_min"] = min(grad_norms)
    
    return stats


@torch.no_grad()
def evaluate_validation(
    dense_model: nn.Module,
    sparse_model: nn.Module,
    bridge_set: BridgeSet,
    val_batches: list,
    accelerator: Accelerator,
    batch_size: int = 16,
    device: str = "cuda",
    kl_approx_n: Optional[int] = None,
) -> dict:
    """
    Evaluate models on validation data.
    
    Args:
        dense_model: The frozen dense model
        sparse_model: The sparse model being trained
        bridge_set: The bridge modules for computing MSE/KL losses
        val_batches: List of token tensors
        accelerator: Accelerator for mixed precision autocast
        batch_size: Batch size for evaluation
        device: Device to use
        
    Returns:
        Dictionary with validation metrics
    """
    sparse_model.eval()
    bridge_set.eval()
    
    total_loss_sparse = 0.0
    total_loss_dense = 0.0
    total_kl_sparse = 0.0
    total_nmse = 0.0
    total_kl_d2s = 0.0
    total_kl_s2d = 0.0
    total_tokens = 0
    n_batches = 0
    
    for i in range(0, len(val_batches), batch_size):
        batch_tensors = val_batches[i:i+batch_size]
        input_ids = torch.stack(batch_tensors).to(device)
        device_type = input_ids.device.type
        
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            # Forward passes with bridge sites to get activations
            # Dense model doesn't have activation sparsity, so pre == post
            y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
            # Sparse model returns pre and post AbsTopK activations
            y_sparse, h_sparse_pre_list, h_sparse_post_list = sparse_model.forward_with_bridge_sites(input_ids)
        
        # Compute losses
        shift_logits_sparse = y_sparse[:, :-1, :].contiguous()
        shift_logits_dense = y_dense[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Cross-entropy losses
        loss_sparse = F.cross_entropy(
            shift_logits_sparse.view(-1, shift_logits_sparse.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        
        loss_dense = F.cross_entropy(
            shift_logits_dense.view(-1, shift_logits_dense.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        
        # KL distillation loss (dense -> sparse)
        kl_sparse = kl_divergence(shift_logits_dense, shift_logits_sparse, topk=kl_approx_n)
        
        # Bridge NMSE loss
        # Encoder targets post-AbsTopK, decoder takes pre-AbsTopK as input
        # Use hard=True for validation (no soft thresholding)
        nmse_result = compute_bridge_nmse_loss(
            h_dense_list, h_sparse_pre_list, h_sparse_post_list, bridge_set,
            sharpness=None, hard=True,
        )
        
        # Hybrid KL losses (simplified for validation - no gradient buffers needed)
        kl_d2s_total = 0.0
        kl_s2d_total = 0.0
        n_sites = len(h_dense_list)
        
        for site_idx in range(n_sites):
            # d2s: encode dense -> run sparse -> KL to dense (hard threshold)
            h_encoded = bridge_set.encode(site_idx, h_dense_list[site_idx], sharpness=None, hard=True)
            # Cast to match model dtype (handles autocast dtype mismatches)
            if h_encoded.dtype != y_dense.dtype:
                h_encoded = h_encoded.to(y_dense.dtype)
            y_hybrid_d2s = sparse_model.forward_from_site(h_encoded, site_idx, input_ids)
            kl_d2s_total += kl_divergence(y_dense, y_hybrid_d2s, topk=kl_approx_n).item()
            
            # s2d: decode pre-AbsTopK sparse -> run dense -> KL to dense
            h_decoded = bridge_set.decode(site_idx, h_sparse_pre_list[site_idx])
            # Cast to match model dtype
            if h_decoded.dtype != y_dense.dtype:
                h_decoded = h_decoded.to(y_dense.dtype)
            y_hybrid_s2d = dense_model.forward_from_site(h_decoded, site_idx, input_ids)
            kl_s2d_total += kl_divergence(y_dense, y_hybrid_s2d, topk=kl_approx_n).item()
        
        total_loss_sparse += loss_sparse.item()
        total_loss_dense += loss_dense.item()
        total_kl_sparse += kl_sparse.item()
        total_nmse += nmse_result.total.item()
        total_kl_d2s += kl_d2s_total
        total_kl_s2d += kl_s2d_total
        total_tokens += shift_labels.numel()
        n_batches += 1
    
    sparse_model.train()
    bridge_set.train()
    
    avg_loss_sparse = total_loss_sparse / total_tokens if total_tokens > 0 else 0.0
    avg_loss_dense = total_loss_dense / total_tokens if total_tokens > 0 else 0.0
    avg_kl_sparse = total_kl_sparse / n_batches if n_batches > 0 else 0.0
    avg_nmse = total_nmse / n_batches if n_batches > 0 else 0.0
    avg_kl_d2s = total_kl_d2s / n_batches if n_batches > 0 else 0.0
    avg_kl_s2d = total_kl_s2d / n_batches if n_batches > 0 else 0.0
    
    return {
        "val/loss_sparse": avg_loss_sparse,
        "val/loss_dense": avg_loss_dense,
        "val/perplexity_sparse": math.exp(min(avg_loss_sparse, 100)),
        "val/perplexity_dense": math.exp(min(avg_loss_dense, 100)),
        "val/kl_sparse": avg_kl_sparse,
        "val/nmse": avg_nmse,
        "val/kl_d2s": avg_kl_d2s,
        "val/kl_s2d": avg_kl_s2d,
        "val/tokens": total_tokens,
    }


def save_checkpoint(
    accelerator: Accelerator,
    sparse_model: nn.Module,
    bridge_set: BridgeSet,
    optimizer: torch.optim.Optimizer,
    scheduler: SharkfinScheduler,
    sparsifier: Optional[WeightSparsifier],
    step: int,
    loss: float,
    checkpoint_dir: str,
    keep_n: int = 5,
    sparse_model_unwrapped: Optional[nn.Module] = None,
    bridge_set_unwrapped: Optional[nn.Module] = None,
):
    """Save a training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"checkpoint-{step}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    if accelerator.is_main_process:
        # Save sparse model (use provided unwrapped ref if available)
        if sparse_model_unwrapped is None:
            sparse_model_unwrapped = accelerator.unwrap_model(sparse_model)
        torch.save(
            sparse_model_unwrapped.state_dict(),
            checkpoint_path / "sparse_model.bin"
        )
        
        # Save bridges (use provided unwrapped ref if available)
        if bridge_set_unwrapped is None:
            bridge_set_unwrapped = accelerator.unwrap_model(bridge_set)
        torch.save(
            bridge_set_unwrapped.state_dict(),
            checkpoint_path / "bridges.bin"
        )
        
        # Save optimizer
        torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.bin")
        
        # Save extra state
        extra_state = {
            "step": step,
            "loss": loss,
            "sparsifier_state": {
                "current_step": sparsifier.state.current_step,
                "current_l0_fraction": sparsifier.state.current_l0_fraction,
            } if sparsifier is not None else None,
            "scheduler_state": {
                "current_step": scheduler.current_step,
            },
        }
        with open(checkpoint_path / "extra_state.json", "w") as f:
            json.dump(extra_state, f)
    
    # Clean up old checkpoints
    if accelerator.is_main_process:
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1])
        )
        while len(checkpoints) > keep_n:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    
    checkpoints = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


def load_checkpoint(
    checkpoint_path: str,
    sparse_model: nn.Module,
    bridge_set: BridgeSet,
    optimizer: torch.optim.Optimizer,
    scheduler: SharkfinScheduler,
    sparsifier: Optional[WeightSparsifier],
    accelerator: Accelerator,
) -> int:
    """Load a training checkpoint and return the step to resume from."""
    checkpoint_path = Path(checkpoint_path)
    accelerator.print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load sparse model
    sparse_model_path = checkpoint_path / "sparse_model.bin"
    if sparse_model_path.exists():
        state_dict = torch.load(sparse_model_path, map_location=accelerator.device, weights_only=True)
        accelerator.unwrap_model(sparse_model).load_state_dict(state_dict)
        accelerator.print("  Loaded sparse model")
    
    # Load bridges
    bridges_path = checkpoint_path / "bridges.bin"
    if bridges_path.exists():
        state_dict = torch.load(bridges_path, map_location=accelerator.device, weights_only=True)
        accelerator.unwrap_model(bridge_set).load_state_dict(state_dict)
        accelerator.print("  Loaded bridges")
    
    # Load optimizer
    optimizer_path = checkpoint_path / "optimizer.bin"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=accelerator.device, weights_only=True))
        accelerator.print("  Loaded optimizer")
    
    # Load extra state
    extra_state_path = checkpoint_path / "extra_state.json"
    if extra_state_path.exists():
        with open(extra_state_path, "r") as f:
            extra_state = json.load(f)
        
        step = extra_state.get("step", 0)
        
        # Restore scheduler
        if "scheduler_state" in extra_state:
            scheduler.current_step = extra_state["scheduler_state"].get("current_step", 0)
        
        # Restore sparsifier
        if sparsifier is not None and "sparsifier_state" in extra_state:
            sparsifier.state.current_step = extra_state["sparsifier_state"].get("current_step", 0)
            sparsifier.state.current_l0_fraction = extra_state["sparsifier_state"].get("current_l0_fraction", 1.0)
        
        accelerator.print(f"  Resuming from step {step}")
        return step
    
    return 0


def _compile_models(dense_model, sparse_model, bridge_set, mode: str, backend: str = "inductor"):
    """
    Helper to compile models with torch.compile.
    
    Separated into a function to allow serialized compilation across ranks.
    
    IMPORTANT: Bridges training uses custom forward methods (forward_with_bridge_sites,
    forward_from_site) instead of the standard forward(). torch.compile only optimizes
    forward() by default, so we must explicitly compile these custom methods.
    
    Args:
        dense_model: The frozen dense model
        sparse_model: The sparse model (may be DDP-wrapped)
        bridge_set: The bridge modules (may be DDP-wrapped)
        mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        backend: Compilation backend ("inductor", "eager", "aot_eager")
    """
    compile_kwargs = {"mode": mode, "backend": backend}
    
    # Get the underlying models (unwrap DDP if needed)
    sparse_underlying = sparse_model.module if hasattr(sparse_model, "module") else sparse_model
    bridge_underlying = bridge_set.module if hasattr(bridge_set, "module") else bridge_set
    
    # Compile custom forward methods that bridges training actually uses
    # These methods are called directly and bypass the standard forward()
    dense_model.forward_with_bridge_sites = torch.compile(
        dense_model.forward_with_bridge_sites, **compile_kwargs
    )
    dense_model.forward_from_site = torch.compile(
        dense_model.forward_from_site, **compile_kwargs
    )
    
    sparse_underlying.forward_with_bridge_sites = torch.compile(
        sparse_underlying.forward_with_bridge_sites, **compile_kwargs
    )
    sparse_underlying.forward_from_site = torch.compile(
        sparse_underlying.forward_from_site, **compile_kwargs
    )
    
    # For bridge_set, the forward methods of individual bridges and the encode/decode
    # methods are used. Compile the main encode_all/decode_all and per-bridge methods.
    # Actually, the BridgeSet.encode and BridgeSet.decode call Bridge.encode/decode
    # which are simple Linear + activation. Let's compile at the BridgeSet level.
    bridge_underlying.encode = torch.compile(bridge_underlying.encode, **compile_kwargs)
    bridge_underlying.decode = torch.compile(bridge_underlying.decode, **compile_kwargs)
    
    return dense_model, sparse_model, bridge_set


def train_bridges(config: FullBridgesConfig):
    """
    Main bridges training function.
    
    Args:
        config: Full bridges configuration object
    """
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="wandb" if config.training.use_wandb else None,
    )
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Initialize W&B on main process
    wandb_run_url = None
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name,
            entity=config.training.wandb_entity,
            config=config.to_dict(),
        )
        wandb_run_url = wandb.run.url
        accelerator.print(f"W&B run: {wandb_run_url}")
        
        # Save config at the start
        config_path = Path(config.training.checkpoint_dir) / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(config_path))
        wandb.save(str(config_path))
    
    # ==========================================================================
    # Load and verify dense model
    # ==========================================================================
    accelerator.print("Loading dense model...")
    dense_model = load_dense_model(config, device=accelerator.device)
    
    # Verify it's actually dense
    verify_model_is_dense(dense_model, "dense_model")
    accelerator.print(f"  Dense model verified: no sparsity")
    
    d_dense = dense_model.config.d_model
    n_layers_dense = dense_model.config.n_layer
    accelerator.print(f"  d_model={d_dense}, n_layers={n_layers_dense}")
    
    # ==========================================================================
    # Create dataloader and tokenizer
    # ==========================================================================
    accelerator.print("Loading dataset and tokenizer...")
    dataloader, tokenizer = create_dataloader(
        dataset_name=config.training.dataset_name,
        tokenizer_name=config.training.tokenizer_name,
        seq_length=config.sparse_model.n_ctx,
        batch_size=config.training.batch_size,
        split=config.training.dataset_split,
        text_column=config.training.text_column,
        num_workers=config.training.num_workers,
        seed=config.training.seed,
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
    )
    
    # Update vocab size from tokenizer
    config.sparse_model.vocab_size = len(tokenizer)
    accelerator.print(f"Vocabulary size: {config.sparse_model.vocab_size}")
    
    # Create validation data
    accelerator.print("Loading validation data...")
    val_batches, val_desc = create_validation_data(
        dataset_name=config.training.dataset_name,
        tokenizer=tokenizer,
        seq_length=config.sparse_model.n_ctx,
        text_column=config.training.text_column,
        val_split=config.training.val_split,
        holdout_fraction=config.training.val_holdout_fraction,
        max_tokens=config.training.val_max_batches * config.sparse_model.n_ctx * 16,
        seed=config.training.seed + 1,
    )
    accelerator.print(f"  {val_desc}")
    
    # ==========================================================================
    # Create sparse model (randomly initialized)
    # ==========================================================================
    accelerator.print("Creating sparse model...")
    sparse_model = create_model(config.sparse_model, config.sparsity)
    
    d_sparse = config.sparse_model.d_model
    n_layers_sparse = config.sparse_model.n_layer
    accelerator.print(f"  d_model={d_sparse}, n_layers={n_layers_sparse}")
    
    # Verify layer counts match (required for bridges)
    assert n_layers_dense == n_layers_sparse, (
        f"Dense and sparse models must have the same number of layers. "
        f"Got {n_layers_dense} vs {n_layers_sparse}"
    )
    
    # Log parameter counts
    param_stats_sparse = count_parameters(sparse_model)
    param_stats_dense = count_parameters(dense_model)
    accelerator.print(f"  Sparse model parameters: {param_stats_sparse['total_params']:,}")
    accelerator.print(f"  Dense model parameters: {param_stats_dense['total_params']:,}")
    
    # ==========================================================================
    # Create bridges
    # ==========================================================================
    accelerator.print("Creating bridges...")
    bridge_set = BridgeSet(
        n_layers=n_layers_sparse,
        d_dense=d_dense,
        d_sparse=d_sparse,
        encoder_afrac=config.bridges.encoder_afrac,
        encoder_type=config.bridges.bridge_act_fn,
        init_log_eps=config.bridges.threshold_init_log_eps,
    )
    accelerator.print(f"  Encoder type: {config.bridges.bridge_act_fn}")
    if config.bridges.bridge_act_fn == "threshold":
        accelerator.print(f"  Initial log_eps: {config.bridges.threshold_init_log_eps}")
    
    param_stats_bridges = count_parameters(bridge_set)
    accelerator.print(f"  Bridge parameters: {param_stats_bridges['total_params']:,}")
    accelerator.print(f"  Number of bridge sites: {bridge_set.n_sites}")
    
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.log({
            "model/sparse_params": param_stats_sparse["total_params"],
            "model/dense_params": param_stats_dense["total_params"],
            "model/bridge_params": param_stats_bridges["total_params"],
        }, step=0)
    
    # ==========================================================================
    # Calculate training steps
    # ==========================================================================
    tokens_per_step = (
        config.training.batch_size 
        * config.sparse_model.n_ctx 
        * config.training.gradient_accumulation_steps
        * accelerator.num_processes
    )
    total_steps = config.training.total_tokens // tokens_per_step
    accelerator.print(f"Total training steps: {total_steps:,}")
    accelerator.print(f"Tokens per step: {tokens_per_step:,}")
    
    # ==========================================================================
    # Create optimizer (only for sparse model and bridges)
    # ==========================================================================
    # Combine parameters from sparse model and bridges
    trainable_params = list(sparse_model.parameters()) + list(bridge_set.parameters())
    
    use_raw_adam = config.optimizer.optimizer_type == "adam"
    
    if use_raw_adam:
        try:
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=config.optimizer.learning_rate,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.eps,
                fused=True,
            )
        except TypeError:
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=config.optimizer.learning_rate,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.eps,
            )
        accelerator.print("Using raw Adam optimizer")
    else:
        try:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.optimizer.learning_rate,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
                fused=True,
            )
        except TypeError:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.optimizer.learning_rate,
                betas=(config.optimizer.beta1, config.optimizer.beta2),
                eps=config.optimizer.eps,
                weight_decay=config.optimizer.weight_decay,
            )
    
    # ==========================================================================
    # Create weight sparsifier (for sparse model only, NOT bridges)
    # ==========================================================================
    sparsifier = WeightSparsifier(
        model=sparse_model,  # Only sparsify the sparse model
        target_l0_fraction=config.sparsity.target_l0_fraction,
        anneal_start_fraction=config.sparsity.sparsity_anneal_start_fraction,
        anneal_end_fraction=config.sparsity.sparsity_anneal_end_fraction,
        min_weights_per_neuron=config.sparsity.min_weights_per_neuron,
        total_steps=total_steps,
        anneal_type=config.sparsity.anneal_type,
    ) if config.sparsity.enable_weight_sparsity else None
    
    # ==========================================================================
    # Create learning rate scheduler
    # ==========================================================================
    scheduler = SharkfinScheduler(
        optimizer=optimizer,
        base_lr=config.optimizer.learning_rate,
        total_steps=total_steps,
        warmup_fraction=config.optimizer.warmup_fraction,
        enable_lr_decay=config.optimizer.enable_lr_decay,
        sparsifier=sparsifier,
        use_sharkfin=config.optimizer.use_sharkfin_schedule,
    )
    
    # ==========================================================================
    # Prepare for distributed training
    # ==========================================================================
    sparse_model, bridge_set, optimizer, dataloader = accelerator.prepare(
        sparse_model, bridge_set, optimizer, dataloader
    )
    
    # Move dense model to device (it's not optimized so not prepared)
    dense_model = dense_model.to(accelerator.device)
    
    # Cast all modules to the same dtype for mixed precision
    # This is needed because hybrid KL losses are computed outside autocast for stability
    if config.training.mixed_precision == "bf16":
        sparse_model = sparse_model.to(torch.bfloat16)
        dense_model = dense_model.to(torch.bfloat16)
        bridge_set = bridge_set.to(torch.bfloat16)
    elif config.training.mixed_precision == "fp16":
        sparse_model = sparse_model.to(torch.float16)
        dense_model = dense_model.to(torch.float16)
        bridge_set = bridge_set.to(torch.float16)
    
    # ==========================================================================
    # Apply torch.compile if enabled (PyTorch 2.0+ JIT compilation)
    # ==========================================================================
    if config.training.use_torch_compile:
        backend = config.training.torch_compile_backend
        mode = config.training.torch_compile_mode
        accelerator.print(f"Applying torch.compile (backend={backend}, mode={mode})...")
        
        # For multi-GPU, serialize torch.compile initialization to avoid race conditions
        # in Triton template registration. Each rank compiles one at a time.
        if accelerator.num_processes > 1:
            for rank in range(accelerator.num_processes):
                if accelerator.process_index == rank:
                    accelerator.print(f"  Rank {rank}: Compiling models...")
                    dense_model, sparse_model, bridge_set = _compile_models(
                        dense_model, sparse_model, bridge_set,
                        mode=mode,
                        backend=backend,
                    )
                # Barrier to ensure serialized compilation
                accelerator.wait_for_everyone()
        else:
            # Single GPU - just compile directly
            dense_model, sparse_model, bridge_set = _compile_models(
                dense_model, sparse_model, bridge_set,
                mode=mode,
                backend=backend,
            )
        
        accelerator.print("  torch.compile applied successfully")
    
    # ==========================================================================
    # Save references to unwrapped models AFTER torch.compile
    # These are needed for custom forward methods (forward_with_bridge_sites, etc.)
    # that may not be directly accessible through the compiled/DDP wrappers.
    # ==========================================================================
    sparse_model_unwrapped = accelerator.unwrap_model(sparse_model)
    bridge_set_unwrapped = accelerator.unwrap_model(bridge_set)
    
    # ==========================================================================
    # Resume from checkpoint if specified
    # ==========================================================================
    start_step = 0
    if config.training.resume_from_checkpoint:
        ckpt_path = config.training.resume_from_checkpoint
        if ckpt_path.lower() == "latest":
            ckpt_path = find_latest_checkpoint(config.training.checkpoint_dir)
            if ckpt_path:
                accelerator.print(f"Found latest checkpoint: {ckpt_path}")
        
        if ckpt_path:
            start_step = load_checkpoint(
                checkpoint_path=ckpt_path,
                sparse_model=sparse_model,
                bridge_set=bridge_set,
                optimizer=optimizer,
                scheduler=scheduler,
                sparsifier=sparsifier,
                accelerator=accelerator,
            )
    
    # ==========================================================================
    # Training loop
    # ==========================================================================
    accelerator.print("Starting bridges training...")
    
    sparse_model.train()
    bridge_set.train()
    
    step = start_step
    tokens_seen = start_step * tokens_per_step
    running_loss = 0.0
    running_loss_components = {
        "ce_sparse": 0.0,
        "kl_sparse": 0.0,
        "nmse": 0.0,
        "kl_d2s": 0.0,
        "kl_s2d": 0.0,
    }
    # Per-site running loss trackers for granular logging
    n_bridge_sites = 2 * n_layers_sparse + 1
    running_kl_per_site = {f"d2s_site{i}": 0.0 for i in range(n_bridge_sites)}
    running_kl_per_site.update({f"s2d_site{i}": 0.0 for i in range(n_bridge_sites)})
    running_mse_per_site = {f"encoder_site{i}": 0.0 for i in range(n_bridge_sites)}
    running_mse_per_site.update({f"decoder_site{i}": 0.0 for i in range(n_bridge_sites)})
    start_time = time.time()
    
    progress_bar = tqdm(
        total=total_steps,
        initial=start_step,
        desc="Training",
        disable=not accelerator.is_main_process,
    )
    
    data_iter = iter(dataloader)
    grad_accum_steps = config.training.gradient_accumulation_steps
    
    # Set initial LR (skip if resuming - scheduler already at correct step)
    if start_step == 0:
        scheduler.step()
    
    # Log threshold encoder config at start of training
    if config.bridges.bridge_act_fn == "threshold" and accelerator.is_main_process:
        anneal_start_step = int(total_steps * config.bridges.threshold_anneal_start_fraction)
        anneal_end_step = int(total_steps * config.bridges.threshold_anneal_end_fraction)
        accelerator.print(f"Threshold encoder config:")
        accelerator.print(f"  Sharpness annealing: {config.bridges.threshold_sharpness_init} -> {config.bridges.threshold_sharpness_final}")
        accelerator.print(f"  Anneal from step {anneal_start_step} ({config.bridges.threshold_anneal_start_fraction*100:.0f}%) to {anneal_end_step} ({config.bridges.threshold_anneal_end_fraction*100:.0f}% of {total_steps})")
        accelerator.print(f"  Initial log_eps: {config.bridges.threshold_init_log_eps} (eps = {math.exp(config.bridges.threshold_init_log_eps):.4f})")
        accelerator.print(f"  Starting from step: {start_step}")
        
        # Verify log_eps parameters are trainable (float32 for precision)
        for i, bridge in enumerate(bridge_set_unwrapped.bridges):
            log_eps = bridge.encoder.log_eps
            accelerator.print(f"  Site {i} log_eps: requires_grad={log_eps.requires_grad}, dtype={log_eps.dtype}")
    
    while step < total_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # Forward/backward pass
        # IMPORTANT: include *all* DDP-wrapped trainable modules here so that
        # gradient accumulation uses no_sync consistently (avoids early all-reduce
        # on `bridge_set` when grad_accum_steps > 1).
        with accelerator.accumulate(sparse_model, bridge_set):
            # Determine autocast dtype
            if config.training.mixed_precision == "bf16":
                autocast_dtype = torch.bfloat16
            elif config.training.mixed_precision == "fp16":
                autocast_dtype = torch.float16
            else:
                autocast_dtype = None
            
            # ==================================================================
            # Forward passes
            # ==================================================================
            # Use pre-saved unwrapped model reference (saved before torch.compile)
            # to access custom forward methods
            
            if autocast_dtype is not None:
                device_type = accelerator.device.type
                if device_type == "cpu" and autocast_dtype == torch.float16:
                    autocast_dtype = None
                if autocast_dtype is not None:
                    with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype):
                        # Dense model forward (no grad)
                        # Dense model doesn't have activation sparsity, so pre == post
                        with torch.no_grad():
                            y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
                        
                        # Sparse model forward - returns pre and post AbsTopK activations
                        y_sparse, h_sparse_pre_list, h_sparse_post_list = sparse_model_unwrapped.forward_with_bridge_sites(input_ids)
                else:
                    with torch.no_grad():
                        y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
                    y_sparse, h_sparse_pre_list, h_sparse_post_list = sparse_model_unwrapped.forward_with_bridge_sites(input_ids)
            else:
                with torch.no_grad():
                    y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
                y_sparse, h_sparse_pre_list, h_sparse_post_list = sparse_model_unwrapped.forward_with_bridge_sites(input_ids)
            
            # ==================================================================
            # Compute losses (outside autocast for numerical stability)
            # ==================================================================
            
            # Compute sharpness for threshold encoder (if using)
            # Stays at sharpness_init until anneal_start_fraction, then linearly anneals
            # to sharpness_final until anneal_end_fraction, then uses hard thresholding
            if config.bridges.bridge_act_fn == "threshold":
                anneal_start_step = int(total_steps * config.bridges.threshold_anneal_start_fraction)
                anneal_end_step = int(total_steps * config.bridges.threshold_anneal_end_fraction)
                
                if step >= anneal_end_step:
                    # After anneal_end_fraction of training, use hard thresholding
                    encoder_sharpness = None
                    encoder_hard = True
                elif step < anneal_start_step:
                    # Before anneal_start_fraction of training, use initial sharpness
                    encoder_sharpness = config.bridges.threshold_sharpness_init
                    encoder_hard = False
                else:
                    # Linearly anneal sharpness from init to final
                    progress = (step - anneal_start_step) / (anneal_end_step - anneal_start_step)
                    encoder_sharpness = (
                        config.bridges.threshold_sharpness_init
                        + progress * (config.bridges.threshold_sharpness_final - config.bridges.threshold_sharpness_init)
                    )
                    encoder_hard = False
            else:
                # AbsTopK encoder doesn't use sharpness
                encoder_sharpness = None
                encoder_hard = True
            
            # Shift for next-token prediction
            shift_logits_sparse = y_sparse[:, :-1, :].contiguous()
            shift_logits_dense = y_dense[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Pre-compute KL target cache for efficiency
            # This avoids recomputing top-k indices and target softmax for each KL call
            kl_target_cache = KLTargetCache(shift_logits_dense, temperature=1.0, topk=config.bridges.kl_approx_n)
            
            # 1. Cross-entropy on sparse model (standard LM loss)
            loss_ce_sparse = F.cross_entropy(
                shift_logits_sparse.view(-1, shift_logits_sparse.size(-1)),
                shift_labels.view(-1),
            )
            
            # 2. KL distillation from dense to sparse (use cache)
            loss_kl_sparse = kl_divergence(
                shift_logits_dense, shift_logits_sparse, target_cache=kl_target_cache
            )
            
            # 3. NMSE reconstruction loss
            # NOTE: Use pre-saved unwrapped bridge_set reference
            # Encoder targets post-AbsTopK, decoder takes pre-AbsTopK as input
            nmse_result = compute_bridge_nmse_loss(
                h_dense_list, h_sparse_pre_list, h_sparse_post_list, bridge_set_unwrapped,
                sharpness=encoder_sharpness, hard=encoder_hard,
            )
            loss_nmse = nmse_result.total
            
            # 4 & 5. Hybrid KL losses
            # NOTE: Cannot use kl_target_cache here because it was computed on
            # shifted logits (batch, seq-1, vocab) but hybrid outputs have shape
            # (batch, seq, vocab). Using the cache would cause shape misalignment.
            # Instead, pass topk directly so top-k indices are recomputed per call.
            # NOTE: Use unwrapped models to access custom forward methods (forward_from_site)
            # Decoder uses pre-AbsTopK activations for s2d direction
            hybrid_result = compute_hybrid_kl_losses(
                dense_model=dense_model,
                sparse_model=sparse_model_unwrapped,
                bridge_set=bridge_set_unwrapped,
                h_dense_list=h_dense_list,
                h_sparse_pre_list=h_sparse_pre_list,
                y_dense=y_dense,
                input_ids=input_ids,
                kl_target_cache=None,  # Don't use cache due to shape mismatch
                sharpness=encoder_sharpness,
                hard=encoder_hard,
                topk=config.bridges.kl_approx_n,
            )
            loss_kl_d2s = hybrid_result.kl_d2s
            loss_kl_s2d = hybrid_result.kl_s2d
            
            # Total loss with configurable coefficients
            total_loss = (
                config.bridges.coef_ce_sparse * loss_ce_sparse
                + config.bridges.coef_kl_sparse * loss_kl_sparse
                + config.bridges.coef_nmse * loss_nmse
                + config.bridges.coef_kl_d2s * loss_kl_d2s
                + config.bridges.coef_kl_s2d * loss_kl_s2d
            )
            
            # Track loss components
            running_loss_components["ce_sparse"] += loss_ce_sparse.detach().item() / grad_accum_steps
            running_loss_components["kl_sparse"] += loss_kl_sparse.detach().item() / grad_accum_steps
            running_loss_components["nmse"] += loss_nmse.detach().item() / grad_accum_steps
            running_loss_components["kl_d2s"] += loss_kl_d2s.detach().item() / grad_accum_steps
            running_loss_components["kl_s2d"] += loss_kl_s2d.detach().item() / grad_accum_steps
            
            # Track per-site losses for granular logging
            for key, val in hybrid_result.get_detailed_losses().items():
                running_kl_per_site[key] += val / grad_accum_steps
            for key, val in nmse_result.get_detailed_losses().items():
                running_mse_per_site[key] += val / grad_accum_steps
            
            # Backward pass
            # Use retain_graph=True to allow gradient buffer release after
            #
            # IMPORTANT: The hybrid KL losses use gradient buffers that release
            # gradients AFTER the main backward pass. In multi-GPU training, DDP
            # synchronizes gradients during backward(), but the gradient buffer
            # release happens after DDP's hooks have fired. This means those
            # gradients would not be synchronized across GPUs.
            #
            # Fix: When we need to sync gradients (last accumulation step), we:
            # 1. Use no_sync() to prevent DDP from syncing during backward
            # 2. Call backward and release_gradients
            # 3. Manually all_reduce the gradients across all processes
            if accelerator.num_processes > 1 and accelerator.sync_gradients:
                # Multi-GPU on sync step: prevent auto-sync, then manually sync
                with accelerator.no_sync(sparse_model), accelerator.no_sync(bridge_set):
                    accelerator.backward(total_loss, retain_graph=True)
                    hybrid_result.release_gradients()
                
                # Manually synchronize all gradients across processes
                for param in sparse_model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)
                for param in bridge_set.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)
            else:
                # Single GPU or accumulation step (no sync needed yet)
                accelerator.backward(total_loss, retain_graph=True)
                hybrid_result.release_gradients()
            
            # Gradient normalization
            if accelerator.sync_gradients and config.optimizer.enable_grad_clip:
                # Normalize gradients for all trainable params
                all_params = list(sparse_model.parameters()) + list(bridge_set.parameters())
                grad_rms = normalize_grad_rms_(
                    all_params,
                    target_rms=config.optimizer.grad_clip_rms,
                )
            else:
                grad_rms = 0.0
            
            # Capture threshold encoder gradient info BEFORE optimizer.zero_grad()
            log_eps_grad_info = {}
            if config.bridges.bridge_act_fn == "threshold" and accelerator.sync_gradients:
                total_grad_norm = 0.0
                for i, bridge in enumerate(bridge_set_unwrapped.bridges):
                    if bridge.encoder.log_eps.grad is not None:
                        grad_norm = bridge.encoder.log_eps.grad.norm().item()
                        log_eps_grad_info[f"site{i}_grad_norm"] = grad_norm
                        total_grad_norm += grad_norm
                log_eps_grad_info["total_grad_norm"] = total_grad_norm
            
            # Optimizer step
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
        
        # Only update after gradient accumulation
        if accelerator.sync_gradients:
            # Apply weight sparsity (to sparse model only, NOT bridges)
            if sparsifier is not None:
                sparsifier.step()
            
            # Manual weight decay for raw Adam
            if use_raw_adam and config.optimizer.weight_decay > 0:
                current_lr = scheduler.get_lr()
                unwrapped_sparse = sparse_model_unwrapped
                # Note: bridges don't get weight decay (they're relatively small)
                with torch.no_grad():
                    for name, param in unwrapped_sparse.named_parameters():
                        if len(param.shape) > 1 and "bigram_table" not in name:
                            param.data -= config.optimizer.weight_decay * current_lr * param.data
            
            # Update learning rate
            scheduler.step()
            
            step += 1
            tokens_seen += tokens_per_step
            running_loss += total_loss.detach().item()
            
            # Logging
            if step % config.training.log_every_n_steps == 0:
                avg_loss = running_loss / config.training.log_every_n_steps
                running_loss = 0.0
                
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_seen / elapsed
                
                log_dict = {
                    "train/loss": avg_loss,
                    "train/loss_ce_sparse": running_loss_components["ce_sparse"] / config.training.log_every_n_steps,
                    "train/loss_kl_sparse": running_loss_components["kl_sparse"] / config.training.log_every_n_steps,
                    "train/loss_nmse": running_loss_components["nmse"] / config.training.log_every_n_steps,
                    "train/loss_kl_d2s": running_loss_components["kl_d2s"] / config.training.log_every_n_steps,
                    "train/loss_kl_s2d": running_loss_components["kl_s2d"] / config.training.log_every_n_steps,
                    "train/learning_rate": scheduler.get_lr(),
                    "train/tokens_seen": tokens_seen,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step": step,
                    "train/grad_rms": grad_rms,
                }
                
                # Log threshold encoder stats if using threshold activation
                if config.bridges.bridge_act_fn == "threshold":
                    log_dict["train/encoder_sharpness"] = encoder_sharpness if encoder_sharpness is not None else float("inf")
                    log_dict["train/encoder_hard"] = float(encoder_hard)
                    
                    # Log mean log_eps and eps values per site
                    # Log log_eps directly to see small changes that might be hidden in exp()
                    for i, bridge in enumerate(bridge_set_unwrapped.bridges):
                        mean_log_eps = bridge.encoder.log_eps.mean().item()
                        mean_eps = bridge.encoder.eps.mean().item()
                        log_dict[f"threshold/site{i}_mean_log_eps"] = mean_log_eps
                        log_dict[f"threshold/site{i}_mean_eps"] = mean_eps
                    
                    # Log aggregate stats across all sites
                    all_log_eps = torch.cat([b.encoder.log_eps for b in bridge_set_unwrapped.bridges])
                    log_dict["threshold/all_sites_mean_log_eps"] = all_log_eps.mean().item()
                    log_dict["threshold/all_sites_std_log_eps"] = all_log_eps.std().item()
                    
                    # Log gradient info (captured before optimizer.zero_grad())
                    if log_eps_grad_info:
                        log_dict["threshold/total_log_eps_grad_norm"] = log_eps_grad_info.get("total_grad_norm", 0.0)
                        for key, val in log_eps_grad_info.items():
                            if key.startswith("site"):
                                log_dict[f"threshold/{key}"] = val
                
                # Add per-site KL losses for granular logging (under KL/ section)
                for key, val in running_kl_per_site.items():
                    log_dict[f"KL/{key}"] = val / config.training.log_every_n_steps
                
                # Add per-site MSE losses for granular logging (under MSE/ section)
                for key, val in running_mse_per_site.items():
                    log_dict[f"MSE/{key}"] = val / config.training.log_every_n_steps
                
                # Reset loss components
                for key in running_loss_components:
                    running_loss_components[key] = 0.0
                for key in running_kl_per_site:
                    running_kl_per_site[key] = 0.0
                for key in running_mse_per_site:
                    running_mse_per_site[key] = 0.0
                
                # Sparsity stats
                if sparsifier is not None and step % config.training.log_sparsity_every_n_steps == 0:
                    sparsity_stats = sparsifier.get_sparsity_stats()
                    log_dict.update({
                        "sparsity/" + k: v for k, v in sparsity_stats.items()
                    })
                
                # Gradient stats
                if step % config.training.log_gradients_every_n_steps == 0:
                    grad_stats = compute_grad_stats(sparse_model_unwrapped)
                    log_dict.update(grad_stats)
                
                # Validation
                if (
                    accelerator.is_main_process
                    and step % config.training.eval_every_n_steps == 0
                    and len(val_batches) > 0
                ):
                    val_stats = evaluate_validation(
                        dense_model=dense_model,
                        sparse_model=sparse_model_unwrapped,
                        bridge_set=bridge_set_unwrapped,
                        val_batches=val_batches[:config.training.val_max_batches],
                        accelerator=accelerator,
                        batch_size=16,
                        device=accelerator.device,
                        kl_approx_n=config.bridges.kl_approx_n,
                    )
                    log_dict.update(val_stats)
                    accelerator.print(
                        f"  Step {step}: sparse_loss={val_stats['val/loss_sparse']:.4f}, "
                        f"dense_loss={val_stats['val/loss_dense']:.4f}, "
                        f"kl={val_stats['val/kl_sparse']:.4f}, nmse={val_stats['val/nmse']:.4f}"
                    )
                
                if accelerator.is_main_process and config.training.use_wandb:
                    wandb.log(log_dict, step=step)
                
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_lr():.2e}",
                    "L0": f"{sparsifier.get_current_l0_fraction():.4f}" if sparsifier else "N/A",
                })
            
            # Checkpointing
            if step % config.training.checkpoint_every_n_steps == 0:
                save_checkpoint(
                    accelerator=accelerator,
                    sparse_model=sparse_model,
                    bridge_set=bridge_set,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sparsifier=sparsifier,
                    step=step,
                    loss=total_loss.item(),
                    checkpoint_dir=config.training.checkpoint_dir,
                    keep_n=config.training.keep_n_checkpoints,
                    sparse_model_unwrapped=sparse_model_unwrapped,
                    bridge_set_unwrapped=bridge_set_unwrapped,
                )
            
            progress_bar.update(1)
    
    # Final checkpoint
    save_checkpoint(
        accelerator=accelerator,
        sparse_model=sparse_model,
        bridge_set=bridge_set,
        optimizer=optimizer,
        scheduler=scheduler,
        sparsifier=sparsifier,
        step=step,
        loss=total_loss.item(),
        checkpoint_dir=config.training.checkpoint_dir,
        keep_n=config.training.keep_n_checkpoints,
        sparse_model_unwrapped=sparse_model_unwrapped,
        bridge_set_unwrapped=bridge_set_unwrapped,
    )
    
    accelerator.print("Bridges training complete!")
    
    # Upload to HuggingFace Hub if configured
    if accelerator.is_main_process and config.training.hf_repo:
        upload_to_hub(
            sparse_model=sparse_model_unwrapped,
            bridge_set=bridge_set_unwrapped,
            config=config,
            repo_id=config.training.hf_repo,
            checkpoint_dir=config.training.checkpoint_dir,
            wandb_url=wandb_run_url,
        )
    
    if accelerator.is_main_process and config.training.use_wandb:
        wandb.finish()


def main():
    """Main entry point for bridges training."""
    import os
    from huggingface_hub import login
    import wandb
    # Login using environment variables (set HF_TOKEN and WANDB_API_KEY)
    if os.environ.get("HF_TOKEN"):
        login(os.environ["HF_TOKEN"])
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    import argparse
    
    parser = argparse.ArgumentParser(description="Train weight-sparse transformer with bridges")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Override config values, e.g., --override bridges.coef_nmse=2.0 training.batch_size=32",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = FullBridgesConfig.from_yaml(args.config)
    
    # Apply overrides
    for override in args.override:
        key, value = override.split("=", 1)
        parts = key.split(".")
        
        # Navigate to the right config object
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Convert value to appropriate type
        attr_name = parts[-1]
        current_value = getattr(obj, attr_name)
        
        if isinstance(current_value, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current_value, int):
            value = int(value)
        elif isinstance(current_value, float):
            value = float(value)
        
        setattr(obj, attr_name, value)
    
    # Run training
    train_bridges(config)


if __name__ == "__main__":
    main()
