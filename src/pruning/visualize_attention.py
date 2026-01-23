"""
Visualize attention patterns for pruned vs unpruned models.

For any attention head containing an active Q, K, or V node,
plots the attention pattern as a heatmap (lower triangular)
with token strings as axis labels.

Usage:
    python -m src.pruning.visualize_attention \
        --checkpoint_dir path/to/best_checkpoint \
        --output_dir outputs/attention_plots
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer

from .run_pruning import load_model


def load_checkpoint_masks(checkpoint_dir: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Load mask state from a checkpoint directory."""
    masks_path = os.path.join(checkpoint_dir, "masks.pt")
    if not os.path.exists(masks_path):
        raise FileNotFoundError(f"No masks.pt found at {masks_path}")
    return torch.load(masks_path, map_location=device, weights_only=True)


def get_active_heads(
    masks: Dict[str, torch.Tensor],
    n_layers: int,
    n_heads: int,
    d_head: int,
) -> List[Tuple[int, int]]:
    """
    Find attention heads that have at least one active Q, K, or V node.
    
    Returns:
        List of (layer, head) tuples
    """
    active_heads = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            # Get the slice indices for this head
            start_idx = head * d_head
            end_idx = (head + 1) * d_head
            
            # Check Q, K, V masks for this head
            q_key = f"layer{layer}_attn_q"
            k_key = f"layer{layer}_attn_k"
            v_key = f"layer{layer}_attn_v"
            
            has_active = False
            for key in [q_key, k_key, v_key]:
                if key in masks:
                    tau = masks[key]
                    # Check if any node in this head is active (tau >= 0)
                    head_mask = tau[start_idx:end_idx]
                    if (head_mask >= 0).any():
                        has_active = True
                        break
            
            if has_active:
                active_heads.append((layer, head))
    
    return active_heads


def compute_attention_pattern(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    masked_model=None,
) -> torch.Tensor:
    """
    Compute attention patterns for a specific layer.
    
    Args:
        model: The base SparseGPT model
        input_ids: (1, seq_len) token IDs
        layer_idx: Which layer to extract attention from
        masked_model: Optional MaskedSparseGPT to use masks during forward
        
    Returns:
        attention: (n_heads, seq_len, seq_len) attention weights
    """
    model.eval()
    
    B, T = input_ids.shape
    device = input_ids.device
    
    with torch.no_grad():
        # Get embeddings
        x = model.wte(input_ids)
        
        if model.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x = x + model.wpe(pos)
        
        x = model.drop(x)
        
        # Forward through blocks up to the target layer
        if masked_model is not None:
            mask_fn = masked_model._get_masking_fn()
        else:
            mask_fn = None
        
        for idx, block in enumerate(model.blocks):
            if idx < layer_idx:
                # Run normally up to target layer
                if masked_model is not None:
                    masked_model._current_layer = idx
                    x = masked_model._forward_block_with_masks(block, x, mask_fn)
                else:
                    x = block(x)
            elif idx == layer_idx:
                # At target layer, compute attention pattern manually
                normed = block.ln_1(x)
                
                if mask_fn is not None:
                    masked_model._current_layer = idx
                    normed = mask_fn(normed, "attn_in")
                
                attn = block.attn
                
                # QKV projection
                qkv = attn.c_attn(normed)
                q, k, v = qkv.split(attn.n_heads * attn.d_head, dim=-1)
                
                # Apply Q/K/V masks if using masked model
                if mask_fn is not None:
                    q = mask_fn(q, "attn_q")
                    k = mask_fn(k, "attn_k")
                    v = mask_fn(v, "attn_v")
                
                # Reshape for attention
                q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                k = k.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
                
                # Compute attention scores
                scale = 1.0 / (attn.d_head ** 0.5)
                attn_scores = (q @ k.transpose(-2, -1)) * scale
                
                # Apply causal mask
                causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
                attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
                
                # Softmax to get attention weights
                attention = F.softmax(attn_scores, dim=-1)
                
                return attention.squeeze(0)  # (n_heads, seq_len, seq_len)
            else:
                break
    
    raise RuntimeError(f"Failed to compute attention at layer {layer_idx}")


def compute_attention_pattern_masked(
    masked_model,
    input_ids: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """
    Compute attention patterns using a MaskedSparseGPT model.
    """
    return compute_attention_pattern(
        masked_model.model,
        input_ids,
        layer_idx,
        masked_model=masked_model,
    )


def plot_attention_heatmap(
    attention: np.ndarray,
    tokens: List[str],
    title: str,
    output_path: str,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot attention pattern as a lower triangular heatmap.
    
    Args:
        attention: (seq_len, seq_len) attention weights
        tokens: List of token strings
        title: Plot title
        output_path: Where to save the figure
        figsize: Figure size
    """
    seq_len = len(tokens)
    
    # Create lower triangular mask
    mask = np.triu(np.ones_like(attention, dtype=bool), k=1)
    attention_masked = np.ma.masked_array(attention, mask=mask)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Custom colormap (white to dark blue)
    cmap = LinearSegmentedColormap.from_list(
        'attention',
        ['#ffffff', '#4a90d9', '#1e3a5f']
    )
    
    # Plot heatmap
    im = ax.imshow(attention_masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Set ticks and labels
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    
    # Truncate long tokens for display
    max_label_len = 10
    x_labels = [t[:max_label_len] + '...' if len(t) > max_label_len else t for t in tokens]
    y_labels = [t[:max_label_len] + '...' if len(t) > max_label_len else t for t in tokens]
    
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Labels
    ax.set_xlabel('Key Position (source)', fontsize=11)
    ax.set_ylabel('Query Position (target)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(
    attn_unpruned: np.ndarray,
    attn_pruned: np.ndarray,
    tokens: List[str],
    layer: int,
    head: int,
    output_path: str,
    figsize: Tuple[int, int] = (20, 9),
):
    """
    Plot side-by-side comparison of unpruned vs pruned attention patterns.
    """
    seq_len = len(tokens)
    
    # Create lower triangular mask
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'attention',
        ['#ffffff', '#4a90d9', '#1e3a5f']
    )
    
    titles = ['Unpruned Model', 'Pruned Model']
    attns = [attn_unpruned, attn_pruned]
    
    for ax, attn, title in zip(axes, attns, titles):
        attn_masked = np.ma.masked_array(attn, mask=mask)
        
        im = ax.imshow(attn_masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        
        # Truncate long tokens
        max_label_len = 8
        x_labels = [t[:max_label_len] + '..' if len(t) > max_label_len else t for t in tokens]
        y_labels = [t[:max_label_len] + '..' if len(t) > max_label_len else t for t in tokens]
        
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=14)
        ax.set_yticklabels(y_labels, fontsize=14)
        
        ax.set_xlabel('Key Position', fontsize=18)
        ax.set_ylabel('Query Position', fontsize=18)
        ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    fig.suptitle(f'Layer {layer}, Head {head} - Attention Patterns', fontsize=24, fontweight='bold', y=1.02)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize attention patterns for pruned vs unpruned models")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the best_checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (defaults to checkpoint_dir/evals)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for task generation")
    parser.add_argument("--max_heads", type=int, default=10,
                        help="Maximum number of heads to plot")
    
    args = parser.parse_args()
    
    # Create output directory (default to checkpoint_dir/evals/attention_plots)
    if args.output_dir is None:
        output_dir = Path(args.checkpoint_dir) / "evals" / "attention_plots"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config from checkpoint
    config_path = os.path.join(args.checkpoint_dir, "..", "sweep_config.json")
    if not os.path.exists(config_path):
        # Try parent directory
        config_path = os.path.join(args.checkpoint_dir, "config.json")
    
    with open(config_path, "r") as f:
        sweep_config = json.load(f)
    
    model_path = sweep_config["model_path"]
    tokenizer_name = sweep_config["tokenizer_name"]
    task_name = sweep_config["task_name"]
    ablation_type = sweep_config.get("ablation_type", "zero")
    
    print(f"Model: {model_path}")
    print(f"Task: {task_name}")
    print(f"Ablation: {ablation_type}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model, model_config_dict = load_model(model_path, args.device)
    
    n_layers = model.config.n_layer
    n_heads = model.config.n_heads
    d_head = model.config.d_head
    d_model = model.config.d_model
    d_mlp = model.config.d_mlp
    
    print(f"Model: {n_layers} layers, {n_heads} heads, d_head={d_head}")
    
    # Load masks
    print("Loading masks...")
    masks = load_checkpoint_masks(args.checkpoint_dir, args.device)
    
    # Find active heads
    active_heads = get_active_heads(masks, n_layers, n_heads, d_head)
    print(f"Found {len(active_heads)} attention heads with active Q/K/V nodes")
    
    if len(active_heads) == 0:
        print("No active heads found!")
        return
    
    # Print which heads are active
    for layer, head in active_heads[:args.max_heads]:
        print(f"  Layer {layer}, Head {head}")
    
    # Create masked model for pruned inference
    from .config import PruningConfig
    from .masked_model import MaskedSparseGPT
    
    config = PruningConfig(
        ablation_type=ablation_type,
        device=args.device,
    )
    
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(args.device)
    
    # Move masks to correct device before loading
    masks_device = {k: v.to(args.device) for k, v in masks.items()}
    masked_model.load_mask_state(masks_device)
    
    # Generate a single task example
    from .tasks import get_task
    
    task = get_task(task_name, tokenizer, seed=args.seed)
    example = task.generate_example()
    
    input_ids = example.positive_ids.unsqueeze(0).to(args.device)
    seq_len = input_ids.shape[1]
    
    # Decode tokens for labels
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]
    
    # Clean up tokens for display
    tokens = [t.replace('\n', '\\n').replace('\t', '\\t') for t in tokens]
    
    print(f"\nPrompt: {tokenizer.decode(input_ids[0].tolist())}")
    print(f"Sequence length: {seq_len}")
    print(f"Tokens: {tokens}")
    
    # Plot attention patterns for active heads
    print(f"\nPlotting attention patterns (max {args.max_heads} heads)...")
    
    for idx, (layer, head) in enumerate(active_heads[:args.max_heads]):
        print(f"  Processing Layer {layer}, Head {head}...")
        
        # Compute unpruned attention
        attn_unpruned = compute_attention_pattern(
            model, input_ids, layer, masked_model=None
        )
        attn_unpruned = attn_unpruned[head].cpu().numpy()
        
        # Compute pruned attention
        attn_pruned = compute_attention_pattern_masked(
            masked_model, input_ids, layer
        )
        attn_pruned = attn_pruned[head].cpu().numpy()
        
        # Plot comparison
        output_path = output_dir / f"attention_layer{layer}_head{head}.png"
        plot_comparison(
            attn_unpruned,
            attn_pruned,
            tokens,
            layer,
            head,
            str(output_path),
        )
    
    # Save metadata
    metadata = {
        "model_path": model_path,
        "task_name": task_name,
        "prompt": tokenizer.decode(input_ids[0].tolist()),
        "tokens": tokens,
        "active_heads": [{"layer": l, "head": h} for l, h in active_heads],
        "plotted_heads": [{"layer": l, "head": h} for l, h in active_heads[:args.max_heads]],
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDone! Plots saved to {output_dir}")
    print(f"Metadata saved to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()

