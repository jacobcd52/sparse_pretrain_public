"""
Interchange intervention evaluation for pruned circuits.

Evaluates how much of the pruned model's behavior depends on specific nodes
by replacing their activations with the unpruned model's activations.

This tests whether the pruned model has truly learned to rely on the circuit
or is just passing through similar activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import nullcontext

from .masked_model import MaskedSparseGPT
from .tasks import BinaryTask
from .config import PruningConfig


@dataclass
class InterchangeEvalConfig:
    """Configuration for interchange intervention evaluation."""
    # Fraction points to evaluate (0 = no interchange, 1 = all interchanged)
    fractions: List[float] = None  # Defaults to [0, 0.5, 1.0] for testing
    
    # Number of random trials per fraction
    num_trials: int = 3
    
    # Evaluation settings
    num_batches: int = 10
    batch_size: int = 64
    seq_length: int = 0  # 0 = dynamic padding
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        if self.fractions is None:
            # Default: 3 equally spaced for testing
            self.fractions = [0.0, 0.5, 1.0]


@dataclass 
class InterchangeResult:
    """Results from interchange intervention evaluation."""
    # Per-layer results: layer_idx -> {fraction -> (mean_loss, std_loss)}
    per_layer_results: Dict[int, Dict[float, Tuple[float, float]]]
    
    # All-layers combined results: fraction -> (mean_loss, std_loss)
    all_layers_results: Dict[float, Tuple[float, float]]
    
    # Baseline loss
    unpruned_clean_loss: float  # Clean unpruned model loss (no ablation)
    
    # FVU metrics
    fvu_global: float  # Global FVU across all layers
    fvu_per_layer: Dict[int, float]  # FVU per layer
    
    # Metadata
    num_active_nodes: int
    total_nodes: int
    fractions: List[float]
    num_trials: int


def get_active_node_indices(
    masked_model: MaskedSparseGPT,
) -> Dict[str, torch.Tensor]:
    """
    Get indices of active nodes (tau >= 0) for each mask location.
    
    Returns:
        Dict mapping mask key (e.g., "layer0_attn_in") to tensor of active indices
    """
    active_indices = {}
    for key, mask in masked_model.masks.masks.items():
        # Get indices where tau >= 0
        active = (mask.tau >= 0).nonzero(as_tuple=True)[0]
        active_indices[key] = active
    return active_indices


def run_forward_with_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    masked_model: MaskedSparseGPT,
    config: PruningConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run forward pass on the unpruned model and capture activations at mask locations.
    
    Args:
        model: The base (unpruned) model
        input_ids: Input token IDs (batch, seq)
        masked_model: The masked model (used to get mask locations)
        config: Pruning config
        
    Returns:
        Tuple of (logits, activations_dict) where activations_dict maps
        mask keys to activation tensors
    """
    device = input_ids.device
    B, T = input_ids.shape
    activations = {}
    
    # Forward through base model manually to capture activations
    x = model.wte(input_ids)
    
    if torch.is_autocast_enabled('cuda'):
        x = x.to(torch.get_autocast_dtype('cuda'))
    
    if model.wpe is not None:
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + model.wpe(pos)
    
    x = model.drop(x)
    
    for layer_idx, block in enumerate(model.blocks):
        # attn_in: after ln_1
        normed = block.ln_1(x)
        if "attn_in" in config.mask_locations:
            activations[f"layer{layer_idx}_attn_in"] = normed.clone()
        
        # Q, K, V projections
        qkv = block.attn.c_attn(normed)
        q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
        
        if "attn_q" in config.mask_locations:
            activations[f"layer{layer_idx}_attn_q"] = q.clone()
        if "attn_k" in config.mask_locations:
            activations[f"layer{layer_idx}_attn_k"] = k.clone()
        if "attn_v" in config.mask_locations:
            activations[f"layer{layer_idx}_attn_v"] = v.clone()
        
        # Full attention forward
        attn_out = block.attn(normed, None)
        if "attn_out" in config.mask_locations:
            activations[f"layer{layer_idx}_attn_out"] = attn_out.clone()
        
        x = x + attn_out
        
        # mlp_in: after ln_2
        normed = block.ln_2(x)
        if "mlp_in" in config.mask_locations:
            activations[f"layer{layer_idx}_mlp_in"] = normed.clone()
        
        # MLP forward to get neuron activations
        mlp_hidden = block.mlp.c_fc(normed)
        mlp_hidden = block.mlp.act_fn(mlp_hidden)
        if "mlp_neuron" in config.mask_locations:
            activations[f"layer{layer_idx}_mlp_neuron"] = mlp_hidden.clone()
        
        mlp_out = block.mlp.c_proj(mlp_hidden)
        mlp_out = block.mlp.dropout(mlp_out)
        if "mlp_out" in config.mask_locations:
            activations[f"layer{layer_idx}_mlp_out"] = mlp_out.clone()
        
        x = x + mlp_out
    
    # Final layer norm and LM head
    x = model.ln_f(x)
    logits = model.lm_head(x)
    
    if hasattr(model, 'bigram_table') and model.bigram_table is not None:
        bigram_logits = F.embedding(input_ids, model.bigram_table)
        logits = logits + bigram_logits
    
    return logits, activations


def run_masked_forward_with_activations(
    masked_model: MaskedSparseGPT,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Run forward pass on the MASKED model and capture activations AFTER masking.
    
    This is different from run_forward_with_activations which runs on the base model.
    Here we capture what the pruned model actually sees at each location after
    the masks and mean-ablation have been applied.
    
    Args:
        masked_model: The masked model
        input_ids: Input token IDs (batch, seq)
        
    Returns:
        Tuple of (logits, activations_dict) where activations_dict maps
        mask keys to activation tensors (post-masking)
    """
    device = input_ids.device
    B, T = input_ids.shape
    model = masked_model.model
    config = masked_model.config
    activations = {}
    
    # Get mask function which properly applies AbsTopK + node masks
    mask_fn = masked_model._get_masking_fn()
    
    # Compute frozen LN scales if enabled
    frozen_ln_scales = None
    if config.freeze_layernorm_scale:
        frozen_ln_scales = masked_model._compute_frozen_ln_scales(input_ids)
    
    # Forward through model with masking
    x = model.wte(input_ids)
    
    if torch.is_autocast_enabled('cuda'):
        x = x.to(torch.get_autocast_dtype('cuda'))
    
    # Apply token mask if enabled
    if masked_model.token_mask is not None:
        token_binary_mask = masked_model.token_mask.get_binary_mask()
        mask_per_position = token_binary_mask[input_ids].unsqueeze(-1)
        
        if masked_model.token_mask.mean_set:
            mean_embed = masked_model.token_mask.mean_activation
            x = x * mask_per_position + mean_embed * (1 - mask_per_position)
        else:
            x = x * mask_per_position
    
    if model.wpe is not None:
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + model.wpe(pos)
    
    x = model.drop(x)
    
    def apply_mask_and_capture(activation: torch.Tensor, layer_idx: int, location: str) -> torch.Tensor:
        """Apply mask (with AbsTopK + mean ablation) and capture the result."""
        # Set current layer for mask_fn
        masked_model._current_layer = layer_idx
        
        # Apply AbsTopK + node mask via mask_fn
        masked_activation = mask_fn(activation, location)
        
        # Capture the masked activation
        if location in config.mask_locations:
            key = f"layer{layer_idx}_{location}"
            activations[key] = masked_activation.clone()
        
        return masked_activation
    
    for layer_idx, block in enumerate(model.blocks):
        # Attention sublayer
        if frozen_ln_scales is not None:
            frozen_scale = frozen_ln_scales[f"layer{layer_idx}_ln_1"]
            normed = masked_model._apply_ln_with_frozen_scale(x, block.ln_1, frozen_scale)
        else:
            normed = block.ln_1(x)
        normed = apply_mask_and_capture(normed, layer_idx, "attn_in")
        
        # Q, K, V projections
        qkv = block.attn.c_attn(normed)
        q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
        
        q = apply_mask_and_capture(q, layer_idx, "attn_q")
        k = apply_mask_and_capture(k, layer_idx, "attn_k")
        v = apply_mask_and_capture(v, layer_idx, "attn_v")
        
        # Compute attention
        q = q.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        k = k.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        v = v.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        
        scale = 1.0 / (block.attn.d_head ** 0.5)
        
        if block.attn.use_flash and block.attn.attn_fn is not None:
            y = block.attn.attn_fn(
                q, k, v,
                dropout_p=block.attn.dropout if block.attn.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        elif block.attn.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=block.attn.dropout if block.attn.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(
                block.attn.causal_mask[:, :, :T, :T] == 0,
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = block.attn.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, block.attn.n_heads * block.attn.d_head)
        attn_out = block.attn.c_proj(y)
        attn_out = block.attn.resid_dropout(attn_out)
        
        attn_out = apply_mask_and_capture(attn_out, layer_idx, "attn_out")
        
        x = x + attn_out
        
        # MLP sublayer
        if frozen_ln_scales is not None:
            frozen_scale = frozen_ln_scales[f"layer{layer_idx}_ln_2"]
            normed = masked_model._apply_ln_with_frozen_scale(x, block.ln_2, frozen_scale)
        else:
            normed = block.ln_2(x)
        normed = apply_mask_and_capture(normed, layer_idx, "mlp_in")
        
        mlp_hidden = block.mlp.c_fc(normed)
        mlp_hidden = block.mlp.act_fn(mlp_hidden)
        mlp_hidden = apply_mask_and_capture(mlp_hidden, layer_idx, "mlp_neuron")
        
        mlp_out = block.mlp.c_proj(mlp_hidden)
        mlp_out = block.mlp.dropout(mlp_out)
        mlp_out = apply_mask_and_capture(mlp_out, layer_idx, "mlp_out")
        
        x = x + mlp_out
    
    # Final layer norm and LM head
    if frozen_ln_scales is not None:
        frozen_scale = frozen_ln_scales["ln_f"]
        x = masked_model._apply_ln_with_frozen_scale(x, model.ln_f, frozen_scale)
    else:
        x = model.ln_f(x)
    logits = model.lm_head(x)
    
    if hasattr(model, 'bigram_table') and model.bigram_table is not None:
        bigram_logits = F.embedding(input_ids, model.bigram_table)
        logits = logits + bigram_logits
    
    return logits, activations


def run_pruned_forward_with_interchange(
    masked_model: MaskedSparseGPT,
    input_ids: torch.Tensor,
    unpruned_activations: Dict[str, torch.Tensor],
    interchange_mask: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Run forward pass on the pruned model, replacing some activations with unpruned values.
    
    Args:
        masked_model: The masked model
        input_ids: Input token IDs
        unpruned_activations: Dict of activations from the unpruned model
        interchange_mask: Dict mapping mask key to boolean tensor indicating which
                         active nodes should be interchanged (True = use unpruned)
                         
    Returns:
        Logits from the pruned model with interchanged activations
    """
    device = input_ids.device
    B, T = input_ids.shape
    model = masked_model.model
    config = masked_model.config
    
    # Get mask function which properly applies AbsTopK + node masks
    mask_fn = masked_model._get_masking_fn()
    
    # Compute frozen LN scales if enabled
    frozen_ln_scales = None
    if config.freeze_layernorm_scale:
        frozen_ln_scales = masked_model._compute_frozen_ln_scales(input_ids)
    
    # Forward through model with interventions
    x = model.wte(input_ids)
    
    if torch.is_autocast_enabled('cuda'):
        x = x.to(torch.get_autocast_dtype('cuda'))
    
    # Apply token mask if enabled
    if masked_model.token_mask is not None:
        token_binary_mask = masked_model.token_mask.get_binary_mask()
        mask_per_position = token_binary_mask[input_ids].unsqueeze(-1)
        
        if masked_model.token_mask.mean_set:
            mean_embed = masked_model.token_mask.mean_activation
            x = x * mask_per_position + mean_embed * (1 - mask_per_position)
        else:
            x = x * mask_per_position
    
    if model.wpe is not None:
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + model.wpe(pos)
    
    x = model.drop(x)
    
    def apply_interchange(activation: torch.Tensor, key: str, location: str) -> torch.Tensor:
        """Apply mask (with AbsTopK) and potentially interchange with unpruned activation."""
        # Get layer index from key
        layer_idx = int(key.split("_")[0].replace("layer", ""))
        
        # Set current layer for mask_fn
        masked_model._current_layer = layer_idx
        
        # Apply AbsTopK + node mask via mask_fn
        activation = mask_fn(activation, location)
        
        # Now apply interchange for selected active nodes
        if key in interchange_mask and key in unpruned_activations:
            ic_mask = interchange_mask[key]  # Boolean tensor, same shape as active nodes
            unpruned_act = unpruned_activations[key]
            
            # ic_mask indicates which nodes to replace with unpruned values
            # We only consider active nodes (where node_mask == 1)
            # For inactive nodes, we keep the mean-ablated value
            
            # Expand ic_mask to match activation shape
            # activation: (batch, seq, num_nodes)
            # ic_mask: (num_nodes,) - True where we should use unpruned
            activation = torch.where(
                ic_mask.unsqueeze(0).unsqueeze(0),  # (1, 1, num_nodes)
                unpruned_act,
                activation
            )
        
        return activation
    
    for layer_idx, block in enumerate(model.blocks):
        # Attention sublayer
        if frozen_ln_scales is not None:
            frozen_scale = frozen_ln_scales[f"layer{layer_idx}_ln_1"]
            normed = masked_model._apply_ln_with_frozen_scale(x, block.ln_1, frozen_scale)
        else:
            normed = block.ln_1(x)
        key = f"layer{layer_idx}_attn_in"
        normed = apply_interchange(normed, key, "attn_in")
        
        # Q, K, V projections
        qkv = block.attn.c_attn(normed)
        q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
        
        q = apply_interchange(q, f"layer{layer_idx}_attn_q", "attn_q")
        k = apply_interchange(k, f"layer{layer_idx}_attn_k", "attn_k")
        v = apply_interchange(v, f"layer{layer_idx}_attn_v", "attn_v")
        
        # Compute attention
        q = q.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        k = k.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        v = v.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        
        scale = 1.0 / (block.attn.d_head ** 0.5)
        
        if block.attn.use_flash and block.attn.attn_fn is not None:
            y = block.attn.attn_fn(
                q, k, v,
                dropout_p=block.attn.dropout if block.attn.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        elif block.attn.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=block.attn.dropout if block.attn.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(
                block.attn.causal_mask[:, :, :T, :T] == 0,
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = block.attn.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, block.attn.n_heads * block.attn.d_head)
        attn_out = block.attn.c_proj(y)
        attn_out = block.attn.resid_dropout(attn_out)
        
        attn_out = apply_interchange(attn_out, f"layer{layer_idx}_attn_out", "attn_out")
        
        x = x + attn_out
        
        # MLP sublayer
        if frozen_ln_scales is not None:
            frozen_scale = frozen_ln_scales[f"layer{layer_idx}_ln_2"]
            normed = masked_model._apply_ln_with_frozen_scale(x, block.ln_2, frozen_scale)
        else:
            normed = block.ln_2(x)
        normed = apply_interchange(normed, f"layer{layer_idx}_mlp_in", "mlp_in")
        
        mlp_hidden = block.mlp.c_fc(normed)
        mlp_hidden = block.mlp.act_fn(mlp_hidden)
        mlp_hidden = apply_interchange(mlp_hidden, f"layer{layer_idx}_mlp_neuron", "mlp_neuron")
        
        mlp_out = block.mlp.c_proj(mlp_hidden)
        mlp_out = block.mlp.dropout(mlp_out)
        mlp_out = apply_interchange(mlp_out, f"layer{layer_idx}_mlp_out", "mlp_out")
        
        x = x + mlp_out
    
    # Final layer norm and LM head
    if frozen_ln_scales is not None:
        frozen_scale = frozen_ln_scales["ln_f"]
        x = masked_model._apply_ln_with_frozen_scale(x, model.ln_f, frozen_scale)
    else:
        x = model.ln_f(x)
    logits = model.lm_head(x)
    
    if hasattr(model, 'bigram_table') and model.bigram_table is not None:
        bigram_logits = F.embedding(input_ids, model.bigram_table)
        logits = logits + bigram_logits
    
    return logits


def compute_task_loss_from_logits(
    logits: torch.Tensor,
    correct_tokens: torch.Tensor,
    eval_positions: torch.Tensor,
    incorrect_tokens: torch.Tensor = None,
    use_binary_loss: bool = False,
) -> float:
    """
    Compute task cross-entropy loss from logits.
    
    Args:
        logits: (batch, seq, vocab) model output logits
        correct_tokens: (batch,) correct token ids
        eval_positions: (batch,) positions to evaluate
        incorrect_tokens: (batch,) incorrect token ids (needed for binary loss)
        use_binary_loss: If True, compute CE over only [correct, incorrect] logits
    
    Returns:
        Average loss (float)
    """
    batch_size = logits.shape[0]
    batch_indices = torch.arange(batch_size, device=logits.device)
    final_logits = logits[batch_indices, eval_positions, :]
    
    if use_binary_loss and incorrect_tokens is not None:
        # Binary CE: softmax over only [correct, incorrect] logits
        correct_logits = final_logits.gather(1, correct_tokens.unsqueeze(1)).squeeze(1)
        incorrect_logits = final_logits.gather(1, incorrect_tokens.unsqueeze(1)).squeeze(1)
        binary_logits = torch.stack([correct_logits, incorrect_logits], dim=1)
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(binary_logits, targets)
    else:
        # Full vocabulary CE
        loss = F.cross_entropy(final_logits, correct_tokens)
    
    return loss.item()


def create_interchange_mask(
    active_indices: Dict[str, torch.Tensor],
    fraction: float,
    layer_filter: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Create a random interchange mask for the given fraction of active nodes.
    
    Args:
        active_indices: Dict mapping mask key to tensor of active node indices
        fraction: Fraction of active nodes to interchange (0-1)
        layer_filter: If provided, only create masks for this layer
        rng: Random number generator
        device: Device for tensors
        
    Returns:
        Dict mapping mask key to boolean tensor (True = interchange this node)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    interchange_masks = {}
    
    for key, indices in active_indices.items():
        # Check layer filter
        if layer_filter is not None:
            layer_idx = int(key.split("_")[0].replace("layer", ""))
            if layer_idx != layer_filter:
                # Don't interchange nodes from other layers
                continue
        
        num_active = len(indices)
        if num_active == 0:
            continue
        
        # Determine how many nodes to interchange
        num_to_interchange = int(round(fraction * num_active))
        
        # Create mask: False for all nodes, then set True for selected ones
        # We need to figure out the full dimension for this mask
        # The mask should be the same size as the full node dimension
        # Get the full dimension from the key
        if "attn_in" in key or "attn_out" in key or "mlp_in" in key or "mlp_out" in key:
            # These use d_model - we need to infer it
            # For now, use max index + 1 as a proxy (will be corrected by the caller)
            full_dim = indices.max().item() + 1 if len(indices) > 0 else 0
        else:
            full_dim = indices.max().item() + 1 if len(indices) > 0 else 0
        
        # Create full mask (all False initially)
        mask = torch.zeros(full_dim, dtype=torch.bool, device=device)
        
        if num_to_interchange > 0:
            # Randomly select which active nodes to interchange
            perm = rng.permutation(num_active)
            selected_positions = perm[:num_to_interchange]
            selected_indices = indices[selected_positions]
            mask[selected_indices] = True
        
        interchange_masks[key] = mask
    
    return interchange_masks


def compute_fvu(
    pruned_activations: Dict[str, torch.Tensor],
    unpruned_activations: Dict[str, torch.Tensor],
    active_indices: Dict[str, torch.Tensor],
    non_padding_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute Fraction of Variance Unexplained (FVU) between pruned and unpruned activations.
    
    FVU is computed per-token then averaged:
        fvu_per_token = (src - tgt).pow(2).sum(-1) / (tgt - tgt.mean([0,1])).pow(2).sum(-1)
        fvu = fvu_per_token.mean()
    
    where src (pruned) and tgt (unpruned) have shape [batch, seq, d].
    
    Args:
        pruned_activations: Dict of pruned model activations
        unpruned_activations: Dict of unpruned model activations
        active_indices: Dict of active node indices per mask location
        non_padding_mask: Boolean tensor (batch, seq) indicating non-padding tokens
        
    Returns:
        Dict with 'global' and 'per_layer' FVU values
    """
    # Collect activations as [batch, seq, d] tensors
    # We'll concatenate along the last dimension (nodes)
    global_src_list = []
    global_tgt_list = []
    
    per_layer_src = {}  # layer_idx -> list of [batch, seq, d] tensors
    per_layer_tgt = {}
    
    for key in pruned_activations:
        if key not in unpruned_activations:
            continue
        if key not in active_indices:
            continue
        
        active_idx = active_indices[key]
        if len(active_idx) == 0:
            continue
        
        src = pruned_activations[key]  # (batch, seq, dim)
        tgt = unpruned_activations[key]
        
        # Select only active nodes
        src = src[..., active_idx]  # (batch, seq, num_active)
        tgt = tgt[..., active_idx]
        
        # Add to global
        global_src_list.append(src)
        global_tgt_list.append(tgt)
        
        # Parse layer
        parts = key.split("_", 1)
        layer_idx = int(parts[0].replace("layer", ""))
        
        # Add to per-layer
        if layer_idx not in per_layer_src:
            per_layer_src[layer_idx] = []
            per_layer_tgt[layer_idx] = []
        per_layer_src[layer_idx].append(src)
        per_layer_tgt[layer_idx].append(tgt)
    
    def compute_fvu_from_tensors(src_list, tgt_list, mask):
        """
        Compute FVU from lists of [batch, seq, d] tensors.
        
        fvu_per_token = (src - tgt).pow(2).sum(-1) / (tgt - tgt.mean([0,1])).pow(2).sum(-1)
        fvu = fvu_per_token.mean() (over non-padding tokens)
        """
        if not src_list:
            return float('nan')
        
        # Concatenate along node dimension: [batch, seq, total_nodes]
        src = torch.cat(src_list, dim=-1)
        tgt = torch.cat(tgt_list, dim=-1)
        
        # Compute tgt mean over batch and seq: [total_nodes]
        # Only use non-padding tokens for the mean
        mask_expanded = mask.unsqueeze(-1)  # [batch, seq, 1]
        num_valid = mask.sum()
        
        if num_valid == 0:
            return float('nan')
        
        # Compute mean of tgt over valid positions
        tgt_masked = tgt * mask_expanded
        tgt_mean = tgt_masked.sum(dim=[0, 1]) / num_valid  # [total_nodes]
        
        # Compute per-token FVU
        # Numerator: (src - tgt).pow(2).sum(-1) -> [batch, seq]
        numerator = (src - tgt).pow(2).sum(-1)
        
        # Denominator: (tgt - tgt_mean).pow(2).sum(-1) -> [batch, seq]
        denominator = (tgt - tgt_mean).pow(2).sum(-1)
        
        # Mask out padding tokens
        numerator = numerator * mask
        denominator = denominator * mask
        
        # Average over valid tokens
        # Avoid division by zero
        total_numer = numerator.sum()
        total_denom = denominator.sum()
        
        if total_denom < 1e-10:
            return float('nan')
        
        return (total_numer / total_denom).item()
    
    results = {
        'global': compute_fvu_from_tensors(global_src_list, global_tgt_list, non_padding_mask),
        'per_layer': {
            layer: compute_fvu_from_tensors(per_layer_src[layer], per_layer_tgt[layer], non_padding_mask)
            for layer in sorted(per_layer_src.keys())
        },
    }
    
    return results


def run_interchange_evaluation(
    masked_model: MaskedSparseGPT,
    base_model: nn.Module,
    task: BinaryTask,
    config: InterchangeEvalConfig,
    pruning_config: PruningConfig,
    show_progress: bool = True,
) -> InterchangeResult:
    """
    Run the full interchange intervention evaluation.
    
    Args:
        masked_model: The pruned/masked model
        base_model: The base (unpruned) model
        task: Binary task for evaluation
        config: Interchange evaluation config
        pruning_config: Pruning config (for mask locations)
        show_progress: Whether to show progress bars
        
    Returns:
        InterchangeResult with all evaluation data
    """
    device = config.device
    masked_model.eval()
    base_model.eval()
    
    # Get active node indices
    active_indices = get_active_node_indices(masked_model)
    
    # Count nodes
    num_active = sum(len(v) for v in active_indices.values())
    total_nodes = masked_model.masks.get_total_nodes()
    n_layers = masked_model.n_layers
    
    # Get pad token id
    pad_id = task.tokenizer.pad_token_id or 0
    
    # Results containers
    per_layer_results = {i: {} for i in range(n_layers)}
    all_layers_results = {}
    
    # Random generator for reproducibility
    rng = np.random.default_rng(42)
    
    # Pre-generate fixed batches for consistent evaluation
    print("Pre-generating evaluation batches...")
    fixed_batches = []
    for _ in range(config.num_batches):
        batch = task.generate_batch(
            batch_size=config.batch_size,
            max_length=config.seq_length,
        )
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
        fixed_batches.append({
            "positive_ids": positive_ids.to(device),
            "correct_tokens": correct_tokens.to(device),
            "eval_positions": eval_positions.to(device),
        })
    
    # Compute unpruned clean loss (baseline)
    print("Computing unpruned clean loss...")
    clean_losses = []
    with torch.no_grad():
        for batch in fixed_batches:
            input_ids = batch["positive_ids"]
            correct_tokens = batch["correct_tokens"]
            eval_positions = batch["eval_positions"]
            
            # Run base model (no masking, no ablation)
            # Use run_forward_with_activations to get logits
            logits, _ = run_forward_with_activations(
                base_model, input_ids, masked_model, pruning_config
            )
            loss = compute_task_loss_from_logits(logits, correct_tokens, eval_positions)
            clean_losses.append(loss)
    unpruned_clean_loss = np.mean(clean_losses)
    print(f"  Unpruned clean loss: {unpruned_clean_loss:.4f}")
    
    # Also store pruned activations for FVU computation
    fvu_pruned_activations = {}
    fvu_unpruned_activations = {}
    fvu_non_padding_masks = []
    
    # First: collect activations for FVU (at 0% interchange)
    print("Collecting activations for FVU computation...")
    with torch.no_grad():
        for batch in tqdm(fixed_batches[:min(3, len(fixed_batches))], desc="FVU batches", disable=not show_progress):
            input_ids = batch["positive_ids"]
            
            # Get non-padding mask
            non_padding = (input_ids != pad_id)
            fvu_non_padding_masks.append(non_padding)
            
            # Run unpruned model (base model without masking)
            _, unpruned_acts = run_forward_with_activations(
                base_model, input_ids, masked_model, pruning_config
            )
            
            # Run masked model and capture activations AFTER masking is applied
            _, pruned_acts = run_masked_forward_with_activations(
                masked_model, input_ids
            )
            
            # Accumulate for FVU
            for key in unpruned_acts:
                if key not in fvu_unpruned_activations:
                    fvu_unpruned_activations[key] = []
                fvu_unpruned_activations[key].append(unpruned_acts[key].detach())
            for key in pruned_acts:
                if key not in fvu_pruned_activations:
                    fvu_pruned_activations[key] = []
                fvu_pruned_activations[key].append(pruned_acts[key].detach())
    
    # Concatenate FVU data
    for key in fvu_unpruned_activations:
        fvu_unpruned_activations[key] = torch.cat(fvu_unpruned_activations[key], dim=0)
    for key in fvu_pruned_activations:
        fvu_pruned_activations[key] = torch.cat(fvu_pruned_activations[key], dim=0)
    fvu_non_padding = torch.cat(fvu_non_padding_masks, dim=0)
    
    # Compute FVU
    print("Computing FVU...")
    fvu_results = compute_fvu(
        fvu_pruned_activations,
        fvu_unpruned_activations,
        active_indices,
        fvu_non_padding,
    )
    
    # Clear FVU data to save memory
    del fvu_pruned_activations, fvu_unpruned_activations
    torch.cuda.empty_cache()
    
    # Evaluate per-layer interchange
    print("Running per-layer interchange evaluation...")
    for layer_idx in tqdm(range(n_layers), desc="Layers", disable=not show_progress):
        for frac in config.fractions:
            losses = []
            for trial in range(config.num_trials):
                trial_losses = []
                
                with torch.no_grad():
                    for batch in fixed_batches:
                        input_ids = batch["positive_ids"]
                        correct_tokens = batch["correct_tokens"]
                        eval_positions = batch["eval_positions"]
                        
                        # Get unpruned activations
                        _, unpruned_acts = run_forward_with_activations(
                            base_model, input_ids, masked_model, pruning_config
                        )
                        
                        # Create interchange mask for this layer only
                        interchange_mask = create_interchange_mask(
                            active_indices, frac, 
                            layer_filter=layer_idx,
                            rng=rng, device=device
                        )
                        
                        # Get the full dimensions for each mask
                        for key in interchange_mask:
                            if key in unpruned_acts:
                                full_dim = unpruned_acts[key].shape[-1]
                                if len(interchange_mask[key]) < full_dim:
                                    # Extend with False values
                                    new_mask = torch.zeros(full_dim, dtype=torch.bool, device=device)
                                    new_mask[:len(interchange_mask[key])] = interchange_mask[key]
                                    interchange_mask[key] = new_mask
                        
                        # Run pruned forward with interchange
                        logits = run_pruned_forward_with_interchange(
                            masked_model, input_ids, unpruned_acts, interchange_mask
                        )
                        
                        loss = compute_task_loss_from_logits(logits, correct_tokens, eval_positions)
                        trial_losses.append(loss)
                
                losses.append(np.mean(trial_losses))
            
            per_layer_results[layer_idx][frac] = (np.mean(losses), np.std(losses))
    
    # Evaluate all-layers interchange
    print("Running all-layers interchange evaluation...")
    for frac in tqdm(config.fractions, desc="Fractions", disable=not show_progress):
        losses = []
        for trial in range(config.num_trials):
            trial_losses = []
            
            with torch.no_grad():
                for batch in fixed_batches:
                    input_ids = batch["positive_ids"]
                    correct_tokens = batch["correct_tokens"]
                    eval_positions = batch["eval_positions"]
                    
                    # Get unpruned activations
                    _, unpruned_acts = run_forward_with_activations(
                        base_model, input_ids, masked_model, pruning_config
                    )
                    
                    # Create interchange mask for all layers
                    interchange_mask = create_interchange_mask(
                        active_indices, frac,
                        layer_filter=None,  # All layers
                        rng=rng, device=device
                    )
                    
                    # Extend masks to full dimensions
                    for key in interchange_mask:
                        if key in unpruned_acts:
                            full_dim = unpruned_acts[key].shape[-1]
                            if len(interchange_mask[key]) < full_dim:
                                new_mask = torch.zeros(full_dim, dtype=torch.bool, device=device)
                                new_mask[:len(interchange_mask[key])] = interchange_mask[key]
                                interchange_mask[key] = new_mask
                    
                    # Run pruned forward with interchange
                    logits = run_pruned_forward_with_interchange(
                        masked_model, input_ids, unpruned_acts, interchange_mask
                    )
                    
                    loss = compute_task_loss_from_logits(logits, correct_tokens, eval_positions)
                    trial_losses.append(loss)
            
            losses.append(np.mean(trial_losses))
        
        all_layers_results[frac] = (np.mean(losses), np.std(losses))
    
    return InterchangeResult(
        per_layer_results=per_layer_results,
        all_layers_results=all_layers_results,
        unpruned_clean_loss=unpruned_clean_loss,
        fvu_global=fvu_results['global'],
        fvu_per_layer=fvu_results['per_layer'],
        num_active_nodes=num_active,
        total_nodes=total_nodes,
        fractions=config.fractions,
        num_trials=config.num_trials,
    )


def plot_interchange_results(
    result: InterchangeResult,
    output_path: Path,
    title_prefix: str = "",
):
    """
    Create plots for interchange intervention results.
    
    Creates:
    1. Per-layer subplot grid with fraction vs loss curves
    2. All-layers combined plot
    3. FVU bar chart by layer
    
    Args:
        result: InterchangeResult from run_interchange_evaluation
        output_path: Path to save the plot (without extension)
        title_prefix: Prefix for plot titles (e.g., "model_name (zero_ablate)\n")
    """
    n_layers = len(result.per_layer_results)
    fractions = result.fractions
    
    # Calculate grid dimensions for per-layer subplots
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    # Create per-layer plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx in range(n_layers):
        row = layer_idx // n_cols
        col = layer_idx % n_cols
        ax = axes[row, col]
        
        layer_data = result.per_layer_results[layer_idx]
        xs = sorted(layer_data.keys())
        means = [layer_data[x][0] for x in xs]
        stds = [layer_data[x][1] for x in xs]
        
        ax.plot(xs, means, 'b-o', linewidth=2, markersize=6)
        ax.fill_between(xs, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='blue')
        
        # Add horizontal line for unpruned clean loss
        ax.axhline(y=result.unpruned_clean_loss, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel("Fraction Interchanged", fontsize=10)
        ax.set_ylabel("Task Loss", fontsize=10)
        ax.set_title(f"Layer {layer_idx}", fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
    
    # Hide empty subplots
    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle(f"{title_prefix}Per-Layer Interchange Intervention\n"
                 f"(Active: {result.num_active_nodes:,} / {result.total_nodes:,} nodes)", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_path}_per_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create all-layers plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    xs = sorted(result.all_layers_results.keys())
    means = [result.all_layers_results[x][0] for x in xs]
    stds = [result.all_layers_results[x][1] for x in xs]
    
    ax.plot(xs, means, 'b-o', linewidth=2, markersize=8, label='Pruned model (with interchange)')
    ax.fill_between(xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.3, color='blue')
    
    # Add horizontal line for unpruned clean loss
    ax.axhline(y=result.unpruned_clean_loss, color='green', linestyle='--', 
               linewidth=2, label=f'Unpruned model: {result.unpruned_clean_loss:.4f}')
    
    ax.set_xlabel("Fraction of Active Nodes Interchanged", fontsize=12)
    ax.set_ylabel("Task Loss", fontsize=12)
    ax.set_title(f"{title_prefix}All-Layers Interchange Intervention\n"
                 f"(Active: {result.num_active_nodes:,} / {result.total_nodes:,} nodes)", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_all_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create FVU plot (per-layer)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    layers = sorted(result.fvu_per_layer.keys())
    fvus = [result.fvu_per_layer[l] for l in layers]
    bars = ax.bar(layers, fvus, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.axhline(y=result.fvu_global, color='red', linestyle='--', linewidth=2, 
               label=f'Global FVU: {result.fvu_global:.4f}')
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("FVU", fontsize=12)
    ax.set_title(f"{title_prefix}Fraction of Variance Unexplained (Pruned vs Unpruned)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_fvu.png", dpi=150, bbox_inches='tight')
    plt.close()


def save_interchange_results(
    result: InterchangeResult,
    output_path: Path,
):
    """Save interchange results to JSON."""
    data = {
        "per_layer_results": {
            str(layer): {str(frac): {"mean": mean, "std": std} 
                        for frac, (mean, std) in fracs.items()}
            for layer, fracs in result.per_layer_results.items()
        },
        "all_layers_results": {
            str(frac): {"mean": mean, "std": std}
            for frac, (mean, std) in result.all_layers_results.items()
        },
        "baselines": {
            "unpruned_clean_loss": result.unpruned_clean_loss,
        },
        "fvu": {
            "global": result.fvu_global,
            "per_layer": {str(k): v for k, v in result.fvu_per_layer.items()},
        },
        "metadata": {
            "num_active_nodes": result.num_active_nodes,
            "total_nodes": result.total_nodes,
            "fractions": result.fractions,
            "num_trials": result.num_trials,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def run_and_save_interchange_eval(
    masked_model: MaskedSparseGPT,
    base_model: nn.Module,
    task: BinaryTask,
    output_dir: Path,
    config: Optional[InterchangeEvalConfig] = None,
    pruning_config: Optional[PruningConfig] = None,
    title_prefix: str = "",
    show_progress: bool = True,
) -> InterchangeResult:
    """
    Convenience function to run interchange evaluation and save all outputs.
    
    Args:
        masked_model: The pruned/masked model
        base_model: The base (unpruned) model  
        task: Binary task for evaluation
        output_dir: Directory to save outputs (plots and JSON)
        config: Interchange evaluation config (uses defaults if None)
        pruning_config: Pruning config (uses masked_model.config if None)
        title_prefix: Prefix for plot titles
        show_progress: Whether to show progress bars
        
    Returns:
        InterchangeResult
    """
    if config is None:
        config = InterchangeEvalConfig()
    if pruning_config is None:
        pruning_config = masked_model.config
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Running Interchange Intervention Evaluation")
    print(f"{'='*60}")
    print(f"Fractions: {config.fractions}")
    print(f"Trials per fraction: {config.num_trials}")
    print(f"Batches: {config.num_batches}")
    print(f"Output dir: {output_dir}")
    print()
    
    result = run_interchange_evaluation(
        masked_model=masked_model,
        base_model=base_model,
        task=task,
        config=config,
        pruning_config=pruning_config,
        show_progress=show_progress,
    )
    
    # Save plots
    plot_path = output_dir / "interchange"
    plot_interchange_results(result, plot_path, title_prefix=title_prefix)
    print(f"Plots saved to {output_dir}/interchange_*.png")
    
    # Save JSON
    json_path = output_dir / "interchange_results.json"
    save_interchange_results(result, json_path)
    print(f"Results saved to {json_path}")
    
    # Print summary
    print(f"\nInterchange Evaluation Summary:")
    print(f"  Active nodes: {result.num_active_nodes:,} / {result.total_nodes:,}")
    print(f"  Global FVU: {result.fvu_global:.4f}")
    
    return result


# ============================================================================
# Ablation Sweep Evaluation
# ============================================================================

@dataclass
class AblationSweepConfig:
    """Configuration for ablation sweep evaluation on unpruned model."""
    # Number of node counts to evaluate
    num_points: int = 11  # e.g., 0, 10%, 20%, ..., 100% of circuit size
    
    # Number of random trials per point
    num_trials: int = 10
    
    # Evaluation settings
    num_batches: int = 10
    batch_size: int = 64
    seq_length: int = 0  # 0 = dynamic padding
    
    # Device
    device: str = "cuda"


@dataclass
class AblationSweepResult:
    """Results from ablation sweep evaluation."""
    # Baseline losses
    clean_loss: float  # Unpruned model, no ablation
    circuit_ablated_loss: float  # Unpruned model with non-circuit nodes ablated
    
    # Sweep results: num_nodes -> (mean_loss, std_loss)
    random_all_results: Dict[int, Tuple[float, float]]  # Random from all nodes
    random_circuit_results: Dict[int, Tuple[float, float]]  # Random from circuit nodes
    
    # Metadata
    circuit_size: int  # Number of active nodes in circuit
    total_nodes: int
    num_trials: int


def run_unpruned_forward_with_ablation(
    model: nn.Module,
    input_ids: torch.Tensor,
    ablation_mask: Dict[str, torch.Tensor],
    mean_cache: Optional[Dict[str, torch.Tensor]] = None,
    mask_locations: List[str] = None,
) -> torch.Tensor:
    """
    Run forward pass on unpruned model with specified nodes ablated (zeroed).
    
    Args:
        model: The base (unpruned) model
        input_ids: Input token IDs (batch, seq)
        ablation_mask: Dict mapping mask keys to boolean tensors indicating which nodes to ablate
        mean_cache: Optional dict of mean activations for mean ablation (if None, uses zero ablation)
        mask_locations: List of mask locations to apply ablation at
        
    Returns:
        Logits tensor
    """
    device = input_ids.device
    B, T = input_ids.shape
    
    if mask_locations is None:
        mask_locations = ["mlp_neuron"]
    
    # Token embeddings
    x = model.wte(input_ids)
    
    if torch.is_autocast_enabled('cuda'):
        x = x.to(torch.get_autocast_dtype('cuda'))
    
    if model.wpe is not None:
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + model.wpe(pos)
    
    x = model.drop(x)
    
    for layer_idx, block in enumerate(model.blocks):
        # attn_in
        normed = block.ln_1(x)
        key = f"layer{layer_idx}_attn_in"
        if "attn_in" in mask_locations and key in ablation_mask:
            mask = ablation_mask[key]
            if mean_cache and key in mean_cache:
                mean_val = mean_cache[key].to(normed.dtype)
                normed = torch.where(mask.unsqueeze(0).unsqueeze(0), mean_val, normed)
            else:
                normed = normed * (~mask).float().unsqueeze(0).unsqueeze(0)
        
        # Q, K, V projections
        qkv = block.attn.c_attn(normed)
        q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
        
        # Apply ablation to Q, K, V
        for name, tensor in [("attn_q", q), ("attn_k", k), ("attn_v", v)]:
            key = f"layer{layer_idx}_{name}"
            if name in mask_locations and key in ablation_mask:
                mask = ablation_mask[key]
                if mean_cache and key in mean_cache:
                    mean_val = mean_cache[key].to(tensor.dtype)
                    if name == "attn_q":
                        q = torch.where(mask.unsqueeze(0).unsqueeze(0), mean_val, tensor)
                    elif name == "attn_k":
                        k = torch.where(mask.unsqueeze(0).unsqueeze(0), mean_val, tensor)
                    else:
                        v = torch.where(mask.unsqueeze(0).unsqueeze(0), mean_val, tensor)
                else:
                    ablate_mult = (~mask).float().unsqueeze(0).unsqueeze(0)
                    if name == "attn_q":
                        q = tensor * ablate_mult
                    elif name == "attn_k":
                        k = tensor * ablate_mult
                    else:
                        v = tensor * ablate_mult
        
        # Reshape for attention
        q = q.view(B, T, model.config.n_heads, model.config.d_head).transpose(1, 2)
        k = k.view(B, T, model.config.n_heads, model.config.d_head).transpose(1, 2)
        v = v.view(B, T, model.config.n_heads, model.config.d_head).transpose(1, 2)
        
        scale = 1.0 / (model.config.d_head ** 0.5)
        
        # Compute attention
        if block.attn.use_flash and block.attn.attn_fn is not None:
            attn_output = block.attn.attn_fn(
                q, k, v,
                dropout_p=block.attn.dropout if model.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        elif block.attn.use_flash:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=block.attn.dropout if model.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(
                block.attn.causal_mask[:, :, :T, :T] == 0,
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = block.attn.attn_dropout(att)
            attn_output = att @ v
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, model.config.n_heads * model.config.d_head)
        
        # Output projection
        attn_output = block.attn.c_proj(attn_output)
        attn_output = block.attn.resid_dropout(attn_output)
        
        # attn_out ablation
        key = f"layer{layer_idx}_attn_out"
        if "attn_out" in mask_locations and key in ablation_mask:
            mask = ablation_mask[key]
            if mean_cache and key in mean_cache:
                mean_val = mean_cache[key].to(attn_output.dtype)
                attn_output = torch.where(mask.unsqueeze(0).unsqueeze(0), mean_val, attn_output)
            else:
                attn_output = attn_output * (~mask).float().unsqueeze(0).unsqueeze(0)
        
        x = x + attn_output
        
        # mlp_in
        normed = block.ln_2(x)
        key = f"layer{layer_idx}_mlp_in"
        if "mlp_in" in mask_locations and key in ablation_mask:
            mask = ablation_mask[key]
            if mean_cache and key in mean_cache:
                mean_val = mean_cache[key].to(normed.dtype)
                normed = torch.where(mask.unsqueeze(0).unsqueeze(0), mean_val, normed)
            else:
                normed = normed * (~mask).float().unsqueeze(0).unsqueeze(0)
        
        # MLP forward
        mlp_hidden = block.mlp.c_fc(normed)
        mlp_hidden = block.mlp.act_fn(mlp_hidden)
        
        # mlp_neuron ablation
        key = f"layer{layer_idx}_mlp_neuron"
        if "mlp_neuron" in mask_locations and key in ablation_mask:
            mask = ablation_mask[key]
            if mean_cache and key in mean_cache:
                mean_val = mean_cache[key].to(mlp_hidden.dtype)
                mlp_hidden = torch.where(mask.unsqueeze(0).unsqueeze(0), mean_val, mlp_hidden)
            else:
                mlp_hidden = mlp_hidden * (~mask).float().unsqueeze(0).unsqueeze(0)
        
        mlp_out = block.mlp.c_proj(mlp_hidden)
        mlp_out = block.mlp.dropout(mlp_out)
        
        # mlp_out ablation
        key = f"layer{layer_idx}_mlp_out"
        if "mlp_out" in mask_locations and key in ablation_mask:
            mask = ablation_mask[key]
            if mean_cache and key in mean_cache:
                mean_val = mean_cache[key].to(mlp_out.dtype)
                mlp_out = torch.where(mask.unsqueeze(0).unsqueeze(0), mean_val, mlp_out)
            else:
                mlp_out = mlp_out * (~mask).float().unsqueeze(0).unsqueeze(0)
        
        x = x + mlp_out
    
    # Final layer norm and LM head
    x = model.ln_f(x)
    logits = model.lm_head(x)
    
    if hasattr(model, 'bigram_table') and model.bigram_table is not None:
        bigram_logits = F.embedding(input_ids, model.bigram_table)
        logits = logits + bigram_logits
    
    return logits


def create_ablation_mask_for_non_circuit(
    masked_model: MaskedSparseGPT,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Create ablation mask that ablates all nodes NOT in the circuit.
    (i.e., nodes where mask tau < 0 in the pruned model)
    
    Returns:
        Dict mapping mask keys to boolean tensors (True = ablate this node)
    """
    ablation_mask = {}
    for key, mask in masked_model.masks.masks.items():
        # Ablate nodes where tau < 0 (not active in circuit)
        ablation_mask[key] = (mask.tau < 0).to(device)
    return ablation_mask


def create_random_ablation_mask(
    node_counts: Dict[str, int],
    num_to_ablate: int,
    from_circuit_only: bool,
    circuit_indices: Dict[str, torch.Tensor],
    rng: np.random.Generator,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Create ablation mask by randomly selecting nodes to ablate.
    
    Args:
        node_counts: Dict mapping mask keys to total node count
        num_to_ablate: Number of nodes to ablate
        from_circuit_only: If True, only select from circuit nodes; else from all nodes
        circuit_indices: Dict mapping mask keys to active circuit indices
        rng: Random number generator
        device: Device for tensors
        
    Returns:
        Dict mapping mask keys to boolean tensors (True = ablate this node)
    """
    ablation_mask = {}
    
    if from_circuit_only:
        # Build list of all (key, idx) pairs for circuit nodes
        all_circuit_nodes = []
        for key, indices in circuit_indices.items():
            for idx in indices.tolist():
                all_circuit_nodes.append((key, idx))
        
        if len(all_circuit_nodes) == 0:
            # No circuit nodes, return empty masks
            for key, count in node_counts.items():
                ablation_mask[key] = torch.zeros(count, dtype=torch.bool, device=device)
            return ablation_mask
        
        # Randomly select nodes to ablate
        num_to_ablate = min(num_to_ablate, len(all_circuit_nodes))
        selected_indices = rng.choice(len(all_circuit_nodes), size=num_to_ablate, replace=False)
        
        # Initialize masks as all False
        for key, count in node_counts.items():
            ablation_mask[key] = torch.zeros(count, dtype=torch.bool, device=device)
        
        # Set selected nodes to True (ablate)
        for idx in selected_indices:
            key, node_idx = all_circuit_nodes[idx]
            ablation_mask[key][node_idx] = True
    
    else:
        # Select from all nodes
        all_nodes = []
        for key, count in node_counts.items():
            for idx in range(count):
                all_nodes.append((key, idx))
        
        # Randomly select nodes to ablate
        num_to_ablate = min(num_to_ablate, len(all_nodes))
        selected_indices = rng.choice(len(all_nodes), size=num_to_ablate, replace=False)
        
        # Initialize masks as all False
        for key, count in node_counts.items():
            ablation_mask[key] = torch.zeros(count, dtype=torch.bool, device=device)
        
        # Set selected nodes to True (ablate)
        for idx in selected_indices:
            key, node_idx = all_nodes[idx]
            ablation_mask[key][node_idx] = True
    
    return ablation_mask


def run_ablation_sweep_evaluation(
    masked_model: MaskedSparseGPT,
    base_model: nn.Module,
    task: BinaryTask,
    config: AblationSweepConfig,
    pruning_config: PruningConfig,
    mean_cache: Optional[Dict[str, torch.Tensor]] = None,
    show_progress: bool = True,
) -> AblationSweepResult:
    """
    Run ablation sweep evaluation on the unpruned model.
    
    Compares:
    1. Clean unpruned model
    2. Unpruned model with non-circuit nodes ablated
    3. Sweep: ablating random nodes from all vs from circuit only
    
    Args:
        masked_model: The pruned/masked model (for circuit definition)
        base_model: The base (unpruned) model to ablate
        task: Binary task for evaluation
        config: Ablation sweep config
        pruning_config: Pruning config
        mean_cache: Optional mean activations for mean ablation
        show_progress: Whether to show progress bars
        
    Returns:
        AblationSweepResult
    """
    device = config.device
    rng = np.random.default_rng(42)
    
    # Get circuit info
    circuit_indices = get_active_node_indices(masked_model)
    circuit_size = sum(len(v) for v in circuit_indices.values())
    
    # Get node counts per location
    node_counts = {}
    for key, mask in masked_model.masks.masks.items():
        node_counts[key] = mask.tau.shape[0]
    total_nodes = sum(node_counts.values())
    
    print(f"Circuit size: {circuit_size:,} / {total_nodes:,} nodes")
    
    # Generate fixed batches
    print("Pre-generating evaluation batches...")
    fixed_batches = []
    for _ in range(config.num_batches):
        batch = task.generate_batch(
            batch_size=config.batch_size,
            max_length=config.seq_length,
        )
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
        fixed_batches.append({
            "positive_ids": positive_ids.to(device),
            "negative_ids": negative_ids.to(device),
            "correct_tokens": correct_tokens.to(device),
            "incorrect_tokens": incorrect_tokens.to(device),
            "eval_positions": eval_positions.to(device),
        })
    
    def compute_loss_with_ablation(ablation_mask):
        """Compute average loss with given ablation mask."""
        losses = []
        with torch.no_grad():
            for batch in fixed_batches:
                input_ids = batch["positive_ids"]
                correct_tokens = batch["correct_tokens"]
                eval_positions = batch["eval_positions"]
                
                logits = run_unpruned_forward_with_ablation(
                    base_model, input_ids, ablation_mask,
                    mean_cache=mean_cache,
                    mask_locations=pruning_config.mask_locations,
                )
                
                loss = compute_task_loss_from_logits(logits, correct_tokens, eval_positions)
                losses.append(loss)
        
        return np.mean(losses)
    
    # 1. Clean unpruned model (no ablation)
    print("Computing clean loss (no ablation)...")
    empty_mask = {key: torch.zeros(count, dtype=torch.bool, device=device) 
                  for key, count in node_counts.items()}
    clean_loss = compute_loss_with_ablation(empty_mask)
    print(f"  Clean loss: {clean_loss:.4f}")
    
    # 2. Ablate non-circuit nodes
    print("Computing loss with non-circuit nodes ablated...")
    non_circuit_mask = create_ablation_mask_for_non_circuit(masked_model, device)
    circuit_ablated_loss = compute_loss_with_ablation(non_circuit_mask)
    num_non_circuit = sum((mask).sum().item() for mask in non_circuit_mask.values())
    print(f"  Non-circuit ablated loss: {circuit_ablated_loss:.4f} ({num_non_circuit:,} nodes ablated)")
    
    # 3. Sweep over number of nodes to ablate
    # Use circuit_size as max, with num_points evenly spaced
    node_counts_to_test = [int(circuit_size * i / (config.num_points - 1)) 
                           for i in range(config.num_points)]
    # Ensure we have 0 and circuit_size exactly
    node_counts_to_test[0] = 0
    node_counts_to_test[-1] = circuit_size
    # Remove duplicates while preserving order
    node_counts_to_test = list(dict.fromkeys(node_counts_to_test))
    
    random_all_results = {}
    random_circuit_results = {}
    
    print(f"Running ablation sweep (circuit={len(node_counts_to_test)} points, random={len(node_counts_to_test)} points)...")
    
    for num_ablate in tqdm(node_counts_to_test, desc="Ablation sweep", disable=not show_progress):
        # Random from all nodes
        losses_all = []
        for trial in range(config.num_trials):
            ablation_mask = create_random_ablation_mask(
                node_counts, num_ablate, 
                from_circuit_only=False,
                circuit_indices=circuit_indices,
                rng=rng, device=device
            )
            loss = compute_loss_with_ablation(ablation_mask)
            losses_all.append(loss)
        random_all_results[num_ablate] = (np.mean(losses_all), np.std(losses_all))
        
        # Random from circuit nodes only
        losses_circuit = []
        for trial in range(config.num_trials):
            ablation_mask = create_random_ablation_mask(
                node_counts, num_ablate,
                from_circuit_only=True,
                circuit_indices=circuit_indices,
                rng=rng, device=device
            )
            loss = compute_loss_with_ablation(ablation_mask)
            losses_circuit.append(loss)
        random_circuit_results[num_ablate] = (np.mean(losses_circuit), np.std(losses_circuit))
    
    return AblationSweepResult(
        clean_loss=clean_loss,
        circuit_ablated_loss=circuit_ablated_loss,
        random_all_results=random_all_results,
        random_circuit_results=random_circuit_results,
        circuit_size=circuit_size,
        total_nodes=total_nodes,
        num_trials=config.num_trials,
    )


def plot_ablation_sweep_results(
    result: AblationSweepResult,
    output_path: Path,
    title_prefix: str = "",
):
    """
    Create plot for ablation sweep results.
    
    X-axis: Number of nodes ablated
    Y-axis: Task loss
    Two lines: random from all nodes vs random from circuit nodes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Random from all nodes
    xs_all = sorted(result.random_all_results.keys())
    means_all = [result.random_all_results[x][0] for x in xs_all]
    stds_all = [result.random_all_results[x][1] for x in xs_all]
    
    ax.plot(xs_all, means_all, 'b-o', linewidth=2, markersize=6, label='Random from all nodes')
    ax.fill_between(xs_all,
                    [m - s for m, s in zip(means_all, stds_all)],
                    [m + s for m, s in zip(means_all, stds_all)],
                    alpha=0.2, color='blue')
    
    # Random from circuit nodes
    xs_circuit = sorted(result.random_circuit_results.keys())
    means_circuit = [result.random_circuit_results[x][0] for x in xs_circuit]
    stds_circuit = [result.random_circuit_results[x][1] for x in xs_circuit]
    
    ax.plot(xs_circuit, means_circuit, 'r-s', linewidth=2, markersize=6, label='Random from circuit nodes')
    ax.fill_between(xs_circuit,
                    [m - s for m, s in zip(means_circuit, stds_circuit)],
                    [m + s for m, s in zip(means_circuit, stds_circuit)],
                    alpha=0.2, color='red')
    
    # Add horizontal lines for baselines
    ax.axhline(y=result.clean_loss, color='green', linestyle='--', linewidth=2,
               label=f'Clean (no ablation): {result.clean_loss:.4f}')
    ax.axhline(y=result.circuit_ablated_loss, color='orange', linestyle='--', linewidth=2,
               label=f'Non-circuit ablated: {result.circuit_ablated_loss:.4f}')
    
    # Add vertical line at circuit size
    ax.axvline(x=result.circuit_size, color='gray', linestyle=':', linewidth=1.5,
               label=f'Circuit size: {result.circuit_size:,}')
    
    ax.set_xlabel("Number of Nodes Ablated", fontsize=12)
    ax.set_ylabel("Task Loss", fontsize=12)
    ax.set_title(f"{title_prefix}Ablation Sweep on Unpruned Model\n"
                 f"(Circuit: {result.circuit_size:,} / {result.total_nodes:,} nodes)", fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_ablation_sweep_results(
    result: AblationSweepResult,
    output_path: Path,
):
    """Save ablation sweep results to JSON."""
    data = {
        "baselines": {
            "clean_loss": result.clean_loss,
            "circuit_ablated_loss": result.circuit_ablated_loss,
        },
        "random_all_results": {
            str(k): {"mean": v[0], "std": v[1]}
            for k, v in result.random_all_results.items()
        },
        "random_circuit_results": {
            str(k): {"mean": v[0], "std": v[1]}
            for k, v in result.random_circuit_results.items()
        },
        "metadata": {
            "circuit_size": result.circuit_size,
            "total_nodes": result.total_nodes,
            "num_trials": result.num_trials,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def run_and_save_ablation_sweep(
    masked_model: MaskedSparseGPT,
    base_model: nn.Module,
    task: BinaryTask,
    output_dir: Path,
    config: Optional[AblationSweepConfig] = None,
    pruning_config: Optional[PruningConfig] = None,
    mean_cache: Optional[Dict[str, torch.Tensor]] = None,
    title_prefix: str = "",
    show_progress: bool = True,
) -> AblationSweepResult:
    """
    Convenience function to run ablation sweep evaluation and save all outputs.
    """
    if config is None:
        config = AblationSweepConfig()
    if pruning_config is None:
        pruning_config = masked_model.config
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Running Ablation Sweep Evaluation")
    print(f"{'='*60}")
    print(f"Points: {config.num_points}")
    print(f"Trials per point: {config.num_trials}")
    print(f"Batches: {config.num_batches}")
    print(f"Output dir: {output_dir}")
    
    result = run_ablation_sweep_evaluation(
        masked_model=masked_model,
        base_model=base_model,
        task=task,
        config=config,
        pruning_config=pruning_config,
        mean_cache=mean_cache,
        show_progress=show_progress,
    )
    
    # Save plot
    plot_path = output_dir / "ablation_sweep.png"
    plot_ablation_sweep_results(result, plot_path, title_prefix=title_prefix)
    print(f"Plot saved to {plot_path}")
    
    # Save JSON
    json_path = output_dir / "ablation_sweep_results.json"
    save_ablation_sweep_results(result, json_path)
    print(f"Results saved to {json_path}")
    
    # Print summary
    print(f"\nAblation Sweep Summary:")
    print(f"  Circuit size: {result.circuit_size:,} / {result.total_nodes:,}")
    print(f"  Clean loss: {result.clean_loss:.4f}")
    print(f"  Non-circuit ablated loss: {result.circuit_ablated_loss:.4f}")
    
    return result


# ============================================================================
# Mask Relaxation Evaluation
# ============================================================================

@dataclass
class MaskRelaxationConfig:
    """Configuration for mask relaxation evaluation.
    
    This evaluation takes a pruned model and gradually "relaxes" the mask by
    flipping masked (inactive) nodes to be active for various fractions.
    
    At fraction=0: Only circuit nodes are active (pruned model behavior)
    At fraction=1: All nodes are active (approaching unpruned model)
    """
    # Number of fraction points to evaluate
    num_points: int = 11  # e.g., 0, 0.1, 0.2, ..., 1.0
    
    # Number of random trials per fraction
    num_trials: int = 10
    
    # Evaluation settings
    num_batches: int = 10
    batch_size: int = 64
    seq_length: int = 0  # 0 = dynamic padding
    
    # Device
    device: str = "cuda"


@dataclass
class MaskRelaxationResult:
    """Results from mask relaxation evaluation."""
    # Baseline losses
    circuit_only_loss: float  # Pruned model loss (only circuit nodes active)
    all_active_loss: float  # Loss with all nodes active (no masking)
    
    # Sweep results: fraction -> (mean_loss, std_loss)
    relaxation_results: Dict[float, Tuple[float, float]]
    
    # Metadata
    circuit_size: int  # Number of active nodes in circuit
    num_masked: int  # Number of masked (inactive) nodes
    total_nodes: int
    num_trials: int
    fractions: List[float]


def get_masked_node_indices(
    masked_model: MaskedSparseGPT,
) -> Dict[str, torch.Tensor]:
    """
    Get indices of masked (inactive) nodes (tau < 0) for each mask location.
    
    Returns:
        Dict mapping mask key (e.g., "layer0_attn_in") to tensor of masked indices
    """
    masked_indices = {}
    for key, mask in masked_model.masks.masks.items():
        # Get indices where tau < 0 (masked/inactive)
        masked = (mask.tau < 0).nonzero(as_tuple=True)[0]
        masked_indices[key] = masked
    return masked_indices


def get_masked_token_indices(
    masked_model: MaskedSparseGPT,
) -> torch.Tensor:
    """
    Get indices of masked (inactive) tokens (tau < 0) from the token embedding mask.
    
    Returns:
        Tensor of masked token indices, or None if no token mask
    """
    if masked_model.token_mask is None:
        return None
    masked = (masked_model.token_mask.tau < 0).nonzero(as_tuple=True)[0]
    return masked


def run_masked_forward_with_relaxation(
    masked_model: MaskedSparseGPT,
    input_ids: torch.Tensor,
    relaxation_mask: Dict[str, torch.Tensor],
    token_relaxation_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Run forward pass on the masked model, with some masked nodes "relaxed" (activated).
    
    Args:
        masked_model: The masked model
        input_ids: Input token IDs (batch, seq)
        relaxation_mask: Dict mapping mask keys to boolean tensors indicating which
                        masked nodes should be relaxed (True = activate this node)
        token_relaxation_mask: Optional boolean tensor (vocab_size,) indicating which
                              masked tokens should be relaxed (True = activate this token)
                        
    Returns:
        Logits tensor
    """
    device = input_ids.device
    B, T = input_ids.shape
    model = masked_model.model
    config = masked_model.config
    
    # Compute frozen LN scales if enabled
    frozen_ln_scales = None
    if config.freeze_layernorm_scale:
        frozen_ln_scales = masked_model._compute_frozen_ln_scales(input_ids)
    
    # Forward through model with relaxation
    x = model.wte(input_ids)
    
    if torch.is_autocast_enabled('cuda'):
        x = x.to(torch.get_autocast_dtype('cuda'))
    
    # Apply token mask if enabled (with relaxation)
    if masked_model.token_mask is not None:
        # Get original binary mask (0 for masked, 1 for active)
        token_binary_mask = masked_model.token_mask.get_binary_mask()  # (vocab_size,)
        
        # Apply token relaxation: additional tokens to activate
        if token_relaxation_mask is not None:
            combined_mask = token_binary_mask + token_relaxation_mask.float()
            combined_mask = (combined_mask >= 1).float()
        else:
            combined_mask = token_binary_mask
        
        mask_per_position = combined_mask[input_ids].unsqueeze(-1)
        
        if masked_model.token_mask.mean_set:
            mean_embed = masked_model.token_mask.mean_activation
            x = x * mask_per_position + mean_embed * (1 - mask_per_position)
        else:
            x = x * mask_per_position
    
    if model.wpe is not None:
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + model.wpe(pos)
    
    x = model.drop(x)
    
    def apply_abstopk(x: torch.Tensor, location: str) -> torch.Tensor:
        """Apply AbsTopK sparsity if enabled for this location."""
        if not masked_model._activation_sparsity_enabled:
            return x
        if location not in masked_model._activation_sparsity_locations:
            return x
        
        dim = x.shape[-1]
        k = max(1, int(dim * masked_model._activation_topk_fraction))
        
        if k >= dim:
            return x
        
        # AbsTopK: keep top k values by absolute value, zero the rest
        _, topk_indices = torch.topk(x.abs(), k, dim=-1, sorted=False)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, x.gather(-1, topk_indices))
        return result
    
    def apply_relaxed_mask(activation: torch.Tensor, layer_idx: int, location: str) -> torch.Tensor:
        """Apply AbsTopK then mask with relaxation for selected masked nodes."""
        # First apply AbsTopK to match base model's activation sparsity
        activation = apply_abstopk(activation, location)
        
        if location not in config.mask_locations:
            return activation
        
        key = f"layer{layer_idx}_{location}"
        
        # Get the node mask module
        mask_module = masked_model.masks.get_mask(layer_idx, location)
        
        # Get original binary mask (0 for masked, 1 for active)
        original_mask = mask_module.get_binary_mask()  # (num_nodes,)
        
        # Apply relaxation: nodes in relaxation_mask should also be active
        if key in relaxation_mask:
            # Combine original mask with relaxation
            # Original: active nodes (tau >= 0)
            # Relaxation: additional nodes to activate
            combined_mask = original_mask + relaxation_mask[key].float()
            combined_mask = (combined_mask >= 1).float()  # Clamp to [0, 1]
        else:
            combined_mask = original_mask
        
        # Apply mean ablation for still-masked nodes
        if mask_module.mean_set:
            mean = mask_module.mean_activation
            return activation * combined_mask + mean * (1 - combined_mask)
        else:
            return activation * combined_mask
    
    for layer_idx, block in enumerate(model.blocks):
        # Attention sublayer
        if frozen_ln_scales is not None:
            frozen_scale = frozen_ln_scales[f"layer{layer_idx}_ln_1"]
            normed = masked_model._apply_ln_with_frozen_scale(x, block.ln_1, frozen_scale)
        else:
            normed = block.ln_1(x)
        normed = apply_relaxed_mask(normed, layer_idx, "attn_in")
        
        # Q, K, V projections
        qkv = block.attn.c_attn(normed)
        q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
        
        q = apply_relaxed_mask(q, layer_idx, "attn_q")
        k = apply_relaxed_mask(k, layer_idx, "attn_k")
        v = apply_relaxed_mask(v, layer_idx, "attn_v")
        
        # Compute attention
        q = q.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        k = k.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        v = v.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
        
        scale = 1.0 / (block.attn.d_head ** 0.5)
        
        if block.attn.use_flash and block.attn.attn_fn is not None:
            y = block.attn.attn_fn(
                q, k, v,
                dropout_p=block.attn.dropout if block.attn.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        elif block.attn.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=block.attn.dropout if block.attn.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(
                block.attn.causal_mask[:, :, :T, :T] == 0,
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = block.attn.attn_dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, block.attn.n_heads * block.attn.d_head)
        attn_out = block.attn.c_proj(y)
        attn_out = block.attn.resid_dropout(attn_out)
        
        attn_out = apply_relaxed_mask(attn_out, layer_idx, "attn_out")
        
        x = x + attn_out
        
        # MLP sublayer
        if frozen_ln_scales is not None:
            frozen_scale = frozen_ln_scales[f"layer{layer_idx}_ln_2"]
            normed = masked_model._apply_ln_with_frozen_scale(x, block.ln_2, frozen_scale)
        else:
            normed = block.ln_2(x)
        normed = apply_relaxed_mask(normed, layer_idx, "mlp_in")
        
        mlp_hidden = block.mlp.c_fc(normed)
        mlp_hidden = block.mlp.act_fn(mlp_hidden)
        mlp_hidden = apply_relaxed_mask(mlp_hidden, layer_idx, "mlp_neuron")
        
        mlp_out = block.mlp.c_proj(mlp_hidden)
        mlp_out = block.mlp.dropout(mlp_out)
        mlp_out = apply_relaxed_mask(mlp_out, layer_idx, "mlp_out")
        
        x = x + mlp_out
    
    # Final layer norm and LM head
    if frozen_ln_scales is not None:
        frozen_scale = frozen_ln_scales["ln_f"]
        x = masked_model._apply_ln_with_frozen_scale(x, model.ln_f, frozen_scale)
    else:
        x = model.ln_f(x)
    logits = model.lm_head(x)
    
    if hasattr(model, 'bigram_table') and model.bigram_table is not None:
        bigram_logits = F.embedding(input_ids, model.bigram_table)
        logits = logits + bigram_logits
    
    return logits


def create_relaxation_mask(
    masked_indices: Dict[str, torch.Tensor],
    node_counts: Dict[str, int],
    fraction: float,
    rng: np.random.Generator,
    device: str = "cuda",
    masked_token_indices: torch.Tensor = None,
    vocab_size: int = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Create a relaxation mask for the given fraction of masked nodes.
    
    Args:
        masked_indices: Dict mapping mask keys to tensors of masked node indices
        node_counts: Dict mapping mask keys to total node count
        fraction: Fraction of masked nodes to relax (activate)
        rng: Random number generator
        device: Device for tensors
        masked_token_indices: Optional tensor of masked token indices
        vocab_size: Vocabulary size (required if masked_token_indices is provided)
        
    Returns:
        Tuple of:
        - Dict mapping mask key to boolean tensor (True = relax/activate this node)
        - Token relaxation mask (boolean tensor, or None if no token mask)
    """
    # Build list of all (key, idx) pairs for masked nodes
    all_masked_nodes = []
    for key, indices in masked_indices.items():
        for idx in indices.tolist():
            all_masked_nodes.append((key, idx))
    
    # Also include masked tokens if present
    masked_token_list = []
    if masked_token_indices is not None and len(masked_token_indices) > 0:
        masked_token_list = masked_token_indices.tolist()
    
    num_masked_nodes = len(all_masked_nodes)
    num_masked_tokens = len(masked_token_list)
    total_masked = num_masked_nodes + num_masked_tokens
    
    if total_masked == 0:
        # No masked nodes/tokens to relax
        node_mask = {key: torch.zeros(count, dtype=torch.bool, device=device)
                     for key, count in node_counts.items()}
        token_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device) if vocab_size else None
        return node_mask, token_mask
    
    # Determine how many total items to relax
    num_to_relax = int(round(fraction * total_masked))
    
    # Randomly select items to relax (from combined pool of nodes + tokens)
    if num_to_relax > 0:
        selected_indices = rng.choice(total_masked, size=num_to_relax, replace=False)
    else:
        selected_indices = []
    
    # Initialize masks as all False
    relaxation_mask = {}
    for key, count in node_counts.items():
        relaxation_mask[key] = torch.zeros(count, dtype=torch.bool, device=device)
    
    token_relaxation_mask = None
    if vocab_size is not None:
        token_relaxation_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    
    # Set selected items to True (relax)
    for idx in selected_indices:
        if idx < num_masked_nodes:
            # This is a node
            key, node_idx = all_masked_nodes[idx]
            relaxation_mask[key][node_idx] = True
        else:
            # This is a token
            token_idx = masked_token_list[idx - num_masked_nodes]
            if token_relaxation_mask is not None:
                token_relaxation_mask[token_idx] = True
    
    return relaxation_mask, token_relaxation_mask


def run_mask_relaxation_evaluation(
    masked_model: MaskedSparseGPT,
    task: BinaryTask,
    config: MaskRelaxationConfig,
    show_progress: bool = True,
) -> MaskRelaxationResult:
    """
    Run mask relaxation evaluation on the masked model.
    
    This gradually relaxes the mask by activating fractions of the masked nodes,
    measuring how the loss changes as more non-circuit nodes are allowed.
    
    Args:
        masked_model: The pruned/masked model
        task: Binary task for evaluation
        config: Mask relaxation config
        show_progress: Whether to show progress bars
        
    Returns:
        MaskRelaxationResult
    """
    device = config.device
    rng = np.random.default_rng(42)
    
    masked_model.eval()
    
    # Get masked (inactive) node indices
    masked_indices = get_masked_node_indices(masked_model)
    num_masked_nodes = sum(len(v) for v in masked_indices.values())
    
    # Get masked token indices (if token mask exists)
    masked_token_indices = get_masked_token_indices(masked_model)
    num_masked_tokens = len(masked_token_indices) if masked_token_indices is not None else 0
    vocab_size = masked_model.vocab_size if masked_model.token_mask is not None else None
    
    num_masked = num_masked_nodes + num_masked_tokens
    
    # Get active (circuit) node indices
    active_indices = get_active_node_indices(masked_model)
    circuit_size = sum(len(v) for v in active_indices.values())
    if masked_model.token_mask is not None:
        num_active_tokens = masked_model.token_mask.get_num_active()
        circuit_size += num_active_tokens
    
    # Get node counts per location
    node_counts = {}
    for key, mask in masked_model.masks.masks.items():
        node_counts[key] = mask.tau.shape[0]
    total_nodes = sum(node_counts.values())
    if vocab_size is not None:
        total_nodes += vocab_size
    
    print(f"Circuit size: {circuit_size:,} active nodes/tokens")
    print(f"Masked: {num_masked_nodes:,} nodes + {num_masked_tokens:,} tokens = {num_masked:,} total")
    print(f"Total nodes: {total_nodes:,}")
    
    # Generate fixed batches for consistent evaluation
    print("Pre-generating evaluation batches...")
    pruning_config = masked_model.config
    use_binary_loss = pruning_config.use_binary_loss
    fixed_batches = []
    for _ in range(config.num_batches):
        batch = task.generate_batch(
            batch_size=config.batch_size,
            max_length=config.seq_length,
        )
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
        fixed_batches.append({
            "positive_ids": positive_ids.to(device),
            "correct_tokens": correct_tokens.to(device),
            "incorrect_tokens": incorrect_tokens.to(device),
            "eval_positions": eval_positions.to(device),
        })
    
    def compute_loss_with_relaxation(relaxation_mask, token_relaxation_mask=None):
        """Compute average loss with given relaxation mask."""
        losses = []
        with torch.no_grad():
            for batch in fixed_batches:
                input_ids = batch["positive_ids"]
                correct_tokens = batch["correct_tokens"]
                incorrect_tokens = batch["incorrect_tokens"]
                eval_positions = batch["eval_positions"]
                
                logits = run_masked_forward_with_relaxation(
                    masked_model, input_ids, relaxation_mask, token_relaxation_mask
                )
                
                loss = compute_task_loss_from_logits(
                    logits, correct_tokens, eval_positions,
                    incorrect_tokens=incorrect_tokens, use_binary_loss=use_binary_loss
                )
                losses.append(loss)
        
        return np.mean(losses)
    
    # 1. Circuit-only loss (no relaxation, fraction=0)
    print("Computing circuit-only loss (fraction=0)...")
    empty_relaxation = {key: torch.zeros(count, dtype=torch.bool, device=device)
                        for key, count in node_counts.items()}
    empty_token_relaxation = torch.zeros(vocab_size, dtype=torch.bool, device=device) if vocab_size else None
    circuit_only_loss = compute_loss_with_relaxation(empty_relaxation, empty_token_relaxation)
    print(f"  Circuit-only loss: {circuit_only_loss:.4f}")
    
    # 2. All-active loss (full relaxation, fraction=1)
    print("Computing all-active loss (fraction=1)...")
    full_relaxation = {}
    for key in node_counts:
        full_relaxation[key] = torch.zeros(node_counts[key], dtype=torch.bool, device=device)
        if key in masked_indices:
            full_relaxation[key][masked_indices[key]] = True
    full_token_relaxation = None
    if vocab_size is not None and masked_token_indices is not None:
        full_token_relaxation = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        full_token_relaxation[masked_token_indices] = True
    all_active_loss = compute_loss_with_relaxation(full_relaxation, full_token_relaxation)
    print(f"  All-active loss: {all_active_loss:.4f}")
    
    # 3. Sweep over fractions
    fractions = [i / (config.num_points - 1) for i in range(config.num_points)]
    
    relaxation_results = {}
    
    print(f"Running relaxation sweep ({len(fractions)} fraction points, {config.num_trials} trials each)...")
    
    for frac in tqdm(fractions, desc="Relaxation sweep", disable=not show_progress):
        losses = []
        for trial in range(config.num_trials):
            relaxation_mask, token_relaxation_mask = create_relaxation_mask(
                masked_indices, node_counts, frac, rng, device,
                masked_token_indices=masked_token_indices,
                vocab_size=vocab_size,
            )
            loss = compute_loss_with_relaxation(relaxation_mask, token_relaxation_mask)
            losses.append(loss)
        
        relaxation_results[frac] = (np.mean(losses), np.std(losses))
    
    return MaskRelaxationResult(
        circuit_only_loss=circuit_only_loss,
        all_active_loss=all_active_loss,
        relaxation_results=relaxation_results,
        circuit_size=circuit_size,
        num_masked=num_masked,
        total_nodes=total_nodes,
        num_trials=config.num_trials,
        fractions=fractions,
    )


def plot_mask_relaxation_results(
    result: MaskRelaxationResult,
    output_path: Path,
    title_prefix: str = "",
):
    """
    Create plot for mask relaxation results.
    
    X-axis: Fraction of masked nodes relaxed (0 = circuit only, 1 = all nodes active)
    Y-axis: Task loss
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Relaxation curve
    fracs = sorted(result.relaxation_results.keys())
    means = [result.relaxation_results[f][0] for f in fracs]
    stds = [result.relaxation_results[f][1] for f in fracs]
    
    ax.plot(fracs, means, 'b-o', linewidth=2, markersize=6, label='Relaxation sweep')
    ax.fill_between(fracs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color='blue')
    
    # Add horizontal lines for baselines
    ax.axhline(y=result.circuit_only_loss, color='red', linestyle='--', linewidth=2,
               label=f'Circuit only (f=0): {result.circuit_only_loss:.4f}')
    ax.axhline(y=result.all_active_loss, color='green', linestyle='--', linewidth=2,
               label=f'All active (f=1): {result.all_active_loss:.4f}')
    
    ax.set_xlabel("Fraction of Masked Nodes Relaxed", fontsize=12)
    ax.set_ylabel("Task Loss", fontsize=12)
    ax.set_title(f"{title_prefix}Mask Relaxation Sweep\n"
                 f"(Circuit: {result.circuit_size:,} active, {result.num_masked:,} masked, {result.total_nodes:,} total)", 
                 fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    # Use linear scale for y-axis
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_mask_relaxation_results(
    result: MaskRelaxationResult,
    output_path: Path,
):
    """Save mask relaxation results to JSON."""
    data = {
        "baselines": {
            "circuit_only_loss": result.circuit_only_loss,
            "all_active_loss": result.all_active_loss,
        },
        "relaxation_results": {
            str(k): {"mean": v[0], "std": v[1]}
            for k, v in result.relaxation_results.items()
        },
        "metadata": {
            "circuit_size": result.circuit_size,
            "num_masked": result.num_masked,
            "total_nodes": result.total_nodes,
            "num_trials": result.num_trials,
            "fractions": result.fractions,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def run_and_save_mask_relaxation(
    masked_model: MaskedSparseGPT,
    task: BinaryTask,
    output_dir: Path,
    config: Optional[MaskRelaxationConfig] = None,
    title_prefix: str = "",
    show_progress: bool = True,
) -> MaskRelaxationResult:
    """
    Convenience function to run mask relaxation evaluation and save all outputs.
    
    Args:
        masked_model: The pruned/masked model
        task: Binary task for evaluation
        output_dir: Directory to save outputs (plot and JSON)
        config: Mask relaxation config (uses defaults if None)
        title_prefix: Prefix for plot titles
        show_progress: Whether to show progress bars
        
    Returns:
        MaskRelaxationResult
    """
    if config is None:
        config = MaskRelaxationConfig()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Running Mask Relaxation Evaluation")
    print(f"{'='*60}")
    print(f"Fraction points: {config.num_points}")
    print(f"Trials per point: {config.num_trials}")
    print(f"Batches: {config.num_batches}")
    print(f"Output dir: {output_dir}")
    
    result = run_mask_relaxation_evaluation(
        masked_model=masked_model,
        task=task,
        config=config,
        show_progress=show_progress,
    )
    
    # Save plot
    plot_path = output_dir / "mask_relaxation_sweep.png"
    plot_mask_relaxation_results(result, plot_path, title_prefix=title_prefix)
    print(f"Plot saved to {plot_path}")
    
    # Save JSON
    json_path = output_dir / "mask_relaxation_results.json"
    save_mask_relaxation_results(result, json_path)
    print(f"Results saved to {json_path}")
    
    # Print summary
    print(f"\nMask Relaxation Summary:")
    print(f"  Circuit size: {result.circuit_size:,} / {result.total_nodes:,}")
    print(f"  Masked nodes: {result.num_masked:,}")
    print(f"  Circuit-only loss (f=0): {result.circuit_only_loss:.4f}")
    print(f"  All-active loss (f=1): {result.all_active_loss:.4f}")
    
    return result

