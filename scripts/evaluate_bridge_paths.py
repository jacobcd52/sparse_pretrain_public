#!/usr/bin/env python3
"""
Evaluate all possible paths through dense and sparse models using bridges.

This script evaluates the KL divergences and cross-entropies for all valid paths
through a trained bridge model. The paths include combinations of:
- Running different numbers of layers on each model
- Crossing bridges (encoder/decoder) at different points
- Paths with at most one "double-back" (encoder then decoder at the same site)

For an L-layer model, bridge sites are:
- Site 0: After embedding
- Site 2i+1: After layer i's attention (before MLP), for i = 0..L-1
- Site 2i+2: After layer i's MLP, for i = 0..L-1

Path types:
1. Pure dense: Just run the dense model start to finish
2. Pure sparse: Just run the sparse model start to finish
3. Single transition (d→s): Dense for some layers, then encoder, then sparse to end
4. Single transition (s→d): Sparse for some layers, then decoder, then dense to end
5. Double-back (d→s→d): Dense, encoder, decoder (no sparse computation), then dense
6. Double-back (s→d→s): Sparse, decoder, encoder (no dense computation), then sparse

Usage:
    python scripts/evaluate_bridge_paths.py \
        --checkpoint /path/to/bridges/checkpoint \
        --output /path/to/output.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from sparse_pretrain.src.config import ModelConfig, SparsityConfig
from sparse_pretrain.src.config_bridges import FullBridgesConfig, BridgesConfig
from sparse_pretrain.src.model import SparseGPT
from sparse_pretrain.src.bridges import BridgeSet, kl_divergence
from sparse_pretrain.src.train_bridges import load_dense_model
from sparse_pretrain.src.data import create_validation_data


# =============================================================================
# Path Representation
# =============================================================================

@dataclass
class PathSegment:
    """A segment of a path through the models.
    
    Each segment represents a portion of the forward pass on one model.
    """
    model: str  # "dense" or "sparse"
    start_site: int  # Bridge site index to start from
    end_site: int  # Bridge site index to end at (exclusive for transitions)


@dataclass
class PathDescription:
    """Full description of a path through the models.
    
    A path consists of segments on different models connected by bridge transitions.
    """
    segments: List[PathSegment]
    transitions: List[Tuple[int, str]]  # List of (site_idx, transition_type) where type is "encode" or "decode"
    name: str  # Human-readable name for the path
    
    def __post_init__(self):
        # Compute complexity score (number of transitions)
        self.complexity = len(self.transitions)


def enumerate_all_paths(n_sites: int) -> List[PathDescription]:
    """
    Enumerate all valid paths through the models.
    
    For L layers, there are 2L+1 bridge sites.
    
    Valid paths:
    1. Pure model: Only one model, no transitions
    2. Single transition: Switch once from one model to another
    3. Double-back: Switch to other model and immediately back (encoder then decoder)
    
    The constraint is: at most one "double-back" per site, meaning we can do
    encoder→decoder or decoder→encoder at a single site, but not
    encoder→decoder→encoder→decoder.
    
    Returns:
        List of PathDescription objects, ordered by complexity then site index.
    """
    paths = []
    
    # 1. Pure paths (no transitions)
    # Pure dense
    paths.append(PathDescription(
        segments=[PathSegment("dense", 0, n_sites)],
        transitions=[],
        name="pure_dense"
    ))
    
    # Pure sparse
    paths.append(PathDescription(
        segments=[PathSegment("sparse", 0, n_sites)],
        transitions=[],
        name="pure_sparse"
    ))
    
    # 2. Single transition paths (d→s or s→d at each site)
    for site in range(n_sites):
        # Dense to sparse (encode at site)
        if site > 0:  # Can't transition at site 0 if we haven't run any layers
            paths.append(PathDescription(
                segments=[
                    PathSegment("dense", 0, site),
                    PathSegment("sparse", site, n_sites)
                ],
                transitions=[(site, "encode")],
                name=f"d{site}->s"
            ))
        elif site == 0:
            # Special case: start from dense embedding, immediately encode to sparse
            paths.append(PathDescription(
                segments=[
                    PathSegment("dense", 0, 0),  # Just embedding
                    PathSegment("sparse", 0, n_sites)
                ],
                transitions=[(0, "encode")],
                name="d0->s"
            ))
        
        # Sparse to dense (decode at site)
        if site > 0:
            paths.append(PathDescription(
                segments=[
                    PathSegment("sparse", 0, site),
                    PathSegment("dense", site, n_sites)
                ],
                transitions=[(site, "decode")],
                name=f"s{site}->d"
            ))
        elif site == 0:
            paths.append(PathDescription(
                segments=[
                    PathSegment("sparse", 0, 0),
                    PathSegment("dense", 0, n_sites)
                ],
                transitions=[(0, "decode")],
                name="s0->d"
            ))
    
    # 3. Double-back paths (go to other model and immediately back at same site)
    for site in range(n_sites):
        # Dense → encode → decode → dense (no sparse computation)
        if site < n_sites:  # Must have remaining layers to run
            paths.append(PathDescription(
                segments=[
                    PathSegment("dense", 0, site),
                    PathSegment("dense", site, n_sites)  # Continue on dense
                ],
                transitions=[(site, "encode"), (site, "decode")],
                name=f"d{site}->enc->dec->d"
            ))
        
        # Sparse → decode → encode → sparse (no dense computation)
        if site < n_sites:
            paths.append(PathDescription(
                segments=[
                    PathSegment("sparse", 0, site),
                    PathSegment("sparse", site, n_sites)  # Continue on sparse
                ],
                transitions=[(site, "decode"), (site, "encode")],
                name=f"s{site}->dec->enc->s"
            ))
    
    # 4. Cross paths with double-back in the middle
    # d -> encode -> sparse for a bit -> decode -> d to end
    for start_site in range(n_sites):
        for end_site in range(start_site + 1, n_sites + 1):
            if end_site < n_sites:  # Must have remaining dense layers
                paths.append(PathDescription(
                    segments=[
                        PathSegment("dense", 0, start_site),
                        PathSegment("sparse", start_site, end_site),
                        PathSegment("dense", end_site, n_sites)
                    ],
                    transitions=[(start_site, "encode"), (end_site, "decode")],
                    name=f"d{start_site}->s{end_site}->d"
                ))
    
    # s -> decode -> dense for a bit -> encode -> s to end
    for start_site in range(n_sites):
        for end_site in range(start_site + 1, n_sites + 1):
            if end_site < n_sites:  # Must have remaining sparse layers
                paths.append(PathDescription(
                    segments=[
                        PathSegment("sparse", 0, start_site),
                        PathSegment("dense", start_site, end_site),
                        PathSegment("sparse", end_site, n_sites)
                    ],
                    transitions=[(start_site, "decode"), (end_site, "encode")],
                    name=f"s{start_site}->d{end_site}->s"
                ))
    
    # Sort by complexity (number of transitions), then by name
    paths.sort(key=lambda p: (p.complexity, p.name))
    
    return paths


# =============================================================================
# ASCII Path Visualization
# =============================================================================

def path_to_ascii(path: PathDescription, n_sites: int, n_layers: int) -> str:
    """
    Create an ASCII visualization of a path.
    
    The visualization shows the dense and sparse models as vertical columns
    with horizontal rungs (bridges) connecting them at each site.
    
    Example for a 2-layer model with path d2->s:
    
    Dense    Sparse
      │        
      ●──────────●  Site 0 (after embed)
      │          
      │        
      ●──────────●  Site 1 (after L0 attn)
      │          
      │        
      ●════>═══>●  Site 2 (after L0 MLP) [transition here]
               │
               │
      ●──────────●  Site 3 (after L1 attn)
               │
               │
      ●──────────●  Site 4 (after L1 MLP)
               ▼
    
    Legend:
      │ = active path segment
      ● = bridge site (node)
      ═══> = encode transition (dense → sparse)
      <═══ = decode transition (sparse → dense)
      <══> = double-back (encode then decode, or decode then encode)
      ─── = bridge connection (not used)
    """
    lines = []
    
    # Track which model is active at each site
    active_model = [None] * (n_sites + 1)
    transitions_at = {}  # site -> list of (trans_type,)
    
    # Process path segments
    for seg in path.segments:
        for site in range(seg.start_site, min(seg.end_site + 1, n_sites + 1)):
            if active_model[site] is None:
                active_model[site] = seg.model
    
    # Process transitions - track all at each site
    for site, trans_type in path.transitions:
        if site not in transitions_at:
            transitions_at[site] = []
        transitions_at[site].append(trans_type)
    
    # Build ASCII representation
    lines.append("  Dense     Sparse")
    lines.append("    │         │")
    
    for site in range(n_sites):
        # Determine what to show for this site
        is_active_dense = active_model[site] == "dense" or (site in transitions_at)
        is_active_sparse = active_model[site] == "sparse" or (site in transitions_at)
        
        # Node line
        d_node = "●" if is_active_dense else "○"
        s_node = "●" if is_active_sparse else "○"
        
        # Bridge between nodes
        if site in transitions_at:
            trans_list = transitions_at[site]
            if len(trans_list) >= 2:
                # Double-back at this site
                bridge = "═══<══>═══"  # Shows both directions
            elif trans_list[0] == "encode":
                bridge = "══════>═══"
            else:  # decode
                bridge = "═══<══════"
        else:
            bridge = "──────────"
        
        # Site label
        if site == 0:
            label = "(embed)"
        elif site % 2 == 1:
            layer = (site - 1) // 2
            label = f"(L{layer} attn)"
        else:
            layer = (site - 2) // 2
            label = f"(L{layer} MLP)"
        
        lines.append(f"    {d_node}{bridge}{s_node}  Site {site} {label}")
        
        # Vertical connectors to next site
        if site < n_sites - 1:
            next_active = active_model[site + 1]
            
            # Determine which model continues after this site
            if site in transitions_at:
                trans_list = transitions_at[site]
                if len(trans_list) >= 2:
                    # Double-back: stays on same model
                    if trans_list[0] == "encode":
                        # d -> enc -> dec -> d: continues on dense
                        d_vert = "│"
                        s_vert = " "
                    else:
                        # s -> dec -> enc -> s: continues on sparse
                        d_vert = " "
                        s_vert = "│"
                else:
                    # Single transition
                    if trans_list[0] == "encode":
                        d_vert = " "
                        s_vert = "│"
                    else:
                        d_vert = "│"
                        s_vert = " "
            else:
                d_vert = "│" if next_active == "dense" or active_model[site] == "dense" else " "
                s_vert = "│" if next_active == "sparse" or active_model[site] == "sparse" else " "
            
            lines.append(f"    {d_vert}         {s_vert}")
    
    # Final arrow
    final_model = path.segments[-1].model
    if final_model == "dense":
        lines.append("    ▼")
    else:
        lines.append("              ▼")
    
    return "\n".join(lines)


def path_to_compact_ascii(path: PathDescription, n_sites: int) -> str:
    """
    Create a compact single-line ASCII visualization of a path.
    
    Examples:
        pure_dense:     D━━━━━━━━━━━━━━━━━▶
        pure_sparse:    S━━━━━━━━━━━━━━━━━▶
        d2->s:          D━━━━▶│━━━━━━━━━━━▶S
        s2->d:          S━━━━▶│━━━━━━━━━━━▶D
        d2->s4->d:      D━━━━▶│━━━━▶│━━━━━▶D
    """
    # Calculate width per site
    width_per_site = 2
    total_width = n_sites * width_per_site
    
    result = []
    current_model = path.segments[0].model
    
    for seg in path.segments:
        seg_width = (seg.end_site - seg.start_site) * width_per_site
        if seg_width > 0:
            result.append("━" * (seg_width - 1))
            if seg != path.segments[-1]:
                result.append("│")
    
    # Add endpoints
    start_model = path.segments[0].model[0].upper()
    end_model = path.segments[-1].model[0].upper()
    
    path_str = "".join(result)
    
    return f"{start_model}{path_str}▶{end_model}"


# =============================================================================
# Path Execution
# =============================================================================

@torch.no_grad()
def execute_path(
    path: PathDescription,
    input_ids: torch.Tensor,
    dense_model: SparseGPT,
    sparse_model: SparseGPT,
    bridge_set: BridgeSet,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Execute a path through the models and return the logits.
    
    Args:
        path: The path description
        input_ids: Input token IDs (batch, seq_len)
        dense_model: The frozen dense model
        sparse_model: The trained sparse model
        bridge_set: The bridge modules
        device: Device to use
        
    Returns:
        Logits of shape (batch, seq_len, vocab_size)
    """
    n_sites = bridge_set.n_sites
    
    # Get initial activations from both models (we may need either)
    # We'll compute on-demand to save memory
    
    def get_activations(model: SparseGPT, input_ids: torch.Tensor):
        """Get activations at all bridge sites for a model."""
        logits, h_pre_list, h_post_list = model.forward_with_bridge_sites(input_ids)
        return logits, h_pre_list, h_post_list
    
    # Determine which model we start with
    current_model = path.segments[0].model
    
    if current_model == "dense":
        # Get dense activations
        _, h_dense_pre, h_dense_post = get_activations(dense_model, input_ids)
        h_current = h_dense_post[0]  # Start from site 0 (post AbsTopK if applicable)
    else:
        # Get sparse activations
        _, h_sparse_pre, h_sparse_post = get_activations(sparse_model, input_ids)
        h_current = h_sparse_post[0]
    
    # Track transition index
    trans_idx = 0
    
    # Process each segment
    for seg_idx, seg in enumerate(path.segments):
        # Apply any transitions at the start of this segment
        while trans_idx < len(path.transitions) and path.transitions[trans_idx][0] == seg.start_site:
            site, trans_type = path.transitions[trans_idx]
            if trans_type == "encode":
                # Encode: dense -> sparse
                h_current = bridge_set.encode(site, h_current, hard=True)
            else:
                # Decode: sparse -> dense
                h_current = bridge_set.decode(site, h_current)
            trans_idx += 1
        
        # Run the model segment
        if seg.end_site > seg.start_site:
            if seg.model == "dense":
                h_current = dense_model.forward_from_site(h_current, seg.start_site, input_ids)
                # This returns logits, but we need intermediate activations if not final segment
                if seg_idx < len(path.segments) - 1:
                    # Re-run to get activations at the end site
                    # Actually, forward_from_site returns logits, not intermediate activations
                    # We need to be smarter here
                    pass
            else:
                h_current = sparse_model.forward_from_site(h_current, seg.start_site, input_ids)
    
    # The final segment's model's forward_from_site already returns logits
    return h_current


@torch.no_grad()
def execute_path_v2(
    path: PathDescription,
    input_ids: torch.Tensor,
    dense_model: SparseGPT,
    sparse_model: SparseGPT,
    bridge_set: BridgeSet,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Execute a path through the models and return the logits.
    
    This version properly handles intermediate activations by computing
    them only when needed.
    
    Args:
        path: The path description
        input_ids: Input token IDs (batch, seq_len)
        dense_model: The frozen dense model
        sparse_model: The trained sparse model
        bridge_set: The bridge modules
        device: Device to use
        
    Returns:
        Logits of shape (batch, seq_len, vocab_size)
    """
    # Special case: pure paths
    if path.name == "pure_dense":
        logits, _, _ = dense_model(input_ids)
        return logits
    
    if path.name == "pure_sparse":
        logits, _, _ = sparse_model(input_ids)
        return logits
    
    # For paths with transitions, we need to track activations
    n_sites = bridge_set.n_sites
    
    # Get activations at all sites from the starting model
    first_model = path.segments[0].model
    if first_model == "dense":
        _, h_pre, h_post = dense_model.forward_with_bridge_sites(input_ids)
        dense_h_pre, dense_h_post = h_pre, h_post
        sparse_h_pre, sparse_h_post = None, None
    else:
        _, h_pre, h_post = sparse_model.forward_with_bridge_sites(input_ids)
        sparse_h_pre, sparse_h_post = h_pre, h_post
        dense_h_pre, dense_h_post = None, None
    
    # Track current activation
    current_h = h_post[path.segments[0].start_site]
    current_model = first_model
    
    # Process transitions and segments
    for site, trans_type in path.transitions:
        # First, if we need to run some layers before the transition
        # Find the segment that ends at this site
        for seg in path.segments:
            if seg.end_site == site and seg.model == current_model:
                if seg.start_site < site:
                    # Need to run from seg.start_site to site
                    if current_model == "dense":
                        # Get activation at site by running partial forward
                        # We use forward_from_site but it returns logits
                        # Instead, we need to manually run layers
                        pass
                break
        
        # Apply transition
        if trans_type == "encode":
            # Encode: dense -> sparse
            current_h = bridge_set.encode(site, current_h, hard=True)
            current_model = "sparse"
        else:
            # Decode: sparse -> dense  
            current_h = bridge_set.decode(site, current_h)
            current_model = "dense"
    
    # Run remaining layers
    final_seg = path.segments[-1]
    if current_model == "dense":
        logits = dense_model.forward_from_site(current_h, final_seg.start_site, input_ids)
    else:
        logits = sparse_model.forward_from_site(current_h, final_seg.start_site, input_ids)
    
    return logits


def _run_model_segment(
    model: SparseGPT,
    h: torch.Tensor,
    start_site: int,
    end_site: int,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Run a model from start_site to end_site and return activation at end_site.
    
    If end_site is the final site, returns logits instead.
    """
    n_layers = len(model.blocks)
    n_sites = 2 * n_layers + 1
    
    if end_site >= n_sites:
        # Run to completion, return logits
        return model.forward_from_site(h, start_site, input_ids)
    
    # Otherwise, we need to run partial and get intermediate activation
    x = h
    act_sparsity_fn = model._get_activation_sparsity_fn()
    
    for layer_idx in range(n_layers):
        block = model.blocks[layer_idx]
        site_before_attn = 2 * layer_idx
        site_after_attn = 2 * layer_idx + 1
        site_after_mlp = 2 * layer_idx + 2
        
        # Skip if we've passed this layer
        if start_site > site_after_mlp:
            continue
        
        # Stop if we've reached the end
        if end_site <= site_before_attn:
            break
        
        # Run attention if needed
        if start_site <= site_before_attn and end_site > site_before_attn:
            if act_sparsity_fn is not None:
                normed = block.ln_1(x)
                normed = act_sparsity_fn(normed, "attn_in")
            else:
                normed = block.ln_1(x)
            
            attn_out = block.attn(normed, act_sparsity_fn)
            
            if act_sparsity_fn is not None:
                attn_out = act_sparsity_fn(attn_out, "attn_out")
            
            x = x + attn_out
        
        # Check if we stop after attention
        if end_site == site_after_attn:
            return x
        
        # Run MLP if needed
        if start_site <= site_after_attn and end_site > site_after_attn:
            if act_sparsity_fn is not None:
                normed = block.ln_2(x)
                normed = act_sparsity_fn(normed, "mlp_in")
            else:
                normed = block.ln_2(x)
            
            mlp_out = block.mlp(normed, act_sparsity_fn)
            
            if act_sparsity_fn is not None:
                mlp_out = act_sparsity_fn(mlp_out, "mlp_out")
            
            x = x + mlp_out
        
        # Check if we stop after MLP
        if end_site == site_after_mlp:
            return x
    
    return x


@torch.no_grad()
def execute_path_v3(
    path: PathDescription,
    input_ids: torch.Tensor,
    dense_model: SparseGPT,
    sparse_model: SparseGPT,
    bridge_set: BridgeSet,
) -> torch.Tensor:
    """
    Execute a path through the models and return the logits.
    
    This version carefully tracks activations through transitions.
    """
    n_sites = bridge_set.n_sites
    
    # Special case: pure paths
    if path.name == "pure_dense":
        logits, _, _ = dense_model(input_ids)
        return logits
    
    if path.name == "pure_sparse":
        logits, _, _ = sparse_model(input_ids)
        return logits
    
    # Get the starting activation
    first_seg = path.segments[0]
    if first_seg.model == "dense":
        # Get dense activations at the first segment's start
        _, h_pre, h_post = dense_model.forward_with_bridge_sites(input_ids)
        current_h = h_post[first_seg.start_site] if first_seg.start_site < len(h_post) else h_post[-1]
    else:
        _, h_pre, h_post = sparse_model.forward_with_bridge_sites(input_ids)
        current_h = h_post[first_seg.start_site] if first_seg.start_site < len(h_post) else h_post[-1]
    
    # Sort transitions by site
    sorted_transitions = sorted(path.transitions, key=lambda x: x[0])
    trans_idx = 0
    
    # Process segments
    for seg_idx, seg in enumerate(path.segments):
        # Get model for this segment
        model = dense_model if seg.model == "dense" else sparse_model
        
        # Determine if there's a transition at the end of this segment
        next_trans_site = None
        if trans_idx < len(sorted_transitions):
            next_trans_site = sorted_transitions[trans_idx][0]
        
        # Run the segment
        if seg_idx == len(path.segments) - 1:
            # Final segment - run to completion
            current_h = model.forward_from_site(current_h, seg.start_site, input_ids)
        else:
            # Intermediate segment - run to end site
            current_h = _run_model_segment(model, current_h, seg.start_site, seg.end_site, input_ids)
        
        # Apply transitions at the end of this segment
        while trans_idx < len(sorted_transitions) and sorted_transitions[trans_idx][0] == seg.end_site:
            _, trans_type = sorted_transitions[trans_idx]
            if trans_type == "encode":
                current_h = bridge_set.encode(seg.end_site, current_h, hard=True)
            else:
                current_h = bridge_set.decode(seg.end_site, current_h)
            trans_idx += 1
    
    return current_h


# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class PathEvalResult:
    """Results from evaluating a single path."""
    path_name: str
    path_ascii: str
    cross_entropy: float
    kl_to_dense: float
    kl_to_sparse: float
    perplexity: float
    num_transitions: int


def evaluate_path(
    path: PathDescription,
    input_ids: torch.Tensor,
    dense_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
    dense_model: SparseGPT,
    sparse_model: SparseGPT,
    bridge_set: BridgeSet,
    kl_topk: int = 64,
) -> PathEvalResult:
    """
    Evaluate a single path.
    
    Args:
        path: The path to evaluate
        input_ids: Input token IDs
        dense_logits: Pre-computed dense model logits
        sparse_logits: Pre-computed sparse model logits
        dense_model: The dense model
        sparse_model: The sparse model
        bridge_set: The bridge modules
        kl_topk: Number of top tokens for KL approximation
        
    Returns:
        PathEvalResult with metrics
    """
    # Execute the path
    path_logits = execute_path_v3(path, input_ids, dense_model, sparse_model, bridge_set)
    
    # Shift for next-token prediction
    shift_path_logits = path_logits[:, :-1, :].contiguous()
    shift_dense_logits = dense_logits[:, :-1, :].contiguous()
    shift_sparse_logits = sparse_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Cross-entropy
    ce_loss = F.cross_entropy(
        shift_path_logits.view(-1, shift_path_logits.size(-1)),
        shift_labels.view(-1),
    ).item()
    
    # KL to dense (dense is target)
    kl_dense = kl_divergence(shift_dense_logits, shift_path_logits, topk=kl_topk).item()
    
    # KL to sparse (sparse is target)
    kl_sparse = kl_divergence(shift_sparse_logits, shift_path_logits, topk=kl_topk).item()
    
    # Perplexity
    perplexity = torch.exp(torch.tensor(ce_loss)).item()
    
    # ASCII visualization
    n_layers = len(dense_model.blocks)
    ascii_viz = path_to_compact_ascii(path, bridge_set.n_sites)
    
    return PathEvalResult(
        path_name=path.name,
        path_ascii=ascii_viz,
        cross_entropy=ce_loss,
        kl_to_dense=kl_dense,
        kl_to_sparse=kl_sparse,
        perplexity=perplexity,
        num_transitions=path.complexity,
    )


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def load_from_hub(
    repo_id: str,
    device: str = "cuda",
) -> Tuple[SparseGPT, SparseGPT, BridgeSet, dict]:
    """
    Load a bridges model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "jacobcd52/ss_bridges_d1024_f0.015625")
        device: Device to load models on
        
    Returns:
        Tuple of (dense_model, sparse_model, bridge_set, config_dict)
    """
    from huggingface_hub import hf_hub_download
    
    print(f"Loading from HuggingFace Hub: {repo_id}")
    
    # Download config
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Download model and bridges
    sparse_model_path = hf_hub_download(repo_id=repo_id, filename="sparse_model.bin")
    bridges_path = hf_hub_download(repo_id=repo_id, filename="bridges.bin")
    
    # Extract configs
    model_cfg = config_dict["model_config"]
    sparsity_cfg = config_dict["sparsity_config"]
    bridges_cfg = config_dict.get("bridges_config", {})
    training_cfg = config_dict.get("training_config", {})
    
    # Create model config
    model_config = ModelConfig(
        n_layer=model_cfg["n_layer"],
        d_model=model_cfg["d_model"],
        n_ctx=model_cfg["n_ctx"],
        d_head=model_cfg["d_head"],
        d_mlp=model_cfg.get("d_mlp"),
        vocab_size=model_cfg["vocab_size"],
        use_rms_norm=model_cfg.get("use_rms_norm", True),
        tie_embeddings=model_cfg.get("tie_embeddings", False),
        use_positional_embeddings=model_cfg.get("use_positional_embeddings", False),
        use_bigram_table=model_cfg.get("use_bigram_table", False),
        use_attention_sinks=model_cfg.get("use_attention_sinks", True),
        activation=model_cfg.get("activation", "gelu"),
        dropout=model_cfg.get("dropout", 0.0),
        use_bias=model_cfg.get("use_bias", True),
    )
    
    # Create sparsity config
    sparsity_config = SparsityConfig(
        enable_weight_sparsity=sparsity_cfg.get("enable_weight_sparsity", True),
        target_l0_fraction=sparsity_cfg.get("target_l0_fraction", 0.015625),
        enable_activation_sparsity=sparsity_cfg.get("enable_activation_sparsity", True),
        activation_topk_fraction=sparsity_cfg.get("activation_topk_fraction", 0.25),
    )
    
    # Load sparse model
    print("Loading sparse model...")
    sparse_model = SparseGPT(model_config, sparsity_config)
    state_dict = torch.load(sparse_model_path, map_location=device, weights_only=True)
    sparse_model.load_state_dict(state_dict)
    sparse_model.to(device)
    sparse_model.eval()
    
    # Load dense model
    print("Loading dense model...")
    dense_model_source = bridges_cfg.get("d_dense", None)
    if dense_model_source is None:
        # Try to find the dense model from training config
        # Common pattern: the dense model is stored as a reference
        # For now, we'll try to load from a known pattern
        raise ValueError("Could not determine dense model source from config")
    
    # If dense_model_source is a HuggingFace repo ID, load from there
    dense_model = SparseGPT.from_pretrained(dense_model_source, device=device)
    dense_model.eval()
    for param in dense_model.parameters():
        param.requires_grad = False
    
    # Load bridges
    print("Loading bridges...")
    d_dense = dense_model.config.d_model
    d_sparse = model_config.d_model
    n_layers = model_config.n_layer
    
    encoder_afrac = bridges_cfg.get("encoder_afrac", 0.25)
    encoder_type = bridges_cfg.get("bridge_act_fn", "abstopk")
    init_log_eps = bridges_cfg.get("threshold_init_log_eps", -1.0)
    
    bridge_set = BridgeSet(
        n_layers=n_layers,
        d_dense=d_dense,
        d_sparse=d_sparse,
        encoder_afrac=encoder_afrac,
        encoder_type=encoder_type,
        init_log_eps=init_log_eps,
    )
    
    state_dict = torch.load(bridges_path, map_location=device, weights_only=True)
    bridge_set.load_state_dict(state_dict)
    bridge_set.to(device)
    bridge_set.eval()
    
    return dense_model, sparse_model, bridge_set, config_dict


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[SparseGPT, SparseGPT, BridgeSet, dict]:
    """
    Load a bridges checkpoint from local path or HuggingFace Hub.
    
    Args:
        checkpoint_path: Path to checkpoint directory or HuggingFace repo ID
        device: Device to load models on
        
    Returns:
        Tuple of (dense_model, sparse_model, bridge_set, config_dict)
    """
    # Check if it's a HuggingFace repo ID
    if "/" in checkpoint_path and not Path(checkpoint_path).exists():
        return load_from_hub(checkpoint_path, device)
    
    checkpoint_dir = Path(checkpoint_path)
    
    # Load config
    config_path = checkpoint_dir / "config.yaml"
    if config_path.exists():
        config = FullBridgesConfig.from_yaml(str(config_path))
        config_dict = config.to_dict()
    else:
        # Try config.json
        config_json_path = checkpoint_dir / "config.json"
        if config_json_path.exists():
            with open(config_json_path, "r") as f:
                config_dict = json.load(f)
        else:
            # Try parent directory
            config_path = checkpoint_dir.parent / "config.yaml"
            if config_path.exists():
                config = FullBridgesConfig.from_yaml(str(config_path))
                config_dict = config.to_dict()
            else:
                raise FileNotFoundError(f"No config found in {checkpoint_dir}")
    
    # Load from config dict
    model_cfg = config_dict.get("model_config", config_dict.get("sparse_model", {}))
    sparsity_cfg = config_dict.get("sparsity_config", config_dict.get("sparsity", {}))
    bridges_cfg = config_dict.get("bridges_config", config_dict.get("bridges", {}))
    training_cfg = config_dict.get("training_config", config_dict.get("training", {}))
    dense_cfg = config_dict.get("dense_model", {})
    
    # Create model config
    model_config = ModelConfig(
        n_layer=model_cfg["n_layer"],
        d_model=model_cfg["d_model"],
        n_ctx=model_cfg["n_ctx"],
        d_head=model_cfg["d_head"],
        d_mlp=model_cfg.get("d_mlp"),
        vocab_size=model_cfg["vocab_size"],
        use_rms_norm=model_cfg.get("use_rms_norm", True),
        tie_embeddings=model_cfg.get("tie_embeddings", False),
        use_positional_embeddings=model_cfg.get("use_positional_embeddings", False),
        use_bigram_table=model_cfg.get("use_bigram_table", False),
        use_attention_sinks=model_cfg.get("use_attention_sinks", True),
        activation=model_cfg.get("activation", "gelu"),
        dropout=model_cfg.get("dropout", 0.0),
        use_bias=model_cfg.get("use_bias", True),
    )
    
    # Create sparsity config
    sparsity_config = SparsityConfig(
        enable_weight_sparsity=sparsity_cfg.get("enable_weight_sparsity", True),
        target_l0_fraction=sparsity_cfg.get("target_l0_fraction", 0.015625),
        enable_activation_sparsity=sparsity_cfg.get("enable_activation_sparsity", True),
        activation_topk_fraction=sparsity_cfg.get("activation_topk_fraction", 0.25),
    )
    
    # Load dense model
    print("Loading dense model...")
    dense_model_source = dense_cfg.get("repo_id") or dense_cfg.get("local_path") or bridges_cfg.get("d_dense")
    if dense_model_source:
        if Path(dense_model_source).exists():
            # Load from local path
            from sparse_pretrain.src.train_bridges import load_dense_model
            config = FullBridgesConfig.from_yaml(str(config_path)) if config_path.exists() else None
            if config:
                dense_model = load_dense_model(config, device=device)
            else:
                raise ValueError("Cannot load dense model without full config")
        else:
            # Load from HuggingFace
            dense_model = SparseGPT.from_pretrained(dense_model_source, device=device)
    else:
        raise ValueError("Could not determine dense model source")
    
    dense_model.eval()
    for param in dense_model.parameters():
        param.requires_grad = False
    
    # Load sparse model
    print("Loading sparse model...")
    sparse_model = SparseGPT(model_config, sparsity_config)
    
    sparse_model_path = checkpoint_dir / "sparse_model.bin"
    if sparse_model_path.exists():
        state_dict = torch.load(sparse_model_path, map_location=device, weights_only=True)
        sparse_model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"sparse_model.bin not found in {checkpoint_dir}")
    
    sparse_model.to(device)
    sparse_model.eval()
    
    # Load bridges
    print("Loading bridges...")
    d_dense = dense_model.config.d_model
    d_sparse = model_config.d_model
    n_layers = model_config.n_layer
    
    encoder_afrac = bridges_cfg.get("encoder_afrac", 0.25)
    encoder_type = bridges_cfg.get("bridge_act_fn", "abstopk")
    init_log_eps = bridges_cfg.get("threshold_init_log_eps", -1.0)
    
    bridge_set = BridgeSet(
        n_layers=n_layers,
        d_dense=d_dense,
        d_sparse=d_sparse,
        encoder_afrac=encoder_afrac,
        encoder_type=encoder_type,
        init_log_eps=init_log_eps,
    )
    
    bridges_path = checkpoint_dir / "bridges.bin"
    if bridges_path.exists():
        state_dict = torch.load(bridges_path, map_location=device, weights_only=True)
        bridge_set.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"bridges.bin not found in {checkpoint_dir}")
    
    bridge_set.to(device)
    bridge_set.eval()
    
    return dense_model, sparse_model, bridge_set, config_dict


def run_evaluation(
    checkpoint_path: str,
    output_path: Optional[str] = None,
    num_batches: int = 10,
    batch_size: int = 16,
    device: str = "cuda",
    kl_topk: Optional[int] = None,
    dataset_name: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full path evaluation on a bridges checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory or HuggingFace repo ID
        output_path: Path to save results JSON
        num_batches: Number of batches to evaluate
        batch_size: Batch size
        device: Device to use
        kl_topk: Number of top tokens for KL approximation (default: from config or 64)
        dataset_name: Dataset to use (default: from config or "data/simplestories-tokenized")
        tokenizer_name: Tokenizer to use (default: from config or "SimpleStories/SimpleStories-1.25M")
        
    Returns:
        Dictionary with evaluation results
    """
    # Load checkpoint
    dense_model, sparse_model, bridge_set, config_dict = load_checkpoint(checkpoint_path, device)
    
    # Extract relevant config values
    bridges_cfg = config_dict.get("bridges_config", config_dict.get("bridges", {}))
    training_cfg = config_dict.get("training_config", config_dict.get("training", {}))
    model_cfg = config_dict.get("model_config", config_dict.get("sparse_model", {}))
    
    # Determine KL top-k
    if kl_topk is None:
        kl_topk = bridges_cfg.get("kl_approx_n") or 64
    
    print(f"Using KL top-k approximation: {kl_topk}")
    
    # Determine tokenizer
    if tokenizer_name is None:
        tokenizer_name = training_cfg.get("tokenizer_name", "SimpleStories/SimpleStories-1.25M")
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dataset
    if dataset_name is None:
        dataset_name = training_cfg.get("dataset_name", "data/simplestories-tokenized")
    
    # Get sequence length
    seq_length = model_cfg.get("n_ctx", 512)
    text_column = training_cfg.get("text_column", "story")
    val_split = training_cfg.get("val_split", "test")
    
    # Load validation data
    print(f"Loading validation data from: {dataset_name}")
    val_batches, val_desc = create_validation_data(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_length=seq_length,
        text_column=text_column,
        val_split=val_split,
        max_tokens=num_batches * batch_size * seq_length,
        seed=42,
    )
    print(f"  {val_desc}")
    print(f"  Loaded {len(val_batches)} validation batches")
    
    # Enumerate all paths
    n_sites = bridge_set.n_sites
    all_paths = enumerate_all_paths(n_sites)
    print(f"\nEnumerated {len(all_paths)} paths through {n_sites} bridge sites")
    
    # Initialize result accumulators
    path_results = {path.name: {
        "cross_entropy": 0.0,
        "kl_to_dense": 0.0,
        "kl_to_sparse": 0.0,
        "num_transitions": path.complexity,
        "path_ascii": path_to_compact_ascii(path, n_sites),
    } for path in all_paths}
    
    num_tokens = 0
    num_evaluated_batches = 0
    
    # Evaluate
    print("\nEvaluating paths...")
    with torch.amp.autocast(device, dtype=torch.bfloat16):
        for batch_idx, batch_tokens in enumerate(tqdm(val_batches[:num_batches], desc="Batches")):
            # Stack batch
            if isinstance(batch_tokens, list):
                input_ids = torch.stack(batch_tokens[:batch_size]).to(device)
            else:
                input_ids = batch_tokens.unsqueeze(0).to(device)
            
            # Pre-compute dense and sparse logits
            dense_logits, _, _ = dense_model(input_ids)
            sparse_logits, _, _ = sparse_model(input_ids)
            
            # Evaluate each path
            for path in all_paths:
                result = evaluate_path(
                    path=path,
                    input_ids=input_ids,
                    dense_logits=dense_logits,
                    sparse_logits=sparse_logits,
                    dense_model=dense_model,
                    sparse_model=sparse_model,
                    bridge_set=bridge_set,
                    kl_topk=kl_topk,
                )
                
                path_results[path.name]["cross_entropy"] += result.cross_entropy
                path_results[path.name]["kl_to_dense"] += result.kl_to_dense
                path_results[path.name]["kl_to_sparse"] += result.kl_to_sparse
            
            num_tokens += input_ids.numel()
            num_evaluated_batches += 1
    
    # Average results
    for name in path_results:
        path_results[name]["cross_entropy"] /= num_evaluated_batches
        path_results[name]["kl_to_dense"] /= num_evaluated_batches
        path_results[name]["kl_to_sparse"] /= num_evaluated_batches
        path_results[name]["perplexity"] = float(torch.exp(torch.tensor(path_results[name]["cross_entropy"])).item())
    
    # Sort by cross-entropy
    sorted_results = OrderedDict(
        sorted(path_results.items(), key=lambda x: x[1]["cross_entropy"])
    )
    
    # Compile final results
    results = {
        "checkpoint": str(checkpoint_path),
        "num_batches": num_evaluated_batches,
        "num_tokens": num_tokens,
        "kl_topk": kl_topk,
        "num_sites": n_sites,
        "num_layers": len(dense_model.blocks),
        "paths": sorted_results,
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("PATH EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Evaluated on {num_evaluated_batches} batches ({num_tokens:,} tokens)")
    print(f"KL approximation using top-{kl_topk} tokens")
    
    # Print training KLs first (the simple ones used in training)
    print("\n" + "-" * 80)
    print("TRAINING LOSS TERMS (KLs used during bridges training)")
    print("-" * 80)
    print(f"\n{'Path':<30} {'CE':>8} {'KL→Dense':>10}  Description")
    print("-" * 80)
    
    # The training uses: pure_sparse, d{i}->s (encode), s{i}->d (decode)
    training_paths = ["pure_sparse", "pure_dense"]
    for name in sorted_results:
        if name.startswith("d") and "->s" in name and "->" not in name[3:]:
            training_paths.append(name)
        elif name.startswith("s") and "->d" in name and "->" not in name[3:]:
            training_paths.append(name)
    
    for name in training_paths:
        if name in sorted_results:
            data = sorted_results[name]
            if name == "pure_sparse":
                desc = "Sparse model alone (coef_ce_sparse, coef_kl_sparse)"
            elif name == "pure_dense":
                desc = "Dense model alone (target)"
            elif name.startswith("d"):
                desc = f"Dense→Sparse hybrid (coef_kl_d2s)"
            else:
                desc = f"Sparse→Dense hybrid (coef_kl_s2d)"
            print(f"{name:<30} {data['cross_entropy']:>8.4f} {data['kl_to_dense']:>10.4f}  {desc}")
    
    # Print all paths
    print("\n" + "-" * 80)
    print("ALL PATHS (sorted by cross-entropy)")
    print("-" * 80)
    print(f"\n{'Path':<30} {'CE':>8} {'Perplexity':>12} {'KL→Dense':>10} {'KL→Sparse':>10}")
    print("-" * 80)
    
    for name, data in sorted_results.items():
        print(f"{name:<30} {data['cross_entropy']:>8.4f} {data['perplexity']:>12.2f} "
              f"{data['kl_to_dense']:>10.4f} {data['kl_to_sparse']:>10.4f}")
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Also print ASCII visualizations for the best paths
    print("\n" + "=" * 80)
    print("PATH VISUALIZATIONS (sorted by cross-entropy)")
    print("=" * 80)
    
    n_layers = len(dense_model.blocks)
    for i, (name, data) in enumerate(list(sorted_results.items())[:10]):
        path = next(p for p in all_paths if p.name == name)
        print(f"\n{i+1}. {name} (CE={data['cross_entropy']:.4f})")
        print(path_to_ascii(path, n_sites, n_layers))
    
    # Generate text file
    if output_path:
        txt_path = Path(output_path).with_suffix('.txt')
        generate_results_txt(results, all_paths, n_sites, n_layers, txt_path)
        print(f"Text summary saved to: {txt_path}")
    
    return results


def generate_results_txt(
    results: Dict[str, Any],
    all_paths: List[PathDescription],
    n_sites: int,
    n_layers: int,
    output_path: Path,
) -> None:
    """
    Generate a compact, readable text file with path evaluation results.
    Uses vertical ladder-style ASCII visualization.
    """
    lines = []
    sorted_results = results["paths"]
    
    # Header
    lines.append(results['checkpoint'].split('/')[-1])
    lines.append("")
    lines.append("CE   KL>D KL>S")
    lines.append("")
    
    def path_to_ladder(path: PathDescription) -> List[str]:
        """
        Create vertical ladder visualization with exactly n_sites lines.
        Left column = Dense, Right column = Sparse
        |     = on dense
            | = on sparse
        |___ = leaving dense (encode or double-back)
         ___| = leaving sparse (decode or double-back)
        """
        # Track which model is active at each site and transitions
        active = {}  # site -> model (the model we're on AT this site)
        trans = {}   # site -> list of transitions happening at this site
        
        for seg in path.segments:
            for s in range(seg.start_site, min(seg.end_site + 1, n_sites)):
                if s not in active:
                    active[s] = seg.model
        
        for site, t in path.transitions:
            if site not in trans:
                trans[site] = []
            trans[site].append(t)
        
        # Build visualization lines - exactly n_sites lines
        viz_lines = []
        viz_lines.append("D   S")
        
        for site in range(n_sites):
            model_here = active.get(site)
            
            if site in trans:
                t_list = trans[site]
                if len(t_list) >= 2:
                    # Double-back: show horizontal line from the side we're on
                    if model_here == "dense":
                        viz_lines.append("|___")
                    else:
                        viz_lines.append(" ___|")
                elif t_list[0] == "encode":
                    # Encode: leaving dense
                    viz_lines.append("|___")
                else:
                    # Decode: leaving sparse
                    viz_lines.append(" ___|")
            else:
                # No transition, just show which side we're on
                if model_here == "dense":
                    viz_lines.append("|    ")
                else:
                    viz_lines.append("    |")
        
        return viz_lines
    
    # Group paths
    pure = [p for p in all_paths if p.complexity == 0]
    single = [p for p in all_paths if p.complexity == 1]
    double = [p for p in all_paths if p.complexity == 2]
    
    def add_section(title: str, paths: List[PathDescription]):
        if not paths:
            return
        lines.append(f"--- {title} ---")
        lines.append("")
        sorted_paths = sorted(paths, key=lambda p: sorted_results.get(p.name, {}).get('cross_entropy', 999))
        for p in sorted_paths:
            if p.name not in sorted_results:
                continue
            data = sorted_results[p.name]
            # Metrics line
            lines.append(f"{data['cross_entropy']:.2f} {data['kl_to_dense']:.2f} {data['kl_to_sparse']:.2f}")
            # Ladder visualization
            ladder = path_to_ladder(p)
            for l in ladder:
                lines.append(l)
            lines.append("")
    
    add_section("Pure Models", pure)
    add_section("Single Transition", single)
    add_section("Double Transition", double)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all paths through dense and sparse models using bridges"
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to bridges checkpoint directory or HuggingFace repo ID"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: path_evaluation.json in current dir)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to evaluate (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--kl-topk",
        type=int,
        default=None,
        help="Number of top tokens for KL approximation (default: from config or 64)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to use (default: from config or data/simplestories-tokenized)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer to use (default: from config or SimpleStories/SimpleStories-1.25M)"
    )
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        if "/" in args.checkpoint and not Path(args.checkpoint).exists():
            # HuggingFace repo ID - save locally
            args.output = f"path_evaluation_{args.checkpoint.split('/')[-1]}.json"
        else:
            args.output = str(Path(args.checkpoint) / "path_evaluation.json")
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        device=args.device,
        kl_topk=args.kl_topk,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
    )


if __name__ == "__main__":
    main()

