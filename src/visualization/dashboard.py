"""
Dashboard generation for circuit nodes.

Computes top and bottom activating examples for each node in the circuit,
similar to SAE feature dashboards but for sparse circuit nodes.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Union

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .circuit_extract import CircuitGraph, CircuitNode


@dataclass
class ActivationExample:
    """A single activation example for a node."""
    
    tokens: List[str]  # String tokens
    token_ids: List[int]  # Token IDs
    activations: List[float]  # Activation value at each position
    max_activation: float  # Maximum activation value
    max_position: int  # Position of maximum activation
    
    # For negative examples
    min_activation: float = 0.0
    min_position: int = 0


@dataclass
class NodeDashboardData:
    """Dashboard data for a single circuit node."""
    
    node_id: str
    layer: int
    location: str
    index: int
    
    # Top activating examples (most positive)
    top_examples: List[ActivationExample] = field(default_factory=list)
    
    # Bottom activating examples (most negative)
    bottom_examples: List[ActivationExample] = field(default_factory=list)
    
    # Statistics
    mean_activation: float = 0.0
    std_activation: float = 0.0
    frequency: float = 0.0  # Fraction of tokens with nonzero activation
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "node_id": self.node_id,
            "layer": self.layer,
            "location": self.location,
            "index": self.index,
            "top_examples": [
                {
                    "tokens": ex.tokens,
                    "activations": ex.activations,
                    "max_activation": ex.max_activation,
                    "max_position": ex.max_position,
                }
                for ex in self.top_examples
            ],
            "bottom_examples": [
                {
                    "tokens": ex.tokens,
                    "activations": ex.activations,
                    "min_activation": ex.min_activation,
                    "min_position": ex.min_position,
                }
                for ex in self.bottom_examples
            ],
            "mean_activation": self.mean_activation,
            "std_activation": self.std_activation,
            "frequency": self.frequency,
        }


def _create_simplestories_iterator(
    tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 256,
    n_samples: int = 1000,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    """
    Create an iterator over SimpleStories dataset using sequence packing.
    
    Uses the same strategy as pretraining: texts are concatenated with EOS tokens
    between them, then chunked into max_seq_len pieces. NO PADDING.
    
    Yields batches of tokenized sequences.
    """
    from datasets import load_dataset
    
    dataset = load_dataset(
        "SimpleStories/SimpleStories",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    eos_token_id = tokenizer.eos_token_id
    token_buffer = []
    batches_yielded = 0
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for example in dataset:
        if batches_yielded >= n_batches:
            break
        
        text = example.get("story", "")
        if not text:
            continue
        
        # Tokenize without special tokens (matching pretraining)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue
        
        # Add tokens to buffer
        token_buffer.extend(tokens)
        
        # Add EOS between examples (matching pretraining)
        if eos_token_id is not None:
            token_buffer.append(eos_token_id)
        
        # Yield complete batches when we have enough tokens
        while len(token_buffer) >= batch_size * max_seq_len:
            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:max_seq_len]
                token_buffer = token_buffer[max_seq_len:]
                batch.append(torch.tensor(chunk, dtype=torch.long))
            
            yield torch.stack(batch)
            batches_yielded += 1
            
            if batches_yielded >= n_batches:
                break


def _create_task_iterator(
    task,
    batch_size: int = 32,
    max_seq_len: int = 256,
    n_samples: int = 1000,
) -> Iterator[torch.Tensor]:
    """
    Create an iterator from a task's generate_batch method.
    
    Yields batches of tokenized sequences (positive examples only).
    """
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for _ in range(n_batches):
        positive_ids, _, _, _, _ = task.generate_batch(batch_size, max_seq_len)
        yield positive_ids


def _hook_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    locations: List[str],
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Run model forward pass and capture activations at specified locations.
    
    Returns dict mapping "layer{i}_{location}" -> activation tensor [batch, seq, dim]
    """
    captured = {}
    hooks = []
    
    # We need to hook into the masked model's forward
    # The locations are captured during the block forward
    
    B, T = input_ids.shape
    
    # Manual forward with activation capture
    x = model.model.wte(input_ids.to(device))
    
    if torch.is_autocast_enabled('cuda'):
        x = x.to(torch.get_autocast_dtype('cuda'))
    
    if model.model.wpe is not None:
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + model.model.wpe(pos)
    
    x = model.model.drop(x)
    
    for layer_idx, block in enumerate(model.model.blocks):
        # Capture at each location
        
        # attn_in (after ln_1)
        normed = block.ln_1(x)
        if "attn_in" in locations:
            captured[f"layer{layer_idx}_attn_in"] = normed.detach().cpu()
        
        # Q, K, V (after projection)
        qkv = block.attn.c_attn(normed)
        n_heads = block.attn.n_heads
        d_head = block.attn.d_head
        q, k, v = qkv.split(n_heads * d_head, dim=-1)
        
        if "attn_q" in locations:
            captured[f"layer{layer_idx}_attn_q"] = q.detach().cpu()
        if "attn_k" in locations:
            captured[f"layer{layer_idx}_attn_k"] = k.detach().cpu()
        if "attn_v" in locations:
            captured[f"layer{layer_idx}_attn_v"] = v.detach().cpu()
        
        # Run attention
        attn_out = block.attn(normed, None)
        if "attn_out" in locations:
            captured[f"layer{layer_idx}_attn_out"] = attn_out.detach().cpu()
        
        x = x + attn_out
        
        # mlp_in (after ln_2)
        normed = block.ln_2(x)
        if "mlp_in" in locations:
            captured[f"layer{layer_idx}_mlp_in"] = normed.detach().cpu()
        
        # mlp_neuron (after activation)
        mlp_hidden = block.mlp.c_fc(normed)
        mlp_hidden = block.mlp.act_fn(mlp_hidden)
        if "mlp_neuron" in locations:
            captured[f"layer{layer_idx}_mlp_neuron"] = mlp_hidden.detach().cpu()
        
        mlp_out = block.mlp.c_proj(mlp_hidden)
        mlp_out = block.mlp.dropout(mlp_out)
        if "mlp_out" in locations:
            captured[f"layer{layer_idx}_mlp_out"] = mlp_out.detach().cpu()
        
        x = x + mlp_out
    
    return captured


def compute_dashboards(
    graph: CircuitGraph,
    model: nn.Module,
    tokenizer,
    data_source: str = "simplestories",
    task = None,
    n_samples: int = 1000,
    batch_size: int = 32,
    max_seq_len: int = 256,
    k_top: int = 10,
    k_bottom: int = 10,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, NodeDashboardData]:
    """
    Compute dashboard data for all nodes in the circuit graph.
    
    Uses a two-pass approach for efficiency:
    1. First pass: collect max/min activations per sample (just indices, not full sequences)
    2. Second pass: retrieve full sequences only for top-k examples
    
    Args:
        graph: The circuit graph with active nodes
        model: The MaskedSparseGPT model (or underlying SparseGPT)
        tokenizer: Tokenizer for decoding
        data_source: "simplestories" or "task"
        task: Task object if data_source="task"
        n_samples: Number of samples to process
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        k_top: Number of top examples to keep
        k_bottom: Number of bottom examples to keep
        device: Device for computation
        verbose: Show progress bars
        
    Returns:
        Dictionary mapping node_id -> NodeDashboardData
    """
    # Get the base model (handle MaskedSparseGPT wrapper)
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    base_model.eval()
    base_model.to(device)
    
    # Determine which locations we need to capture
    locations = set()
    for node in graph.nodes.values():
        if node.location not in ["resid_pre", "resid_mid", "resid_post"]:
            locations.add(node.location)
    locations = list(locations)
    
    # Build a mapping from node_id to a contiguous index for efficient tensor storage
    node_list = []
    node_id_to_idx = {}
    for node in graph.nodes.values():
        if node.location not in ["resid_pre", "resid_mid", "resid_post"]:
            node_id_to_idx[node.node_id] = len(node_list)
            node_list.append(node)
    
    n_nodes = len(node_list)
    if n_nodes == 0:
        return {}
    
    # Group nodes by location key for efficient activation extraction
    nodes_by_loc = {}
    for node in node_list:
        key = f"layer{node.layer}_{node.location}"
        if key not in nodes_by_loc:
            nodes_by_loc[key] = []
        nodes_by_loc[key].append(node)
    
    # ========== FIRST PASS: Collect all samples and compute statistics ==========
    # Store all input_ids for later retrieval
    all_input_ids = []
    
    # Statistics: [n_nodes] tensors
    sum_acts = torch.zeros(n_nodes, dtype=torch.float64)
    sum_sq_acts = torch.zeros(n_nodes, dtype=torch.float64)
    count_nonzero = torch.zeros(n_nodes, dtype=torch.int64)
    total_tokens = 0
    
    # Max/min per sample: [n_samples, n_nodes] - we'll build these as lists then stack
    all_max_vals = []
    all_max_pos = []
    all_min_vals = []
    all_min_pos = []
    
    # Create data iterator
    if data_source == "simplestories":
        data_iter = _create_simplestories_iterator(
            tokenizer, batch_size, max_seq_len, n_samples
        )
    elif data_source == "task" and task is not None:
        data_iter = _create_task_iterator(task, batch_size, max_seq_len, n_samples)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Pass 1: Collecting activations for {n_nodes} nodes over {n_samples} samples...")
    iterator = tqdm(data_iter, total=n_batches, desc="Collecting activations") if verbose else data_iter
    
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for batch_idx, input_ids in enumerate(iterator):
            if batch_idx >= n_batches:
                break
            
            input_ids = input_ids.to(device)
            B, T = input_ids.shape
            total_tokens += B * T
            
            # Store input_ids for later
            all_input_ids.append(input_ids.cpu())
            
            # Capture activations - returns dict of [B, T, dim] tensors
            acts = _hook_activations_simple(base_model, input_ids, locations, device)
            
            # Initialize batch tensors for max/min
            batch_max_vals = torch.zeros(B, n_nodes)
            batch_max_pos = torch.zeros(B, n_nodes, dtype=torch.long)
            batch_min_vals = torch.zeros(B, n_nodes)
            batch_min_pos = torch.zeros(B, n_nodes, dtype=torch.long)
            
            # Process each location - extract relevant node activations
            for loc_key, nodes in nodes_by_loc.items():
                if loc_key not in acts:
                    continue
                
                act_tensor = acts[loc_key]  # [B, T, dim]
                
                # Get indices for all nodes in this location
                node_indices = torch.tensor([node.index for node in nodes], dtype=torch.long)
                node_global_indices = torch.tensor([node_id_to_idx[node.node_id] for node in nodes], dtype=torch.long)
                
                # Extract activations for all relevant indices at once: [B, T, n_loc_nodes]
                node_acts = act_tensor[:, :, node_indices]  # [B, T, n_loc_nodes]
                
                # Compute statistics (vectorized) - use float() to convert from fp16
                sum_acts[node_global_indices] += node_acts.float().sum(dim=(0, 1)).double().cpu()
                sum_sq_acts[node_global_indices] += (node_acts.float() ** 2).sum(dim=(0, 1)).double().cpu()
                count_nonzero[node_global_indices] += (node_acts != 0).sum(dim=(0, 1)).long().cpu()
                
                # Find max/min per sequence: [B, n_loc_nodes]
                max_vals, max_pos = node_acts.max(dim=1)
                min_vals, min_pos = node_acts.min(dim=1)
                
                batch_max_vals[:, node_global_indices] = max_vals.float().cpu()
                batch_max_pos[:, node_global_indices] = max_pos.cpu()
                batch_min_vals[:, node_global_indices] = min_vals.float().cpu()
                batch_min_pos[:, node_global_indices] = min_pos.cpu()
            
            all_max_vals.append(batch_max_vals)
            all_max_pos.append(batch_max_pos)
            all_min_vals.append(batch_min_vals)
            all_min_pos.append(batch_min_pos)
    
    # Stack all batches: [n_samples, n_nodes]
    all_input_ids = torch.cat(all_input_ids, dim=0)  # [n_samples, T]
    all_max_vals = torch.cat(all_max_vals, dim=0)  # [n_samples, n_nodes]
    all_max_pos = torch.cat(all_max_pos, dim=0)
    all_min_vals = torch.cat(all_min_vals, dim=0)
    all_min_pos = torch.cat(all_min_pos, dim=0)
    
    actual_n_samples = all_max_vals.shape[0]
    
    # ========== SECOND PASS: Find top-k and extract examples ==========
    if verbose:
        print(f"Pass 2: Finding top-{k_top} examples for each node...")
    
    # Use torch.topk to find top-k and bottom-k sample indices for each node
    # For top-k: highest max activations
    # For bottom-k: lowest min activations
    
    effective_k_top = min(k_top, actual_n_samples)
    effective_k_bottom = min(k_bottom, actual_n_samples)
    
    # top-k indices per node: [k_top, n_nodes]
    top_vals, top_sample_idx = torch.topk(all_max_vals, effective_k_top, dim=0)
    
    # bottom-k: find samples with lowest min values
    bottom_vals, bottom_sample_idx = torch.topk(-all_min_vals, effective_k_bottom, dim=0)
    bottom_vals = -bottom_vals  # Convert back to actual min values
    
    # ========== BUILD DASHBOARDS ==========
    if verbose:
        print("Building dashboard data...")
    
    dashboards = {}
    
    node_iterator = tqdm(enumerate(node_list), total=n_nodes, desc="Building dashboards") if verbose else enumerate(node_list)
    
    for node_idx, node in node_iterator:
        node_id = node.node_id
        
        # Compute statistics
        mean_act = sum_acts[node_idx].item() / total_tokens if total_tokens > 0 else 0
        var_act = sum_sq_acts[node_idx].item() / total_tokens - mean_act ** 2 if total_tokens > 0 else 0
        std_act = max(0, var_act) ** 0.5
        frequency = count_nonzero[node_idx].item() / total_tokens if total_tokens > 0 else 0
        
        # Extract top examples
        top_examples = []
        for k_idx in range(effective_k_top):
            sample_idx = top_sample_idx[k_idx, node_idx].item()
            max_val = top_vals[k_idx, node_idx].item()
            max_pos = all_max_pos[sample_idx, node_idx].item()
            
            # Get the actual tokens for this sample
            tok_ids = all_input_ids[sample_idx].tolist()
            str_tokens = [tokenizer.decode([t]) for t in tok_ids]
            
            # We don't store per-position activations anymore to save memory
            # Just store placeholder - the max position is what matters for highlighting
            top_examples.append(ActivationExample(
                tokens=str_tokens,
                token_ids=tok_ids,
                activations=[],  # Empty - we only care about max position for display
                max_activation=max_val,
                max_position=max_pos,
            ))
        
        # Extract bottom examples
        bottom_examples = []
        for k_idx in range(effective_k_bottom):
            sample_idx = bottom_sample_idx[k_idx, node_idx].item()
            min_val = bottom_vals[k_idx, node_idx].item()
            min_pos = all_min_pos[sample_idx, node_idx].item()
            
            # Get the actual tokens for this sample
            tok_ids = all_input_ids[sample_idx].tolist()
            str_tokens = [tokenizer.decode([t]) for t in tok_ids]
            
            bottom_examples.append(ActivationExample(
                tokens=str_tokens,
                token_ids=tok_ids,
                activations=[],
                max_activation=min_val,
                max_position=min_pos,
                min_activation=min_val,
                min_position=min_pos,
            ))
        
        dashboards[node_id] = NodeDashboardData(
            node_id=node_id,
            layer=node.layer,
            location=node.location,
            index=node.index,
            top_examples=top_examples,
            bottom_examples=bottom_examples,
            mean_activation=mean_act,
            std_activation=std_act,
            frequency=frequency,
        )
    
    return dashboards


def _hook_activations_simple(
    model: nn.Module,
    input_ids: torch.Tensor,
    locations: List[str],
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Simplified activation capture that works with SparseGPT directly.
    """
    captured = {}
    
    B, T = input_ids.shape
    
    # Token embeddings
    x = model.wte(input_ids)
    
    if torch.is_autocast_enabled('cuda'):
        x = x.to(torch.get_autocast_dtype('cuda'))
    
    if model.wpe is not None:
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        x = x + model.wpe(pos)
    
    x = model.drop(x)
    
    for layer_idx, block in enumerate(model.blocks):
        # attn_in (after ln_1)
        normed = block.ln_1(x)
        if "attn_in" in locations:
            captured[f"layer{layer_idx}_attn_in"] = normed.detach().cpu()
        
        # Q, K, V (after projection)
        qkv = block.attn.c_attn(normed)
        n_heads = block.attn.n_heads
        d_head = block.attn.d_head
        q, k, v = qkv.split(n_heads * d_head, dim=-1)
        
        if "attn_q" in locations:
            captured[f"layer{layer_idx}_attn_q"] = q.detach().cpu()
        if "attn_k" in locations:
            captured[f"layer{layer_idx}_attn_k"] = k.detach().cpu()
        if "attn_v" in locations:
            captured[f"layer{layer_idx}_attn_v"] = v.detach().cpu()
        
        # Run attention
        attn_out = block.attn(normed, None)
        if "attn_out" in locations:
            captured[f"layer{layer_idx}_attn_out"] = attn_out.detach().cpu()
        
        x = x + attn_out
        
        # mlp_in (after ln_2)
        normed = block.ln_2(x)
        if "mlp_in" in locations:
            captured[f"layer{layer_idx}_mlp_in"] = normed.detach().cpu()
        
        # mlp_neuron (after activation)
        mlp_hidden = block.mlp.c_fc(normed)
        mlp_hidden = block.mlp.act_fn(mlp_hidden)
        if "mlp_neuron" in locations:
            captured[f"layer{layer_idx}_mlp_neuron"] = mlp_hidden.detach().cpu()
        
        mlp_out = block.mlp.c_proj(mlp_hidden)
        mlp_out = block.mlp.dropout(mlp_out)
        if "mlp_out" in locations:
            captured[f"layer{layer_idx}_mlp_out"] = mlp_out.detach().cpu()
        
        x = x + mlp_out
    
    return captured


def save_dashboards(
    dashboards: Dict[str, NodeDashboardData],
    path: str,
):
    """Save dashboard data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {node_id: dash.to_dict() for node_id, dash in dashboards.items()}
    
    with open(path, "w") as f:
        json.dump(data, f)


def load_dashboards(path: str) -> Dict[str, NodeDashboardData]:
    """Load dashboard data from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    
    dashboards = {}
    for node_id, d in data.items():
        dash = NodeDashboardData(
            node_id=d["node_id"],
            layer=d["layer"],
            location=d["location"],
            index=d["index"],
            mean_activation=d.get("mean_activation", 0.0),
            std_activation=d.get("std_activation", 0.0),
            frequency=d.get("frequency", 0.0),
        )
        
        # Load top examples
        for ex in d.get("top_examples", []):
            dash.top_examples.append(ActivationExample(
                tokens=ex["tokens"],
                token_ids=ex.get("token_ids", []),
                activations=ex["activations"],
                max_activation=ex["max_activation"],
                max_position=ex["max_position"],
            ))
        
        # Load bottom examples
        for ex in d.get("bottom_examples", []):
            dash.bottom_examples.append(ActivationExample(
                tokens=ex["tokens"],
                token_ids=ex.get("token_ids", []),
                activations=ex["activations"],
                min_activation=ex.get("min_activation", 0.0),
                min_position=ex.get("min_position", 0),
            ))
        
        dashboards[node_id] = dash
    
    return dashboards

