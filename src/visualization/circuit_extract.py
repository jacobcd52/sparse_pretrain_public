"""
Circuit extraction from pruned MaskedSparseGPT models.

Extracts the computational graph of active nodes and their connecting edges
based on pruning masks and model weights.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class CircuitNode:
    """A node in the circuit graph."""
    
    node_id: str  # Unique identifier: f"{layer}_{location}_{index}"
    layer: int  # Layer index (0-based)
    location: str  # Node location type (e.g., "attn_q", "mlp_neuron", "resid_pre")
    index: int  # Index within the location (e.g., which dimension)
    
    # Optional metadata
    head_idx: Optional[int] = None  # For attention nodes: which head
    head_dim_idx: Optional[int] = None  # For attention nodes: which head dimension
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, CircuitNode):
            return False
        return self.node_id == other.node_id


@dataclass 
class CircuitEdge:
    """An edge connecting two circuit nodes."""
    
    source_id: str  # Source node ID
    target_id: str  # Target node ID
    weight: float  # Edge weight (from model weights or 1.0 for identity connections)
    edge_type: str  # Type of edge (e.g., "W_up", "W_V", "identity", "qk_match")
    
    def __hash__(self):
        return hash((self.source_id, self.target_id))
    
    def __eq__(self, other):
        if not isinstance(other, CircuitEdge):
            return False
        return self.source_id == other.source_id and self.target_id == other.target_id


@dataclass
class CircuitGraph:
    """The complete circuit graph extracted from a pruned model."""
    
    nodes: Dict[str, CircuitNode] = field(default_factory=dict)
    edges: List[CircuitEdge] = field(default_factory=list)
    
    # Model metadata
    n_layers: int = 0
    d_model: int = 0
    d_mlp: int = 0
    n_heads: int = 0
    d_head: int = 0
    
    # Mask locations used
    mask_locations: List[str] = field(default_factory=list)
    
    def add_node(self, node: CircuitNode):
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, edge: CircuitEdge):
        """Add an edge to the graph."""
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[CircuitNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_location(self, layer: int, location: str) -> List[CircuitNode]:
        """Get all nodes at a specific layer and location."""
        return [
            n for n in self.nodes.values()
            if n.layer == layer and n.location == location
        ]
    
    def get_edges_from(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source_id == node_id]
    
    def get_edges_to(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target_id == node_id]
    
    def summary(self) -> Dict:
        """Get a summary of the graph."""
        # Count nodes by location
        location_counts = {}
        for node in self.nodes.values():
            key = f"L{node.layer}_{node.location}"
            location_counts[key] = location_counts.get(key, 0) + 1
        
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "n_layers": self.n_layers,
            "location_counts": location_counts,
        }
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "layer": n.layer,
                    "location": n.location,
                    "index": n.index,
                    "head_idx": n.head_idx,
                    "head_dim_idx": n.head_dim_idx,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "weight": e.weight,
                    "edge_type": e.edge_type,
                }
                for e in self.edges
            ],
            "metadata": {
                "n_layers": self.n_layers,
                "d_model": self.d_model,
                "d_mlp": self.d_mlp,
                "n_heads": self.n_heads,
                "d_head": self.d_head,
                "mask_locations": self.mask_locations,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CircuitGraph":
        """Load from dictionary."""
        graph = cls()
        
        # Load metadata
        meta = data.get("metadata", {})
        graph.n_layers = meta.get("n_layers", 0)
        graph.d_model = meta.get("d_model", 0)
        graph.d_mlp = meta.get("d_mlp", 0)
        graph.n_heads = meta.get("n_heads", 0)
        graph.d_head = meta.get("d_head", 0)
        graph.mask_locations = meta.get("mask_locations", [])
        
        # Load nodes
        for n in data.get("nodes", []):
            node = CircuitNode(
                node_id=n["node_id"],
                layer=n["layer"],
                location=n["location"],
                index=n["index"],
                head_idx=n.get("head_idx"),
                head_dim_idx=n.get("head_dim_idx"),
            )
            graph.add_node(node)
        
        # Load edges
        for e in data.get("edges", []):
            edge = CircuitEdge(
                source_id=e["source_id"],
                target_id=e["target_id"],
                weight=e["weight"],
                edge_type=e["edge_type"],
            )
            graph.add_edge(edge)
        
        return graph


def _make_node_id(layer: int, location: str, index: int) -> str:
    """Create a unique node ID."""
    return f"{layer}_{location}_{index}"


def _get_active_indices(mask_tensor: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """Get indices where mask value > threshold (i.e., active nodes).
    
    For binary masks (0 or 1), threshold=0.5 correctly identifies active nodes.
    """
    return (mask_tensor > threshold).nonzero(as_tuple=True)[0].tolist()


def extract_circuit(
    checkpoint_path: str,
    edge_threshold: float = 0.0,
    include_resid_nodes: bool = True,
    device: str = "cpu",
) -> CircuitGraph:
    """
    Extract circuit graph from a pruned model checkpoint.
    
    Args:
        checkpoint_path: Path to the pruning output directory containing:
            - binary_masks.pt or checkpoint.pt with mask state
            - Model weights (loaded from original model path in results.json)
        edge_threshold: Minimum absolute weight to include an edge
        include_resid_nodes: Whether to include residual stream nodes
        device: Device for loading tensors
        
    Returns:
        CircuitGraph with active nodes and their connecting edges
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load masks
    binary_masks_path = checkpoint_path / "binary_masks.pt"
    checkpoint_pt_path = checkpoint_path / "checkpoint.pt"
    
    if binary_masks_path.exists():
        masks = torch.load(binary_masks_path, map_location=device, weights_only=True)
    elif checkpoint_pt_path.exists():
        checkpoint = torch.load(checkpoint_pt_path, map_location=device, weights_only=True)
        # Extract mask state from checkpoint
        masks = {}
        if "mask_state" in checkpoint:
            for key, tau in checkpoint["mask_state"].items():
                # Convert tau to binary mask
                masks[key] = (tau >= 0).float()
        else:
            raise ValueError(f"Checkpoint at {checkpoint_pt_path} does not contain mask_state")
    else:
        raise FileNotFoundError(
            f"Could not find masks. Expected either:\n"
            f"  - {binary_masks_path}\n"
            f"  - {checkpoint_pt_path}"
        )
    
    # Load results to get model path
    results_path = checkpoint_path / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Could not find {results_path}")
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    model_path = results.get("model_path")
    if not model_path:
        raise ValueError("results.json does not contain model_path")
    
    # Load model
    from ..pruning.run_pruning import load_model
    model, config_dict = load_model(model_path, device=device)
    
    # Get model dimensions
    model_config = model.config
    n_layers = model_config.n_layer
    d_model = model_config.d_model
    d_mlp = model_config.d_mlp
    n_heads = model_config.n_heads
    d_head = model_config.d_head
    
    # Determine mask locations from the loaded masks
    mask_locations = set()
    for key in masks.keys():
        # Key format: "layer{i}_{location}"
        parts = key.split("_", 1)
        if len(parts) == 2:
            loc = parts[1]
            mask_locations.add(loc)
    mask_locations = sorted(list(mask_locations))
    
    # Create graph
    graph = CircuitGraph(
        n_layers=n_layers,
        d_model=d_model,
        d_mlp=d_mlp,
        n_heads=n_heads,
        d_head=d_head,
        mask_locations=mask_locations,
    )
    
    # === Step 1: Create nodes for all active mask elements ===
    
    for layer in range(n_layers):
        for loc in mask_locations:
            key = f"layer{layer}_{loc}"
            if key not in masks:
                continue
            
            mask = masks[key]
            active_indices = _get_active_indices(mask)
            
            for idx in active_indices:
                # Determine head info for attention nodes
                head_idx = None
                head_dim_idx = None
                if loc in ["attn_q", "attn_k", "attn_v"]:
                    head_idx = idx // d_head
                    head_dim_idx = idx % d_head
                
                node = CircuitNode(
                    node_id=_make_node_id(layer, loc, idx),
                    layer=layer,
                    location=loc,
                    index=idx,
                    head_idx=head_idx,
                    head_dim_idx=head_dim_idx,
                )
                graph.add_node(node)
    
    # === Step 2: Create residual nodes connected to active nodes ===
    
    if include_resid_nodes:
        # Track which resid dimensions are needed
        resid_pre_needed = {}  # layer -> set of indices
        resid_mid_needed = {}  # layer -> set of indices
        resid_post_needed = {}  # layer -> set of indices (same as next layer's resid_pre)
        
        for layer in range(n_layers):
            resid_pre_needed[layer] = set()
            resid_mid_needed[layer] = set()
            resid_post_needed[layer] = set()
        
        # resid_pre connects to attn_in (d_model)
        # attn_in connects to Q, K, V via W_QKV
        # attn_out connects to resid_mid (d_model)
        # resid_mid connects to mlp_in (d_model)
        # mlp_in connects to mlp_neuron via W_up
        # mlp_out connects to resid_post (d_model)
        
        for node in list(graph.nodes.values()):
            layer = node.layer
            loc = node.location
            idx = node.index
            
            # attn_in nodes need corresponding resid_pre
            if loc == "attn_in":
                resid_pre_needed[layer].add(idx)
            
            # attn_out nodes contribute to resid_mid
            if loc == "attn_out":
                resid_mid_needed[layer].add(idx)
            
            # mlp_in nodes need corresponding resid_mid
            if loc == "mlp_in":
                resid_mid_needed[layer].add(idx)
            
            # mlp_out nodes contribute to resid_post
            if loc == "mlp_out":
                resid_post_needed[layer].add(idx)
        
        # Create resid nodes
        for layer in range(n_layers):
            for idx in resid_pre_needed[layer]:
                node = CircuitNode(
                    node_id=_make_node_id(layer, "resid_pre", idx),
                    layer=layer,
                    location="resid_pre",
                    index=idx,
                )
                graph.add_node(node)
            
            for idx in resid_mid_needed[layer]:
                node = CircuitNode(
                    node_id=_make_node_id(layer, "resid_mid", idx),
                    layer=layer,
                    location="resid_mid",
                    index=idx,
                )
                graph.add_node(node)
            
            for idx in resid_post_needed[layer]:
                node = CircuitNode(
                    node_id=_make_node_id(layer, "resid_post", idx),
                    layer=layer,
                    location="resid_post",
                    index=idx,
                )
                graph.add_node(node)
    
    # === Step 3: Create edges based on model weights ===
    
    for layer in range(n_layers):
        block = model.blocks[layer]
        
        # --- Attention edges ---
        
        # W_QKV: attn_in -> (Q, K, V)
        # c_attn weight shape: [3 * n_heads * d_head, d_model]
        W_QKV = block.attn.c_attn.weight.data  # [3*H*D, d_model]
        qkv_dim = n_heads * d_head
        W_Q = W_QKV[:qkv_dim, :]  # [H*D, d_model]
        W_K = W_QKV[qkv_dim:2*qkv_dim, :]
        W_V = W_QKV[2*qkv_dim:3*qkv_dim, :]
        
        # attn_in -> attn_q via W_Q
        _add_weight_edges(
            graph, layer, layer,
            "attn_in", "attn_q",
            W_Q.T,  # [d_model, H*D]
            edge_threshold, "W_Q"
        )
        
        # attn_in -> attn_k via W_K
        _add_weight_edges(
            graph, layer, layer,
            "attn_in", "attn_k",
            W_K.T,
            edge_threshold, "W_K"
        )
        
        # attn_in -> attn_v via W_V
        _add_weight_edges(
            graph, layer, layer,
            "attn_in", "attn_v",
            W_V.T,
            edge_threshold, "W_V"
        )
        
        # Q-K matching: q and k at same head_idx and head_dim
        # This is implicit in attention computation
        for q_node in graph.get_nodes_by_location(layer, "attn_q"):
            for k_node in graph.get_nodes_by_location(layer, "attn_k"):
                if (q_node.head_idx == k_node.head_idx and 
                    q_node.head_dim_idx == k_node.head_dim_idx):
                    # They interact in attention score computation
                    edge = CircuitEdge(
                        source_id=q_node.node_id,
                        target_id=k_node.node_id,
                        weight=1.0,
                        edge_type="qk_match",
                    )
                    graph.add_edge(edge)
        
        # V -> attention output contribution (per head)
        # W_O: [H*D, d_model]
        W_O = block.attn.c_proj.weight.data  # [d_model, H*D]
        
        # attn_v -> attn_out via W_O (transposed since it's the output projection)
        _add_weight_edges(
            graph, layer, layer,
            "attn_v", "attn_out",
            W_O.T,  # [H*D, d_model]
            edge_threshold, "W_O"
        )
        
        # --- MLP edges ---
        
        # W_up (c_fc): mlp_in -> mlp_neuron
        W_up = block.mlp.c_fc.weight.data  # [d_mlp, d_model]
        _add_weight_edges(
            graph, layer, layer,
            "mlp_in", "mlp_neuron",
            W_up.T,  # [d_model, d_mlp]
            edge_threshold, "W_up"
        )
        
        # W_down (c_proj): mlp_neuron -> mlp_out
        W_down = block.mlp.c_proj.weight.data  # [d_model, d_mlp]
        _add_weight_edges(
            graph, layer, layer,
            "mlp_neuron", "mlp_out",
            W_down.T,  # [d_mlp, d_model]
            edge_threshold, "W_down"
        )
        
        # --- Residual stream edges (identity connections) ---
        
        if include_resid_nodes:
            # resid_pre -> attn_in (through layer norm, weight ~1)
            _add_identity_edges(
                graph, layer, layer,
                "resid_pre", "attn_in",
                "ln1"
            )
            
            # attn_out -> resid_mid (residual add)
            _add_identity_edges(
                graph, layer, layer,
                "attn_out", "resid_mid",
                "resid_add"
            )
            
            # resid_pre also flows to resid_mid (skip connection)
            _add_identity_edges(
                graph, layer, layer,
                "resid_pre", "resid_mid",
                "skip"
            )
            
            # resid_mid -> mlp_in (through layer norm)
            _add_identity_edges(
                graph, layer, layer,
                "resid_mid", "mlp_in",
                "ln2"
            )
            
            # mlp_out -> resid_post (residual add)
            _add_identity_edges(
                graph, layer, layer,
                "mlp_out", "resid_post",
                "resid_add"
            )
            
            # resid_mid also flows to resid_post (skip connection)
            _add_identity_edges(
                graph, layer, layer,
                "resid_mid", "resid_post",
                "skip"
            )
            
            # Cross-layer: resid_post of layer L -> resid_pre of layer L+1
            if layer < n_layers - 1:
                _add_identity_edges(
                    graph, layer, layer + 1,
                    "resid_post", "resid_pre",
                    "cross_layer"
                )
    
    return graph


def _add_weight_edges(
    graph: CircuitGraph,
    src_layer: int,
    tgt_layer: int,
    src_loc: str,
    tgt_loc: str,
    weight_matrix: torch.Tensor,
    threshold: float,
    edge_type: str,
):
    """
    Add edges between nodes based on weight matrix entries.
    
    Args:
        graph: The circuit graph
        src_layer, tgt_layer: Source and target layer indices
        src_loc, tgt_loc: Source and target location types
        weight_matrix: Weight matrix [src_dim, tgt_dim]
        threshold: Minimum absolute weight to include
        edge_type: Type label for the edge
    """
    src_nodes = graph.get_nodes_by_location(src_layer, src_loc)
    tgt_nodes = graph.get_nodes_by_location(tgt_layer, tgt_loc)
    
    if not src_nodes or not tgt_nodes:
        return
    
    # Build index maps for efficiency
    src_idx_to_node = {n.index: n for n in src_nodes}
    tgt_idx_to_node = {n.index: n for n in tgt_nodes}
    
    # Find nonzero weights connecting active nodes
    weight_matrix = weight_matrix.cpu()
    
    for src_idx, src_node in src_idx_to_node.items():
        for tgt_idx, tgt_node in tgt_idx_to_node.items():
            if src_idx < weight_matrix.shape[0] and tgt_idx < weight_matrix.shape[1]:
                w = weight_matrix[src_idx, tgt_idx].item()
                if abs(w) > threshold:
                    edge = CircuitEdge(
                        source_id=src_node.node_id,
                        target_id=tgt_node.node_id,
                        weight=w,
                        edge_type=edge_type,
                    )
                    graph.add_edge(edge)


def _add_identity_edges(
    graph: CircuitGraph,
    src_layer: int,
    tgt_layer: int,
    src_loc: str,
    tgt_loc: str,
    edge_type: str,
):
    """
    Add identity edges (weight=1) between nodes at the same index.
    Used for residual connections and layer norm pass-through.
    """
    src_nodes = graph.get_nodes_by_location(src_layer, src_loc)
    tgt_nodes = graph.get_nodes_by_location(tgt_layer, tgt_loc)
    
    if not src_nodes or not tgt_nodes:
        return
    
    # Build target index map
    tgt_idx_to_node = {n.index: n for n in tgt_nodes}
    
    for src_node in src_nodes:
        if src_node.index in tgt_idx_to_node:
            tgt_node = tgt_idx_to_node[src_node.index]
            edge = CircuitEdge(
                source_id=src_node.node_id,
                target_id=tgt_node.node_id,
                weight=1.0,
                edge_type=edge_type,
            )
            graph.add_edge(edge)


def save_circuit(graph: CircuitGraph, path: str):
    """Save circuit graph to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(graph.to_dict(), f, indent=2)


def load_circuit(path: str) -> CircuitGraph:
    """Load circuit graph from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return CircuitGraph.from_dict(data)

