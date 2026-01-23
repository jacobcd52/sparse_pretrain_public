"""
Generate Pareto curves of cross-entropy loss vs circuit size.

This module prunes attribution graphs to various sizes and measures the
cross-entropy loss at each size, producing a Pareto frontier of
accuracy vs sparsity.

The workflow:
1. Build an attribution graph with top-K logits as terminal nodes
2. Prune the graph at various edge thresholds (node_threshold=1)
3. For each pruned graph, run the forward pass to get logits
4. Compute cross-entropy loss using softmax over the K logits
5. Plot CE loss vs circuit size (number of edges or nodes)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ParetoPoint:
    """A single point on the Pareto curve."""
    edge_threshold: float
    node_threshold: float
    n_edges: int
    n_nodes: int
    n_features: int
    ce_loss: float
    accuracy: float  # Whether the top logit is correct
    top_logit_token: int
    target_token: int


@dataclass 
class ParetoResult:
    """Results from a Pareto curve computation."""
    points: List[ParetoPoint]
    prompt: str
    target_token: int
    baseline_ce_loss: float  # CE loss with full (unpruned) graph
    true_ce_loss: float  # CE loss from the actual model
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "target_token": self.target_token,
            "baseline_ce_loss": self.baseline_ce_loss,
            "true_ce_loss": self.true_ce_loss,
            "points": [
                {
                    "edge_threshold": p.edge_threshold,
                    "n_edges": p.n_edges,
                    "n_nodes": p.n_nodes,
                    "n_features": p.n_features,
                    "ce_loss": p.ce_loss,
                    "accuracy": p.accuracy,
                    "top_logit_token": p.top_logit_token,
                    "target_token": p.target_token,
                }
                for p in self.points
            ],
        }


class GraphPareto:
    """
    Compute Pareto curves of CE loss vs circuit size.
    
    Example:
        >>> gp = GraphPareto(replacement_model, transcoders)
        >>> result = gp.compute_pareto(prompt, target_token, n_thresholds=20)
        >>> gp.plot_pareto(result)
    """
    
    def __init__(
        self,
        replacement_model,
        transcoders,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            replacement_model: SparseGPTReplacementModel
            transcoders: List of CrossLayerTranscoder or CLTAdapter
            device: Device to run on
            dtype: Data type
        """
        self.replacement_model = replacement_model
        self.model = replacement_model.model
        self.cfg = replacement_model.cfg
        self.device = device
        self.dtype = dtype
        
        if hasattr(transcoders, 'transcoders'):
            self.transcoders = transcoders.transcoders
        else:
            self.transcoders = transcoders
        
        self.n_layers = self.cfg.n_layers
        self.n_heads = self.cfg.n_heads
        self.d_head = self.cfg.d_head
        self.qkv_size = self.n_heads * self.d_head
        self.vocab_size = self.cfg.d_vocab
        
        # Cache original biases
        self.b_O_orig = [block.attn.b_O.clone() for block in self.model.blocks]
        self.b_V_orig = [block.attn.b_QKV[2*self.qkv_size:].clone() for block in self.model.blocks]
    
    def _get_true_ce_loss(self, prompt: str, target_token: int, top_k: int = 64) -> float:
        """Compute true CE loss from the model (no graph)."""
        tokens = self.replacement_model.to_tokens(prompt)
        
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.dtype):
            logits = self.model(tokens)
        
        final_logits = logits[0, -1].float()  # [vocab]
        
        # Get top-k logits
        top_vals, top_idx = torch.topk(final_logits, top_k)
        
        # Check if target is in top-k
        target_in_topk = (top_idx == target_token).any()
        
        if target_in_topk:
            # Compute softmax over top-k
            probs = F.softmax(top_vals, dim=0)
            target_pos = (top_idx == target_token).nonzero().item()
            ce_loss = -torch.log(probs[target_pos] + 1e-10).item()
        else:
            # Target not in top-k, assign high loss
            ce_loss = 10.0
        
        return ce_loss
    
    def _compute_bias_contributions_to_logits(
        self,
        prompt: str,
        logit_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Compute b_O and b_V contributions to specified logits."""
        tokens = self.replacement_model.to_tokens(prompt)
        n_pos = tokens.shape[1]
        final_pos = n_pos - 1
        
        for p in self.model.parameters():
            p.requires_grad = False
        for layer in range(self.n_layers):
            self.model.blocks[layer].attn.b_O.requires_grad = True
            self.model.blocks[layer].attn.b_QKV.requires_grad = True
        
        captured = [None]
        def cap(acts, hook):
            captured[0] = acts
            return acts
        self.model.unembed.hook_post.add_hook(cap, is_permanent=False)
        
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            _ = self.model(tokens)
        
        all_logits = captured[0][0, final_pos]
        self.model.unembed.hook_post.remove_hooks()
        
        b_O_tensors = [self.model.blocks[l].attn.b_O for l in range(self.n_layers)]
        b_QKV_tensors = [self.model.blocks[l].attn.b_QKV for l in range(self.n_layers)]
        
        # Compute mean bias contribution
        logit_sum = all_logits.float().sum()
        grads = torch.autograd.grad(
            logit_sum, b_O_tensors + b_QKV_tensors,
            allow_unused=True, retain_graph=True
        )
        
        mean_bias_contrib = 0.0
        for layer in range(self.n_layers):
            if grads[layer] is not None:
                mean_grad = grads[layer].float() / self.vocab_size
                mean_bias_contrib += (mean_grad @ self.b_O_orig[layer].float()).item()
            if grads[self.n_layers + layer] is not None:
                b_V_grad = grads[self.n_layers + layer][2*self.qkv_size:].float() / self.vocab_size
                mean_bias_contrib += (b_V_grad @ self.b_V_orig[layer].float()).item()
        
        # Compute individual contributions
        n_logits = len(logit_tokens)
        bias_contribs = torch.zeros(n_logits, device=self.device, dtype=torch.float32)
        
        for i, tok in enumerate(logit_tokens):
            tok_id = tok.item()
            grads = torch.autograd.grad(
                all_logits[tok_id].float(), b_O_tensors + b_QKV_tensors,
                allow_unused=True, retain_graph=True
            )
            
            total = 0.0
            for layer in range(self.n_layers):
                if grads[layer] is not None:
                    total += (grads[layer].float() @ self.b_O_orig[layer].float()).item()
                if grads[self.n_layers + layer] is not None:
                    b_V_grad = grads[self.n_layers + layer][2*self.qkv_size:].float()
                    total += (b_V_grad @ self.b_V_orig[layer].float()).item()
            
            bias_contribs[i] = total
        
        return bias_contribs, mean_bias_contrib
    
    def _forward_pruned_graph(
        self,
        graph,
        prompt: str,
        edge_threshold: float,
        logit_bias_contribs: torch.Tensor,
        mean_bias_contrib: float,
    ) -> Tuple[torch.Tensor, int, int, int]:
        """
        Run forward pass on pruned graph.
        
        Returns:
            Tuple of (logit_values, n_edges, n_nodes, n_features)
        """
        n_features = len(graph.active_features)
        n_layers = self.n_layers
        n_pos = graph.n_pos
        n_logits = len(graph.logit_tokens)
        
        logit_start = n_features + n_layers * n_pos + n_pos
        
        # Apply edge threshold
        adj = graph.adjacency_matrix.float().to(self.device)
        edge_mask = adj.abs() >= edge_threshold
        pruned_adj = adj * edge_mask
        
        # Count edges and nodes (excluding logit nodes)
        n_edges = edge_mask.sum().item()
        
        # Count active nodes (nodes with at least one edge)
        has_incoming = edge_mask.any(dim=1)
        has_outgoing = edge_mask.any(dim=0)
        active_nodes = has_incoming | has_outgoing
        # Count feature nodes + error nodes + embedding nodes (not logits)
        n_circuit_nodes = n_features + n_layers * n_pos + n_pos
        n_nodes = active_nodes[:n_circuit_nodes].sum().item()
        
        # Count active features
        n_active_features = active_nodes[:n_features].sum().item()
        
        # Get logit edge sums from pruned adjacency
        logit_edge_sums = pruned_adj[logit_start:logit_start + n_logits].sum(dim=1)
        
        # Add bias contributions (these are the same regardless of pruning)
        logit_values = logit_edge_sums + (logit_bias_contribs.to(self.device) - mean_bias_contrib)
        
        return logit_values, int(n_edges), int(n_nodes), int(n_active_features)
    
    def _compute_ce_loss(
        self,
        logit_values: torch.Tensor,
        logit_tokens: torch.Tensor,
        target_token: int,
    ) -> Tuple[float, int]:
        """
        Compute cross-entropy loss from graph logits.
        
        Args:
            logit_values: Reconstructed logit values [n_logits]
            logit_tokens: Token IDs for each logit [n_logits]
            target_token: The target (next) token
            
        Returns:
            Tuple of (ce_loss, predicted_token)
        """
        # Check if target is in the logit tokens
        target_mask = logit_tokens == target_token
        target_in_logits = target_mask.any()
        
        if not target_in_logits:
            # Target not in top-k, assign high loss
            top_logit_idx = logit_values.argmax().item()
            predicted_token = logit_tokens[top_logit_idx].item()
            return 10.0, predicted_token
        
        # Compute softmax over the logits
        probs = F.softmax(logit_values.float(), dim=0)
        
        # Get probability of target
        target_idx = target_mask.nonzero().item()
        target_prob = probs[target_idx]
        
        # Cross-entropy loss
        ce_loss = -torch.log(target_prob + 1e-10).item()
        
        # Get predicted token
        top_logit_idx = logit_values.argmax().item()
        predicted_token = logit_tokens[top_logit_idx].item()
        
        return ce_loss, predicted_token
    
    def _forward_pruned_graph_2d(
        self,
        graph,
        prompt: str,
        node_threshold: float,
        edge_threshold: float,
        logit_bias_contribs: torch.Tensor,
        mean_bias_contrib: float,
    ) -> Tuple[torch.Tensor, int, int, int]:
        """
        Run forward pass on pruned graph with both node and edge thresholds.
        
        Uses circuit_tracer's prune_graph function which does influence-based pruning:
        - node_threshold: Keep nodes that contribute to this fraction of total influence
                         (0 = keep none, 1 = keep all)
        - edge_threshold: Keep edges that contribute to this fraction of total influence
                         (0 = keep none, 1 = keep all)
        
        Returns:
            Tuple of (logit_values, n_edges, n_nodes, n_features)
        """
        from circuit_tracer.graph import prune_graph
        
        n_features = len(graph.active_features)
        n_layers = self.n_layers
        n_pos = graph.n_pos
        n_logits = len(graph.logit_tokens)
        
        logit_start = n_features + n_layers * n_pos + n_pos
        
        # Use circuit_tracer's influence-based pruning
        # Note: their thresholds are cumulative influence fractions (0.8 = keep 80%)
        # We pass through directly since we're now using the same semantics
        node_mask, edge_mask, _ = prune_graph(graph, node_threshold, edge_threshold)
        
        # Apply masks to get pruned adjacency
        adj = graph.adjacency_matrix.float().to(self.device)
        pruned_adj = adj.clone()
        pruned_adj[~edge_mask] = 0
        
        # Count edges and nodes (excluding logit nodes)
        n_edges = edge_mask.sum().item()
        # Count feature nodes + error nodes + embedding nodes (not logits)
        n_circuit_nodes = n_features + n_layers * n_pos + n_pos
        n_nodes = node_mask[:n_circuit_nodes].sum().item()
        n_active_features = node_mask[:n_features].sum().item()
        
        # Get logit edge sums
        logit_edge_sums = pruned_adj[logit_start:logit_start + n_logits].sum(dim=1)
        
        # Add bias contributions
        logit_values = logit_edge_sums + (logit_bias_contribs.to(self.device) - mean_bias_contrib)
        
        return logit_values, int(n_edges), int(n_nodes), int(n_active_features)
    
    def compute_pareto(
        self,
        prompt: str,
        target_token: int,
        top_k: int = 64,
        n_thresholds: int = 20,
        edge_thresholds: Optional[List[float]] = None,
    ) -> ParetoResult:
        """
        Compute Pareto curve of CE loss vs circuit size (edge threshold only).
        
        Args:
            prompt: Input prompt
            target_token: Target (next) token to predict
            top_k: Number of top logits to use as terminal nodes
            n_thresholds: Number of edge threshold values to try
            edge_thresholds: Optional explicit list of thresholds
            
        Returns:
            ParetoResult with all points on the curve
        """
        from circuit_tracer.attribution.attribute import attribute
        
        # Build attribution graph with top-k logits
        print(f"Building attribution graph with top-{top_k} logits...")
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            graph = attribute(
                prompt, 
                self.replacement_model, 
                max_n_logits=top_k,
                desired_logit_prob=0.99,
            )
        
        print(f"Graph: {len(graph.active_features)} features, {len(graph.logit_tokens)} logits")
        
        # Precompute bias contributions (same for all thresholds)
        print("Computing bias contributions...")
        logit_bias_contribs, mean_bias_contrib = self._compute_bias_contributions_to_logits(
            prompt, graph.logit_tokens
        )
        
        # Get true CE loss from model
        true_ce_loss = self._get_true_ce_loss(prompt, target_token, top_k)
        print(f"True CE loss: {true_ce_loss:.4f}")
        
        # Compute baseline (unpruned) CE loss
        baseline_logits, _, _, _ = self._forward_pruned_graph(
            graph, prompt, 0.0, logit_bias_contribs, mean_bias_contrib
        )
        baseline_ce_loss, _ = self._compute_ce_loss(
            baseline_logits, graph.logit_tokens, target_token
        )
        print(f"Baseline CE loss (full graph): {baseline_ce_loss:.4f}")
        
        # Generate edge thresholds
        if edge_thresholds is None:
            # Get range of edge magnitudes
            edge_mags = graph.adjacency_matrix.abs()
            max_edge = edge_mags.max().item()
            min_nonzero = edge_mags[edge_mags > 0].min().item() if (edge_mags > 0).any() else 0
            
            # Log-spaced thresholds from min to max
            edge_thresholds = np.logspace(
                np.log10(min_nonzero + 1e-10),
                np.log10(max_edge),
                n_thresholds
            ).tolist()
            edge_thresholds = [0.0] + edge_thresholds  # Include 0 (full graph)
        
        # Compute Pareto points
        print(f"Computing {len(edge_thresholds)} Pareto points...")
        points = []
        
        for threshold in tqdm(edge_thresholds, desc="Edge thresholds"):
            logit_values, n_edges, n_nodes, n_features = self._forward_pruned_graph(
                graph, prompt, threshold, logit_bias_contribs, mean_bias_contrib
            )
            
            ce_loss, predicted_token = self._compute_ce_loss(
                logit_values, graph.logit_tokens, target_token
            )
            
            points.append(ParetoPoint(
                edge_threshold=threshold,
                node_threshold=0.0,  # 1D mode: no node threshold
                n_edges=n_edges,
                n_nodes=n_nodes,
                n_features=n_features,
                ce_loss=ce_loss,
                accuracy=float(predicted_token == target_token),
                top_logit_token=predicted_token,
                target_token=target_token,
            ))
        
        return ParetoResult(
            points=points,
            prompt=prompt,
            target_token=target_token,
            baseline_ce_loss=baseline_ce_loss,
            true_ce_loss=true_ce_loss,
        )
    
    def compute_pareto_2d(
        self,
        prompt: str,
        target_token: int,
        top_k: int = 64,
        node_thresholds: Optional[List[float]] = None,
        edge_thresholds: Optional[List[float]] = None,
        n_node_thresholds: int = 10,
        n_edge_thresholds: int = 10,
    ) -> ParetoResult:
        """
        Compute Pareto curve by varying both node and edge thresholds.
        
        This generates a 2D grid of (node_threshold, edge_threshold) pairs,
        then extracts the Pareto frontier.
        
        Args:
            prompt: Input prompt
            target_token: Target token to predict
            top_k: Number of top logits
            node_thresholds: List of node thresholds (0 to 1)
            edge_thresholds: List of edge thresholds
            n_node_thresholds: Number of node threshold values if not specified
            n_edge_thresholds: Number of edge threshold values if not specified
            
        Returns:
            ParetoResult with Pareto-optimal points only
        """
        from circuit_tracer.attribution.attribute import attribute
        
        # Build graph
        print(f"Building attribution graph with top-{top_k} logits...")
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            graph = attribute(
                prompt,
                self.replacement_model,
                max_n_logits=top_k,
                desired_logit_prob=0.99,
            )
        
        print(f"Graph: {len(graph.active_features)} features, {len(graph.logit_tokens)} logits")
        
        # Precompute bias contributions
        print("Computing bias contributions...")
        logit_bias_contribs, mean_bias_contrib = self._compute_bias_contributions_to_logits(
            prompt, graph.logit_tokens
        )
        
        # Get true and baseline CE loss
        true_ce_loss = self._get_true_ce_loss(prompt, target_token, top_k)
        print(f"True CE loss: {true_ce_loss:.4f}")
        
        baseline_logits, _, _, _ = self._forward_pruned_graph_2d(
            graph, prompt, 0.0, 0.0, logit_bias_contribs, mean_bias_contrib
        )
        baseline_ce_loss, _ = self._compute_ce_loss(
            baseline_logits, graph.logit_tokens, target_token
        )
        print(f"Baseline CE loss (full graph): {baseline_ce_loss:.4f}")
        
        # Generate thresholds
        # circuit_tracer semantics: threshold = fraction of influence to keep
        # node_threshold=1.0 keeps all nodes, 0.0 keeps none
        # edge_threshold=1.0 keeps all edges, 0.0 keeps none
        if node_thresholds is None:
            # Linear-spaced from small (aggressive pruning) to 1 (full graph)
            node_thresholds = np.linspace(0.1, 1.0, n_node_thresholds).tolist()
        
        if edge_thresholds is None:
            # Linear-spaced from small to 1
            edge_thresholds = np.linspace(0.1, 1.0, n_edge_thresholds).tolist()
        
        # Generate all points
        total_combos = len(node_thresholds) * len(edge_thresholds)
        print(f"Computing {total_combos} threshold combinations...")
        
        all_points = []
        for node_thresh in tqdm(node_thresholds, desc="Node thresholds"):
            for edge_thresh in edge_thresholds:
                logit_values, n_edges, n_nodes, n_features = self._forward_pruned_graph_2d(
                    graph, prompt, node_thresh, edge_thresh,
                    logit_bias_contribs, mean_bias_contrib
                )
                
                ce_loss, predicted_token = self._compute_ce_loss(
                    logit_values, graph.logit_tokens, target_token
                )
                
                all_points.append(ParetoPoint(
                    edge_threshold=edge_thresh,
                    node_threshold=node_thresh,
                    n_edges=n_edges,
                    n_nodes=n_nodes,
                    n_features=n_features,
                    ce_loss=ce_loss,
                    accuracy=float(predicted_token == target_token),
                    top_logit_token=predicted_token,
                    target_token=target_token,
                ))
        
        # Extract Pareto frontier for both n_edges and n_nodes
        pareto_edges = self._extract_pareto_frontier(all_points, x_attr="n_edges")
        pareto_nodes = self._extract_pareto_frontier(all_points, x_attr="n_nodes")
        
        # Combine and deduplicate
        seen = set()
        pareto_combined = []
        for p in pareto_edges + pareto_nodes:
            key = (p.n_edges, p.n_nodes, p.ce_loss)
            if key not in seen:
                seen.add(key)
                pareto_combined.append(p)
        
        # Sort by n_edges
        pareto_combined.sort(key=lambda p: p.n_edges)
        
        print(f"Extracted {len(pareto_combined)} Pareto-optimal points from {len(all_points)} total")
        print(f"  ({len(pareto_edges)} for edges, {len(pareto_nodes)} for nodes)")
        
        result = ParetoResult(
            points=pareto_combined,
            prompt=prompt,
            target_token=target_token,
            baseline_ce_loss=baseline_ce_loss,
            true_ce_loss=true_ce_loss,
        )
        # Store all points for plotting
        result.all_points = all_points
        return result
    
    def _extract_pareto_frontier(
        self,
        points: List[ParetoPoint],
        x_attr: str = "n_nodes",
    ) -> List[ParetoPoint]:
        """
        Extract Pareto-optimal points (lowest CE loss for each circuit size).
        
        A point is Pareto-optimal if no other point has both:
        - Lower or equal CE loss AND lower circuit size, OR
        - Lower CE loss AND lower or equal circuit size
        
        Args:
            points: All candidate points
            x_attr: Attribute to use for x-axis ("n_nodes" or "n_edges")
            
        Returns:
            List of Pareto-optimal points, sorted by x_attr
        """
        if not points:
            return []
        
        # Sort by x value
        sorted_points = sorted(points, key=lambda p: getattr(p, x_attr))
        
        # Extract Pareto frontier
        pareto = []
        min_ce_loss = float('inf')
        
        # Scan from right to left (largest to smallest x)
        for point in reversed(sorted_points):
            if point.ce_loss < min_ce_loss:
                pareto.append(point)
                min_ce_loss = point.ce_loss
        
        # Reverse to get ascending x order
        pareto.reverse()
        
        return pareto
    
    def plot_pareto(
        self,
        result: ParetoResult,
        x_axis: str = "n_edges",
        save_path: Optional[str] = None,
        all_points: Optional[List[ParetoPoint]] = None,
        y_max: float = 5.0,
    ):
        """
        Plot the Pareto curve.
        
        Args:
            result: ParetoResult from compute_pareto (contains Pareto-optimal points)
            x_axis: What to use for x-axis: "n_edges", "n_nodes", or "n_features"
            save_path: Optional path to save the figure
            all_points: Optional list of ALL points (not just Pareto-optimal) to show as scatter
            y_max: Maximum y-axis value
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use all_points if provided, otherwise use result.points
        points_to_use = all_points if all_points else result.points
        
        all_x = np.array([getattr(p, x_axis) for p in points_to_use])
        all_y = np.array([p.ce_loss for p in points_to_use])
        all_edge_thresh = np.array([p.edge_threshold for p in points_to_use])
        all_node_thresh = np.array([p.node_threshold for p in points_to_use])
        all_n_edges = np.array([p.n_edges for p in points_to_use])
        all_n_nodes = np.array([p.n_nodes for p in points_to_use])
        
        # Find the full graph point: highest thresholds = (1.0, 1.0) = keep everything
        # Use max threshold sum to handle floating point issues
        threshold_sum = all_edge_thresh + all_node_thresh
        max_thresh_sum = threshold_sum.max()
        full_graph_mask = threshold_sum == max_thresh_sum
        
        # Points to plot as regular scatter (excluding full graph)
        scatter_mask = ~full_graph_mask
        scatter_x = all_x[scatter_mask]
        scatter_y = all_y[scatter_mask]
        scatter_node_thresh = all_node_thresh[scatter_mask]
        
        # Full graph point
        full_graph_x = all_x[full_graph_mask]
        full_graph_y = all_y[full_graph_mask]
        
        if len(scatter_x) == 0 and len(full_graph_x) == 0:
            print("No valid points to plot")
            return fig
        
        # Plot regular points as scatter, colored by node_threshold
        if len(scatter_x) > 0:
            scatter = ax.scatter(scatter_x, scatter_y, c=scatter_node_thresh, 
                                cmap='viridis', alpha=0.7, s=30)
            cbar = plt.colorbar(scatter, ax=ax, label='Node Threshold')
        
        # Plot full graph point as star
        if len(full_graph_x) > 0:
            ax.scatter(full_graph_x, full_graph_y, c='red', marker='*', s=200, 
                      zorder=5, label='Full Graph', edgecolors='black', linewidths=1)
            # Add text label
            ax.annotate('Full Graph', (full_graph_x[0], full_graph_y[0]), 
                       textcoords="offset points", xytext=(10, 5), fontsize=10)
        
        # Reference line for true model CE loss
        ax.axhline(y=result.true_ce_loss, color='g', linestyle='--', linewidth=2, label='True Model CE Loss')
        
        ax.set_xlabel(x_axis.replace('_', ' ').title())
        ax.set_ylabel('Cross-Entropy Loss')
        ax.set_title(f'CE Loss vs Circuit Size\n"{result.prompt[:50]}..."')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log scale for x-axis - show full range
        ax.set_xscale('log')
        
        # Set y-axis max
        ax.set_ylim(top=y_max)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
        return fig


def test_graph_pareto(
    prompt: str = "When Rita went to the woods,",
    model_name: str = "jacobcd52/ss_d128_f1",
    clt_repo: str = "jacobcd52/ss_d128_f1_clt_k30_e32",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    top_k: int = 64,
    n_thresholds: int = 15,
    use_2d: bool = True,
):
    """
    Test the Pareto curve computation.
    
    Args:
        use_2d: If True, use 2D grid search over node and edge thresholds
                and extract Pareto frontier. If False, vary edge threshold only.
    """
    import sys
    sys.path.insert(0, '/root/global_circuits/sparse_pretrain')
    sys.path.insert(0, '/root/global_circuits/dictionary_learning')
    sys.path.insert(0, '/root/global_circuits/circuit_tracer')
    sys.path.insert(0, '/root/global_circuits/sparse_pretrain/conversion_utils')
    sys.path.insert(0, '/root/global_circuits/dictionary_learning/conversion_utils')
    
    from src.hooked_model import HookedSparseGPT
    from trainers.cross_layer import CrossLayerTranscoder
    from wrapper import SparseGPTReplacementModel
    from clt_adapter import CLTAdapter
    
    print(f"Loading model: {model_name}")
    model = HookedSparseGPT.from_pretrained(model_name, device=device, dtype=dtype)
    n_layers = model.cfg.n_layers
    
    print(f"Loading {n_layers} transcoders from: {clt_repo}")
    transcoders = []
    for layer_idx in range(n_layers):
        tc = CrossLayerTranscoder.from_hf(clt_repo, subfolder=f'transcoder_layer_{layer_idx}', device=device)
        transcoders.append(tc.to(dtype))
    
    # Zero b_dec
    print("Zeroing b_dec...")
    with torch.no_grad():
        for tc in transcoders:
            tc.b_dec.zero_()
    
    clt_adapter = CLTAdapter(transcoders, scan=clt_repo)
    replacement_model = SparseGPTReplacementModel(model, clt_adapter)
    
    # Get target token (next token after prompt)
    tokens = replacement_model.to_tokens(prompt)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
        logits = model(tokens)
    target_token = logits[0, -1].argmax().item()
    target_str = model.tokenizer.decode([target_token])
    print(f"\nTarget token: {target_token} ('{target_str}')")
    
    # Compute Pareto curve
    print(f"\n{'='*60}")
    print(f"Computing Pareto curve for: '{prompt}'")
    print(f"Top-K logits: {top_k}")
    print(f"Mode: {'2D (node + edge threshold)' if use_2d else '1D (edge threshold only)'}")
    print(f"{'='*60}\n")
    
    gp = GraphPareto(replacement_model, transcoders, device=device, dtype=dtype)
    
    if use_2d:
        # 2D grid search with Pareto extraction
        result = gp.compute_pareto_2d(
            prompt, target_token, top_k=top_k,
            n_node_thresholds=50,
            n_edge_thresholds=50,
        )
    else:
        result = gp.compute_pareto(prompt, target_token, top_k=top_k, n_thresholds=n_thresholds)
    
    # Print results
    print(f"\n{'='*60}")
    print("PARETO CURVE RESULTS")
    print(f"{'='*60}")
    print(f"\nTrue CE Loss:     {result.true_ce_loss:.4f}")
    print(f"Baseline CE Loss: {result.baseline_ce_loss:.4f}")
    
    print(f"\n{'Threshold':<12} {'Edges':<10} {'Nodes':<10} {'Features':<10} {'CE Loss':<10} {'Correct'}")
    print("-" * 70)
    
    for p in result.points:
        thresh_str = f"{p.edge_threshold:.2e}" if p.edge_threshold > 0 and p.edge_threshold < 0.01 else f"{p.edge_threshold:.4f}"
        print(f"{thresh_str:<12} {p.n_edges:<10} {p.n_nodes:<10} {p.n_features:<10} {p.ce_loss:<10.4f} {'✓' if p.accuracy else '✗'}")
    
    # Try to plot (may fail if no display)
    try:
        import os
        save_dir = "/root/global_circuits/misc_outputs"
        os.makedirs(save_dir, exist_ok=True)
        
        # Get all points if available (from 2D search)
        all_points = getattr(result, 'all_points', None)
        
        # Plot with edges on x-axis
        save_path = os.path.join(save_dir, "pareto_curve_edges.png")
        gp.plot_pareto(result, x_axis="n_edges", save_path=save_path, all_points=all_points)
        
        # Plot with nodes on x-axis
        save_path = os.path.join(save_dir, "pareto_curve_nodes.png")
        gp.plot_pareto(result, x_axis="n_nodes", save_path=save_path, all_points=all_points)
    except Exception as e:
        print(f"\nCould not plot: {e}")
    
    return result


if __name__ == "__main__":
    test_graph_pareto(use_2d=True)

