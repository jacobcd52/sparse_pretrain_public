"""
Forward pass through an attribution graph.

This module implements a forward pass through a circuit_tracer attribution graph,
computing feature activations and logits from the graph's edge weights.

The forward pass formulas are:
- Features: activation = ReLU(edge_sum + encoder_bias + b_O_contrib + b_V_contrib)
- Logits: mean_centered_logit = edge_sum + (bias_contrib - mean_bias_contrib)

Where:
- edge_sum: sum of incoming edges from adjacency matrix
- encoder_bias: encoder.bias[id] - encoder.weight[id] @ b_enc
- b_O_contrib, b_V_contrib: computed via gradient injection with frozen model
- mean_bias_contrib: mean over all vocab tokens (computed efficiently via sum trick)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GraphForwardResult:
    """Result of a forward pass through the attribution graph."""
    
    # Reconstructed values
    feature_activations: torch.Tensor  # [n_features]
    logit_values: torch.Tensor  # [n_logits] (mean-centered)
    
    # Ground truth for comparison
    true_feature_activations: torch.Tensor  # [n_features]
    true_logit_values: torch.Tensor  # [n_logits] (mean-centered)
    
    # Error metrics
    feature_abs_pct_error: float
    logit_abs_pct_error: float


class GraphForward:
    """
    Computes a forward pass through an attribution graph.
    
    This class takes an attribution graph and the model/transcoders used to create it,
    and computes feature activations and logits by propagating through the graph's edges.
    
    Example:
        >>> graph = attribute(prompt, replacement_model, max_n_logits=10)
        >>> gf = GraphForward(replacement_model, transcoders)
        >>> result = gf.forward(graph, prompt)
        >>> print(f"Feature error: {result.feature_abs_pct_error:.1f}%")
        >>> print(f"Logit error: {result.logit_abs_pct_error:.1f}%")
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
            replacement_model: SparseGPTReplacementModel wrapping the model
            transcoders: List of CrossLayerTranscoder or CLTAdapter
            device: Device to run computations on
            dtype: Data type for computations
        """
        self.replacement_model = replacement_model
        self.model = replacement_model.model
        self.cfg = replacement_model.cfg
        self.device = device
        self.dtype = dtype
        
        # Handle both list of transcoders and CLTAdapter
        if hasattr(transcoders, 'transcoders'):
            self.transcoders = transcoders.transcoders
        else:
            self.transcoders = transcoders
        
        self.n_layers = self.cfg.n_layers
        self.n_heads = self.cfg.n_heads
        self.d_head = self.cfg.d_head
        self.qkv_size = self.n_heads * self.d_head
        self.vocab_size = self.cfg.d_vocab
        
        # Cache original biases (before any zeroing)
        self.b_O_orig = [block.attn.b_O.clone() for block in self.model.blocks]
        self.b_V_orig = [block.attn.b_QKV[2*self.qkv_size:].clone() for block in self.model.blocks]
    
    def _compute_encoder_biases(self, graph) -> torch.Tensor:
        """
        Compute encoder bias contribution for each active feature.
        
        encoder_bias = encoder.bias[id] - encoder.weight[id] @ b_enc
        
        Returns:
            Tensor of shape [n_features] with encoder bias for each feature
        """
        n_features = len(graph.active_features)
        encoder_biases = torch.zeros(n_features, device=self.device, dtype=torch.float32)
        
        for idx in range(n_features):
            feat = graph.active_features[idx]
            layer, pos, feat_id = feat[0].item(), feat[1].item(), feat[2].item()
            
            tc = self.transcoders[layer]
            enc_weight = tc.encoder.weight[feat_id].float()
            b_enc = tc.b_enc.float()
            
            encoder_biases[idx] = tc.encoder.bias[feat_id].float() - (enc_weight @ b_enc)
        
        return encoder_biases
    
    def _compute_bias_contributions_to_features(
        self, 
        graph, 
        prompt: str,
    ) -> torch.Tensor:
        """
        Compute b_O and b_V contributions to each feature via gradient injection.
        
        Returns:
            Tensor of shape [n_features] with total bias contribution
        """
        n_features = len(graph.active_features)
        bias_contribs = torch.zeros(n_features, device=self.device, dtype=torch.float32)
        
        tokens = self.replacement_model.to_tokens(prompt)
        
        # Group features by layer for efficiency
        for layer in range(self.n_layers):
            layer_mask = graph.active_features[:, 0] == layer
            layer_indices = torch.where(layer_mask)[0]
            
            if len(layer_indices) == 0:
                continue
            
            # Enable gradients on biases
            for p in self.model.parameters():
                p.requires_grad = False
            for l in range(layer + 1):
                self.model.blocks[l].attn.b_O.requires_grad = True
                self.model.blocks[l].attn.b_QKV.requires_grad = True
            
            # Run forward and capture mlp_in
            captured = [None]
            def cap(acts, hook):
                captured[0] = acts
                return acts
            self.model.blocks[layer].hook_mlp_in.add_hook(cap, is_permanent=False)
            
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                _ = self.model(tokens)
            
            mlp_in = captured[0][0]  # [seq, d_model]
            self.model.blocks[layer].hook_mlp_in.remove_hooks()
            
            tc = self.transcoders[layer]
            
            for idx in layer_indices:
                feat = graph.active_features[idx]
                feat_pos, feat_id = feat[1].item(), feat[2].item()
                
                enc_weight = tc.encoder.weight[feat_id].float()
                pre_act = enc_weight @ (mlp_in[feat_pos].float() - tc.b_enc.float())
                
                b_O_tensors = [self.model.blocks[l].attn.b_O for l in range(layer + 1)]
                b_QKV_tensors = [self.model.blocks[l].attn.b_QKV for l in range(layer + 1)]
                
                grads = torch.autograd.grad(
                    pre_act, b_O_tensors + b_QKV_tensors, 
                    allow_unused=True, retain_graph=True
                )
                
                total = 0.0
                for l in range(layer + 1):
                    if grads[l] is not None:
                        total += (grads[l].float() @ self.b_O_orig[l].float()).item()
                    if grads[layer + 1 + l] is not None:
                        b_V_grad = grads[layer + 1 + l][2*self.qkv_size:].float()
                        total += (b_V_grad @ self.b_V_orig[l].float()).item()
                
                bias_contribs[idx] = total
        
        return bias_contribs
    
    def _compute_bias_contributions_to_logits(
        self,
        graph,
        prompt: str,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute b_O and b_V contributions to each target logit.
        
        Also computes the mean bias contribution over all vocab tokens.
        
        Returns:
            Tuple of:
                - Tensor of shape [n_logits] with bias contribution per logit
                - Float with mean bias contribution across all vocab
        """
        tokens = self.replacement_model.to_tokens(prompt)
        n_pos = tokens.shape[1]
        final_pos = n_pos - 1
        
        # Enable gradients on all biases
        for p in self.model.parameters():
            p.requires_grad = False
        for layer in range(self.n_layers):
            self.model.blocks[layer].attn.b_O.requires_grad = True
            self.model.blocks[layer].attn.b_QKV.requires_grad = True
        
        # Run forward and capture logits
        captured = [None]
        def cap(acts, hook):
            captured[0] = acts
            return acts
        self.model.unembed.hook_post.add_hook(cap, is_permanent=False)
        
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            _ = self.model(tokens)
        
        all_logits = captured[0][0, final_pos]  # [vocab_size]
        self.model.unembed.hook_post.remove_hooks()
        
        b_O_tensors = [self.model.blocks[l].attn.b_O for l in range(self.n_layers)]
        b_QKV_tensors = [self.model.blocks[l].attn.b_QKV for l in range(self.n_layers)]
        
        # Compute mean bias contribution using sum trick
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
        
        # Compute individual logit contributions
        n_logits = len(graph.logit_tokens)
        bias_contribs = torch.zeros(n_logits, device=self.device, dtype=torch.float32)
        
        for i, tok in enumerate(graph.logit_tokens):
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
    
    def forward(self, graph, prompt: str) -> GraphForwardResult:
        """
        Run a forward pass through the attribution graph.
        
        Args:
            graph: Attribution graph from circuit_tracer
            prompt: The prompt used to create the graph
            
        Returns:
            GraphForwardResult with reconstructed and true values
        """
        n_features = len(graph.active_features)
        n_layers = self.n_layers
        n_pos = graph.n_pos
        n_logits = len(graph.logit_tokens)
        
        # Node layout in adjacency matrix:
        # [0, n_features): feature nodes
        # [n_features, n_features + n_layers*n_pos): error nodes
        # [n_features + n_layers*n_pos, n_features + n_layers*n_pos + n_pos): token nodes
        # [n_features + n_layers*n_pos + n_pos, ...): logit nodes
        error_start = n_features
        token_start = error_start + n_layers * n_pos
        logit_start = token_start + n_pos
        
        # =====================================================================
        # Compute feature activations
        # =====================================================================
        
        # Get edge sums for features
        feature_edge_sums = graph.adjacency_matrix[:n_features].float().sum(dim=1)
        
        # Get encoder biases
        encoder_biases = self._compute_encoder_biases(graph)
        
        # Get b_O and b_V contributions to features
        feature_bias_contribs = self._compute_bias_contributions_to_features(graph, prompt)
        
        # Compute reconstructed activations: ReLU(edge_sum + encoder_bias + bias_contrib)
        reconstructed_features = torch.relu(
            feature_edge_sums.to(self.device) + 
            encoder_biases.to(self.device) + 
            feature_bias_contribs.to(self.device)
        )
        
        # True activations from graph
        true_features = graph.activation_values.float().to(self.device)
        
        # =====================================================================
        # Compute logits
        # =====================================================================
        
        # Get edge sums for logits
        logit_edge_sums = graph.adjacency_matrix[logit_start:logit_start + n_logits].float().sum(dim=1)
        
        # Get b_O and b_V contributions to logits
        logit_bias_contribs, mean_bias_contrib = self._compute_bias_contributions_to_logits(graph, prompt)
        
        # Compute reconstructed logits (mean-centered)
        reconstructed_logits = (
            logit_edge_sums.to(self.device) + 
            (logit_bias_contribs - mean_bias_contrib).to(self.device)
        )
        
        # Get true logits (mean-centered)
        tokens = self.replacement_model.to_tokens(prompt)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=self.dtype):
            logits = self.model(tokens)
        
        final_logits = logits[0, -1].float()
        mean_logit = final_logits.mean()
        true_logits = (final_logits[graph.logit_tokens] - mean_logit).to(self.device)
        
        # =====================================================================
        # Compute error metrics
        # =====================================================================
        
        # Feature error
        feature_diffs = (reconstructed_features - true_features).abs()
        feature_pct_errors = feature_diffs / (true_features.abs() + 1e-6) * 100
        feature_abs_pct_error = feature_pct_errors.mean().item()
        
        # Logit error
        logit_diffs = (reconstructed_logits - true_logits).abs()
        logit_pct_errors = logit_diffs / (true_logits.abs() + 1e-6) * 100
        logit_abs_pct_error = logit_pct_errors.mean().item()
        
        return GraphForwardResult(
            feature_activations=reconstructed_features,
            logit_values=reconstructed_logits,
            true_feature_activations=true_features,
            true_logit_values=true_logits,
            feature_abs_pct_error=feature_abs_pct_error,
            logit_abs_pct_error=logit_abs_pct_error,
        )
    
    def forward_with_ablation(
        self, 
        graph, 
        prompt: str,
        feature_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run forward pass with optional feature ablation.
        
        This is a simplified version that just scales edge contributions.
        For accurate ablation, you would need to recompute downstream effects.
        
        Args:
            graph: Attribution graph
            prompt: Prompt string
            feature_mask: Binary mask [n_features], 1 = keep, 0 = ablate
            
        Returns:
            Reconstructed logits [n_logits] (mean-centered)
        """
        n_features = len(graph.active_features)
        n_layers = self.n_layers
        n_pos = graph.n_pos
        n_logits = len(graph.logit_tokens)
        
        logit_start = n_features + n_layers * n_pos + n_pos
        
        if feature_mask is None:
            feature_mask = torch.ones(n_features, device=self.device)
        
        # Get edge contributions to logits
        # Rows: logit nodes, Cols: all source nodes
        logit_edges = graph.adjacency_matrix[logit_start:logit_start + n_logits].float()
        
        # Apply ablation mask to feature columns
        logit_edges[:, :n_features] *= feature_mask.unsqueeze(0)
        
        # Sum edges
        logit_edge_sums = logit_edges.sum(dim=1)
        
        # Get bias contributions (these don't change with ablation in the linear approx)
        logit_bias_contribs, mean_bias_contrib = self._compute_bias_contributions_to_logits(graph, prompt)
        
        return logit_edge_sums + (logit_bias_contribs - mean_bias_contrib)


def test_graph_forward(
    prompt: str = "When Rita went to the woods,",
    model_name: str = "jacobcd52/ss_d128_f1",
    clt_repo: str = "jacobcd52/ss_d128_f1_clt_k30_e32",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_n_logits: int = 10,
):
    """
    Test the graph forward pass by verifying reconstruction matches ground truth.
    
    This loads the model and CLTs, creates an unpruned attribution graph,
    and verifies that the forward pass reproduces the original activations.
    """
    import sys
    sys.path.insert(0, '/root/global_circuits/my_sparse_pretrain')
    sys.path.insert(0, '/root/global_circuits/dictionary_learning')
    sys.path.insert(0, '/root/global_circuits/circuit_tracer')
    sys.path.insert(0, '/root/global_circuits/my_sparse_pretrain/conversion_utils')
    sys.path.insert(0, '/root/global_circuits/dictionary_learning/conversion_utils')
    
    from src.hooked_model import HookedSparseGPT
    from trainers.cross_layer import CrossLayerTranscoder
    from wrapper import SparseGPTReplacementModel
    from clt_adapter import CLTAdapter
    from circuit_tracer.attribution.attribute import attribute
    
    print(f"Loading model: {model_name}")
    model = HookedSparseGPT.from_pretrained(model_name, device=device, dtype=dtype)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    qkv_size = n_heads * d_head
    
    print(f"Loading {n_layers} transcoders from: {clt_repo}")
    transcoders = []
    for layer_idx in range(n_layers):
        tc = CrossLayerTranscoder.from_hf(clt_repo, subfolder=f'transcoder_layer_{layer_idx}', device=device)
        transcoders.append(tc.to(dtype))
    
    # Zero b_dec (absorbed into error nodes)
    print("Zeroing b_dec...")
    with torch.no_grad():
        for tc in transcoders:
            tc.b_dec.zero_()
    
    # Create adapter and replacement model
    clt_adapter = CLTAdapter(transcoders, scan=clt_repo)
    replacement_model = SparseGPTReplacementModel(model, clt_adapter)
    
    # Create attribution graph
    print(f"\nCreating attribution graph for: '{prompt}'")
    with torch.autocast(device_type='cuda', dtype=dtype):
        graph = attribute(prompt, replacement_model, max_n_logits=max_n_logits)
    
    print(f"Graph created:")
    print(f"  Active features: {len(graph.active_features)}")
    print(f"  Logit tokens: {len(graph.logit_tokens)}")
    print(f"  Adjacency matrix: {graph.adjacency_matrix.shape}")
    
    # Run forward pass
    print("\nRunning graph forward pass...")
    gf = GraphForward(replacement_model, transcoders, device=device, dtype=dtype)
    result = gf.forward(graph, prompt)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nFeature Activations:")
    print(f"  Mean absolute % error: {result.feature_abs_pct_error:.2f}%")
    
    print(f"\nLogits (mean-centered):")
    print(f"  Mean absolute % error: {result.logit_abs_pct_error:.2f}%")
    
    print(f"\nSample comparisons (first 5 logits):")
    print(f"  {'Token':<10} {'Reconstructed':>15} {'True':>15} {'Diff':>10}")
    print(f"  {'-'*50}")
    for i in range(min(5, len(graph.logit_tokens))):
        tok = graph.logit_tokens[i].item()
        recon = result.logit_values[i].item()
        true = result.true_logit_values[i].item()
        diff = recon - true
        print(f"  {tok:<10} {recon:>15.4f} {true:>15.4f} {diff:>10.4f}")
    
    # Verify acceptable error
    print("\n" + "="*60)
    if result.feature_abs_pct_error < 10 and result.logit_abs_pct_error < 5:
        print("✓ PASS: Reconstruction errors within acceptable bounds")
        print(f"  (Feature < 10%, Logit < 5%)")
    else:
        print("✗ FAIL: Reconstruction errors too high")
        print(f"  Feature: {result.feature_abs_pct_error:.2f}% (threshold: 10%)")
        print(f"  Logit: {result.logit_abs_pct_error:.2f}% (threshold: 5%)")
    print("="*60)
    
    return result


if __name__ == "__main__":
    test_graph_forward()

