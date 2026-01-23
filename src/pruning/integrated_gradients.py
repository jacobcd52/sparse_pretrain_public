"""
Integrated Gradients for estimating node importance in weight-sparse models.

Computes node importance by integrating gradients along a path from the original
activation to zero (zero ablation). The importance tells us how much the loss
increases when a node is ablated.

Based on the formula:
    IG_i(x) = (x_i - x'_i) × ∫₀¹ (∂F/∂x_i)(x' + α(x-x')) dα

Where:
    - x = input (original activation)
    - x' = baseline (zero)
    - F = task loss
    - α = interpolation constant

In practice, we use the Riemann sum approximation:
    IG_i ≈ (x_i - x'_i) × Σₖ (∂F/∂x_i)(x' + (k/m)(x-x')) × (1/m)

Since x' = 0:
    IG_i ≈ x_i × Σₖ (∂F/∂x_i)((k/m)×x) × (1/m)

Note: Positive importance means ablating the node INCREASES the loss, i.e.,
the node is helpful for the task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import json
import numpy as np

from .tasks import BinaryTask
from .config import PruningConfig


@dataclass
class IGConfig:
    """Configuration for integrated gradients computation."""
    n_steps: int = 50  # Number of interpolation steps
    n_samples: int = 8  # Number of batches to average over
    batch_size: int = 64  # Batch size for task examples


def _apply_abstopk(x: torch.Tensor, k_fraction: float) -> torch.Tensor:
    """Apply AbsTopK sparsity: keep top k values by absolute value, zero the rest."""
    dim = x.shape[-1]
    k = max(1, int(dim * k_fraction))
    
    if k >= dim:
        return x
    
    _, topk_indices = torch.topk(x.abs(), k, dim=-1, sorted=False)
    result = torch.zeros_like(x)
    result.scatter_(-1, topk_indices, x.gather(-1, topk_indices))
    return result


class IntegratedGradientsComputer:
    """
    Computes integrated gradients for node importance estimation.
    
    Processes one node location at a time (e.g., layer0_attn_q) to ensure
    accurate gradient computation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task: BinaryTask,
        config: IGConfig,
        use_binary_loss: bool = True,
        device: str = "cuda",
    ):
        self.model = model
        self.task = task
        self.config = config
        self.use_binary_loss = use_binary_loss
        self.device = device
        
        # Get model dimensions
        model_config = model.config
        self.n_layers = model_config.n_layer
        self.d_model = model_config.d_model
        self.d_mlp = model_config.d_mlp
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_head
        
        # Get AbsTopK config
        self._activation_sparsity_enabled = False
        self._activation_topk_fraction = 0.25
        self._activation_sparsity_locations = set()
        
        if hasattr(model, 'sparsity_config'):
            sc = model.sparsity_config
            self._activation_sparsity_enabled = sc.enable_activation_sparsity
            self._activation_topk_fraction = sc.activation_topk_fraction
            if sc.activation_sparsity_locations:
                self._activation_sparsity_locations = set(sc.activation_sparsity_locations.split(','))
        
        # Define all node locations
        self.node_locations = [
            "attn_in", "attn_q", "attn_k", "attn_v", "attn_out",
            "mlp_in", "mlp_neuron", "mlp_out"
        ]
    
    def _get_dim_for_location(self, loc: str) -> int:
        """Get the activation dimension for a location type."""
        if loc in ["attn_in", "attn_out", "mlp_in", "mlp_out"]:
            return self.d_model
        elif loc in ["attn_q", "attn_k", "attn_v"]:
            return self.n_heads * self.d_head
        elif loc == "mlp_neuron":
            return self.d_mlp
        else:
            raise ValueError(f"Unknown location type: {loc}")
    
    def _should_apply_abstopk(self, location: str) -> bool:
        """Check if AbsTopK should be applied at this location."""
        return self._activation_sparsity_enabled and location in self._activation_sparsity_locations
    
    def compute_task_loss(
        self,
        logits: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        eval_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute task loss from logits."""
        batch_size = logits.shape[0]
        batch_indices = torch.arange(batch_size, device=logits.device)
        final_logits = logits[batch_indices, eval_positions, :]
        
        if self.use_binary_loss:
            correct_logits = final_logits.gather(1, correct_tokens.unsqueeze(1)).squeeze(1)
            incorrect_logits = final_logits.gather(1, incorrect_tokens.unsqueeze(1)).squeeze(1)
            binary_logits = torch.stack([correct_logits, incorrect_logits], dim=1)
            targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(binary_logits, targets)
        else:
            loss = F.cross_entropy(final_logits, correct_tokens)
        
        return loss
    
    def _forward_with_patching(
        self,
        input_ids: torch.Tensor,
        target_layer: int,
        target_location: str,
        patch_value: Optional[torch.Tensor] = None,
        scale_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional patching at a specific location.
        
        The patching happens AFTER AbsTopK (if applicable), so we're patching
        the sparse activations, not the dense ones.
        
        Args:
            input_ids: Input token IDs
            target_layer: Layer to patch (-1 means no patching)
            target_location: Location type to patch
            patch_value: Value to use for patching (if None, uses scale_factor)
            scale_factor: Scale factor to multiply activations (alternative to patch_value)
            
        Returns:
            logits: Model output logits
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        x = self.model.wte(input_ids)
        
        if torch.is_autocast_enabled('cuda'):
            x = x.to(torch.get_autocast_dtype('cuda'))
        
        if self.model.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x = x + self.model.wpe(pos)
        
        x = self.model.drop(x)
        
        # Process through each transformer block
        for layer_idx, block in enumerate(self.model.blocks):
            is_target_layer = (layer_idx == target_layer)
            
            # Attention sublayer
            normed = block.ln_1(x)
            
            # Apply AbsTopK then patch for attn_in
            if self._should_apply_abstopk("attn_in"):
                normed = _apply_abstopk(normed, self._activation_topk_fraction)
            if is_target_layer and target_location == "attn_in":
                if patch_value is not None:
                    normed = patch_value
                elif scale_factor is not None:
                    normed = normed * scale_factor
            
            # QKV projection
            qkv = block.attn.c_attn(normed)
            q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
            
            # Apply AbsTopK then patch for Q
            if self._should_apply_abstopk("attn_q"):
                q = _apply_abstopk(q, self._activation_topk_fraction)
            if is_target_layer and target_location == "attn_q":
                if patch_value is not None:
                    q = patch_value
                elif scale_factor is not None:
                    q = q * scale_factor
            
            # Apply AbsTopK then patch for K
            if self._should_apply_abstopk("attn_k"):
                k = _apply_abstopk(k, self._activation_topk_fraction)
            if is_target_layer and target_location == "attn_k":
                if patch_value is not None:
                    k = patch_value
                elif scale_factor is not None:
                    k = k * scale_factor
            
            # Apply AbsTopK then patch for V
            if self._should_apply_abstopk("attn_v"):
                v = _apply_abstopk(v, self._activation_topk_fraction)
            if is_target_layer and target_location == "attn_v":
                if patch_value is not None:
                    v = patch_value
                elif scale_factor is not None:
                    v = v * scale_factor
            
            # Reshape for attention
            q = q.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
            k = k.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
            v = v.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
            
            scale = 1.0 / (block.attn.d_head ** 0.5)
            
            # Compute attention
            if block.attn.use_flash and block.attn.attn_fn is not None:
                y = block.attn.attn_fn(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
            elif block.attn.use_flash:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
            else:
                att = (q @ k.transpose(-2, -1)) * scale
                att = att.masked_fill(block.attn.causal_mask[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                y = att @ v
            
            y = y.transpose(1, 2).contiguous().view(B, T, block.attn.n_heads * block.attn.d_head)
            attn_out = block.attn.c_proj(y)
            
            # Apply AbsTopK then patch for attn_out
            if self._should_apply_abstopk("attn_out"):
                attn_out = _apply_abstopk(attn_out, self._activation_topk_fraction)
            if is_target_layer and target_location == "attn_out":
                if patch_value is not None:
                    attn_out = patch_value
                elif scale_factor is not None:
                    attn_out = attn_out * scale_factor
            
            x = x + attn_out
            
            # MLP sublayer
            normed = block.ln_2(x)
            
            # Apply AbsTopK then patch for mlp_in
            if self._should_apply_abstopk("mlp_in"):
                normed = _apply_abstopk(normed, self._activation_topk_fraction)
            if is_target_layer and target_location == "mlp_in":
                if patch_value is not None:
                    normed = patch_value
                elif scale_factor is not None:
                    normed = normed * scale_factor
            
            # MLP forward
            mlp_hidden = block.mlp.c_fc(normed)
            mlp_hidden = block.mlp.act_fn(mlp_hidden)
            
            # Apply AbsTopK then patch for mlp_neuron
            if self._should_apply_abstopk("mlp_neuron"):
                mlp_hidden = _apply_abstopk(mlp_hidden, self._activation_topk_fraction)
            if is_target_layer and target_location == "mlp_neuron":
                if patch_value is not None:
                    mlp_hidden = patch_value
                elif scale_factor is not None:
                    mlp_hidden = mlp_hidden * scale_factor
            
            mlp_out = block.mlp.c_proj(mlp_hidden)
            
            # Apply AbsTopK then patch for mlp_out
            if self._should_apply_abstopk("mlp_out"):
                mlp_out = _apply_abstopk(mlp_out, self._activation_topk_fraction)
            if is_target_layer and target_location == "mlp_out":
                if patch_value is not None:
                    mlp_out = patch_value
                elif scale_factor is not None:
                    mlp_out = mlp_out * scale_factor
            
            x = x + mlp_out
        
        # Final layer norm and LM head
        x = self.model.ln_f(x)
        logits = self.model.lm_head(x)
        
        if hasattr(self.model, 'bigram_table') and self.model.bigram_table is not None:
            bigram_logits = F.embedding(input_ids, self.model.bigram_table)
            logits = logits + bigram_logits
        
        return logits
    
    def _capture_activations(
        self,
        input_ids: torch.Tensor,
        target_layer: int,
        target_location: str,
    ) -> torch.Tensor:
        """
        Capture activations at a specific location AFTER AbsTopK.
        
        Returns the activations that we would patch.
        """
        B, T = input_ids.shape
        device = input_ids.device
        captured = None
        
        with torch.no_grad():
            # Token embeddings
            x = self.model.wte(input_ids)
            
            if torch.is_autocast_enabled('cuda'):
                x = x.to(torch.get_autocast_dtype('cuda'))
            
            if self.model.wpe is not None:
                pos = torch.arange(0, T, dtype=torch.long, device=device)
                x = x + self.model.wpe(pos)
            
            x = self.model.drop(x)
            
            for layer_idx, block in enumerate(self.model.blocks):
                is_target_layer = (layer_idx == target_layer)
                
                # Attention sublayer
                normed = block.ln_1(x)
                if self._should_apply_abstopk("attn_in"):
                    normed = _apply_abstopk(normed, self._activation_topk_fraction)
                if is_target_layer and target_location == "attn_in":
                    captured = normed.clone()
                
                qkv = block.attn.c_attn(normed)
                q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
                
                if self._should_apply_abstopk("attn_q"):
                    q = _apply_abstopk(q, self._activation_topk_fraction)
                if is_target_layer and target_location == "attn_q":
                    captured = q.clone()
                
                if self._should_apply_abstopk("attn_k"):
                    k = _apply_abstopk(k, self._activation_topk_fraction)
                if is_target_layer and target_location == "attn_k":
                    captured = k.clone()
                
                if self._should_apply_abstopk("attn_v"):
                    v = _apply_abstopk(v, self._activation_topk_fraction)
                if is_target_layer and target_location == "attn_v":
                    captured = v.clone()
                
                q = q.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
                k = k.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
                v = v.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
                
                scale = 1.0 / (block.attn.d_head ** 0.5)
                if block.attn.use_flash and block.attn.attn_fn is not None:
                    y = block.attn.attn_fn(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
                elif block.attn.use_flash:
                    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=scale)
                else:
                    att = (q @ k.transpose(-2, -1)) * scale
                    att = att.masked_fill(block.attn.causal_mask[:, :, :T, :T] == 0, float('-inf'))
                    att = F.softmax(att, dim=-1)
                    y = att @ v
                
                y = y.transpose(1, 2).contiguous().view(B, T, block.attn.n_heads * block.attn.d_head)
                attn_out = block.attn.c_proj(y)
                
                if self._should_apply_abstopk("attn_out"):
                    attn_out = _apply_abstopk(attn_out, self._activation_topk_fraction)
                if is_target_layer and target_location == "attn_out":
                    captured = attn_out.clone()
                
                x = x + attn_out
                
                # MLP sublayer
                normed = block.ln_2(x)
                if self._should_apply_abstopk("mlp_in"):
                    normed = _apply_abstopk(normed, self._activation_topk_fraction)
                if is_target_layer and target_location == "mlp_in":
                    captured = normed.clone()
                
                mlp_hidden = block.mlp.c_fc(normed)
                mlp_hidden = block.mlp.act_fn(mlp_hidden)
                
                if self._should_apply_abstopk("mlp_neuron"):
                    mlp_hidden = _apply_abstopk(mlp_hidden, self._activation_topk_fraction)
                if is_target_layer and target_location == "mlp_neuron":
                    captured = mlp_hidden.clone()
                
                mlp_out = block.mlp.c_proj(mlp_hidden)
                
                if self._should_apply_abstopk("mlp_out"):
                    mlp_out = _apply_abstopk(mlp_out, self._activation_topk_fraction)
                if is_target_layer and target_location == "mlp_out":
                    captured = mlp_out.clone()
                
                x = x + mlp_out
        
        return captured
    
    def compute_ig_for_location(
        self,
        layer: int,
        location: str,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Compute integrated gradients for all nodes at a specific location.
        
        The formula is:
            IG_i = (x_i - 0) × ∫₀¹ (∂L/∂x_i)(α×x) dα
                 ≈ x_i × Σₖ (∂L/∂x_i)((k/m)×x) × (1/m)
        
        We compute this for each position (b, t) and then average.
        
        Args:
            layer: Layer index
            location: Location type (e.g., "attn_q")
            show_progress: Whether to show progress bar
            
        Returns:
            importance: (n_nodes,) tensor of importance scores
        """
        n_nodes = self._get_dim_for_location(location)
        n_steps = self.config.n_steps
        n_samples = self.config.n_samples
        
        # Accumulator for importance
        total_importance = torch.zeros(n_nodes, device=self.device)
        
        # Generate batches
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Layer {layer} {location}", leave=False)
        
        for _ in iterator:
            # Generate a batch of task examples
            pos_ids, neg_ids, correct, incorrect, eval_pos = self.task.generate_batch(
                self.config.batch_size
            )
            pos_ids = pos_ids.to(self.device)
            correct = correct.to(self.device)
            incorrect = incorrect.to(self.device)
            eval_pos = eval_pos.to(self.device)
            
            B, T = pos_ids.shape
            
            # Capture original activations (after AbsTopK)
            with torch.no_grad():
                original_acts = self._capture_activations(pos_ids, layer, location)
            
            # Accumulator for integrated gradients (sum over steps)
            # Shape: (B, T, n_nodes)
            integrated_grads = torch.zeros_like(original_acts)
            
            for step in range(n_steps):
                # Scale factor: k/m where k goes from 0 to m-1
                # This is a left Riemann sum that includes alpha=0
                alpha = step / n_steps
                
                # Create scaled activation tensor that requires grad
                scaled_acts = (original_acts * alpha).detach().requires_grad_(True)
                
                # Forward pass with patched activations
                logits = self._forward_with_patching(
                    pos_ids, layer, location, patch_value=scaled_acts
                )
                
                # Compute loss
                loss = self.compute_task_loss(logits, correct, incorrect, eval_pos)
                
                # Backward to get gradients w.r.t. scaled_acts
                loss.backward()
                
                # Gradient has shape (B, T, n_nodes)
                grad = scaled_acts.grad  # (B, T, n_nodes)
                
                # Accumulate gradients (will multiply by 1/m at the end)
                integrated_grads = integrated_grads + grad
            
            # Complete the IG formula: IG = x × (Σ_k grad_k) / m
            # Shape: (B, T, n_nodes)
            ig_values = original_acts * integrated_grads / n_steps
            
            # Sum over batch and sequence to get total importance per node
            # This matches ablation which zeros the node at ALL positions
            batch_importance = ig_values.sum(dim=(0, 1))  # (n_nodes,)
            total_importance += batch_importance
        
        # Average over samples (different batches of data)
        total_importance /= n_samples
        
        # Negate: IG measures "contribution to loss", but we want "importance" where
        # positive = ablating increases loss. If a node contributes positively to loss
        # (IG > 0), ablating it should decrease loss, so importance = -IG.
        return -total_importance
    
    def compute_all_importances(
        self,
        show_progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute integrated gradients for all nodes in the model.
        
        Returns:
            Dict mapping "layer{i}_{location}" to importance tensors
        """
        importances = {}
        
        total_locations = self.n_layers * len(self.node_locations)
        pbar = tqdm(total=total_locations, desc="Computing IG") if show_progress else None
        
        for layer in range(self.n_layers):
            for location in self.node_locations:
                key = f"layer{layer}_{location}"
                importance = self.compute_ig_for_location(
                    layer, location, show_progress=False
                )
                importances[key] = importance
                
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"location": key})
        
        if pbar is not None:
            pbar.close()
        
        return importances
    
    def compute_ablation_importance(
        self,
        layer: int,
        location: str,
        node_idx: int,
        n_batches: int = 5,
    ) -> float:
        """
        Compute exact ablation importance for a single node.
        
        This zeros out a single node and measures the change in loss.
        
        Args:
            layer: Layer index
            location: Location type
            node_idx: Index of the node to ablate
            n_batches: Number of batches to average over
            
        Returns:
            importance: Loss increase when node is ablated
        """
        total_baseline_loss = 0.0
        total_ablated_loss = 0.0
        
        for _ in range(n_batches):
            # Generate batch
            pos_ids, _, correct, incorrect, eval_pos = self.task.generate_batch(
                self.config.batch_size
            )
            pos_ids = pos_ids.to(self.device)
            correct = correct.to(self.device)
            incorrect = incorrect.to(self.device)
            eval_pos = eval_pos.to(self.device)
            
            with torch.no_grad():
                # Baseline loss (no patching)
                logits = self._forward_with_patching(pos_ids, -1, "", None)
                baseline_loss = self.compute_task_loss(logits, correct, incorrect, eval_pos)
                total_baseline_loss += baseline_loss.item()
                
                # Capture original activations
                original_acts = self._capture_activations(pos_ids, layer, location)
                
                # Create ablated activations (zero out the specific node)
                ablated_acts = original_acts.clone()
                ablated_acts[..., node_idx] = 0.0
                
                # Ablated loss
                logits = self._forward_with_patching(
                    pos_ids, layer, location, patch_value=ablated_acts
                )
                ablated_loss = self.compute_task_loss(logits, correct, incorrect, eval_pos)
                total_ablated_loss += ablated_loss.item()
        
        baseline_loss = total_baseline_loss / n_batches
        ablated_loss = total_ablated_loss / n_batches
        
        # Importance = how much loss increases when ablated
        return ablated_loss - baseline_loss
    
    def validate_ig(
        self,
        n_nodes_per_location: int = 10,
        n_ablation_batches: int = 10,
        show_progress: bool = True,
    ) -> Dict:
        """
        Validate integrated gradients by comparing with exact ablation.
        
        Randomly samples nodes from each location and computes both IG
        and exact ablation importance, then compares them.
        
        Args:
            n_nodes_per_location: Number of nodes to sample per location
            n_ablation_batches: Number of batches for ablation computation
            show_progress: Whether to show progress bar
            
        Returns:
            Dict with validation results including correlation, slope, R², etc.
        """
        # First compute all IG importances
        ig_importances = self.compute_all_importances(show_progress=show_progress)
        
        # Sample nodes and compute ablation importances
        validation_data = []
        
        for layer in range(self.n_layers):
            for location in self.node_locations:
                key = f"layer{layer}_{location}"
                ig_imp = ig_importances[key]
                n_nodes = len(ig_imp)
                
                # Sample random nodes
                sample_indices = torch.randperm(n_nodes)[:n_nodes_per_location].tolist()
                
                for node_idx in sample_indices:
                    # Compute ablation importance
                    abl_imp = self.compute_ablation_importance(
                        layer, location, node_idx, n_ablation_batches
                    )
                    
                    validation_data.append({
                        "key": key,
                        "node_idx": node_idx,
                        "ig_imp": ig_imp[node_idx].item(),
                        "ablation_imp": abl_imp,
                    })
        
        # Compute statistics
        ig_vals = np.array([d["ig_imp"] for d in validation_data])
        abl_vals = np.array([d["ablation_imp"] for d in validation_data])
        
        # Linear regression
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(ig_vals, abl_vals)
        
        # Pearson correlation
        pearson_r = np.corrcoef(ig_vals, abl_vals)[0, 1]
        
        results = {
            "n_samples": len(validation_data),
            "pearson_r": float(pearson_r),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "slope": float(slope),
            "intercept": float(intercept),
            "nodes": validation_data,
        }
        
        # Also compute baseline loss for reference
        pos_ids, _, correct, incorrect, eval_pos = self.task.generate_batch(
            self.config.batch_size
        )
        pos_ids = pos_ids.to(self.device)
        correct = correct.to(self.device)
        incorrect = incorrect.to(self.device)
        eval_pos = eval_pos.to(self.device)
        
        with torch.no_grad():
            logits = self._forward_with_patching(pos_ids, -1, "", None)
            baseline_loss = self.compute_task_loss(logits, correct, incorrect, eval_pos)
        
        results["baseline_loss"] = float(baseline_loss.item())
        
        return results, ig_importances


def analyze_circuit_importance(
    importances: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
) -> Dict:
    """
    Analyze the importance of nodes in a discovered circuit.
    
    Args:
        importances: Dict mapping keys to importance tensors
        masks: Dict mapping keys to binary mask tensors (1 = in circuit)
        
    Returns:
        Dict with analysis results
    """
    # Compute total importance of circuit vs total
    total_importance = 0.0
    circuit_importance = 0.0
    total_abs_importance = 0.0
    circuit_abs_importance = 0.0
    
    all_importances = []
    circuit_importances = []
    
    for key, imp in importances.items():
        if key not in masks:
            continue
        
        mask = masks[key]
        
        # Ensure mask is binary
        mask_binary = (mask > 0).float()
        
        # Accumulate
        total_importance += imp.sum().item()
        circuit_importance += (imp * mask_binary).sum().item()
        
        total_abs_importance += imp.abs().sum().item()
        circuit_abs_importance += (imp.abs() * mask_binary).sum().item()
        
        # Store individual node importances with circuit membership
        for i in range(len(imp)):
            all_importances.append(imp[i].item())
            if mask_binary[i] > 0:
                circuit_importances.append(imp[i].item())
    
    # Compute fractions
    importance_frac = circuit_importance / (total_importance + 1e-10)
    abs_importance_frac = circuit_abs_importance / (total_abs_importance + 1e-10)
    
    # Compute ranks
    all_importances = np.array(all_importances)
    sorted_indices = np.argsort(-all_importances)  # Descending order
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(all_importances))
    
    # Get ranks of circuit nodes
    circuit_mask_flat = []
    for key in importances.keys():
        if key not in masks:
            continue
        mask = masks[key]
        mask_binary = (mask > 0).float()
        circuit_mask_flat.extend(mask_binary.tolist())
    
    circuit_mask_flat = np.array(circuit_mask_flat)
    circuit_ranks = ranks[circuit_mask_flat > 0]
    
    return {
        "importance_frac": importance_frac,
        "abs_importance_frac": abs_importance_frac,
        "total_importance": total_importance,
        "circuit_importance": circuit_importance,
        "total_abs_importance": total_abs_importance,
        "circuit_abs_importance": circuit_abs_importance,
        "n_total_nodes": len(all_importances),
        "n_circuit_nodes": len(circuit_importances),
        "circuit_ranks": circuit_ranks.tolist(),
        "min_rank": int(circuit_ranks.min()) if len(circuit_ranks) > 0 else None,
        "max_rank": int(circuit_ranks.max()) if len(circuit_ranks) > 0 else None,
        "mean_rank": float(circuit_ranks.mean()) if len(circuit_ranks) > 0 else None,
        "median_rank": float(np.median(circuit_ranks)) if len(circuit_ranks) > 0 else None,
    }


def plot_validation_scatter(
    validation_results: Dict,
    output_path: str,
    title: str = "IG vs Ablation Importance Validation",
):
    """Create scatter plot comparing IG and ablation importances."""
    import matplotlib.pyplot as plt
    
    nodes = validation_results["nodes"]
    ig_vals = [d["ig_imp"] for d in nodes]
    abl_vals = [d["ablation_imp"] for d in nodes]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(ig_vals, abl_vals, alpha=0.5, s=20)
    
    # Add regression line
    slope = validation_results["slope"]
    intercept = validation_results["intercept"]
    x_range = np.array([min(ig_vals), max(ig_vals)])
    ax.plot(x_range, slope * x_range + intercept, 'r-', 
            label=f'Fit: y = {slope:.3f}x + {intercept:.4f}')
    
    # Add y=x line for reference
    all_vals = ig_vals + abl_vals
    lim_min = min(all_vals) - 0.01
    lim_max = max(all_vals) + 0.01
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3, label='y = x')
    
    ax.set_xlabel("Integrated Gradients Importance", fontsize=12)
    ax.set_ylabel("Ablation Importance (loss increase)", fontsize=12)
    ax.set_title(f"{title}\nR² = {validation_results['r_squared']:.3f}, "
                 f"r = {validation_results['pearson_r']:.3f}", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rank_distribution(
    analysis_results: Dict,
    output_path: str,
    max_rank: int = 100,
    title: str = "Circuit Node Importance Ranks",
):
    """Create visualization showing which importance ranks are in the circuit."""
    import matplotlib.pyplot as plt
    
    circuit_ranks = np.array(analysis_results["circuit_ranks"])
    n_total = analysis_results["n_total_nodes"]
    
    # Limit to max_rank or max circuit rank (whichever is smaller for visibility)
    max_circuit_rank = circuit_ranks.max() if len(circuit_ranks) > 0 else 0
    display_max = min(max_rank, max_circuit_rank + 10)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create bars for each rank
    ranks_in_circuit = set(circuit_ranks[circuit_ranks < display_max])
    
    colors = ['green' if i in ranks_in_circuit else 'lightgray' 
              for i in range(display_max)]
    
    ax.bar(range(display_max), [1] * display_max, color=colors, width=1.0, edgecolor='none')
    
    ax.set_xlabel("Importance Rank (0 = most important)", fontsize=12)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_title(f"{title}\n"
                 f"Circuit: {analysis_results['n_circuit_nodes']} nodes, "
                 f"Ranks {analysis_results['min_rank']}-{analysis_results['max_rank']}, "
                 f"Mean: {analysis_results['mean_rank']:.1f}", fontsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='In circuit'),
        Patch(facecolor='lightgray', label='Not in circuit'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlim(-0.5, display_max - 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_importance_histogram(
    analysis_results: Dict,
    importances: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    output_path: str,
    title: str = "Node Importance Distribution",
):
    """Create histogram showing importance distribution for circuit vs non-circuit nodes."""
    import matplotlib.pyplot as plt
    
    circuit_imps = []
    non_circuit_imps = []
    
    for key, imp in importances.items():
        if key not in masks:
            continue
        
        mask = masks[key]
        mask_binary = (mask > 0).float()
        
        for i in range(len(imp)):
            if mask_binary[i] > 0:
                circuit_imps.append(imp[i].item())
            else:
                non_circuit_imps.append(imp[i].item())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use same bins for both
    all_imps = circuit_imps + non_circuit_imps
    bins = np.linspace(min(all_imps), max(all_imps), 50)
    
    ax.hist(non_circuit_imps, bins=bins, alpha=0.5, label='Not in circuit', density=True)
    ax.hist(circuit_imps, bins=bins, alpha=0.7, label='In circuit', density=True)
    
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{title}\n"
                 f"Importance fraction: {analysis_results['importance_frac']:.2%}, "
                 f"Abs importance fraction: {analysis_results['abs_importance_frac']:.2%}",
                 fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

