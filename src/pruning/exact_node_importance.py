"""
Exact node importance via one-by-one ablation.

This is the "ground truth" approach: for each node, we zero-ablate it and measure
how much the loss increases. This is slow but exact.

Optimizations:
1. Skip nodes with zero activation everywhere (due to AbsTopK sparsity)
2. Disable gradients for memory efficiency
3. Use large batch size since no gradient memory needed
4. Single large batch instead of multiple small batches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import time

from .tasks import BinaryTask


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


class ExactNodeImportanceComputer:
    """
    Computes exact node importance by ablating each node one at a time.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task: BinaryTask,
        batch_size: int = 256,
        use_binary_loss: bool = True,
        device: str = "cuda",
    ):
        self.model = model
        self.task = task
        self.batch_size = batch_size
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
        
        # Cache for batch data and activations
        self._cached_batch = None
        self._cached_activations = None
        self._cached_baseline_loss = None
    
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
    ) -> torch.Tensor:
        """Forward pass with optional patching at a specific location."""
        B, T = input_ids.shape
        device = input_ids.device
        
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
                normed = patch_value
            
            qkv = block.attn.c_attn(normed)
            q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
            
            if self._should_apply_abstopk("attn_q"):
                q = _apply_abstopk(q, self._activation_topk_fraction)
            if is_target_layer and target_location == "attn_q":
                q = patch_value
            
            if self._should_apply_abstopk("attn_k"):
                k = _apply_abstopk(k, self._activation_topk_fraction)
            if is_target_layer and target_location == "attn_k":
                k = patch_value
            
            if self._should_apply_abstopk("attn_v"):
                v = _apply_abstopk(v, self._activation_topk_fraction)
            if is_target_layer and target_location == "attn_v":
                v = patch_value
            
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
                attn_out = patch_value
            
            x = x + attn_out
            
            # MLP sublayer
            normed = block.ln_2(x)
            
            if self._should_apply_abstopk("mlp_in"):
                normed = _apply_abstopk(normed, self._activation_topk_fraction)
            if is_target_layer and target_location == "mlp_in":
                normed = patch_value
            
            mlp_hidden = block.mlp.c_fc(normed)
            mlp_hidden = block.mlp.act_fn(mlp_hidden)
            
            if self._should_apply_abstopk("mlp_neuron"):
                mlp_hidden = _apply_abstopk(mlp_hidden, self._activation_topk_fraction)
            if is_target_layer and target_location == "mlp_neuron":
                mlp_hidden = patch_value
            
            mlp_out = block.mlp.c_proj(mlp_hidden)
            
            if self._should_apply_abstopk("mlp_out"):
                mlp_out = _apply_abstopk(mlp_out, self._activation_topk_fraction)
            if is_target_layer and target_location == "mlp_out":
                mlp_out = patch_value
            
            x = x + mlp_out
        
        x = self.model.ln_f(x)
        logits = self.model.lm_head(x)
        
        if hasattr(self.model, 'bigram_table') and self.model.bigram_table is not None:
            bigram_logits = F.embedding(input_ids, self.model.bigram_table)
            logits = logits + bigram_logits
        
        return logits
    
    def _capture_all_activations(
        self,
        input_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Capture activations at all locations (after AbsTopK)."""
        B, T = input_ids.shape
        device = input_ids.device
        activations = {}
        
        x = self.model.wte(input_ids)
        
        if torch.is_autocast_enabled('cuda'):
            x = x.to(torch.get_autocast_dtype('cuda'))
        
        if self.model.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x = x + self.model.wpe(pos)
        
        x = self.model.drop(x)
        
        for layer_idx, block in enumerate(self.model.blocks):
            # Attention sublayer
            normed = block.ln_1(x)
            if self._should_apply_abstopk("attn_in"):
                normed = _apply_abstopk(normed, self._activation_topk_fraction)
            activations[f"layer{layer_idx}_attn_in"] = normed.clone()
            
            qkv = block.attn.c_attn(normed)
            q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
            
            if self._should_apply_abstopk("attn_q"):
                q = _apply_abstopk(q, self._activation_topk_fraction)
            activations[f"layer{layer_idx}_attn_q"] = q.clone()
            
            if self._should_apply_abstopk("attn_k"):
                k = _apply_abstopk(k, self._activation_topk_fraction)
            activations[f"layer{layer_idx}_attn_k"] = k.clone()
            
            if self._should_apply_abstopk("attn_v"):
                v = _apply_abstopk(v, self._activation_topk_fraction)
            activations[f"layer{layer_idx}_attn_v"] = v.clone()
            
            q_reshaped = q.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
            k_reshaped = k.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
            v_reshaped = v.view(B, T, block.attn.n_heads, block.attn.d_head).transpose(1, 2)
            
            scale = 1.0 / (block.attn.d_head ** 0.5)
            if block.attn.use_flash and block.attn.attn_fn is not None:
                y = block.attn.attn_fn(q_reshaped, k_reshaped, v_reshaped, dropout_p=0.0, is_causal=True, scale=scale)
            elif block.attn.use_flash:
                y = F.scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped, dropout_p=0.0, is_causal=True, scale=scale)
            else:
                att = (q_reshaped @ k_reshaped.transpose(-2, -1)) * scale
                att = att.masked_fill(block.attn.causal_mask[:, :, :T, :T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                y = att @ v_reshaped
            
            y = y.transpose(1, 2).contiguous().view(B, T, block.attn.n_heads * block.attn.d_head)
            attn_out = block.attn.c_proj(y)
            
            if self._should_apply_abstopk("attn_out"):
                attn_out = _apply_abstopk(attn_out, self._activation_topk_fraction)
            activations[f"layer{layer_idx}_attn_out"] = attn_out.clone()
            
            x = x + attn_out
            
            # MLP sublayer
            normed = block.ln_2(x)
            if self._should_apply_abstopk("mlp_in"):
                normed = _apply_abstopk(normed, self._activation_topk_fraction)
            activations[f"layer{layer_idx}_mlp_in"] = normed.clone()
            
            mlp_hidden = block.mlp.c_fc(normed)
            mlp_hidden = block.mlp.act_fn(mlp_hidden)
            
            if self._should_apply_abstopk("mlp_neuron"):
                mlp_hidden = _apply_abstopk(mlp_hidden, self._activation_topk_fraction)
            activations[f"layer{layer_idx}_mlp_neuron"] = mlp_hidden.clone()
            
            mlp_out = block.mlp.c_proj(mlp_hidden)
            
            if self._should_apply_abstopk("mlp_out"):
                mlp_out = _apply_abstopk(mlp_out, self._activation_topk_fraction)
            activations[f"layer{layer_idx}_mlp_out"] = mlp_out.clone()
            
            x = x + mlp_out
        
        return activations
    
    def setup_batch(self, seed: int = 42):
        """Generate and cache a single large batch for all ablation tests."""
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        
        pos_ids, neg_ids, correct, incorrect, eval_pos = self.task.generate_batch(
            self.batch_size
        )
        self._cached_batch = {
            'pos_ids': pos_ids.to(self.device),
            'correct': correct.to(self.device),
            'incorrect': incorrect.to(self.device),
            'eval_pos': eval_pos.to(self.device),
        }
        
        # Capture all activations
        with torch.no_grad():
            self._cached_activations = self._capture_all_activations(
                self._cached_batch['pos_ids']
            )
            
            # Compute baseline loss
            logits = self._forward_with_patching(
                self._cached_batch['pos_ids'], -1, "", None
            )
            self._cached_baseline_loss = self.compute_task_loss(
                logits,
                self._cached_batch['correct'],
                self._cached_batch['incorrect'],
                self._cached_batch['eval_pos'],
            ).item()
        
        return self._cached_baseline_loss
    
    def compute_importance_for_location(
        self,
        layer: int,
        location: str,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute exact importance for all nodes at a specific location.
        
        Returns:
            importance: (n_nodes,) tensor of importance scores
            n_skipped: number of nodes skipped (zero activation everywhere)
        """
        assert self._cached_batch is not None, "Call setup_batch() first"
        
        key = f"layer{layer}_{location}"
        original_acts = self._cached_activations[key]  # (B, T, n_nodes)
        n_nodes = original_acts.shape[-1]
        
        # Find nodes that are zero everywhere (can skip these)
        # A node is zero everywhere if its activation is 0 at all (B, T) positions
        node_is_zero = (original_acts.abs().sum(dim=(0, 1)) == 0)  # (n_nodes,)
        n_skipped = node_is_zero.sum().item()
        
        importance = torch.zeros(n_nodes, device=self.device)
        
        iterator = range(n_nodes)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Layer {layer} {location}", leave=False)
        
        with torch.no_grad():
            for node_idx in iterator:
                # Skip if this node is zero everywhere
                if node_is_zero[node_idx]:
                    importance[node_idx] = 0.0
                    continue
                
                # Create ablated activations
                ablated_acts = original_acts.clone()
                ablated_acts[..., node_idx] = 0.0
                
                # Forward pass with ablated activations
                logits = self._forward_with_patching(
                    self._cached_batch['pos_ids'],
                    layer,
                    location,
                    patch_value=ablated_acts,
                )
                
                # Compute ablated loss
                ablated_loss = self.compute_task_loss(
                    logits,
                    self._cached_batch['correct'],
                    self._cached_batch['incorrect'],
                    self._cached_batch['eval_pos'],
                ).item()
                
                # Importance = loss increase when ablated
                importance[node_idx] = ablated_loss - self._cached_baseline_loss
        
        return importance, n_skipped
    
    def compute_all_importances(
        self,
        show_progress: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        """
        Compute exact importance for all nodes in the model.
        
        Returns:
            importances: Dict mapping "layer{i}_{location}" to importance tensors
            skipped: Dict mapping "layer{i}_{location}" to number of skipped nodes
        """
        importances = {}
        skipped = {}
        
        total_locations = self.n_layers * len(self.node_locations)
        
        for layer in range(self.n_layers):
            for location in self.node_locations:
                key = f"layer{layer}_{location}"
                if show_progress:
                    print(f"Processing {key}...")
                
                importance, n_skipped = self.compute_importance_for_location(
                    layer, location, show_progress=show_progress
                )
                importances[key] = importance
                skipped[key] = n_skipped
                
                if show_progress:
                    n_nodes = len(importance)
                    print(f"  {key}: {n_nodes} nodes, {n_skipped} skipped ({100*n_skipped/n_nodes:.1f}%)")
        
        return importances, skipped


def estimate_runtime(
    model: nn.Module,
    task: BinaryTask,
    batch_size: int = 256,
    use_binary_loss: bool = True,
    device: str = "cuda",
    test_duration: float = 60.0,
) -> Dict:
    """
    Estimate total runtime by running for a short duration and extrapolating.
    
    Args:
        model: The model to evaluate
        task: The task
        batch_size: Batch size
        use_binary_loss: Whether to use binary CE loss
        device: Device
        test_duration: How long to run the test (seconds)
        
    Returns:
        Dict with timing estimates
    """
    computer = ExactNodeImportanceComputer(
        model=model,
        task=task,
        batch_size=batch_size,
        use_binary_loss=use_binary_loss,
        device=device,
    )
    
    print("Setting up batch...")
    baseline_loss = computer.setup_batch()
    print(f"Baseline loss: {baseline_loss:.4f}")
    
    # Count total nodes
    total_nodes = 0
    total_nonzero = 0
    for layer in range(computer.n_layers):
        for location in computer.node_locations:
            key = f"layer{layer}_{location}"
            acts = computer._cached_activations[key]
            n_nodes = acts.shape[-1]
            n_nonzero = (acts.abs().sum(dim=(0, 1)) > 0).sum().item()
            total_nodes += n_nodes
            total_nonzero += n_nonzero
    
    print(f"Total nodes: {total_nodes}")
    print(f"Non-zero nodes: {total_nonzero} ({100*total_nonzero/total_nodes:.1f}%)")
    
    # Time a subset of nodes
    print(f"\nRunning timing test for {test_duration:.0f} seconds...")
    
    start_time = time.time()
    nodes_processed = 0
    
    for layer in range(computer.n_layers):
        for location in computer.node_locations:
            key = f"layer{layer}_{location}"
            original_acts = computer._cached_activations[key]
            n_nodes = original_acts.shape[-1]
            node_is_zero = (original_acts.abs().sum(dim=(0, 1)) == 0)
            
            with torch.no_grad():
                for node_idx in range(n_nodes):
                    elapsed = time.time() - start_time
                    if elapsed >= test_duration:
                        break
                    
                    if node_is_zero[node_idx]:
                        nodes_processed += 1
                        continue
                    
                    # Ablate and compute loss
                    ablated_acts = original_acts.clone()
                    ablated_acts[..., node_idx] = 0.0
                    
                    logits = computer._forward_with_patching(
                        computer._cached_batch['pos_ids'],
                        layer,
                        location,
                        patch_value=ablated_acts,
                    )
                    
                    _ = computer.compute_task_loss(
                        logits,
                        computer._cached_batch['correct'],
                        computer._cached_batch['incorrect'],
                        computer._cached_batch['eval_pos'],
                    )
                    
                    nodes_processed += 1
            
            elapsed = time.time() - start_time
            if elapsed >= test_duration:
                break
        
        elapsed = time.time() - start_time
        if elapsed >= test_duration:
            break
    
    elapsed = time.time() - start_time
    nodes_per_second = nodes_processed / elapsed
    
    # Estimate total time
    estimated_total_seconds = total_nonzero / nodes_per_second
    estimated_total_minutes = estimated_total_seconds / 60
    estimated_total_hours = estimated_total_minutes / 60
    
    results = {
        'total_nodes': total_nodes,
        'total_nonzero': total_nonzero,
        'nodes_processed': nodes_processed,
        'elapsed_seconds': elapsed,
        'nodes_per_second': nodes_per_second,
        'estimated_total_seconds': estimated_total_seconds,
        'estimated_total_minutes': estimated_total_minutes,
        'estimated_total_hours': estimated_total_hours,
        'baseline_loss': baseline_loss,
        'batch_size': batch_size,
    }
    
    print(f"\nTiming Results:")
    print(f"  Processed {nodes_processed} nodes in {elapsed:.1f}s")
    print(f"  Rate: {nodes_per_second:.2f} nodes/second")
    print(f"\nEstimated Total Runtime:")
    print(f"  {estimated_total_seconds:.0f} seconds")
    print(f"  {estimated_total_minutes:.1f} minutes")
    print(f"  {estimated_total_hours:.2f} hours")
    
    return results

