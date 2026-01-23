"""
Masked model wrapper for circuit pruning.

Wraps a SparseGPT model with node masks at all specified locations.
Provides mean ablation for masked nodes.

Based on Appendix A.5 of Gao et al. (2025).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Iterator
from tqdm import tqdm
import copy

from .node_mask import NodeMaskCollection, NodeMask
from .config import PruningConfig


class MaskedSparseGPT(nn.Module):
    """
    Wrapper around SparseGPT that applies learnable node masks.
    
    This module:
    1. Wraps a frozen SparseGPT model
    2. Inserts learnable masks at each node location
    3. Applies mean ablation for masked-out nodes
    4. Provides methods for computing task loss with mask sparsity penalty
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: PruningConfig,
    ):
        super().__init__()
        
        self.config = config
        
        # Store the base model (frozen during pruning)
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get model dimensions
        model_config = model.config
        self.n_layers = model_config.n_layer
        self.d_model = model_config.d_model
        self.d_mlp = model_config.d_mlp
        self.n_heads = model_config.n_heads
        self.d_head = model_config.d_head
        
        # Create mask collection
        self.masks = NodeMaskCollection(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_mlp=self.d_mlp,
            n_heads=self.n_heads,
            d_head=self.d_head,
            mask_locations=config.mask_locations,
            init_noise_scale=config.init_noise_scale,
            init_noise_bias=config.init_noise_bias,
            temperature=config.heaviside_temp,
        )
        
        # Storage for captured activations during mean computation
        self._activation_cache = {}
        self._hooks = []
        
        # Current layer being processed (for masking function)
        self._current_layer = 0
        
        # Get AbsTopK config from base model's sparsity_config
        self._activation_sparsity_enabled = False
        self._activation_topk_fraction = 0.25
        self._activation_sparsity_locations = set()
        self._abstopk_cache = {}  # Cache AbsTopK modules by dimension
        
        if hasattr(model, 'sparsity_config'):
            sc = model.sparsity_config
            self._activation_sparsity_enabled = sc.enable_activation_sparsity
            self._activation_topk_fraction = sc.activation_topk_fraction
            if sc.activation_sparsity_locations:
                self._activation_sparsity_locations = set(sc.activation_sparsity_locations.split(','))
        
        # Token embedding mask (optional)
        self.token_mask = None
        self.vocab_size = model_config.vocab_size
        if config.mask_token_embeds:
            self.token_mask = NodeMask(
                num_nodes=self.vocab_size,
                init_noise_scale=config.init_noise_scale,
                init_noise_bias=config.init_noise_bias,
                temperature=config.heaviside_temp,
            )
        
        # Frozen layer norm scales storage (used when freeze_layernorm_scale=True)
        # Stores the normalization scale (1/RMS or 1/std) from unpruned forward pass
        self._frozen_ln_scales = None
    
    def _apply_abstopk(self, x: torch.Tensor, location: str) -> torch.Tensor:
        """Apply AbsTopK sparsity if enabled for this location."""
        if not self._activation_sparsity_enabled:
            return x
        if location not in self._activation_sparsity_locations:
            return x
        
        dim = x.shape[-1]
        k = max(1, int(dim * self._activation_topk_fraction))
        
        # Cache AbsTopK computation
        if dim not in self._abstopk_cache:
            # Store k value for this dimension
            self._abstopk_cache[dim] = k
        
        k = self._abstopk_cache[dim]
        
        # AbsTopK: keep top k values by absolute value, zero the rest
        if k >= dim:
            return x
        
        _, topk_indices = torch.topk(x.abs(), k, dim=-1, sorted=False)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, x.gather(-1, topk_indices))
        return result
    
    def _get_masking_fn(self):
        """
        Create a masking function compatible with SparseGPT's activation_sparsity_fn.
        
        This function is called at each activation location with (activation, location_type).
        First applies AbsTopK (to match base model), then applies node mask.
        """
        def apply_mask(x: torch.Tensor, location: str) -> torch.Tensor:
            # First apply AbsTopK to match base model's activation sparsity
            x = self._apply_abstopk(x, location)
            
            # Then apply our node mask (if this location is being masked)
            if location not in self.config.mask_locations:
                return x
            
            # Get the mask for this layer and location
            mask_module = self.masks.get_mask(self._current_layer, location)
            
            # Apply mask with mean ablation
            return mask_module(x)
        
        return apply_mask
    
    def _compute_ln_rms_scale(self, x: torch.Tensor, ln_module: nn.Module) -> torch.Tensor:
        """
        Compute the RMS normalization scale for a layer norm module.
        
        For RMSNorm: scale = 1 / sqrt(mean(x^2) + eps)
        For LayerNorm: scale = 1 / sqrt(var(x) + eps)
        
        Args:
            x: Input tensor of shape (..., d_model)
            ln_module: The layer norm module (RMSNorm or LayerNorm)
            
        Returns:
            scale: The normalization scale of shape (..., 1)
        """
        # Default eps value (PyTorch RMSNorm may have eps=None)
        default_eps = 1e-6
        
        if isinstance(ln_module, nn.RMSNorm):
            # RMSNorm: scale = 1 / sqrt(mean(x^2) + eps)
            eps = ln_module.eps if ln_module.eps is not None else default_eps
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
            return 1.0 / rms
        else:
            # LayerNorm: scale = 1 / sqrt(var(x) + eps)
            eps = ln_module.eps if ln_module.eps is not None else default_eps
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            return 1.0 / torch.sqrt(var + eps)
    
    def _apply_ln_with_frozen_scale(
        self, 
        x: torch.Tensor, 
        ln_module: nn.Module, 
        frozen_scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply layer norm using a frozen scale but the current input.
        
        This allows gradients to flow through x while using frozen normalization statistics.
        
        For RMSNorm: output = x * frozen_scale * weight
        For LayerNorm: output = (x - mean(x)) * frozen_scale * weight + bias
        
        Args:
            x: Current input tensor (masked residual stream)
            ln_module: The layer norm module (for weight/bias parameters)
            frozen_scale: The frozen normalization scale from unpruned forward
            
        Returns:
            Normalized output with gradients flowing through x
        """
        if isinstance(ln_module, nn.RMSNorm):
            # RMSNorm: just scale and apply weight
            # Note: frozen_scale is already 1/rms, so we multiply
            return x * frozen_scale * ln_module.weight
        else:
            # LayerNorm: center, scale, apply weight and bias
            mean = x.mean(dim=-1, keepdim=True)
            centered = x - mean
            normed = centered * frozen_scale
            output = normed * ln_module.weight
            if ln_module.bias is not None:
                output = output + ln_module.bias
            return output
    
    def _compute_frozen_ln_scales(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass WITHOUT any masking to capture layer norm scales.
        
        The scale is the normalization factor (1/RMS or 1/std) computed from unpruned activations.
        These scales can be used in the masked forward pass when freeze_layernorm_scale=True.
        
        Args:
            input_ids: (batch, seq) token IDs
            
        Returns:
            Dictionary mapping layer norm keys to their scales (detached).
            Keys are: "layer{i}_ln_1", "layer{i}_ln_2" for each layer, and "ln_f" for final.
        """
        frozen_scales = {}
        
        B, T = input_ids.shape
        device = input_ids.device
        
        with torch.no_grad():
            # Token embeddings (no masking - this is the unpruned forward)
            x = self.model.wte(input_ids)
            
            # Cast to autocast dtype if enabled
            if torch.is_autocast_enabled('cuda'):
                x = x.to(torch.get_autocast_dtype('cuda'))
            
            # Add positional embeddings if enabled
            if self.model.wpe is not None:
                pos = torch.arange(0, T, dtype=torch.long, device=device)
                x = x + self.model.wpe(pos)
            
            x = self.model.drop(x)
            
            # Process through each transformer block WITHOUT masks
            for layer_idx, block in enumerate(self.model.blocks):
                # Capture ln_1 scale (before attention)
                frozen_scales[f"layer{layer_idx}_ln_1"] = self._compute_ln_rms_scale(x, block.ln_1).detach()
                
                # Run attention (unpruned) - need full ln output for attention
                normed_1 = block.ln_1(x)
                attn_out = block.attn(normed_1, None)
                x = x + attn_out
                
                # Capture ln_2 scale (before MLP)
                frozen_scales[f"layer{layer_idx}_ln_2"] = self._compute_ln_rms_scale(x, block.ln_2).detach()
                
                # Run MLP (unpruned) - need full ln output for MLP
                normed_2 = block.ln_2(x)
                mlp_out = block.mlp(normed_2)
                x = x + mlp_out
            
            # Capture final layer norm scale
            frozen_scales["ln_f"] = self._compute_ln_rms_scale(x, self.model.ln_f).detach()
        
        return frozen_scales
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_logits_only: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with masks applied.
        
        Args:
            input_ids: (batch, seq) token IDs
            return_logits_only: If True, return only logits; else return (logits, loss, hidden)
            
        Returns:
            logits: (batch, seq, vocab) if return_logits_only
        """
        # If freeze_layernorm_scale is enabled, first compute frozen LN scales
        if self.config.freeze_layernorm_scale:
            self._frozen_ln_scales = self._compute_frozen_ln_scales(input_ids)
        else:
            self._frozen_ln_scales = None
        
        # We need to manually process through the model to apply masks at each layer
        # This requires modifying the forward to track the current layer
        
        B, T = input_ids.shape
        device = input_ids.device
        model_config = self.model.config
        
        # Token embeddings
        x = self.model.wte(input_ids)
        
        # Cast to autocast dtype if enabled
        if torch.is_autocast_enabled('cuda'):
            x = x.to(torch.get_autocast_dtype('cuda'))
        
        # Apply token mask if enabled
        # For each token t in input_ids, if token_mask[t] = 0, replace embedding with mean
        if self.token_mask is not None:
            # Get binary mask for each token type: (vocab_size,)
            token_binary_mask = self.token_mask.get_binary_mask()
            # Index into mask for each token in input: (B, T)
            mask_per_position = token_binary_mask[input_ids]
            # Expand to match embedding dim: (B, T, 1)
            mask_per_position = mask_per_position.unsqueeze(-1)
            
            if self.token_mask.mean_set:
                # Mean ablation: replace masked tokens with dataset mean embedding
                # mean_activation has shape (d_model,) - single mean embedding
                mean_embed = self.token_mask.mean_activation  # (d_model,)
                x = x * mask_per_position + mean_embed * (1 - mask_per_position)
            else:
                # Zero ablation
                x = x * mask_per_position
        
        # Add positional embeddings if enabled
        if self.model.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            x = x + self.model.wpe(pos)
        
        x = self.model.drop(x)
        
        # Get our masking function
        mask_fn = self._get_masking_fn()
        
        # Process through each transformer block
        for layer_idx, block in enumerate(self.model.blocks):
            self._current_layer = layer_idx
            x = self._forward_block_with_masks(block, x, mask_fn)
        
        # Final layer norm
        # Apply frozen scale if enabled (gradients still flow through x)
        if self._frozen_ln_scales is not None:
            frozen_scale = self._frozen_ln_scales["ln_f"]
            x = self._apply_ln_with_frozen_scale(x, self.model.ln_f, frozen_scale)
        else:
            x = self.model.ln_f(x)
        
        # LM head
        logits = self.model.lm_head(x)
        
        # Add bigram logits if enabled
        if hasattr(self.model, 'bigram_table') and self.model.bigram_table is not None:
            bigram_logits = F.embedding(input_ids, self.model.bigram_table)
            logits = logits + bigram_logits
        
        if return_logits_only:
            return logits
        else:
            return logits, None, None
    
    def _forward_block_with_masks(
        self,
        block: nn.Module,
        x: torch.Tensor,
        mask_fn,
    ) -> torch.Tensor:
        """Forward through a single block with masks applied."""
        
        layer_idx = self._current_layer
        
        # Attention sublayer
        # Apply attn_in mask after layer norm
        if self._frozen_ln_scales is not None:
            # Apply layer norm with frozen scale but current (masked) input
            # This allows gradients to flow through x while using frozen normalization
            frozen_scale = self._frozen_ln_scales[f"layer{layer_idx}_ln_1"]
            normed = self._apply_ln_with_frozen_scale(x, block.ln_1, frozen_scale)
        else:
            normed = block.ln_1(x)
        normed = mask_fn(normed, "attn_in")
        
        # Get attention output with Q/K/V masks
        attn_out = self._forward_attention_with_masks(block.attn, normed, mask_fn)
        
        # Apply attn_out mask
        attn_out = mask_fn(attn_out, "attn_out")
        
        # Residual connection
        x = x + attn_out
        
        # MLP sublayer
        # Apply mlp_in mask after layer norm
        if self._frozen_ln_scales is not None:
            # Apply layer norm with frozen scale but current (masked) input
            frozen_scale = self._frozen_ln_scales[f"layer{layer_idx}_ln_2"]
            normed = self._apply_ln_with_frozen_scale(x, block.ln_2, frozen_scale)
        else:
            normed = block.ln_2(x)
        normed = mask_fn(normed, "mlp_in")
        
        # Get MLP output with neuron mask
        mlp_out = self._forward_mlp_with_masks(block.mlp, normed, mask_fn)
        
        # Apply mlp_out mask
        mlp_out = mask_fn(mlp_out, "mlp_out")
        
        # Residual connection
        x = x + mlp_out
        
        return x
    
    def _forward_attention_with_masks(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        mask_fn,
    ) -> torch.Tensor:
        """Forward through attention with Q/K/V masks."""
        B, T, C = x.shape
        
        # QKV projection
        qkv = attn.c_attn(x)
        q, k, v = qkv.split(attn.n_heads * attn.d_head, dim=-1)
        
        # Apply Q/K/V masks
        q = mask_fn(q, "attn_q")
        k = mask_fn(k, "attn_k")
        v = mask_fn(v, "attn_v")
        
        # Reshape for attention
        q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        k = k.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        v = v.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        
        scale = 1.0 / (attn.d_head ** 0.5)
        
        # Compute attention
        if attn.use_flash and attn.attn_fn is not None:
            # Use attention with sinks
            y = attn.attn_fn(
                q, k, v,
                dropout_p=attn.dropout if attn.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        elif attn.use_flash:
            # Standard flash attention
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=attn.dropout if attn.training else 0.0,
                is_causal=True,
                scale=scale,
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(
                attn.causal_mask[:, :, :T, :T] == 0,
                float('-inf')
            )
            att = F.softmax(att, dim=-1)
            att = attn.attn_dropout(att)
            y = att @ v
        
        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, attn.n_heads * attn.d_head)
        
        # Output projection
        y = attn.c_proj(y)
        y = attn.resid_dropout(y)
        
        return y
    
    def _forward_mlp_with_masks(
        self,
        mlp: nn.Module,
        x: torch.Tensor,
        mask_fn,
    ) -> torch.Tensor:
        """Forward through MLP with neuron mask."""
        # FC layer
        x = mlp.c_fc(x)
        
        # Activation
        x = mlp.act_fn(x)
        
        # Apply neuron mask (post-activation)
        x = mask_fn(x, "mlp_neuron")
        
        # Output projection
        x = mlp.c_proj(x)
        x = mlp.dropout(x)
        
        return x
    
    def compute_task_loss(
        self,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        eval_positions: Optional[torch.Tensor] = None,
        use_binary_loss: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute task cross-entropy loss for a batch of binary task examples.
        
        Args:
            positive_ids: (batch, seq) positive example sequences (right-padded)
            negative_ids: (batch, seq) negative example sequences (same context, different answer)
            correct_tokens: (batch,) correct completion tokens
            incorrect_tokens: (batch,) incorrect completion tokens
            eval_positions: (batch,) position to evaluate logits for each example.
                           If None, uses last position (-1) for all examples.
            use_binary_loss: If True, compute CE over only [correct, incorrect] logits.
                            If None, uses self.config.use_binary_loss.
            
        Returns:
            loss: Task cross-entropy loss
            metrics: Dictionary of metrics
        """
        batch_size = positive_ids.shape[0]
        
        # Determine whether to use binary loss
        if use_binary_loss is None:
            use_binary_loss = self.config.use_binary_loss
        
        # Forward pass on positive examples
        logits = self.forward(positive_ids)  # (batch, seq, vocab)
        
        # Get logits at the correct position for each batch element
        if eval_positions is not None:
            # Index into logits at the specified position for each batch element
            # eval_positions: (batch,) containing positions like [45, 32, 50, ...]
            # We need logits[i, eval_positions[i], :] for each i
            batch_indices = torch.arange(batch_size, device=logits.device)
            final_logits = logits[batch_indices, eval_positions, :]  # (batch, vocab)
        else:
            # Fallback: use last position (assumes all sequences have same length)
            final_logits = logits[:, -1, :]  # (batch, vocab)
        
        # Get correct and incorrect logits
        correct_logits = final_logits.gather(1, correct_tokens.unsqueeze(1)).squeeze(1)
        incorrect_logits = final_logits.gather(1, incorrect_tokens.unsqueeze(1)).squeeze(1)
        
        if use_binary_loss:
            # Binary cross-entropy: softmax over only [correct, incorrect] logits
            # Stack to get (batch, 2) tensor where index 0 is correct, index 1 is incorrect
            binary_logits = torch.stack([correct_logits, incorrect_logits], dim=1)  # (batch, 2)
            # CE loss for predicting class 0 (correct)
            targets = torch.zeros(batch_size, dtype=torch.long, device=positive_ids.device)
            loss = F.cross_entropy(binary_logits, targets)
            
            # Binary accuracy
            with torch.no_grad():
                binary_probs = F.softmax(binary_logits, dim=1)
                binary_accuracy = (binary_probs[:, 0] > 0.5).float().mean().item()
        else:
            # Full vocabulary cross-entropy loss
            loss = F.cross_entropy(final_logits, correct_tokens)
            binary_accuracy = None
        
        # Compute metrics
        with torch.no_grad():
            pred_tokens = final_logits.argmax(dim=-1)
            accuracy = (pred_tokens == correct_tokens).float().mean().item()
            logit_diff = (correct_logits - incorrect_logits).mean().item()
        
        metrics = {
            "task_loss": loss.item(),
            "accuracy": accuracy,
            "logit_diff": logit_diff,
        }
        
        if binary_accuracy is not None:
            metrics["binary_accuracy"] = binary_accuracy
        
        return loss, metrics
    
    def compute_full_loss(
        self,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
        correct_tokens: torch.Tensor,
        incorrect_tokens: torch.Tensor,
        k_coef: float,
        eval_positions: Optional[torch.Tensor] = None,
        use_binary_loss: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute full pruning loss: task_loss + k_coef * num_active_nodes.
        
        Args:
            positive_ids, negative_ids, correct_tokens, incorrect_tokens: Task data
            k_coef: Coefficient for sparsity penalty
            eval_positions: (batch,) position to evaluate logits for each example
            use_binary_loss: If True, compute CE over only [correct, incorrect] logits.
            
        Returns:
            total_loss: Combined loss
            metrics: Dictionary of metrics
        """
        # Task loss
        task_loss, metrics = self.compute_task_loss(
            positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions,
            use_binary_loss=use_binary_loss
        )
        
        # Sparsity penalty
        sparsity_loss = self.masks.get_sparsity_loss()
        
        # Add token mask sparsity if enabled
        token_sparsity_loss = torch.tensor(0.0, device=sparsity_loss.device)
        if self.token_mask is not None:
            token_sparsity_loss = self.token_mask.get_binary_mask().sum()
            sparsity_loss = sparsity_loss + token_sparsity_loss
        
        # Total loss
        total_loss = task_loss + k_coef * sparsity_loss
        
        # Update metrics
        metrics["sparsity_loss"] = sparsity_loss.item()
        metrics["total_loss"] = total_loss.item()
        metrics["num_active_nodes"] = self.masks.get_total_active_nodes()
        metrics["total_nodes"] = self.masks.get_total_nodes()
        
        # Add token mask stats if enabled
        if self.token_mask is not None:
            metrics["num_active_tokens"] = self.token_mask.get_num_active()
            metrics["total_tokens"] = self.vocab_size
            metrics["token_sparsity_loss"] = token_sparsity_loss.item()
        
        return total_loss, metrics
    
    def clamp_mask_parameters(self):
        """Clamp all mask parameters to [-1, 1]."""
        self.masks.clamp_all_parameters()
        if self.token_mask is not None:
            self.token_mask.clamp_parameters()
    
    def compute_mean_cache(
        self,
        data_iterator: Iterator[torch.Tensor],
        num_batches: int = 100,
        show_progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mean activations at each node location.
        
        Args:
            data_iterator: Iterator yielding batches of input_ids
            num_batches: Number of batches to process
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping location keys to mean tensors
        """
        device = next(self.model.parameters()).device
        
        # Initialize accumulators
        means = {}
        counts = {}
        
        for layer in range(self.n_layers):
            for loc in self.config.mask_locations:
                key = f"layer{layer}_{loc}"
                dim = self.masks._get_dim_for_location(loc)
                means[key] = torch.zeros(dim, device=device)
                counts[key] = 0
        
        # Token embedding mean cache (single mean embedding over all tokens)
        if self.token_mask is not None:
            means["token_embed"] = torch.zeros(self.d_model, device=device)
            counts["token_embed"] = 0
        
        # Capture hook
        captured = {}
        
        def make_capture_fn(layer_idx):
            def capture_fn(x: torch.Tensor, location: str) -> torch.Tensor:
                if location in self.config.mask_locations:
                    key = f"layer{layer_idx}_{location}"
                    # Store flattened activation
                    captured[key] = x.detach().reshape(-1, x.shape[-1])
                return x  # Pass through unchanged
            return capture_fn
        
        self.model.eval()
        
        iterator = tqdm(data_iterator, total=num_batches, desc="Computing mean cache") if show_progress else data_iterator
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                if batch_idx >= num_batches:
                    break
                
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                else:
                    input_ids = batch.to(device)
                
                # Forward with capture (no masks applied)
                B, T = input_ids.shape
                
                x = self.model.wte(input_ids)
                
                if torch.is_autocast_enabled('cuda'):
                    x = x.to(torch.get_autocast_dtype('cuda'))
                
                # Accumulate token embedding means (before positional embeddings)
                if self.token_mask is not None:
                    # x has shape (B, T, d_model)
                    # Compute mean over all positions in this batch
                    batch_mean = x.reshape(-1, self.d_model).mean(dim=0)  # (d_model,)
                    batch_count = B * T
                    
                    # Running mean update
                    old_count = counts["token_embed"]
                    new_count = old_count + batch_count
                    if old_count == 0:
                        means["token_embed"] = batch_mean
                    else:
                        delta = batch_mean - means["token_embed"]
                        means["token_embed"] = means["token_embed"] + delta * (batch_count / new_count)
                    counts["token_embed"] = new_count
                
                if self.model.wpe is not None:
                    pos = torch.arange(0, T, dtype=torch.long, device=device)
                    x = x + self.model.wpe(pos)
                
                x = self.model.drop(x)
                
                for layer_idx, block in enumerate(self.model.blocks):
                    captured.clear()
                    capture_fn = make_capture_fn(layer_idx)
                    
                    # Capture at each location
                    # attn_in
                    normed = block.ln_1(x)
                    capture_fn(normed, "attn_in")
                    
                    # Q, K, V
                    qkv = block.attn.c_attn(normed)
                    q, k, v = qkv.split(block.attn.n_heads * block.attn.d_head, dim=-1)
                    capture_fn(q, "attn_q")
                    capture_fn(k, "attn_k")
                    capture_fn(v, "attn_v")
                    
                    # Run attention
                    attn_out = block.attn(normed, None)
                    capture_fn(attn_out, "attn_out")
                    
                    x = x + attn_out
                    
                    # mlp_in
                    normed = block.ln_2(x)
                    capture_fn(normed, "mlp_in")
                    
                    # MLP forward to get neuron activations
                    mlp_hidden = block.mlp.c_fc(normed)
                    mlp_hidden = block.mlp.act_fn(mlp_hidden)
                    capture_fn(mlp_hidden, "mlp_neuron")
                    
                    mlp_out = block.mlp.c_proj(mlp_hidden)
                    mlp_out = block.mlp.dropout(mlp_out)
                    capture_fn(mlp_out, "mlp_out")
                    
                    x = x + mlp_out
                    
                    # Update running means
                    for key, act in captured.items():
                        batch_mean = act.mean(dim=0)
                        batch_count = act.shape[0]
                        
                        old_count = counts[key]
                        new_count = old_count + batch_count
                        
                        if old_count == 0:
                            means[key] = batch_mean
                        else:
                            delta = batch_mean - means[key]
                            means[key] = means[key] + delta * (batch_count / new_count)
                        
                        counts[key] = new_count
        
        # Token embedding mean is already computed via running mean, no finalization needed
        
        return means
    
    def set_means_from_dict(self, mean_dict: Dict[str, torch.Tensor]):
        """Set mean activations for all masks from a dictionary."""
        self.masks.set_means_from_cache(mean_dict)
        
        # Set token embedding mean if present
        # token_mask.mean_activation is (d_model,) - single mean embedding
        if self.token_mask is not None and "token_embed" in mean_dict:
            # Re-register the buffer with the correct shape
            self.token_mask.register_buffer("mean_activation", mean_dict["token_embed"].clone())
            self.token_mask.mean_set = True
    
    def get_mask_state(self) -> Dict[str, torch.Tensor]:
        """Get the current state of all masks (tau values)."""
        state = {key: mask.tau.clone() for key, mask in self.masks.masks.items()}
        if self.token_mask is not None:
            state["token_embed"] = self.token_mask.tau.clone()
        return state
    
    def load_mask_state(self, state: Dict[str, torch.Tensor]):
        """Load mask state from a dictionary."""
        for key, tau in state.items():
            if key == "token_embed":
                if self.token_mask is not None:
                    self.token_mask.tau.data.copy_(tau)
            elif key in self.masks.masks:
                self.masks.masks[key].tau.data.copy_(tau)
    
    def get_circuit_mask(self) -> Dict[str, torch.Tensor]:
        """Get binary circuit mask (which nodes are active)."""
        circuit = {
            key: (mask.tau >= 0).float()
            for key, mask in self.masks.masks.items()
        }
        if self.token_mask is not None:
            circuit["token_embed"] = (self.token_mask.tau >= 0).float()
        return circuit
    
    def count_edges(self) -> int:
        """
        Count number of edges in the current circuit.
        
        An edge is a nonzero weight connecting two active nodes.
        This is the primary interpretability metric from the paper.
        """
        # This requires access to the model weights and is more complex
        # For now, return number of active nodes as a proxy
        return self.masks.get_total_active_nodes()

