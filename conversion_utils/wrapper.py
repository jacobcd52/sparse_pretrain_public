"""
Wrapper classes for using HookedSparseGPT with circuit_tracer.

This module provides SparseGPTReplacementModel which wraps HookedSparseGPT
and configures it identically to circuit_tracer's ReplacementModel:
- MLPs are wrapped with ReplacementMLP (adds hook_in, hook_out)
- Unembed is wrapped with ReplacementUnembed (adds hook_pre, hook_post)
- Skip connection logic detaches MLP outputs and routes gradients through grad hooks
- Attention patterns and LayerNorm scales are frozen via stop_gradient
- All parameters are frozen except embeddings
"""

from functools import partial
from typing import Optional, List, Union, Callable, Tuple
import warnings

import torch
import torch.nn as nn
from transformer_lens.hook_points import HookPoint


class ReplacementMLP(nn.Module):
    """Wrapper for HookedSparseGPT MLP that adds hook_in and hook_out hook points.
    
    This matches circuit_tracer's ReplacementMLP.
    """

    def __init__(self, old_mlp: nn.Module):
        super().__init__()
        self.old_mlp = old_mlp
        self.hook_in = HookPoint()
        self.hook_out = HookPoint()

    def forward(self, x, activation_sparsity_fn=None):
        x = self.hook_in(x)
        # HookedSparseGPT's MLP.forward takes activation_sparsity_fn
        mlp_out = self.old_mlp(x, activation_sparsity_fn)
        return self.hook_out(mlp_out)


class ReplacementUnembed(nn.Module):
    """Wrapper for HookedSparseGPT Unembed that adds hook_pre and hook_post.
    
    This matches circuit_tracer's ReplacementUnembed.
    """

    def __init__(self, old_unembed: nn.Module):
        super().__init__()
        self.old_unembed = old_unembed
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    @property
    def W_U(self):
        return self.old_unembed.W_U

    @property
    def b_U(self):
        return self.old_unembed.b_U

    def forward(self, x):
        x = self.hook_pre(x)
        x = self.old_unembed(x)
        return self.hook_post(x)


class SparseGPTReplacementModel(nn.Module):
    """
    Wrapper for HookedSparseGPT with CLT transcoders.
    
    This wrapper configures the model identically to circuit_tracer's ReplacementModel:
    1. Wraps MLPs with ReplacementMLP (adds hook_in, hook_out)
    2. Wraps unembed with ReplacementUnembed (adds hook_pre, hook_post)
    3. Configures skip connection: caches input, detaches output, adds grad_hook
    4. Freezes attention patterns and LayerNorm scales via stop_gradient
    5. Freezes all parameters, then re-enables gradient on embeddings
    
    Example:
        >>> from sparse_pretrain.src import HookedSparseGPT
        >>> from dictionary_learning.conversion_utils import CLTAdapter
        >>> 
        >>> model = HookedSparseGPT.from_pretrained("jacobcd52/ss_d128_f1", device="cuda")
        >>> clt_adapter = CLTAdapter.from_pretrained("jacobcd52/ss_d128_f1_clt_k30_e32", ...)
        >>> replacement_model = SparseGPTReplacementModel(model, clt_adapter)
    """
    
    def __init__(self, model, transcoders):
        """
        Args:
            model: HookedSparseGPT model
            transcoders: CLTAdapter wrapping the cross-layer transcoders
        """
        super().__init__()
        self.model = model
        self.transcoders = transcoders
        self.tokenizer = model.tokenizer
        self.cfg = model.cfg
        
        # Hook point names from transcoders
        self.feature_input_hook = transcoders.feature_input_hook
        self.original_feature_output_hook = transcoders.feature_output_hook
        # circuit_tracer expects ".hook_out_grad" suffix for backward hooks
        self.feature_output_hook = transcoders.feature_output_hook + ".hook_out_grad"
        self.skip_transcoder = transcoders.skip_connection
        self.scan = transcoders.scan
        
        # Replace MLPs and unembed with wrappers (like circuit_tracer)
        self._configure_replacement_model()
    
    def _configure_replacement_model(self):
        """
        Configure the model like circuit_tracer's ReplacementModel:
        1. Wrap MLPs with ReplacementMLP
        2. Wrap unembed with ReplacementUnembed
        3. Configure gradient flow
        4. Rebuild hook_dict
        """
        # Wrap each block's MLP
        for block in self.model.blocks:
            block.mlp = ReplacementMLP(block.mlp)
        
        # Wrap unembed
        self.model.unembed = ReplacementUnembed(self.model.unembed)
        
        # Configure gradient flow
        self._configure_gradient_flow()
        
        # Rebuild hook_dict to include new hook points
        self.model.setup()
    
    def _configure_gradient_flow(self):
        """
        Configure gradient flow for attribution.
        
        This matches circuit_tracer's ReplacementModel._configure_gradient_flow():
        1. Configure skip connection for each layer
        2. Stop gradients on attention patterns and layernorm scales
        3. Freeze all parameters
        4. Enable gradients on embeddings
        """
        # Configure skip connection for each layer
        for layer in range(self.cfg.n_layers):
            self._configure_skip_connection(self.model.blocks[layer])
        
        # Stop gradients on attention patterns and layernorm scales
        def stop_gradient(acts, hook):
            return acts.detach()
        
        for block in self.model.blocks:
            block.attn.hook_pattern.add_hook(stop_gradient, is_permanent=True)
            block.ln1.hook_scale.add_hook(stop_gradient, is_permanent=True)
            block.ln2.hook_scale.add_hook(stop_gradient, is_permanent=True)
            # Note: HookedSparseGPT doesn't have ln1_post/ln2_post
        
        # ln_final - add hook once (outside the loop, unlike circuit_tracer's bug)
        self.model.ln_final.hook_scale.add_hook(stop_gradient, is_permanent=True)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Re-enable gradients on embeddings
        def enable_gradient(tensor, hook):
            tensor.requires_grad = True
            return tensor
        
        self.model.hook_embed.add_hook(enable_gradient, is_permanent=True)
    
    def _configure_skip_connection(self, block):
        """
        Configure skip connection gradient flow for a single block.
        
        This matches circuit_tracer's ReplacementModel._configure_skip_connection():
        - Cache activations at feature input hook (mlp.hook_in)
        - At feature output hook (mlp.hook_out), replace with:
          grad_hook(skip + (acts - skip).detach())
        - Since our CLTs don't have skip connections (W_skip), skip = 0
        
        This effectively:
        - Detaches the MLP output from the computation graph
        - Routes gradients through the grad_hook for attribution
        """
        cached = {}
        
        def cache_activations(acts, hook):
            cached["acts"] = acts
        
        def add_skip_connection(acts: torch.Tensor, hook: HookPoint, grad_hook: HookPoint):
            # We add grad_hook because we need a way to hook into the gradients of the output
            # of this function. If we put the backwards hook here at hook, the grads will be 0
            # because we detached acts.
            skip_input_activation = cached.pop("acts")
            
            # Check if transcoder has skip connection
            if hasattr(self.transcoders, "W_skip") and self.transcoders.W_skip is not None:
                skip = self.transcoders.compute_skip(skip_input_activation)
            else:
                # No skip connection - gradients only flow through grad_hook
                skip = skip_input_activation * 0
            
            return grad_hook(skip + (acts - skip).detach())
        
        # Add cache hook at feature input (now it's mlp.hook_in due to ReplacementMLP)
        block.mlp.hook_in.add_hook(cache_activations, is_permanent=True)
        
        # Add skip connection hook at feature output (mlp.hook_out)
        # Create the grad hook for backward pass
        block.mlp.hook_out.hook_out_grad = HookPoint()
        
        # Add the skip connection hook
        block.mlp.hook_out.add_hook(
            partial(add_skip_connection, grad_hook=block.mlp.hook_out.hook_out_grad),
            is_permanent=True,
        )
    
    # =========================================================================
    # Properties delegated to model
    # =========================================================================
    
    @property
    def W_E(self):
        """Embedding matrix [d_vocab, d_model]."""
        return self.model.W_E
    
    @property 
    def W_U(self):
        """Unembedding matrix [d_model, d_vocab]."""
        return self.model.W_U
    
    @property
    def blocks(self):
        """Transformer blocks."""
        return self.model.blocks
    
    @property
    def unembed(self):
        """Unembed module (with hook_pre, hook_post, W_U, b_U)."""
        return self.model.unembed
    
    @property
    def embed(self):
        """Embedding module."""
        return self.model.embed
    
    @property
    def hook_embed(self):
        """Embedding hook point."""
        return self.model.hook_embed
    
    @property
    def ln_final(self):
        """Final layer norm."""
        return self.model.ln_final
    
    @property
    def hook_dict(self):
        """Dictionary mapping hook names to hook points."""
        return self.model.hook_dict
    
    @property
    def device(self):
        return self.cfg.device
    
    @property
    def dtype(self):
        return self.cfg.dtype
    
    # =========================================================================
    # Methods delegated to model
    # =========================================================================
    
    def ensure_tokenized(self, prompt: Union[str, torch.Tensor, List[int]], prepend_bos: bool = False) -> torch.Tensor:
        """Convert prompt to tokens, handling various input types.
        
        Args:
            prompt: String, tensor, or list of token ids
            prepend_bos: If True, prepend a special token (BOS/PAD). Default False.
        """
        return self.model.ensure_tokenized(prompt, prepend_bos=prepend_bos)
    
    def forward(self, input_ids, stop_at_layer=None, **kwargs):
        """Forward pass through the model."""
        return self.model(input_ids, stop_at_layer=stop_at_layer, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make the wrapper callable like the model."""
        return self.forward(*args, **kwargs)
    
    def run_with_cache(self, input_ids, names_filter=None, return_type="logits", **kwargs):
        """Run forward pass and cache activations at hook points."""
        return self.model.run_with_cache(
            input_ids, 
            names_filter=names_filter, 
            return_type=return_type, 
            **kwargs
        )
    
    def run_with_hooks(self, input_ids, fwd_hooks=None, return_type=None, **kwargs):
        """Run forward pass with custom hooks."""
        return self.model.run_with_hooks(
            input_ids, 
            fwd_hooks=fwd_hooks, 
            return_type=return_type or "logits", 
            **kwargs
        )
    
    def hooks(self, fwd_hooks=None, bwd_hooks=None, reset_hooks_end=True, clear_contexts=False):
        """Context manager for temporarily adding hooks (from HookedRootModule)."""
        if fwd_hooks is None:
            fwd_hooks = []
        if bwd_hooks is None:
            bwd_hooks = []
        return self.model.hooks(
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd_hooks,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        )
    
    def get_caching_hooks(self, names_filter=None, incl_bwd=False, device=None, remove_batch_dim=False):
        """Get caching hooks for specified hook points (from HookedRootModule)."""
        return self.model.get_caching_hooks(
            names_filter=names_filter,
            incl_bwd=incl_bwd,
            device=device,
            remove_batch_dim=remove_batch_dim,
        )
    
    def add_hook(self, name, hook, dir="fwd", is_permanent=False, level=None, prepend=False):
        """Add a hook by name (from HookedRootModule)."""
        return self.model.add_hook(
            name=name,
            hook=hook,
            dir=dir,
            is_permanent=is_permanent,
            level=level,
            prepend=prepend,
        )
    
    def reset_hooks(self, including_permanent=False, level=None):
        """Remove all hooks (from HookedRootModule)."""
        return self.model.reset_hooks(including_permanent=including_permanent, level=level)
    
    def setup(self):
        """Setup hook dict (from HookedRootModule)."""
        return self.model.setup()
    
    def to_tokens(self, text, **kwargs):
        """Tokenize text."""
        return self.model.to_tokens(text, **kwargs)
    
    def to_str_tokens(self, tokens, **kwargs):
        """Convert tokens to string representation."""
        return self.model.to_str_tokens(tokens, **kwargs)
    
    def _get_hook_point(self, name: str) -> Optional[HookPoint]:
        """Get a hook point by name."""
        parts = name.split(".")
        obj = self.model
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj if isinstance(obj, HookPoint) else None
    
    @torch.no_grad()
    def setup_attribution(self, inputs: Union[str, torch.Tensor], *, zero_first_pos: bool = False):
        """
        Precompute transcoder activations and error vectors for attribution.
        
        This method implements the interface expected by circuit_tracer's attribute() function.
        
        Args:
            inputs: Either a string prompt or tensor of token ids
            zero_first_pos: If True, zero out features and errors at position 0
            
        Returns:
            AttributionContext with all components needed for attribution
        """
        # Import here to avoid circular imports
        from circuit_tracer.attribution.context import AttributionContext
        
        # Ensure we have tokens
        if isinstance(inputs, str):
            tokens = self.ensure_tokenized(inputs)
        else:
            tokens = inputs.squeeze()
        
        assert isinstance(tokens, torch.Tensor), "Tokens must be a tensor"
        assert tokens.ndim == 1, "Tokens must be a 1D tensor"
        
        # Cache activations at feature input and output hooks
        # CLTs are trained on hook_resid_mid (before LayerNorm), not mlp.hook_in
        # The feature_input_hook should match "hook_resid_mid"
        # The feature_output_hook should match "mlp.hook_out"
        resid_mid_cache, resid_mid_hooks, _ = self.get_caching_hooks(
            lambda name: self.feature_input_hook in name
        )
        # Use original_feature_output_hook to get mlp.hook_out, not hook_out_grad
        mlp_out_cache, mlp_out_hooks, _ = self.get_caching_hooks(
            lambda name: self.original_feature_output_hook in name and ".hook_out_grad" not in name
        )
        
        # Run forward pass with caching (add batch dim for forward pass)
        logits = self.run_with_hooks(tokens.unsqueeze(0), fwd_hooks=resid_mid_hooks + mlp_out_hooks)
        
        # Stack cached activations: [n_layers, n_pos, d_model]
        resid_mid_cache = torch.cat(list(resid_mid_cache.values()), dim=0)
        mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)
        
        # Compute attribution components using transcoders
        # CLTs are trained on resid_mid (before LayerNorm)
        attribution_data = self.transcoders.compute_attribution_components(
            resid_mid_cache, zero_first_pos=zero_first_pos
        )
        
        # Compute error vectors (residual not captured by CLT)
        error_vectors = mlp_out_cache - attribution_data["reconstruction"]
        
        if zero_first_pos:
            error_vectors[:, 0] = 0
        
        # Get token embeddings
        token_vectors = self.W_E[tokens].detach()  # [n_pos, d_model]
        
        return AttributionContext(
            activation_matrix=attribution_data["activation_matrix"],
            logits=logits,
            error_vectors=error_vectors,
            token_vectors=token_vectors,
            decoder_vecs=attribution_data["decoder_vecs"],
            encoder_vecs=attribution_data["encoder_vecs"],
            encoder_to_decoder_map=attribution_data["encoder_to_decoder_map"],
            decoder_locations=attribution_data["decoder_locations"],
        )
