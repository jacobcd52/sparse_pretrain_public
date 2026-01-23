"""
HookedSparseGPT - TransformerLens-compatible wrapper for SparseGPT.

Provides hook points for interpretability tools like:
- dictionary_learning (SAE training)
- circuit-tracer (attribution graphs)

Based on TransformerLens HookedTransformer interface.
"""

import math
from typing import Optional, Tuple, Union, List, Dict, Callable, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from transformer_lens.hook_points import HookedRootModule, HookPoint

from .config import ModelConfig, SparsityConfig
from .model import AbsTopK


# =============================================================================
# Configuration Wrapper
# =============================================================================

@dataclass
class HookedSparseGPTConfig:
    """
    TransformerLens-compatible config wrapper.
    
    Maps SparseGPT config attributes to TransformerLens naming conventions.
    """
    # Core dimensions
    n_layers: int
    d_model: int
    n_ctx: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int
    
    # Model features
    act_fn: str
    use_rms_norm: bool
    use_attention_sinks: bool
    use_bias: bool
    
    # Activation sparsity settings (from SparsityConfig)
    enable_activation_sparsity: bool = False
    activation_topk_fraction: float = 0.25
    activation_sparsity_locations: str = "attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,attn_v,attn_k,attn_q"
    
    # Naming
    model_name: str = "sparse_gpt"
    tokenizer_name: Optional[str] = None
    
    # Device/dtype
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    
    # TransformerLens compatibility
    normalization_type: str = "RMS"  # or "LN"
    default_prepend_bos: bool = False
    tokenizer_prepends_bos: Optional[bool] = None
    
    # circuit_tracer compatibility
    output_logits_soft_cap: float = 0.0  # No soft cap by default
    
    @classmethod
    def from_sparse_gpt_config(
        cls,
        model_config: ModelConfig,
        sparsity_config: Optional[SparsityConfig] = None,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "HookedSparseGPTConfig":
        """Create from SparseGPT ModelConfig and optional SparsityConfig."""
        # Default sparsity settings
        enable_activation_sparsity = False
        activation_topk_fraction = 0.25
        activation_sparsity_locations = "attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,attn_v,attn_k,attn_q"
        
        if sparsity_config is not None:
            enable_activation_sparsity = sparsity_config.enable_activation_sparsity
            activation_topk_fraction = sparsity_config.activation_topk_fraction
            activation_sparsity_locations = sparsity_config.activation_sparsity_locations
        
        return cls(
            n_layers=model_config.n_layer,
            d_model=model_config.d_model,
            n_ctx=model_config.n_ctx,
            d_head=model_config.d_head,
            n_heads=model_config.n_heads,
            d_mlp=model_config.d_mlp,
            d_vocab=model_config.vocab_size,
            act_fn=model_config.activation,
            use_rms_norm=model_config.use_rms_norm,
            use_attention_sinks=model_config.use_attention_sinks,
            use_bias=model_config.use_bias,
            enable_activation_sparsity=enable_activation_sparsity,
            activation_topk_fraction=activation_topk_fraction,
            activation_sparsity_locations=activation_sparsity_locations,
            tokenizer_name=tokenizer_name,
            device=device,
            dtype=dtype,
            normalization_type="RMS" if model_config.use_rms_norm else "LN",
        )


# =============================================================================
# Hooked Normalization Layers
# =============================================================================

class HookedRMSNorm(nn.Module):
    """RMSNorm with hook points for scale and normalized output."""
    
    def __init__(self, d_model: int, eps: float = None, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        # Match PyTorch's nn.RMSNorm: when eps=None, use machine epsilon
        self.eps = eps if eps is not None else torch.finfo(dtype).eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))
        
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        scale = self.hook_scale(1.0 / rms)
        
        # Normalize
        x_normed = x * scale
        x_normed = self.hook_normalized(x_normed)
        
        return x_normed * self.weight


class HookedLayerNorm(nn.Module):
    """LayerNorm with hook points for scale and normalized output."""
    
    def __init__(self, d_model: int, eps: float = 1e-5, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=dtype))
        
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean and variance
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        scale = self.hook_scale(1.0 / torch.sqrt(var + self.eps))
        
        # Normalize
        x_normed = (x - mean) * scale
        x_normed = self.hook_normalized(x_normed)
        
        return x_normed * self.weight + self.bias


# =============================================================================
# Hooked Unembed (for circuit_tracer compatibility)
# =============================================================================

class HookedUnembed(nn.Module):
    """
    Unembedding layer with hook points for circuit_tracer compatibility.
    
    Provides hook_pre and hook_post around the unembedding operation,
    as well as W_U and b_U properties matching TransformerLens conventions.
    """
    
    def __init__(
        self,
        d_model: int,
        d_vocab: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        
        # Weight matrix: [d_vocab, d_model] (like nn.Linear)
        self.weight = nn.Parameter(torch.empty(d_vocab, d_model, dtype=dtype))
        
        # Optional bias
        if bias:
            self.b_U = nn.Parameter(torch.zeros(d_vocab, dtype=dtype))
        else:
            self.register_parameter('b_U', None)
        
        # Hook points for circuit_tracer
        self.hook_pre = HookPoint()   # Before unembedding: [batch, pos, d_model]
        self.hook_post = HookPoint()  # After unembedding: [batch, pos, d_vocab]
        
        # Initialize weights
        nn.init.normal_(self.weight, std=0.02)
    
    @property
    def W_U(self) -> torch.Tensor:
        """Unembedding matrix [d_model, d_vocab] (transposed for convenience)."""
        return self.weight.T
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hooks.
        
        Args:
            x: Input tensor [batch, pos, d_model]
            
        Returns:
            Logits [batch, pos, d_vocab]
        """
        x = self.hook_pre(x)
        
        # Linear transformation: x @ W_U = x @ weight.T
        logits = x @ self.weight.T
        if self.b_U is not None:
            logits = logits + self.b_U
        
        logits = self.hook_post(logits)
        return logits


# =============================================================================
# Hooked MLP
# =============================================================================

class HookedMLP(nn.Module):
    """MLP with hook points for pre and post activation."""
    
    def __init__(self, cfg: HookedSparseGPTConfig):
        super().__init__()
        self.cfg = cfg
        
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp, dtype=cfg.dtype))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model, dtype=cfg.dtype))
        
        if cfg.use_bias:
            self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp, dtype=cfg.dtype))
            self.b_out = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))
        else:
            self.register_parameter('b_in', None)
            self.register_parameter('b_out', None)
        
        # Activation function
        if cfg.act_fn == "gelu":
            self.act_fn = F.gelu
        else:
            self.act_fn = F.relu
        
        # Hook points
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp] - before activation
        self.hook_post = HookPoint()  # [batch, pos, d_mlp] - after activation
    
    def forward(
        self, 
        x: torch.Tensor,
        activation_sparsity_fn: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Input projection
        pre_act = x @ self.W_in
        if self.b_in is not None:
            pre_act = pre_act + self.b_in
        pre_act = self.hook_pre(pre_act)
        
        # Activation
        post_act = self.act_fn(pre_act)
        post_act = self.hook_post(post_act)
        
        # Apply activation sparsity to neurons (after activation, before output projection)
        if activation_sparsity_fn is not None:
            post_act = activation_sparsity_fn(post_act, "mlp_neuron")
        
        # Output projection
        out = post_act @ self.W_out
        if self.b_out is not None:
            out = out + self.b_out
        
        return out


# =============================================================================
# Hooked Attention (with Attention Sink support)
# =============================================================================

class HookedAttention(nn.Module):
    """
    Multi-head causal self-attention with full hook points.
    
    Supports attention sinks (learnable per-head bias to a dummy position).
    Implements manual attention computation to expose hook_pattern.
    """
    
    def __init__(self, cfg: HookedSparseGPTConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        
        # QKV projection (combined)
        qkv_dim = 3 * cfg.d_head * cfg.n_heads
        self.W_QKV = nn.Parameter(torch.empty(cfg.d_model, qkv_dim, dtype=cfg.dtype))
        if cfg.use_bias:
            self.b_QKV = nn.Parameter(torch.zeros(qkv_dim, dtype=cfg.dtype))
        else:
            self.register_parameter('b_QKV', None)
        
        # Output projection
        self.W_O = nn.Parameter(torch.empty(cfg.n_heads * cfg.d_head, cfg.d_model, dtype=cfg.dtype))
        if cfg.use_bias:
            self.b_O = nn.Parameter(torch.zeros(cfg.d_model, dtype=cfg.dtype))
        else:
            self.register_parameter('b_O', None)
        
        # Attention sink (learnable logit for dummy position)
        if cfg.use_attention_sinks:
            self.sink_logit = nn.Parameter(torch.zeros(cfg.n_heads, dtype=cfg.dtype))
        else:
            self.register_parameter('sink_logit', None)
        
        # Hook points
        self.hook_q = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_k = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_v = HookPoint()  # [batch, pos, n_heads, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, n_heads, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, n_heads, query_pos, key_pos] - after softmax
        self.hook_z = HookPoint()  # [batch, pos, n_heads, d_head] - attention output per head
        self.hook_result = HookPoint()  # [batch, pos, n_heads, d_model] - after W_O per head
        
        # Causal mask (registered as buffer)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        activation_sparsity_fn: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        
        # QKV projection
        qkv = x @ self.W_QKV
        if self.b_QKV is not None:
            qkv = qkv + self.b_QKV
        
        # Split into Q, K, V
        qkv_size = self.n_heads * self.d_head
        q, k, v = qkv.split(qkv_size, dim=-1)
        
        # Apply activation sparsity to Q, K, V (before reshape, like original SparseGPT)
        if activation_sparsity_fn is not None:
            q = activation_sparsity_fn(q, "attn_q")
            k = activation_sparsity_fn(k, "attn_k")
            v = activation_sparsity_fn(v, "attn_v")
        
        # Reshape to [batch, pos, n_heads, d_head]
        q = q.view(B, T, self.n_heads, self.d_head)
        k = k.view(B, T, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)
        
        # Hook Q, K, V
        q = self.hook_q(q)
        k = self.hook_k(k)
        v = self.hook_v(v)
        
        # Transpose to [batch, n_heads, pos, d_head] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.d_head)
        
        if self.sink_logit is not None:
            # With attention sinks: prepend dummy KV position
            # k_aug: [B, H, T+1, D], v_aug: [B, H, T+1, D]
            k_sink = torch.zeros((B, self.n_heads, 1, self.d_head), dtype=k.dtype, device=k.device)
            v_sink = torch.zeros((B, self.n_heads, 1, self.d_head), dtype=v.dtype, device=v.device)
            k_aug = torch.cat([k_sink, k], dim=2)
            v_aug = torch.cat([v_sink, v], dim=2)
            
            # Attention scores: [B, H, T, T+1]
            attn_scores = (q @ k_aug.transpose(-2, -1)) * scale
            
            # Build mask: sink (col 0) always visible, causal for real positions
            mask = torch.zeros((T, T + 1), dtype=torch.bool, device=x.device)
            mask[:, 0] = True  # Sink always visible
            mask[:, 1:] = self.causal_mask[:T, :T]  # Causal for real keys
            
            # Apply mask
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Add learnable sink bias to column 0
            sink_bias = self.sink_logit.view(1, self.n_heads, 1, 1)
            attn_scores[:, :, :, 0:1] = attn_scores[:, :, :, 0:1] + sink_bias
            
            # Hook attention scores
            attn_scores = self.hook_attn_scores(attn_scores)
            
            # Softmax
            pattern = F.softmax(attn_scores, dim=-1)
            pattern = self.hook_pattern(pattern)
            
            # Apply attention
            z = pattern @ v_aug  # [B, H, T, D]
        else:
            # Standard causal attention (no sinks)
            attn_scores = (q @ k.transpose(-2, -1)) * scale
            
            # Apply causal mask
            attn_scores = attn_scores.masked_fill(
                ~self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0),
                float('-inf')
            )
            
            # Hook attention scores
            attn_scores = self.hook_attn_scores(attn_scores)
            
            # Softmax
            pattern = F.softmax(attn_scores, dim=-1)
            pattern = self.hook_pattern(pattern)
            
            # Apply attention
            z = pattern @ v  # [B, H, T, D]
        
        # Reshape z for hook: [B, T, H, D]
        z = z.transpose(1, 2)
        z = self.hook_z(z)
        
        # Reshape for output projection: [B, T, H*D]
        z_flat = z.reshape(B, T, self.n_heads * self.d_head)
        
        # Output projection
        out = z_flat @ self.W_O
        if self.b_O is not None:
            out = out + self.b_O
        
        return out


# =============================================================================
# Hooked Transformer Block
# =============================================================================

class HookedTransformerBlock(nn.Module):
    """Transformer block with full hook points."""
    
    def __init__(self, cfg: HookedSparseGPTConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        # Normalization
        norm_cls = HookedRMSNorm if cfg.use_rms_norm else HookedLayerNorm
        self.ln1 = norm_cls(cfg.d_model, dtype=cfg.dtype)
        self.ln2 = norm_cls(cfg.d_model, dtype=cfg.dtype)
        
        # Attention and MLP
        self.attn = HookedAttention(cfg, layer_idx)
        self.mlp = HookedMLP(cfg)
        
        # Block-level hook points
        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_in = HookPoint()  # [batch, pos, d_model]
        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]
    
    def forward(
        self, 
        x: torch.Tensor,
        activation_sparsity_fn: Optional[Callable[[torch.Tensor, str], torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Pre-attention residual
        x = self.hook_resid_pre(x)
        if activation_sparsity_fn is not None:
            x = activation_sparsity_fn(x, "resid_pre")
        
        # Attention sublayer
        attn_in = self.ln1(x)
        if activation_sparsity_fn is not None:
            attn_in = activation_sparsity_fn(attn_in, "attn_in")
        attn_in = self.hook_attn_in(attn_in)
        attn_out = self.attn(attn_in, activation_sparsity_fn)
        if activation_sparsity_fn is not None:
            attn_out = activation_sparsity_fn(attn_out, "attn_out")
        attn_out = self.hook_attn_out(attn_out)
        
        # First residual connection
        x = x + attn_out
        x = self.hook_resid_mid(x)
        if activation_sparsity_fn is not None:
            x = activation_sparsity_fn(x, "resid_mid")
        
        # MLP sublayer
        mlp_in = self.ln2(x)
        if activation_sparsity_fn is not None:
            mlp_in = activation_sparsity_fn(mlp_in, "mlp_in")
        mlp_in = self.hook_mlp_in(mlp_in)
        mlp_out = self.mlp(mlp_in, activation_sparsity_fn)
        if activation_sparsity_fn is not None:
            mlp_out = activation_sparsity_fn(mlp_out, "mlp_out")
        mlp_out = self.hook_mlp_out(mlp_out)
        
        # Second residual connection
        x = x + mlp_out
        x = self.hook_resid_post(x)
        
        return x


# =============================================================================
# Main HookedSparseGPT Class
# =============================================================================

class HookedSparseGPT(HookedRootModule):
    """
    TransformerLens-compatible wrapper for SparseGPT models.
    
    Provides all hook points needed for interpretability tools like
    dictionary_learning and circuit-tracer.
    
    Usage:
        model = HookedSparseGPT.from_pretrained("username/model-name")
        logits, cache = model.run_with_cache(tokens)
    """
    
    def __init__(
        self,
        cfg: HookedSparseGPTConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        # Embedding
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_model, dtype=cfg.dtype)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            HookedTransformerBlock(cfg, layer_idx)
            for layer_idx in range(cfg.n_layers)
        ])
        
        # Final layer norm
        norm_cls = HookedRMSNorm if cfg.use_rms_norm else HookedLayerNorm
        self.ln_final = norm_cls(cfg.d_model, dtype=cfg.dtype)
        
        # Unembedding (with hook points for circuit_tracer)
        self.unembed = HookedUnembed(cfg.d_model, cfg.d_vocab, bias=False, dtype=cfg.dtype)
        
        # Setup activation sparsity
        self._setup_activation_sparsity()
        
        # Setup hook dict (required by HookedRootModule)
        self.setup()
    
    def _setup_activation_sparsity(self):
        """Setup activation sparsity functions based on config."""
        if not self.cfg.enable_activation_sparsity:
            self._activation_sparsity_fn = None
            return
        
        # Parse locations where activation sparsity is applied
        locations = set(self.cfg.activation_sparsity_locations.split(","))
        k_fraction = self.cfg.activation_topk_fraction
        
        # Create AbsTopK modules for different dimensions (lazily cached)
        self._activation_sparsity_locations = locations
        self._activation_sparsity_k_fraction = k_fraction
        self._activation_sparsity_cache: Dict[int, AbsTopK] = {}
    
    def _get_activation_sparsity_fn(self) -> Optional[Callable[[torch.Tensor, str], torch.Tensor]]:
        """Get the activation sparsity function."""
        if not self.cfg.enable_activation_sparsity:
            return None
        
        def apply_activation_sparsity(x: torch.Tensor, location: str) -> torch.Tensor:
            if location not in self._activation_sparsity_locations:
                return x
            
            dim = x.shape[-1]
            k = max(1, int(dim * self._activation_sparsity_k_fraction))
            
            # Cache AbsTopK modules by dimension
            if dim not in self._activation_sparsity_cache:
                self._activation_sparsity_cache[dim] = AbsTopK(k)
            
            return self._activation_sparsity_cache[dim](x)
        
        return apply_activation_sparsity
    
    @property
    def W_E(self) -> torch.Tensor:
        """Embedding matrix [d_vocab, d_model]."""
        return self.embed.weight
    
    @property
    def W_U(self) -> torch.Tensor:
        """Unembedding matrix [d_model, d_vocab]."""
        return self.unembed.W_U
    
    @property
    def b_U(self) -> torch.Tensor:
        """Unembedding bias [d_vocab]."""
        if self.unembed.b_U is not None:
            return self.unembed.b_U
        return torch.zeros(self.cfg.d_vocab, device=self.W_U.device, dtype=self.W_U.dtype)
    
    def to_tokens(
        self,
        input: Union[str, List[str]],
        prepend_bos: bool = False,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> Int[torch.Tensor, "batch pos"]:
        """Convert string(s) to tokens."""
        assert self.tokenizer is not None, "Tokenizer not set"
        
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
            add_special_tokens=False,  # Don't add EOS/BOS automatically
        )["input_ids"]
        
        if prepend_bos and self.tokenizer.bos_token_id is not None:
            bos = torch.full((tokens.shape[0], 1), self.tokenizer.bos_token_id, dtype=tokens.dtype)
            tokens = torch.cat([bos, tokens], dim=1)
        
        if move_to_device and self.cfg.device:
            tokens = tokens.to(self.cfg.device)
        
        return tokens
    
    def to_str_tokens(
        self,
        input: Union[str, Int[torch.Tensor, "pos"], Int[torch.Tensor, "1 pos"], List[int]],
        prepend_bos: bool = False,
    ) -> List[str]:
        """
        Convert input to a list of string tokens.
        
        Matches TransformerLens HookedTransformer.to_str_tokens behavior.
        
        Args:
            input: Either a string (will be tokenized), or token IDs as a tensor or list
            prepend_bos: If True and input is a string, prepend BOS token
            
        Returns:
            List of string tokens (one string per token)
        """
        assert self.tokenizer is not None, "Tokenizer not set"
        
        if isinstance(input, str):
            # Tokenize the string first
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, move_to_device=False)
            tokens = tokens.squeeze(0)  # Remove batch dim
        elif isinstance(input, torch.Tensor):
            tokens = input.squeeze()  # Remove any extra dims
            if tokens.dim() == 0:
                tokens = tokens.unsqueeze(0)
        elif isinstance(input, list):
            tokens = input
        else:
            raise ValueError(f"Invalid input type: {type(input)}")
        
        # Convert to list of ints for tokenizer
        if isinstance(tokens, torch.Tensor):
            token_list = tokens.tolist()
        else:
            token_list = list(tokens)
        
        # Use convert_ids_to_tokens which preserves the original token representation
        str_tokens = self.tokenizer.convert_ids_to_tokens(token_list)
        
        # Check if this tokenizer encodes spaces (e.g. GPT-2 uses Ġ, SentencePiece uses ▁)
        # If not, we need to infer spaces by comparing decoded text
        has_space_markers = any(
            tok is not None and (tok.startswith("Ġ") or tok.startswith("▁"))
            for tok in str_tokens[:min(10, len(str_tokens))]
        )
        
        if has_space_markers:
            # Tokenizer encodes spaces - just handle byte tokens and None
            processed_tokens = []
            for i, tok in enumerate(str_tokens):
                if tok is None:
                    processed_tokens.append(self.tokenizer.decode([token_list[i]]))
                elif tok.startswith("<0x") and tok.endswith(">"):
                    try:
                        byte_val = int(tok[3:-1], 16)
                        processed_tokens.append(chr(byte_val))
                    except:
                        processed_tokens.append(tok)
                else:
                    processed_tokens.append(tok)
            return processed_tokens
        
        # Tokenizer doesn't encode spaces - use heuristic for word-level tokenizers
        # Handle WordPiece (##) and add space before word tokens
        
        processed_tokens = []
        for i, (tok, tok_id) in enumerate(zip(str_tokens, token_list)):
            if tok is None:
                tok = self.tokenizer.decode([tok_id])
            
            # Handle WordPiece continuation tokens (##prefix)
            if tok.startswith("##"):
                # This is a subword continuation - no space, remove ##
                processed_tokens.append(tok[2:])
            elif i == 0:
                processed_tokens.append(tok)
            else:
                # Add space before word tokens (not punctuation)
                # A token needs a leading space if it starts with a letter/number
                first_char = tok[0] if tok else ''
                if first_char.isalnum():
                    processed_tokens.append("▁" + tok)
                else:
                    processed_tokens.append(tok)
        
        return processed_tokens
    
    def ensure_tokenized(
        self,
        prompt: Union[str, List[int], Int[torch.Tensor, "pos"]],
        prepend_bos: bool = False,
    ) -> Int[torch.Tensor, "pos"]:
        """
        Convert prompt to 1-D tensor of token ids with optional special token handling.
        
        Args:
            prompt: String, tensor, or list of token ids representing a single sequence
            prepend_bos: If True, prepend a special token (BOS/PAD) to the input.
                        Default is False (no special token added).
            
        Returns:
            1-D tensor of token ids
            
        Raises:
            TypeError: If prompt is not str, tensor, or list
            ValueError: If tensor has wrong shape (must be 1-D or 2-D with batch size 1)
        """
        import warnings
        
        if isinstance(prompt, str):
            # Don't add special tokens automatically - we handle BOS manually below if needed
            tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        elif isinstance(prompt, torch.Tensor):
            tokens = prompt.squeeze()
        elif isinstance(prompt, list):
            tokens = torch.tensor(prompt, dtype=torch.long).squeeze()
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")
        
        if tokens.ndim > 1:
            raise ValueError(f"Tensor must be 1-D, got shape {tokens.shape}")
        
        if not prepend_bos:
            return tokens.to(self.cfg.device)
        
        # Check if a special token is already present at the beginning
        if tokens[0] in self.tokenizer.all_special_ids:
            return tokens.to(self.cfg.device)
        
        # Prepend a special token to avoid artifacts at position 0
        candidate_bos_token_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        ]
        candidate_bos_token_ids += self.tokenizer.all_special_ids
        
        dummy_bos_token_id = next(filter(None, candidate_bos_token_ids), None)
        if dummy_bos_token_id is None:
            warnings.warn(
                "No suitable special token found for BOS token replacement. "
                "The first token will be ignored."
            )
        else:
            tokens = torch.cat([torch.tensor([dummy_bos_token_id], device=tokens.device), tokens])
        
        return tokens.to(self.cfg.device)
    
    def forward(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        return_type: Optional[str] = "logits",
        stop_at_layer: Optional[int] = None,
        prepend_bos: bool = False,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Float[torch.Tensor, "batch pos d_model"],
        Tuple[Float[torch.Tensor, "batch pos d_vocab"], Float[torch.Tensor, ""]],
    ]:
        """
        Forward pass with TransformerLens-compatible interface.
        
        Args:
            input: Token IDs or strings (will be tokenized)
            return_type: "logits", "loss", "both", or None
            stop_at_layer: If set, return residual stream at this layer (exclusive)
            prepend_bos: Whether to prepend BOS token (only for string input)
            
        Returns:
            Depends on return_type:
            - "logits": logits tensor
            - "loss": loss scalar (requires integer input as labels)
            - "both": (logits, loss) tuple
            - None: None (useful for just running hooks)
            
            If stop_at_layer is set, returns residual stream instead.
        """
        # Handle string input
        if isinstance(input, str) or isinstance(input, list):
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input
        
        if tokens.device != self.W_E.device:
            tokens = tokens.to(self.W_E.device)
        
        B, T = tokens.shape
        
        # Embedding
        x = self.embed(tokens)
        x = self.hook_embed(x)
        
        # Get activation sparsity function
        act_sparsity_fn = self._get_activation_sparsity_fn()
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            if stop_at_layer is not None and i >= stop_at_layer:
                break
            x = block(x, act_sparsity_fn)
        
        # If stopping early, return residual stream
        if stop_at_layer is not None:
            return x
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Unembedding
        logits = self.unembed(x)
        
        # Return based on return_type
        if return_type is None:
            return None
        elif return_type == "logits":
            return logits
        elif return_type == "loss":
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return loss
        elif return_type == "both":
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            return logits, loss
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
    
    def run_with_cache(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        return_type: Optional[str] = "logits",
        names_filter: Optional[Union[Callable[[str], bool], List[str]]] = None,
        prepend_bos: bool = False,
    ) -> Tuple[
        Union[None, Float[torch.Tensor, "batch pos d_vocab"], Float[torch.Tensor, "batch pos d_model"]],
        Dict[str, torch.Tensor],
    ]:
        """
        Run forward pass and cache activations at all hook points.
        
        Args:
            input: Token IDs or strings (will be tokenized)
            return_type: "logits", "loss", "both", or None
            names_filter: Optional filter for which hooks to cache.
                         If None, caches all hooks. If a callable, only caches
                         hooks where names_filter(hook_name) returns True.
                         If a list, only caches hooks whose names are in the list.
            prepend_bos: Whether to prepend BOS token (only for string input)
            
        Returns:
            Tuple of (output, cache) where:
            - output: Depends on return_type (logits, loss, or both)
            - cache: Dict mapping hook names to cached tensors
        """
        cache = {}
        
        def make_cache_hook(name: str):
            def hook_fn(tensor, hook):
                cache[name] = tensor.detach().clone()
                return tensor
            return hook_fn
        
        # Collect all hook points that we add hooks to
        hooked_modules = []
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                # Apply filter if provided
                if names_filter is not None:
                    if callable(names_filter):
                        if not names_filter(name):
                            continue
                    else:
                        # names_filter is a list of names
                        if name not in names_filter:
                            continue
                module.add_hook(make_cache_hook(name))
                hooked_modules.append(module)
        
        try:
            # Run forward pass
            output = self.forward(
                input,
                return_type=return_type,
                prepend_bos=prepend_bos,
            )
        finally:
            # Remove all hooks from the modules we added to
            for module in hooked_modules:
                module.remove_hooks()
        
        return output, cache
    
    def run_with_hooks(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        fwd_hooks: Optional[List[Tuple[str, Callable]]] = None,
        return_type: Optional[str] = "logits",
        prepend_bos: bool = False,
        **kwargs,
    ) -> Union[
        None,
        Float[torch.Tensor, "batch pos d_vocab"],
        Float[torch.Tensor, "batch pos d_model"],
    ]:
        """
        Run forward pass with custom hooks.
        
        Args:
            input: Token IDs or strings (will be tokenized)
            fwd_hooks: List of (hook_name, hook_fn) tuples. Each hook_fn
                      receives (tensor, hook) and should return tensor.
            return_type: "logits", "loss", "both", or None
            prepend_bos: Whether to prepend BOS token (only for string input)
            
        Returns:
            Forward pass output (depends on return_type)
        """
        if fwd_hooks is None:
            fwd_hooks = []
        
        hooked_modules = []
        
        # Register hooks
        for hook_name, hook_fn in fwd_hooks:
            # Find the hook point by name
            hook_point = self._get_hook_point_by_name(hook_name)
            if hook_point is not None:
                hook_point.add_hook(hook_fn)
                hooked_modules.append(hook_point)
        
        try:
            output = self.forward(
                input,
                return_type=return_type,
                prepend_bos=prepend_bos,
            )
        finally:
            # Remove all hooks from the modules we added to
            for module in hooked_modules:
                module.remove_hooks()
        
        return output
    
    def _get_hook_point_by_name(self, name: str) -> Optional[HookPoint]:
        """Get a hook point by its name."""
        parts = name.split(".")
        obj = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj if isinstance(obj, HookPoint) else None
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HookedSparseGPT":
        """
        Load a pretrained SparseGPT model as HookedSparseGPT.
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "username/model-name")
            device: Device to load model on
            dtype: Data type for model weights
            
        Returns:
            HookedSparseGPT instance with loaded weights and tokenizer
        """
        import json
        from huggingface_hub import hf_hub_download
        
        # Download config and weights
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        
        # Load config
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        # Check bigram table is disabled
        model_cfg = config_dict["model_config"]
        if model_cfg.get("use_bigram_table", False):
            raise ValueError(
                "HookedSparseGPT does not support models with bigram_table enabled. "
                "The bigram table is not compatible with TransformerLens-style interpretability tools."
            )
        
        # Create ModelConfig
        sparse_config = ModelConfig(
            n_layer=model_cfg["n_layer"],
            d_model=model_cfg["d_model"],
            n_ctx=model_cfg["n_ctx"],
            d_head=model_cfg["d_head"],
            d_mlp=model_cfg["d_mlp"],
            vocab_size=model_cfg["vocab_size"],
            use_rms_norm=model_cfg["use_rms_norm"],
            tie_embeddings=model_cfg.get("tie_embeddings", False),
            use_positional_embeddings=model_cfg.get("use_positional_embeddings", False),
            use_bigram_table=False,  # Asserted above
            use_attention_sinks=model_cfg.get("use_attention_sinks", True),
            activation=model_cfg.get("activation", "gelu"),
            dropout=model_cfg.get("dropout", 0.0),
            use_bias=model_cfg.get("use_bias", True),
        )
        
        # Load sparsity config if present
        sparsity_cfg = None
        if "sparsity_config" in config_dict:
            sparsity_dict = config_dict["sparsity_config"]
            sparsity_cfg = SparsityConfig(
                enable_weight_sparsity=sparsity_dict.get("enable_weight_sparsity", False),
                target_l0_fraction=sparsity_dict.get("target_l0_fraction", 0.1),
                enable_activation_sparsity=sparsity_dict.get("enable_activation_sparsity", False),
                activation_topk_fraction=sparsity_dict.get("activation_topk_fraction", 0.25),
                activation_sparsity_locations=sparsity_dict.get(
                    "activation_sparsity_locations", 
                    "attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,attn_v,attn_k,attn_q"
                ),
            )
        
        # Get tokenizer name
        tokenizer_name = config_dict.get("training_config", {}).get("tokenizer_name")
        
        # Create hooked config
        hooked_cfg = HookedSparseGPTConfig.from_sparse_gpt_config(
            sparse_config,
            sparsity_config=sparsity_cfg,
            tokenizer_name=tokenizer_name,
            device=device,
            dtype=dtype,
        )
        
        # Load tokenizer
        tokenizer = None
        if tokenizer_name:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
            except Exception as e:
                print(f"Warning: Could not load tokenizer '{tokenizer_name}': {e}")
        
        # Create model
        model = cls(hooked_cfg, tokenizer=tokenizer)
        
        # Load weights from SparseGPT format
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model._load_sparse_gpt_weights(state_dict)
        
        model.to(device)
        model.eval()
        
        return model
    
    def _load_sparse_gpt_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load weights from SparseGPT state dict into HookedSparseGPT.
        
        Maps SparseGPT weight names to HookedSparseGPT structure.
        """
        # Embedding
        self.embed.weight.data.copy_(state_dict["wte.weight"])
        
        # Unembedding (lm_head)
        self.unembed.weight.data.copy_(state_dict["lm_head.weight"])
        
        # Final layer norm
        if "ln_f.weight" in state_dict:
            self.ln_final.weight.data.copy_(state_dict["ln_f.weight"])
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            
            # Layer norms (SparseGPT uses ln_1/ln_2, we use ln1/ln2)
            block.ln1.weight.data.copy_(state_dict[f"{prefix}ln_1.weight"])
            block.ln2.weight.data.copy_(state_dict[f"{prefix}ln_2.weight"])
            
            # LayerNorm also has bias
            if hasattr(block.ln1, 'bias') and block.ln1.bias is not None:
                if f"{prefix}ln_1.bias" in state_dict:
                    block.ln1.bias.data.copy_(state_dict[f"{prefix}ln_1.bias"])
            if hasattr(block.ln2, 'bias') and block.ln2.bias is not None:
                if f"{prefix}ln_2.bias" in state_dict:
                    block.ln2.bias.data.copy_(state_dict[f"{prefix}ln_2.bias"])
            
            # Attention
            # SparseGPT uses c_attn for combined QKV, c_proj for output
            attn_weight = state_dict[f"{prefix}attn.c_attn.weight"]
            if f"{prefix}attn.c_attn.bias" in state_dict:
                attn_bias = state_dict[f"{prefix}attn.c_attn.bias"]
            else:
                attn_bias = None
            
            # Transpose because nn.Linear weight is (out, in) but we use x @ W
            block.attn.W_QKV.data.copy_(attn_weight.T)
            if attn_bias is not None and block.attn.b_QKV is not None:
                block.attn.b_QKV.data.copy_(attn_bias)
            
            # Attention output projection
            out_weight = state_dict[f"{prefix}attn.c_proj.weight"]
            if f"{prefix}attn.c_proj.bias" in state_dict:
                out_bias = state_dict[f"{prefix}attn.c_proj.bias"]
            else:
                out_bias = None
            
            # Transpose because nn.Linear weight is (out, in) but we use x @ W
            block.attn.W_O.data.copy_(out_weight.T)
            if out_bias is not None and block.attn.b_O is not None:
                block.attn.b_O.data.copy_(out_bias)
            
            # Attention sink logit
            if block.attn.sink_logit is not None and f"{prefix}attn.attn_fn.sink_logit" in state_dict:
                block.attn.sink_logit.data.copy_(state_dict[f"{prefix}attn.attn_fn.sink_logit"])
            
            # MLP
            # SparseGPT uses c_fc for input, c_proj for output
            block.mlp.W_in.data.copy_(state_dict[f"{prefix}mlp.c_fc.weight"].T)
            block.mlp.W_out.data.copy_(state_dict[f"{prefix}mlp.c_proj.weight"].T)
            
            if block.mlp.b_in is not None and f"{prefix}mlp.c_fc.bias" in state_dict:
                block.mlp.b_in.data.copy_(state_dict[f"{prefix}mlp.c_fc.bias"])
            if block.mlp.b_out is not None and f"{prefix}mlp.c_proj.bias" in state_dict:
                block.mlp.b_out.data.copy_(state_dict[f"{prefix}mlp.c_proj.bias"])

