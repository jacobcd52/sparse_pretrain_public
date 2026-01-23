"""
Bridge modules for coupling dense and sparse models.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

Bridges are encoder/decoder pairs that translate between dense and sparse
residual stream activations at each sublayer location.
"""

import math
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import AbsTopK, SparseGPT


# =============================================================================
# Gradient Buffering Optimization
# =============================================================================

class GradientBuffer:
    """
    A gradient buffer that accumulates gradients and releases them on demand.
    
    This is used for efficient hybrid pass computation in bridges training.
    Instead of L separate backward passes, we get:
    - L small backwards: From each KL loss to the buffer checkpoint (accumulated)
    - 1 big backward: From checkpoint through the rest of the network
    
    Usage:
        buffer = GradientBuffer(h)
        
        # Use buffer.accumulator for all hybrid forward passes
        for i in range(n_sites):
            y_hybrid = model.forward_from_site(buffer.accumulator, ...)
            loss += kl_divergence(y_target, y_hybrid)
        
        # First backward accumulates gradients (must use retain_graph=True)
        loss.backward(retain_graph=True)
        
        # Then release gradients through the buffer
        buffer.release_gradients()
    """
    
    def __init__(self, x: torch.Tensor):
        """
        Create a gradient buffer for tensor x.
        
        Args:
            x: Tensor to buffer gradients for. Must require gradients.
        """
        self.original = x
        self.grad_buf = None
        self._released = False
        
        # Create the accumulator using a custom autograd function
        buffer_ref = self  # Reference to self for the inner class
        
        class _BufferedGradAccum(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # No need to save_for_backward since we don't use it
                # No need to clone - we're not modifying x, just passing it through
                return x
            
            @staticmethod
            def backward(ctx, grad):
                if buffer_ref.grad_buf is None:
                    buffer_ref.grad_buf = grad.clone()
                else:
                    buffer_ref.grad_buf = buffer_ref.grad_buf + grad
                # Return None to stop gradient propagation
                return None
        
        self.accumulator = _BufferedGradAccum.apply(x)
        self._AccumClass = _BufferedGradAccum
    
    def release_gradients(self, retain_graph: bool = True):
        """
        Release accumulated gradients back through the original tensor.
        
        Call this after backward() on the loss that uses the accumulator.
        
        Args:
            retain_graph: Whether to keep the computation graph after backward.
                Use False for the last release to free memory.
        """
        if self._released:
            return
        self._released = True
        
        if self.grad_buf is None:
            return
        
        # Continue backprop through the computation graph of original
        # This propagates gradients to all tensors that original depends on
        if self.original.grad_fn is not None:
            self.original.backward(self.grad_buf, retain_graph=retain_graph)
        
        # Clear references to free memory
        self.grad_buf = None
        self.original = None
        self.accumulator = None


class BridgeEncoder(nn.Module):
    """
    Encoder that maps dense residual activations to sparse residual activations.
    
    Structure: Linear + AbsTopK
    
    The AbsTopK enforces activation sparsity on the encoded representation,
    matching the sparse model's residual stream characteristics.
    """
    
    def __init__(
        self,
        d_dense: int,
        d_sparse: int,
        afrac: float = 0.25,
    ):
        """
        Args:
            d_dense: Dimension of dense model's residual stream
            d_sparse: Dimension of sparse model's residual stream
            afrac: Fraction of activations to keep in AbsTopK
        """
        super().__init__()
        self.linear = nn.Linear(d_dense, d_sparse, bias=True)
        k = max(1, int(d_sparse * afrac))
        self.topk = AbsTopK(k)
        
    def forward(self, x: torch.Tensor, sharpness: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            x: Dense activations of shape (..., d_dense)
            sharpness: Ignored for AbsTopK encoder (only used by ThresholdEncoder)
            
        Returns:
            Sparse-space activations of shape (..., d_sparse)
        """
        return self.topk(self.linear(x))


class ThresholdEncoder(nn.Module):
    """
    Encoder that maps dense residual activations to sparse residual activations.
    
    Structure: Linear + Learned Threshold
    
    Uses a per-dimension learnable threshold (epsilon) to zero out small activations.
    During training, uses soft thresholding with sigmoid for gradient flow.
    At eval or when hard=True, uses true hard thresholding.
    
    The threshold is learned via log_eps parameters for numerical stability.
    
    NOTE: log_eps is always kept in float32 for numerical precision, even when the
    rest of the model uses bfloat16. bfloat16 has insufficient precision for the
    small gradient updates that log_eps receives.
    """
    
    def __init__(
        self,
        d_dense: int,
        d_sparse: int,
        init_log_eps: float = -1.0,
    ):
        """
        Args:
            d_dense: Dimension of dense model's residual stream
            d_sparse: Dimension of sparse model's residual stream
            init_log_eps: Initial value for log(epsilon) parameters
        """
        super().__init__()
        self.linear = nn.Linear(d_dense, d_sparse, bias=True)
        # Per-dimension learnable log(epsilon) for threshold
        # IMPORTANT: Always use float32 for log_eps - bfloat16 doesn't have enough
        # precision for the small updates this parameter receives
        self.log_eps = nn.Parameter(torch.full((d_sparse,), init_log_eps, dtype=torch.float32))
        self.d_sparse = d_sparse
    
    def _apply(self, fn):
        """Override _apply to keep log_eps in float32."""
        # First apply to everything normally
        super()._apply(fn)
        # Then restore log_eps to float32 if it was converted
        if self.log_eps.dtype != torch.float32:
            self.log_eps.data = self.log_eps.data.float()
        return self
    
    @property
    def eps(self) -> torch.Tensor:
        """Get the threshold values (epsilon) from log_eps."""
        return torch.exp(self.log_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        sharpness: Optional[float] = None,
        hard: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: Dense activations of shape (..., d_dense)
            sharpness: Sharpness for soft thresholding. If None, uses hard thresholding.
                During training, pass sharpness to enable gradient flow to log_eps.
            hard: If True (default), use hard thresholding. If False and sharpness is
                provided, use soft thresholding with sigmoid.
            
        Returns:
            Sparse-space activations of shape (..., d_sparse)
        """
        z = self.linear(x)
        # eps is float32, but we need to compute in the dtype of z for the mask
        # However, we want gradient to flow to log_eps in float32
        eps = self.eps  # Shape: (d_sparse,), always float32
        
        # Determine whether to use hard or soft thresholding
        # Use soft thresholding only when explicitly requested (hard=False and sharpness provided)
        use_hard = hard or sharpness is None
        
        if use_hard:
            # Hard threshold: zero if |z| < eps
            # Compare in float32 for precision, then convert mask to z's dtype
            mask = (torch.abs(z.float()) >= eps).to(z.dtype)
        else:
            # Soft threshold: smooth approximation for gradient flow to eps
            # Compute in float32 for gradient precision to log_eps
            abs_z = torch.abs(z.float())
            mask = torch.sigmoid(sharpness * (abs_z - eps)).to(z.dtype)
        
        return z * mask


class BridgeDecoder(nn.Module):
    """
    Decoder that maps sparse residual activations to dense residual activations.
    
    Structure: Linear (no activation sparsity)
    """
    
    def __init__(
        self,
        d_sparse: int,
        d_dense: int,
    ):
        """
        Args:
            d_sparse: Dimension of sparse model's residual stream
            d_dense: Dimension of dense model's residual stream
        """
        super().__init__()
        self.linear = nn.Linear(d_sparse, d_dense, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sparse activations of shape (..., d_sparse)
            
        Returns:
            Dense-space activations of shape (..., d_dense)
        """
        return self.linear(x)


class Bridge(nn.Module):
    """
    A single bridge consisting of an encoder and decoder pair.
    
    Encoder: dense → sparse (Linear + AbsTopK or Linear + Threshold)
    Decoder: sparse → dense (Linear)
    """
    
    def __init__(
        self,
        d_dense: int,
        d_sparse: int,
        encoder_afrac: float = 0.25,
        encoder_type: str = "abstopk",
        init_log_eps: float = -1.0,
    ):
        """
        Args:
            d_dense: Dimension of dense model's residual stream
            d_sparse: Dimension of sparse model's residual stream
            encoder_afrac: Fraction of activations to keep (for AbsTopK encoder)
            encoder_type: "abstopk" or "threshold"
            init_log_eps: Initial log(epsilon) for threshold encoder
        """
        super().__init__()
        
        if encoder_type == "abstopk":
            self.encoder = BridgeEncoder(d_dense, d_sparse, encoder_afrac)
        elif encoder_type == "threshold":
            self.encoder = ThresholdEncoder(d_dense, d_sparse, init_log_eps)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Must be 'abstopk' or 'threshold'")
        
        self.decoder = BridgeDecoder(d_sparse, d_dense)
        self.encoder_type = encoder_type
        
    def encode(
        self,
        h_dense: torch.Tensor,
        sharpness: Optional[float] = None,
        hard: bool = True,
    ) -> torch.Tensor:
        """Map dense activations to sparse space.
        
        Args:
            h_dense: Dense activations
            sharpness: Sharpness for soft thresholding (ThresholdEncoder only)
            hard: Whether to use hard thresholding (ThresholdEncoder only)
        """
        if self.encoder_type == "threshold":
            return self.encoder(h_dense, sharpness=sharpness, hard=hard)
        else:
            return self.encoder(h_dense)
    
    def decode(self, h_sparse: torch.Tensor) -> torch.Tensor:
        """Map sparse activations to dense space."""
        return self.decoder(h_sparse)


class BridgeSet(nn.Module):
    """
    Collection of all bridges for coupling dense and sparse models.
    
    For an L-layer model, there are 2L+1 bridge sites:
    - Site 0: After embedding
    - Site 2i+1: After layer i's attention (before MLP), for i = 0..L-1
    - Site 2i+2: After layer i's MLP, for i = 0..L-1
    
    Each site has both an encoder (dense→sparse) and decoder (sparse→dense).
    """
    
    def __init__(
        self,
        n_layers: int,
        d_dense: int,
        d_sparse: int,
        encoder_afrac: float = 0.25,
        encoder_type: str = "abstopk",
        init_log_eps: float = -1.0,
    ):
        """
        Args:
            n_layers: Number of transformer layers
            d_dense: Dimension of dense model's residual stream
            d_sparse: Dimension of sparse model's residual stream
            encoder_afrac: Fraction of activations to keep in encoder's AbsTopK
            encoder_type: "abstopk" or "threshold"
            init_log_eps: Initial log(epsilon) for threshold encoder
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.n_sites = 2 * n_layers + 1
        self.d_dense = d_dense
        self.d_sparse = d_sparse
        self.encoder_type = encoder_type
        
        # Create bridges for each site
        self.bridges = nn.ModuleList([
            Bridge(d_dense, d_sparse, encoder_afrac, encoder_type, init_log_eps)
            for _ in range(self.n_sites)
        ])
    
    def encode(
        self,
        site_idx: int,
        h_dense: torch.Tensor,
        sharpness: Optional[float] = None,
        hard: bool = True,
    ) -> torch.Tensor:
        """Encode dense activations at a specific site.
        
        Args:
            site_idx: Bridge site index
            h_dense: Dense activations
            sharpness: Sharpness for soft thresholding (ThresholdEncoder only)
            hard: Whether to use hard thresholding (ThresholdEncoder only)
        """
        return self.bridges[site_idx].encode(h_dense, sharpness=sharpness, hard=hard)
    
    def decode(self, site_idx: int, h_sparse: torch.Tensor) -> torch.Tensor:
        """Decode sparse activations at a specific site."""
        return self.bridges[site_idx].decode(h_sparse)
    
    def encode_all(
        self,
        h_dense_list: List[torch.Tensor],
        sharpness: Optional[float] = None,
        hard: bool = True,
    ) -> List[torch.Tensor]:
        """Encode dense activations at all sites.
        
        Args:
            h_dense_list: List of dense activations at each site
            sharpness: Sharpness for soft thresholding (ThresholdEncoder only)
            hard: Whether to use hard thresholding (ThresholdEncoder only)
        """
        assert len(h_dense_list) == self.n_sites
        return [
            self.bridges[i].encode(h_dense_list[i], sharpness=sharpness, hard=hard)
            for i in range(self.n_sites)
        ]
    
    def decode_all(self, h_sparse_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode sparse activations at all sites."""
        assert len(h_sparse_list) == self.n_sites
        return [
            self.bridges[i].decode(h_sparse_list[i])
            for i in range(self.n_sites)
        ]


def nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Normalized Mean Squared Error (FVU - Fraction of Variance Unexplained).
    
    First computes per-token FVU, then averages:
        fvu_per_token = (pred - target).pow(2).sum(-1) / (target - target.mean(batch_and_seq_dims)).pow(2).sum(-1)
        fvu = fvu_per_token.mean()
    
    The denominator is detached to prevent weird optimization effects where
    one model learns high-variance directions to make NMSE artificially small.
    
    Args:
        pred: Predicted activations, shape (..., features)
        target: Target activations, shape (..., features)
        eps: Small constant for numerical stability
        
    Returns:
        Scalar NMSE/FVU loss
    """
    # Compute mean over all dimensions except the last (features)
    # This is the "batch_and_seq_dims" mean
    batch_and_seq_dims = tuple(range(target.dim() - 1))
    target_mean = target.mean(dim=batch_and_seq_dims, keepdim=True)
    
    # Per-token squared error summed over features
    squared_error = (pred - target).pow(2).sum(-1)
    
    # Per-token squared deviation from global mean, summed over features
    squared_deviation = (target - target_mean).pow(2).sum(-1)
    
    # Per-token FVU (detach denominator to prevent weird optimization)
    fvu_per_token = squared_error / (squared_deviation.detach() + eps)
    
    # Average over all tokens
    return fvu_per_token.mean()


class BridgeNMSEResult:
    """
    Result container for bridge NMSE loss computation with per-site breakdown.
    """
    
    def __init__(
        self,
        total: torch.Tensor,
        encoder_losses: List[torch.Tensor],
        decoder_losses: List[torch.Tensor],
    ):
        self.total = total
        self.encoder_losses = encoder_losses  # Per-site encoder NMSE
        self.decoder_losses = decoder_losses  # Per-site decoder NMSE
    
    def get_detailed_losses(self) -> dict:
        """Return a dict of per-site losses for logging."""
        result = {}
        for i, loss in enumerate(self.encoder_losses):
            result[f"encoder_site{i}"] = loss.detach().item()
        for i, loss in enumerate(self.decoder_losses):
            result[f"decoder_site{i}"] = loss.detach().item()
        return result


def compute_bridge_nmse_loss(
    h_dense_list: List[torch.Tensor],
    h_sparse_pre_list: List[torch.Tensor],
    h_sparse_post_list: List[torch.Tensor],
    bridge_set: BridgeSet,
    sharpness: Optional[float] = None,
    hard: bool = True,
) -> BridgeNMSEResult:
    """
    Compute the total NMSE loss for all bridge sites.
    
    L_NMSE = sum_i [ NMSE(encoder_i(h^d_i), h^s_post_i) + NMSE(decoder_i(h^s_pre_i), h^d_i) ]
    
    The encoder predicts the POST-AbsTopK sparse activations (the sparsified version).
    The decoder takes the PRE-AbsTopK sparse activations as input (the dense version
    before sparsification, so it can learn a richer mapping).
    
    Args:
        h_dense_list: List of dense activations at each bridge site
        h_sparse_pre_list: List of sparse activations BEFORE AbsTopK at each bridge site
        h_sparse_post_list: List of sparse activations AFTER AbsTopK at each bridge site
        bridge_set: The bridge modules
        sharpness: Sharpness for soft thresholding (ThresholdEncoder only)
        hard: Whether to use hard thresholding (ThresholdEncoder only)
        
    Returns:
        BridgeNMSEResult with total loss and per-site breakdown
    """
    total_loss = 0.0
    n_sites = len(h_dense_list)
    encoder_losses = []
    decoder_losses = []
    
    for i in range(n_sites):
        h_d = h_dense_list[i]
        h_s_pre = h_sparse_pre_list[i]
        h_s_post = h_sparse_post_list[i]
        
        # Encoder loss: predict post-AbsTopK sparse activations from dense
        # IMPORTANT: Detach h_s_post so encoder loss only trains the encoder,
        # not the sparse model. Otherwise sparse model learns to produce
        # "predictable" activations rather than useful representations.
        h_s_pred = bridge_set.encode(i, h_d, sharpness=sharpness, hard=hard)
        encoder_loss = nmse_loss(h_s_pred, h_s_post.detach())
        encoder_losses.append(encoder_loss)
        
        # Decoder loss: predict dense from pre-AbsTopK sparse activations
        h_d_pred = bridge_set.decode(i, h_s_pre)
        decoder_loss = nmse_loss(h_d_pred, h_d)
        decoder_losses.append(decoder_loss)
        
        total_loss = total_loss + encoder_loss + decoder_loss
    
    return BridgeNMSEResult(total_loss, encoder_losses, decoder_losses)


class KLTargetCache:
    """
    Pre-computed values for efficient KL divergence computation.
    
    When computing multiple KL divergences against the same target distribution
    (e.g., y_dense for all hybrid passes), we can pre-compute the top-k indices
    and target softmax once, then reuse them for all source distributions.
    
    This saves ~4L+2 top-k operations per step for an L-layer model.
    """
    
    def __init__(
        self,
        logits_target: torch.Tensor,
        temperature: float = 1.0,
        topk: Optional[int] = None,
    ):
        """
        Pre-compute target distribution values.
        
        Args:
            logits_target: Target logits, shape (batch, seq, vocab)
            temperature: Temperature for softmax
            topk: Number of top tokens to use (None for full vocab)
        """
        self.temperature = temperature
        self.topk = topk
        
        # Apply temperature scaling
        logits_scaled = logits_target / temperature
        
        # Flatten to (batch * seq, vocab) for easier processing
        self.orig_shape = logits_scaled.shape
        if logits_scaled.dim() == 3:
            logits_scaled = logits_scaled.reshape(-1, self.orig_shape[-1])
        
        if topk is not None and topk < logits_scaled.shape[-1]:
            # Pre-compute top-k indices (no gradient needed for indices)
            _, self.topk_indices = torch.topk(logits_scaled, topk, dim=-1)  # (N, k)
            
            # Pre-compute target softmax over top-k
            logits_target_topk = torch.gather(logits_scaled, dim=-1, index=self.topk_indices)
            self.p_target = F.softmax(logits_target_topk, dim=-1)  # (N, k)
            self.use_topk = True
        else:
            # Full vocabulary - pre-compute full softmax
            self.p_target = F.softmax(logits_scaled, dim=-1)
            self.topk_indices = None
            self.use_topk = False


def kl_divergence(
    logits_target: torch.Tensor,
    logits_source: torch.Tensor,
    temperature: float = 1.0,
    topk: Optional[int] = None,
    target_cache: Optional[KLTargetCache] = None,
) -> torch.Tensor:
    """
    Compute KL divergence KL(target || source) with optional top-k approximation.
    
    When topk is specified, only computes KL over the top-k most probable tokens
    from the target distribution. This is more efficient for large vocabularies
    but may miss errors on tokens outside the target's top-k.
    
    The target distribution is what we're trying to match.
    The source distribution is what we're learning.
    
    Args:
        logits_target: Logits from the target distribution (e.g., dense model)
            Shape: (batch, seq, vocab) or (batch * seq, vocab)
            Can be None if target_cache is provided.
        logits_source: Logits from the source distribution (e.g., hybrid pass)
            Shape: same as logits_target
        temperature: Temperature for softmax (default 1.0)
        topk: If specified, only compute KL over top-k tokens from target.
            Default None computes exact KL over full vocabulary.
        target_cache: Pre-computed target values for efficiency. If provided,
            logits_target is ignored and the cached values are used.
        
    Returns:
        KL divergence (scalar)
    """
    if target_cache is not None:
        # Use pre-computed target values
        temperature = target_cache.temperature
        
        # Apply temperature scaling to source
        logits_source = logits_source / temperature
        
        # Flatten source to match cache shape
        if logits_source.dim() == 3:
            logits_source = logits_source.reshape(-1, logits_source.shape[-1])
        
        if target_cache.use_topk:
            # Gather source logits at pre-computed top-k indices
            logits_source_topk = torch.gather(logits_source, dim=-1, index=target_cache.topk_indices)
            log_p_source = F.log_softmax(logits_source_topk, dim=-1)
            kl = F.kl_div(log_p_source, target_cache.p_target, reduction='batchmean')
        else:
            log_p_source = F.log_softmax(logits_source, dim=-1)
            kl = F.kl_div(log_p_source, target_cache.p_target, reduction='batchmean')
        
        return kl * (temperature ** 2)
    
    # Original implementation when no cache provided
    # Apply temperature scaling
    logits_target = logits_target / temperature
    logits_source = logits_source / temperature
    
    if topk is not None and topk < logits_target.shape[-1]:
        # Top-k approximation: only compute KL over the top-k tokens from target
        # This captures most of the probability mass efficiently
        
        # Flatten to (batch * seq, vocab) for easier processing
        orig_shape = logits_target.shape
        if logits_target.dim() == 3:
            logits_target = logits_target.reshape(-1, orig_shape[-1])
            logits_source = logits_source.reshape(-1, orig_shape[-1])
        
        # Get top-k indices from target distribution
        _, topk_indices = torch.topk(logits_target, topk, dim=-1)  # (N, k)
        
        # Gather the top-k logits from both distributions
        logits_target_topk = torch.gather(logits_target, dim=-1, index=topk_indices)  # (N, k)
        logits_source_topk = torch.gather(logits_source, dim=-1, index=topk_indices)  # (N, k)
        
        # Compute softmax only over the top-k (renormalized)
        p_target = F.softmax(logits_target_topk, dim=-1)
        log_p_source = F.log_softmax(logits_source_topk, dim=-1)
        
        # KL divergence over the top-k tokens
        kl = F.kl_div(log_p_source, p_target, reduction='batchmean')
    else:
        # Exact KL over full vocabulary
        p_target = F.softmax(logits_target, dim=-1)
        log_p_source = F.log_softmax(logits_source, dim=-1)
        kl = F.kl_div(log_p_source, p_target, reduction='batchmean')
    
    # Scale back by temperature^2 for proper gradient scaling
    return kl * (temperature ** 2)


class HybridKLResult:
    """
    Result container for hybrid KL loss computation.
    
    When using gradient buffering, this holds both the losses and the
    gradient buffers that must be released after backward().
    """
    
    def __init__(
        self,
        kl_d2s: torch.Tensor,
        kl_s2d: torch.Tensor,
        kl_d2s_per_site: Optional[List[torch.Tensor]] = None,
        kl_s2d_per_site: Optional[List[torch.Tensor]] = None,
        gradient_buffers: Optional[List[GradientBuffer]] = None,
    ):
        self.kl_d2s = kl_d2s
        self.kl_s2d = kl_s2d
        self.kl_d2s_per_site = kl_d2s_per_site or []
        self.kl_s2d_per_site = kl_s2d_per_site or []
        self.gradient_buffers = gradient_buffers or []
        self._released = False
    
    def release_gradients(self):
        """Release accumulated gradients from all buffers.
        
        Uses retain_graph=True for all but the last buffer to allow
        shared computation graphs, then retain_graph=False for the
        last buffer to free memory.
        """
        if self._released:
            return
        self._released = True
        
        n_buffers = len(self.gradient_buffers)
        for i, buffer in enumerate(self.gradient_buffers):
            # Only keep graph for all but the last buffer
            retain = (i < n_buffers - 1)
            buffer.release_gradients(retain_graph=retain)
        
        # Clear buffer list to free references
        self.gradient_buffers = []
    
    @property
    def total(self) -> torch.Tensor:
        """Sum of d2s and s2d losses."""
        return self.kl_d2s + self.kl_s2d
    
    def get_detailed_losses(self) -> dict:
        """Return a dict of per-site KL losses for logging."""
        result = {}
        for i, loss in enumerate(self.kl_d2s_per_site):
            result[f"d2s_site{i}"] = loss.detach().item()
        for i, loss in enumerate(self.kl_s2d_per_site):
            result[f"s2d_site{i}"] = loss.detach().item()
        return result


def compute_hybrid_kl_losses(
    dense_model: SparseGPT,
    sparse_model: SparseGPT,
    bridge_set: "BridgeSet",
    h_dense_list: List[torch.Tensor],
    h_sparse_pre_list: List[torch.Tensor],
    y_dense: torch.Tensor,
    input_ids: torch.Tensor,
    kl_target_cache: Optional[KLTargetCache] = None,
    sharpness: Optional[float] = None,
    hard: bool = True,
) -> "HybridKLResult":
    """
    Compute KL losses for hybrid forward passes.
    
    This function computes two types of hybrid KL losses:
    1. d→s: Encode dense activations, run sparse model to end, KL to dense logits
    2. s→d: Decode sparse activations, run dense model to end, KL to dense logits
    
    Uses gradient buffering optimization to accumulate gradients efficiently.
    
    Args:
        dense_model: Frozen dense model
        sparse_model: Sparse model being trained
        bridge_set: Bridge modules
        h_dense_list: Dense activations at each bridge site
        h_sparse_pre_list: Sparse activations BEFORE AbsTopK at each bridge site
            (used as decoder input for s2d direction)
        y_dense: Dense model logits (target)
        input_ids: Input token IDs (for bigram table)
        kl_target_cache: Optional pre-computed KL target values for efficiency.
            If provided, reuses cached top-k indices and target softmax.
        sharpness: Sharpness for soft thresholding (ThresholdEncoder only)
        hard: Whether to use hard thresholding (ThresholdEncoder only)
        
    Returns:
        HybridKLResult containing kl_d2s, kl_s2d losses, per-site losses, and gradient buffers.
        IMPORTANT: Call result.release_gradients() after backward() to propagate
        gradients through the bridges.
    """
    kl_d2s, kl_s2d, kl_d2s_per_site, kl_s2d_per_site, buffers = _compute_hybrid_kl_losses_buffered(
        dense_model, sparse_model, bridge_set,
        h_dense_list, h_sparse_pre_list, y_dense, input_ids,
        kl_target_cache, sharpness, hard
    )
    return HybridKLResult(kl_d2s, kl_s2d, kl_d2s_per_site, kl_s2d_per_site, buffers)


def _compute_hybrid_kl_losses_buffered(
    dense_model: SparseGPT,
    sparse_model: SparseGPT,
    bridge_set: "BridgeSet",
    h_dense_list: List[torch.Tensor],
    h_sparse_pre_list: List[torch.Tensor],
    y_dense: torch.Tensor,
    input_ids: torch.Tensor,
    kl_target_cache: Optional[KLTargetCache] = None,
    sharpness: Optional[float] = None,
    hard: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List["GradientBuffer"]]:
    """
    Optimized implementation using gradient buffering.
    
    This accumulates gradients at checkpoint locations, reducing the number
    of backward passes through the later layers of the model.
    
    IMPORTANT: This returns the gradient buffers that must be released after
    calling backward() on the total loss. Use the wrapper function
    `compute_hybrid_kl_losses` which handles this automatically.
    
    Returns:
        Tuple of (kl_d2s, kl_s2d, kl_d2s_per_site, kl_s2d_per_site, gradient_buffers)
    """
    n_sites = len(h_dense_list)
    gradient_buffers = []
    
    # =========================================================================
    # KL for dense→sparse hybrid passes (d2s)
    # =========================================================================
    kl_d2s_total = torch.tensor(0.0, device=y_dense.device, dtype=y_dense.dtype)
    kl_d2s_per_site = []
    
    for i in range(n_sites):
        # Encode dense activation to sparse space
        h_encoded = bridge_set.encode(i, h_dense_list[i], sharpness=sharpness, hard=hard)
        
        # Cast to match model dtype (handles autocast dtype mismatches)
        # Get target dtype from y_dense (which has the model's output dtype)
        if h_encoded.dtype != y_dense.dtype:
            h_encoded = h_encoded.to(y_dense.dtype)
        
        # Create gradient buffer
        buffer = GradientBuffer(h_encoded)
        gradient_buffers.append(buffer)
        
        # Run sparse model from site i to end using the accumulator
        y_hybrid = sparse_model.forward_from_site(buffer.accumulator, i, input_ids)
        
        # KL(dense || hybrid) - gradients will accumulate in buffer
        # Use cache if provided for efficiency
        kl_site = kl_divergence(y_dense, y_hybrid, target_cache=kl_target_cache)
        kl_d2s_per_site.append(kl_site)
        kl_d2s_total = kl_d2s_total + kl_site
    
    # =========================================================================
    # KL for sparse→dense hybrid passes (s2d)
    # =========================================================================
    kl_s2d_total = torch.tensor(0.0, device=y_dense.device, dtype=y_dense.dtype)
    kl_s2d_per_site = []
    
    for i in range(n_sites):
        # Decode pre-AbsTopK sparse activation to dense space
        h_decoded = bridge_set.decode(i, h_sparse_pre_list[i])
        
        # Cast to match model dtype (handles autocast dtype mismatches)
        if h_decoded.dtype != y_dense.dtype:
            h_decoded = h_decoded.to(y_dense.dtype)
        
        # Create gradient buffer
        buffer = GradientBuffer(h_decoded)
        gradient_buffers.append(buffer)
        
        # Run dense model from site i to end using the accumulator
        y_hybrid = dense_model.forward_from_site(buffer.accumulator, i, input_ids)
        
        # KL(dense || hybrid) - gradients will accumulate in buffer
        # Use cache if provided for efficiency
        kl_site = kl_divergence(y_dense, y_hybrid, target_cache=kl_target_cache)
        kl_s2d_per_site.append(kl_site)
        kl_s2d_total = kl_s2d_total + kl_site
    
    return kl_d2s_total, kl_s2d_total, kl_d2s_per_site, kl_s2d_per_site, gradient_buffers


def verify_model_is_dense(model: nn.Module, model_name: str = "model") -> None:
    """
    Verify that a model has no weight or activation sparsity enabled.
    
    Raises AssertionError if the model has sparsity enabled.
    
    Args:
        model: The model to check
        model_name: Name for error messages
    """
    # Check sparsity config if available
    if hasattr(model, 'sparsity_config'):
        config = model.sparsity_config
        
        assert not config.enable_weight_sparsity, (
            f"{model_name} has weight sparsity enabled (enable_weight_sparsity=True). "
            "The dense model must have no weight sparsity."
        )
        
        assert not config.enable_activation_sparsity, (
            f"{model_name} has activation sparsity enabled (enable_activation_sparsity=True). "
            "The dense model must have no activation sparsity."
        )
    
    # Also check for actual sparse weights (in case loaded model has zeroed weights)
    total_params = 0
    nonzero_params = 0
    
    for name, param in model.named_parameters():
        if len(param.shape) >= 2:  # Only check weight matrices
            total_params += param.numel()
            nonzero_params += (param != 0).sum().item()
    
    if total_params > 0:
        sparsity = 1.0 - (nonzero_params / total_params)
        # Allow small sparsity from numerical zeros, but flag significant sparsity
        if sparsity > 0.01:  # More than 1% zeros is suspicious
            print(f"WARNING: {model_name} has {sparsity*100:.1f}% zero weights. "
                  "This may indicate weight sparsity was applied.")
