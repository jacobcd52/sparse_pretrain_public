"""
Weight sparsity utilities for sparse transformer training.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

Key concepts:
- After each optimizer step, zero out all but the largest magnitude entries
- Every matrix has the SAME fraction of nonzero elements
- Sparsity is annealed from dense to target L0 over training
- Minimum j nonzero values per neuron/channel to prevent dead neurons
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SparsityState:
    """Tracks the current state of sparsity during training."""
    
    current_l0_fraction: float = 1.0  # Start fully dense
    target_l0_fraction: float = 0.001
    total_steps: int = 0
    current_step: int = 0
    anneal_start_step: int = 0  # Step when annealing starts
    anneal_end_step: int = 0  # Step when annealing ends


class WeightSparsifier:
    """
    Applies weight sparsity to model parameters after optimizer steps.
    
    Paper Section 2.1: "After each optimizer step, zero out all but the largest 
    magnitude entries in each weight matrix"
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_l0_fraction: float = 0.015625,
        anneal_start_fraction: float = 0.01,
        anneal_end_fraction: float = 0.5,
        min_weights_per_neuron: int = 4,
        total_steps: int = 1000,
        anneal_type: str = "linear",
    ):
        """
        Args:
            model: The model to sparsify
            target_l0_fraction: Target fraction of nonzero weights (default: 1/64 from authors' code)
            anneal_start_fraction: When to START annealing (fraction of training)
            anneal_end_fraction: When to END annealing (fraction of training)
            min_weights_per_neuron: Minimum nonzero weights per neuron (paper: j=4)
            total_steps: Total training steps
            anneal_type: Type of annealing schedule ("linear" or "exp")
        """
        self.model = model
        self.target_l0_fraction = target_l0_fraction
        self.min_weights_per_neuron = min_weights_per_neuron
        self.anneal_type = anneal_type
        
        self.state = SparsityState(
            current_l0_fraction=1.0,  # Always start fully dense
            target_l0_fraction=target_l0_fraction,
            total_steps=total_steps,
            anneal_start_step=int(total_steps * anneal_start_fraction),
            anneal_end_step=int(total_steps * anneal_end_fraction),
        )
        
        # Cache parameter info for efficient sparsification
        self._param_info: List[Dict] = []
        self._collect_sparse_params()
    
    def _collect_sparse_params(self):
        """Collect information about parameters that should be sparsified."""
        self._param_info = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip 1D parameters (biases, norm weights) - authors only sparsify 2D matrices
            if len(param.shape) < 2:
                continue
            
            # Skip bigram table (it's meant to be dense)
            if "bigram" in name.lower():
                continue
            
            info = {
                "name": name,
                "param": param,
                "is_bias": "bias" in name.lower(),
                "is_embedding": "wte" in name or "wpe" in name,
                "is_lm_head": "lm_head" in name,
            }
            
            # Determine the "neuron" axis for minimum weights constraint
            # For weight matrices: rows are output neurons, columns are input neurons
            # We want to ensure each output neuron has at least j nonzero input weights
            info["neuron_axis"] = 0  # Each row is an output neuron
            
            self._param_info.append(info)
    
    def get_current_l0_fraction(self) -> float:
        """Get the current L0 fraction based on annealing schedule."""
        step = self.state.current_step
        
        # Before annealing starts - fully dense
        if step < self.state.anneal_start_step:
            return 1.0
        
        # After annealing ends - at target
        if step >= self.state.anneal_end_step:
            return self.target_l0_fraction
        
        # During annealing - interpolate from 1.0 to target
        anneal_duration = self.state.anneal_end_step - self.state.anneal_start_step
        progress = (step - self.state.anneal_start_step) / max(1, anneal_duration)
        
        if self.anneal_type == "exp":
            # Exponential annealing: l0 = target^progress
            # At progress=0: 1.0, at progress=1: target
            return self.target_l0_fraction ** progress
        else:
            # Linear annealing (default)
            return 1.0 - progress * (1.0 - self.target_l0_fraction)
    
    def get_sharkfin_lr_multiplier(self) -> float:
        """
        Get the learning rate multiplier for the "sharkfin" schedule.
        
        Paper Section 4.3: "The LR schedule is the product of two factors:
        1. Normal warmup-decay schedule
        2. Factor of 1/√L0"
        
        Returns:
            Multiplier to apply to base learning rate
        """
        l0_fraction = self.get_current_l0_fraction()
        # 1/sqrt(L0) where L0 is the fraction of nonzero weights
        return 1.0 / math.sqrt(l0_fraction)
    
    @torch.no_grad()
    def apply_sparsity(self):
        """
        Apply sparsity mask to all parameters after optimizer step.
        
        This zeros out all but the largest magnitude entries, ensuring:
        1. Same fraction of nonzero elements for every matrix
        2. At least min_weights_per_neuron nonzero per neuron/channel
        """
        l0_fraction = self.get_current_l0_fraction()
        self.state.current_l0_fraction = l0_fraction
        
        # If fully dense, nothing to do
        if l0_fraction >= 1.0:
            return
        
        for info in self._param_info:
            param = info["param"]
            self._sparsify_parameter(param, l0_fraction, info)
    
    @torch.no_grad()
    def _sparsify_parameter(
        self,
        param: torch.Tensor,
        l0_fraction: float,
        info: Dict,
    ):
        """
        Sparsify a single parameter tensor.
        
        Args:
            param: The parameter to sparsify (modified in-place)
            l0_fraction: Target fraction of nonzero elements
            info: Parameter metadata
        """
        if l0_fraction >= 1.0:
            return
        
        # Calculate number of elements to keep
        total_elements = param.numel()
        k = max(1, int(total_elements * l0_fraction))
        
        # Apply sparsification with minimum per neuron constraint
        if self.min_weights_per_neuron > 0 and len(param.shape) == 2:
            self._sparsify_with_min_per_neuron(param, k, info)
        else:
            # Simple top-k by magnitude (no min per neuron constraint)
            self._sparsify_simple(param, k)
    
    @torch.no_grad()
    def _sparsify_simple(self, param: torch.Tensor, k: int):
        """Simple top-k sparsification by magnitude."""
        flat = param.view(-1)
        
        if k >= flat.numel():
            return
        
        # Find threshold for top-k
        _, topk_indices = torch.topk(flat.abs(), k, sorted=False)
        
        # Create mask and apply
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        
        flat.mul_(mask.float())
    
    @torch.no_grad()
    def _sparsify_with_min_per_neuron(
        self,
        param: torch.Tensor,
        k: int,
        info: Dict,
    ):
        """
        Sparsify while ensuring minimum weights per neuron.
        
        Paper Section 2.3: "Never zero out values that would cause a neuron or 
        attention channel to have fewer than j nonzero values"
        
        Args:
            param: 2D weight matrix (modified in-place)
            k: Total number of elements to keep
            info: Parameter metadata
        """
        n_neurons, n_inputs = param.shape
        j = self.min_weights_per_neuron
        
        # First, identify the j largest weights per neuron (these are protected)
        abs_param = param.abs()
        
        if j >= n_inputs:
            # All weights are protected
            return
        
        # Get indices of top-j weights per neuron
        _, topj_indices = torch.topk(abs_param, j, dim=1, sorted=False)
        
        # Create mask for protected weights
        protected_mask = torch.zeros_like(param, dtype=torch.bool)
        protected_mask.scatter_(1, topj_indices, True)
        
        # Count protected weights
        n_protected = protected_mask.sum().item()
        
        # Calculate remaining budget
        remaining_k = max(0, k - n_protected)
        
        if remaining_k <= 0:
            # Only keep protected weights
            param.mul_(protected_mask.float())
            return
        
        # For non-protected weights, select top-k by magnitude
        unprotected_abs = abs_param.clone()
        unprotected_abs[protected_mask] = -1  # Exclude protected from consideration
        
        flat_unprotected = unprotected_abs.view(-1)
        
        if remaining_k < (flat_unprotected >= 0).sum():
            # Find top remaining_k among non-protected
            _, topk_indices = torch.topk(flat_unprotected, remaining_k, sorted=False)
            
            additional_mask = torch.zeros_like(flat_unprotected, dtype=torch.bool)
            additional_mask.scatter_(0, topk_indices, True)
            additional_mask = additional_mask.view_as(param)
        else:
            # Keep all non-protected
            additional_mask = ~protected_mask
        
        # Combine masks
        final_mask = protected_mask | additional_mask
        param.mul_(final_mask.float())
    
    def step(self):
        """Call after each optimizer step to apply sparsity and update state."""
        self.apply_sparsity()
        self.state.current_step += 1
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get statistics about current sparsity levels."""
        # Compute anneal progress
        anneal_duration = self.state.anneal_end_step - self.state.anneal_start_step
        if anneal_duration > 0 and self.state.current_step >= self.state.anneal_start_step:
            steps_into_anneal = self.state.current_step - self.state.anneal_start_step
            anneal_progress = min(1.0, steps_into_anneal / anneal_duration)
        else:
            anneal_progress = 0.0
        
        stats = {
            "current_l0_fraction": self.get_current_l0_fraction(),
            "target_l0_fraction": self.target_l0_fraction,
            "sharkfin_lr_multiplier": self.get_sharkfin_lr_multiplier(),
            "anneal_progress": anneal_progress,
        }
        
        # Compute actual sparsity for each parameter
        total_params = 0
        total_nonzero = 0
        
        for info in self._param_info:
            param = info["param"]
            n_params = param.numel()
            n_nonzero = (param != 0).sum().item()
            total_params += n_params
            total_nonzero += n_nonzero
        
        if total_params > 0:
            stats["actual_l0_fraction"] = total_nonzero / total_params
            stats["actual_nonzero_params"] = total_nonzero
            stats["total_sparse_params"] = total_params
        
        return stats


class SharkfinScheduler:
    """
    Learning rate scheduler that implements the "sharkfin" schedule.
    
    Paper Section 4.3: "The LR schedule is the product of two factors:
    1. Normal warmup-decay schedule
    2. Factor of 1/√L0"
    
    This scheduler wraps a base scheduler and multiplies by 1/√L0.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        total_steps: int,
        warmup_fraction: float = 0.01,
        enable_lr_decay: bool = True,
        sparsifier: Optional[WeightSparsifier] = None,
        use_sharkfin: bool = True,
    ):
        """
        Args:
            optimizer: The optimizer to schedule
            base_lr: Base learning rate (before sharkfin multiplier)
            total_steps: Total training steps
            warmup_fraction: Fraction of steps for warmup (paper: 1%)
            enable_lr_decay: Whether to decay LR after warmup (paper default: True)
            sparsifier: WeightSparsifier to get L0 fraction from
            use_sharkfin: Whether to apply the 1/√L0 multiplier
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.enable_lr_decay = enable_lr_decay
        self.sparsifier = sparsifier
        self.use_sharkfin = use_sharkfin
        
        self.current_step = 0
    
    def get_warmup_decay_factor(self) -> float:
        """Get the warmup-decay factor (without sharkfin)."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / max(1, self.warmup_steps)
        elif self.enable_lr_decay:
            # Linear decay
            decay_steps = self.total_steps - self.warmup_steps
            steps_since_warmup = self.current_step - self.warmup_steps
            return max(0.0, 1.0 - steps_since_warmup / max(1, decay_steps))
        else:
            # No decay - stay at 1.0 after warmup
            return 1.0
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        warmup_decay = self.get_warmup_decay_factor()
        
        if self.use_sharkfin and self.sparsifier is not None:
            sharkfin = self.sparsifier.get_sharkfin_lr_multiplier()
        else:
            sharkfin = 1.0
        
        return self.base_lr * warmup_decay * sharkfin
    
    def step(self):
        """Update learning rate."""
        lr = self.get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
    
    def get_last_lr(self) -> List[float]:
        """Get last computed learning rate."""
        return [self.get_lr()]


def clip_grad_rms_(parameters, max_rms: float = 1.0) -> float:
    """
    Clip gradients by RMS (root mean square).
    
    Paper Section 4.2: "Clip root-mean-square of gradient to 1"
    "This is ESSENTIAL for training stability"
    
    Args:
        parameters: Iterable of parameters with gradients
        max_rms: Maximum RMS value
        
    Returns:
        The gradient RMS before clipping
    """
    parameters = list(parameters)
    
    # Compute RMS of all gradients
    total_sq_sum = 0.0
    total_count = 0
    
    for p in parameters:
        if p.grad is not None:
            total_sq_sum += p.grad.data.pow(2).sum().item()
            total_count += p.grad.numel()
    
    if total_count == 0:
        return 0.0
    
    rms = math.sqrt(total_sq_sum / total_count)
    
    # Clip if necessary
    if rms > max_rms:
        clip_coef = max_rms / rms
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    
    return rms


def normalize_grad_rms_(parameters, target_rms: float = 1.0) -> float:
    """
    Normalize gradients to have RMS = 1.
    
    This is what the authors' code actually does - it ALWAYS normalizes,
    not just clips. This is different from standard gradient clipping.
    
    Args:
        parameters: Iterable of parameters with gradients
        
    Returns:
        The gradient RMS before normalizing
    """
    parameters = list(parameters)
    
    # Compute total gradient norm (L2)
    total_norm_sq = 0.0
    total_count = 0
    
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += p.grad.data.pow(2).sum().item()
            total_count += p.grad.numel()
    
    if total_count == 0:
        return 0.0
    
    # RMS = norm / sqrt(count)
    total_norm = math.sqrt(total_norm_sq)
    rms = total_norm / math.sqrt(total_count)
    
    # Normalize all gradients to have RMS = target_rms (always, not just when > 1)
    if rms > 1e-8:  # Avoid division by zero
        scale = float(target_rms) / (rms + 1e-5)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)
    
    return rms


def apply_manual_weight_decay_(
    parameters,
    weight_decay: float,
    learning_rate: float,
    exclude_names: Optional[List[str]] = None,
):
    """
    Apply weight decay manually after optimizer step.
    
    This is how the authors' code does it - NOT using AdamW's built-in
    weight decay, but manually subtracting: p.data -= wd * lr * p.data
    
    This means weight decay scales with learning rate (unlike AdamW).
    
    Args:
        parameters: Iterable of (name, param) tuples
        weight_decay: Weight decay coefficient
        learning_rate: Current learning rate
        exclude_names: Parameter names to exclude (e.g., bigram_table)
    """
    if exclude_names is None:
        exclude_names = []
    
    with torch.no_grad():
        for name, p in parameters:
            if len(p.shape) > 1:  # Only apply to 2D+ params (not biases)
                # Check exclusions
                should_exclude = any(excl in name for excl in exclude_names)
                if not should_exclude:
                    p.data.mul_(1.0 - weight_decay * learning_rate)

