"""
Node mask implementation for circuit pruning.

Implements learnable masks with:
- Parameters τ clamped to [-1, 1]
- Heaviside step function in forward pass
- Sigmoid-derivative surrogate gradient in backward pass

Based on Appendix A.5 of Gao et al. (2025).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass


class HeavisideSTE(torch.autograd.Function):
    """
    Heaviside step function with Straight-Through Estimator using sigmoid derivative.
    
    Forward: H(x) = 1 if x >= 0 else 0
    Backward: Uses sigmoid derivative as surrogate gradient
    
    The temperature controls the sharpness of the sigmoid in the backward pass.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.temperature = temperature
        # Heaviside: 1 if x >= 0, else 0
        return (x >= 0).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        temperature = ctx.temperature
        
        # Sigmoid surrogate gradient: σ'(x/T) / T = σ(x/T) * (1 - σ(x/T)) / T
        scaled_x = x / temperature
        sigmoid_x = torch.sigmoid(scaled_x)
        surrogate_grad = sigmoid_x * (1 - sigmoid_x) / temperature
        
        return grad_output * surrogate_grad, None  # None for temperature


def heaviside_ste(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply Heaviside step function with STE gradient."""
    return HeavisideSTE.apply(x, temperature)


@dataclass
class NodeLocation:
    """Identifies a node location in the model."""
    layer: int
    location_type: str  # e.g., "attn_in", "mlp_neuron", etc.
    dim: int  # Dimension of the activation at this location


class NodeMask(nn.Module):
    """
    A single node mask for a specific location in the model.
    
    The mask has one learnable parameter per node (element of the activation).
    Parameters are clamped to [-1, 1] and passed through Heaviside to get binary masks.
    """
    
    def __init__(
        self,
        num_nodes: int,
        init_noise_scale: float = 0.01,
        init_noise_bias: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.temperature = temperature
        
        # Initialize with Gaussian noise centered at init_noise_bias
        # Then clamp to [-1, 1]
        init_values = torch.randn(num_nodes) * init_noise_scale + init_noise_bias
        init_values = init_values.clamp(-1.0, 1.0)
        
        # Learnable mask parameters
        self.tau = nn.Parameter(init_values)
        
        # Register buffer for mean activations (set later)
        self.register_buffer("mean_activation", torch.zeros(num_nodes))
        self.mean_set = False
    
    def set_mean(self, mean: torch.Tensor):
        """Set the mean activation for mean ablation."""
        self.mean_activation.copy_(mean)
        self.mean_set = True
    
    def clamp_parameters(self):
        """Clamp parameters to [-1, 1] (call after each optimizer step)."""
        with torch.no_grad():
            self.tau.clamp_(-1.0, 1.0)
    
    def get_binary_mask(self) -> torch.Tensor:
        """Get the current binary mask (0 or 1 for each node)."""
        return heaviside_ste(self.tau, self.temperature)
    
    def get_num_active(self) -> int:
        """Get number of currently active (non-masked) nodes."""
        with torch.no_grad():
            return int((self.tau >= 0).sum().item())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to activation.
        
        For active nodes (mask=1): keep original activation
        For masked nodes (mask=0): replace with mean activation
        
        x: (..., num_nodes) - activation tensor
        Returns: (..., num_nodes) - masked activation
        """
        mask = self.get_binary_mask()  # (num_nodes,)
        
        # Broadcast mask to match x shape
        # x shape: (batch, seq, num_nodes) or (batch, seq, heads, head_dim) etc.
        # We apply mask to the last dimension
        
        if not self.mean_set:
            # If mean not set, just apply mask (zero ablation)
            return x * mask
        else:
            # Mean ablation: x * mask + mean * (1 - mask)
            mean = self.mean_activation
            return x * mask + mean * (1 - mask)


class NodeMaskCollection(nn.Module):
    """
    Collection of node masks for all locations in the model.
    
    Organizes masks by layer and location type.
    """
    
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_mlp: int,
        n_heads: int,
        d_head: int,
        mask_locations: List[str],
        init_noise_scale: float = 0.01,
        init_noise_bias: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.d_head = d_head
        self.mask_locations = mask_locations
        self.temperature = temperature
        
        # Create masks for each layer and location
        self.masks = nn.ModuleDict()
        
        for layer in range(n_layers):
            for loc in mask_locations:
                # Determine dimension for this location
                dim = self._get_dim_for_location(loc)
                
                key = f"layer{layer}_{loc}"
                self.masks[key] = NodeMask(
                    num_nodes=dim,
                    init_noise_scale=init_noise_scale,
                    init_noise_bias=init_noise_bias,
                    temperature=temperature,
                )
    
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
    
    def get_mask(self, layer: int, location: str) -> NodeMask:
        """Get the mask for a specific layer and location."""
        key = f"layer{layer}_{location}"
        return self.masks[key]
    
    def clamp_all_parameters(self):
        """Clamp all mask parameters to [-1, 1]."""
        for mask in self.masks.values():
            mask.clamp_parameters()
    
    def get_total_active_nodes(self) -> int:
        """Get total number of active nodes across all masks."""
        total = 0
        for mask in self.masks.values():
            total += mask.get_num_active()
        return total
    
    def get_total_nodes(self) -> int:
        """Get total number of nodes across all masks."""
        total = 0
        for mask in self.masks.values():
            total += mask.num_nodes
        return total
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Get the sparsity penalty (number of active nodes).
        
        This is differentiable due to the sigmoid surrogate gradient.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for mask in self.masks.values():
            # Sum of binary masks (with gradient through STE)
            total = total + mask.get_binary_mask().sum()
        return total
    
    def set_means_from_cache(self, mean_cache: Dict[str, torch.Tensor]):
        """
        Set mean activations from a cache dictionary.
        
        Args:
            mean_cache: Dict mapping "layer{i}_{loc}" to mean tensors
        """
        for key, mask in self.masks.items():
            if key in mean_cache:
                mask.set_mean(mean_cache[key])
            else:
                print(f"Warning: No mean cache for {key}")
    
    def get_mask_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of active nodes per layer and location."""
        summary = {}
        for key, mask in self.masks.items():
            parts = key.split("_", 1)
            layer = parts[0]
            loc = parts[1]
            if layer not in summary:
                summary[layer] = {}
            summary[layer][loc] = mask.get_num_active()
        return summary
    
    def get_all_tau_values(self) -> torch.Tensor:
        """Get all tau values concatenated into a single tensor."""
        return torch.cat([mask.tau for mask in self.masks.values()])
    
    def apply_threshold(self, threshold: float):
        """
        Apply a threshold to all masks to make them binary.
        
        Nodes with tau >= threshold are kept, others are masked.
        """
        with torch.no_grad():
            for mask in self.masks.values():
                # Set tau to +1 or -1 based on threshold
                above = mask.tau >= threshold
                mask.tau.fill_(-1.0)
                mask.tau[above] = 1.0
    
    def keep_top_k(self, k: int):
        """
        Keep only the top k nodes (by tau value) across all masks.
        
        Selects the k nodes with highest tau values across ALL nodes,
        regardless of whether they are currently active (tau >= 0) or not.
        This allows exploring circuits of any size during discretization.
        
        All other nodes are masked out.
        """
        # Gather ALL tau values (not just active ones)
        all_taus = []
        tau_to_mask = []  # Maps index in all_taus to (mask_key, node_index)
        
        for key, mask in self.masks.items():
            for i in range(mask.num_nodes):
                tau_val = mask.tau[i].item()
                all_taus.append(tau_val)
                tau_to_mask.append((key, i))
        
        num_total = len(all_taus)
        
        # If k >= total nodes, keep all nodes as-is
        if k >= num_total:
            return
        
        if k <= 0:
            # Deactivate all nodes
            with torch.no_grad():
                for mask in self.masks.values():
                    mask.tau.fill_(-1.0)
            return
        
        all_taus = torch.tensor(all_taus)
        
        # Find the k-th largest tau among ALL nodes
        sorted_taus, sorted_indices = torch.sort(all_taus, descending=True)
        threshold = sorted_taus[k - 1].item()
        
        # Build set of (key, index) pairs to keep active
        keep_active = set()
        for i in range(k):
            idx = sorted_indices[i].item()
            keep_active.add(tau_to_mask[idx])
        
        # Apply: set tau to -1 for all nodes, then +1 for those in keep_active
        with torch.no_grad():
            for key, mask in self.masks.items():
                for i in range(mask.num_nodes):
                    if (key, i) in keep_active:
                        mask.tau[i] = 1.0
                    else:
                        mask.tau[i] = -1.0

