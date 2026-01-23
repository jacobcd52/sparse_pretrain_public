"""
Configuration for graph/circuit pruning.

Based on Appendix A.5 Table 2 hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class PruningConfig:
    """Configuration for graph/circuit pruning."""
    
    # === Target settings ===
    target_loss: float = 0.15  # Target task loss to achieve
    
    # === Mask optimization hyperparameters (Table 2 centers) ===
    k_coef: float = 1e-4  # Weight for mask sparsity penalty
    init_noise_scale: float = 1e-2  # Gaussian init noise std
    init_noise_bias: float = 1e-1  # Gaussian init noise mean
    heaviside_temp: float = 1.0  # Temperature for sigmoid surrogate gradient
    
    # === Optimizer settings ===
    lr: float = 3e-3  # Learning rate
    weight_decay: float = 1e-3  # AdamW weight decay
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.95  # Adam beta2 (inv_beta2 = 0.05 in paper)
    eps: float = 1e-8  # Adam epsilon
    grad_clip_norm: float = 1.0  # Gradient clipping (RMS)
    
    # === LR schedule ===
    lr_warmup_frac: float = 0.0  # Fraction of training for warmup (paper says "no warmup")
    # Note: Paper text says no warmup but Table 2 has lr_warmup_frac=0.05 as search center
    # We default to 0 per the text, but this is configurable
    
    # === Training settings ===
    num_steps: int = 2000  # Number of optimization steps
    batch_size: int = 64  # Number of task datapoints per batch (= 128 sequences)
    seq_length: int = 256  # Maximum sequence length
    
    # === Mean cache settings ===
    mean_cache_num_batches: int = 100  # Batches to estimate mean activations
    mean_cache_batch_size: int = 32  # Batch size for mean estimation
    
    # === Ablation settings ===
    # Options: "zero", "mean_pretrain", "mean_task"
    # - zero: masked nodes → 0
    # - mean_pretrain: masked nodes → mean activation over SimpleStories
    # - mean_task: masked nodes → mean activation over task dataset
    ablation_type: str = "mean_pretrain"
    
    # === Token embedding mask ===
    # If True, also learn a mask over the vocabulary (one mask per token type)
    # Masked tokens have their embeddings replaced with the mean embedding
    mask_token_embeds: bool = False
    
    # === Binary loss (for tense task) ===
    # If True, compute CE loss over only [correct, incorrect] logits
    # instead of full vocabulary. Useful when task requires relative comparison.
    use_binary_loss: bool = False
    
    # === Freeze layer norm scale ===
    # If True, during the forward pass we first run the unpruned model to capture
    # layer norm outputs, then use those saved outputs in the pruned forward pass.
    # This prevents the layer norm statistics from changing due to masking.
    freeze_layernorm_scale: bool = False
    
    # === Discretization settings ===
    discretization_tolerance: float = 0.01  # Tolerance for target loss in bisection
    discretization_max_iters: int = 50  # Max bisection iterations
    
    # === Calibration settings ===
    calibration_steps: int = 16  # LBFGS steps for logit calibration
    
    # === Logging ===
    log_every: int = 50  # Log every N steps
    
    # === Device ===
    device: str = "cuda"
    
    # === Paths ===
    output_dir: str = "pruning_outputs"
    
    # === Node locations to mask ===
    # These are the locations where boolean masks are inserted (Appendix A.5)
    mask_locations: List[str] = field(default_factory=lambda: [
        "attn_in",      # After RMSNorm before attention
        "attn_q",       # After Q projection
        "attn_k",       # After K projection  
        "attn_v",       # After V projection
        "attn_out",     # At end of attention block (before residual add)
        "mlp_in",       # After RMSNorm before MLP
        "mlp_neuron",   # After MLP activation (post-GELU)
        "mlp_out",      # At end of MLP block (before residual add)
    ])
    
    @classmethod
    def from_yaml(cls, path: str) -> "PruningConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        from dataclasses import asdict
        return asdict(self)

