"""
Configuration dataclasses for weight-sparse transformer training.
Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Core dimensions (from authors' reference code)
    n_layer: int = 8
    d_model: int = 1024  # Authors' default; paper mentions 2048 for some experiments
    n_ctx: int = 256  # context length / block size
    d_head: int = 16  # small for monosemanticity (paper uses 16, standard GPT-2 uses 64)
    d_mlp: Optional[int] = None  # defaults to 4 * d_model if None
    
    # Vocab (will be set from tokenizer)
    vocab_size: Optional[int] = None
    
    # Normalization
    use_rms_norm: bool = True  # Paper uses RMSNorm for zero-value privilege
    
    # Embeddings
    tie_embeddings: bool = False  # Paper uses UNTIED embeddings
    use_positional_embeddings: bool = False  # Paper doesn't use positional embeddings
    
    # Bigram table (paper Section 1.5)
    use_bigram_table: bool = True  # Dense d_vocab x d_vocab matrix added to logits
    
    # Attention sinks (paper Section 1.6)
    use_attention_sinks: bool = True  # Per-head learnable attention denominator bias
    
    # Activation function
    activation: Literal["gelu", "relu"] = "gelu"
    
    # Dropout (paper doesn't mention dropout, so default to 0)
    dropout: float = 0.0
    
    # Use biases in linear layers
    use_bias: bool = True
    
    # Flash attention
    use_flash_attention: bool = True
    
    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
    
    @property
    def n_heads(self) -> int:
        """Number of attention heads."""
        assert self.d_model % self.d_head == 0, f"d_model ({self.d_model}) must be divisible by d_head ({self.d_head})"
        return self.d_model // self.d_head


@dataclass
class SparsityConfig:
    """Weight and activation sparsity configuration."""
    
    # Weight sparsity
    enable_weight_sparsity: bool = True
    target_l0_fraction: float = 0.015625  # 1/64 from authors' code (paper mentions ~1 in 1000 for sparsest)
    
    # Sparsity annealing (paper Section 2.2)
    # Always anneals from fully dense (1.0) to target over the specified range
    sparsity_anneal_start_fraction: float = 0.01  # When to START annealing (fraction of training)
    sparsity_anneal_end_fraction: float = 0.5  # When to END annealing (fraction of training)
    anneal_type: Literal["linear", "exp"] = "linear"  # Annealing schedule type
    
    # Minimum nonzero weights per neuron (paper Section 2.3)
    min_weights_per_neuron: int = 4  # j parameter from paper
    
    # Activation sparsity (paper Section 3)
    enable_activation_sparsity: bool = True
    activation_topk_fraction: float = 0.25  # Keep top 1/4 activations (paper default)
    
    # Which locations to apply activation sparsity (paper Section 1.7 and 3.2)
    activation_sparsity_locations: str = "attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,attn_v,attn_k,attn_q"


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    
    # Optimizer type: "adamw" (PyTorch AdamW) or "adam" (raw Adam with manual weight decay like circuit_sparsity)
    optimizer_type: Literal["adamw", "adam"] = "adamw"
    
    # AdamW parameters (paper Section 4.1)
    learning_rate: float = 1.28e-2  # From authors' reference code
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 0.1  # NOTE: Unusually large epsilon! (paper Section 4.1)
    
    # Gradient clipping (paper Section 4.2) - ESSENTIAL for stability
    enable_grad_clip: bool = True
    grad_clip_rms: float = 1.0  # Clip RMS of gradient to this value
    
    # Learning rate schedule (paper Section 4.3-4.5)
    warmup_fraction: float = 0.01  # 1% warmup (paper default)
    enable_lr_decay: bool = True  # Linear decay after warmup
    use_sharkfin_schedule: bool = False  # Authors don't use this in their simple interface


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Dataset
    dataset_name: str = ""  # HuggingFace dataset name/path
    dataset_split: str = "train"
    text_column: str = "text"
    
    # Tokenizer
    tokenizer_name: str = ""  # HuggingFace tokenizer name/path
    
    # Training
    total_tokens: int = 100_000_000  # From authors' code (100M tokens)
    batch_size: int = 64  # Per-device batch size (global_bs in authors' code)
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_steps: int = 1000
    keep_n_checkpoints: int = 5
    
    # Logging frequencies (from authors' code)
    log_every_n_steps: int = 1  # log_every in authors' code
    log_gradients_every_n_steps: int = 10  # log_gradients_every
    log_weights_every_n_steps: int = 100  # log_weights_every
    log_sparsity_every_n_steps: int = 50  # log_sparsity_every
    eval_every_n_steps: int = 20  # val_every in authors' code
    
    # Validation
    val_split: Optional[str] = "test"  # Use "test" split if available, else hold out from train
    val_holdout_fraction: float = 0.01  # Fraction to hold out if no test split (1%)
    val_max_batches: int = 20  # Max batches to use for validation (val_max_steps in authors' code)
    
    # W&B
    wandb_project: str = "circuit_sparsity"  # From authors' code
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    use_wandb: bool = True
    
    # Reproducibility
    seed: int = 0  # From authors' code
    
    # HuggingFace Hub
    hf_repo: Optional[str] = None  # If set, upload final checkpoint to this repo (e.g., "username/model-name")
    
    # Data loading
    num_workers: int = 8  # Number of dataloader workers (increase for multi-GPU training)
    
    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint dir (e.g., "checkpoints/checkpoint-5000") or "latest"
    
    # torch.compile (PyTorch 2.0+ JIT compilation for speedup)
    use_torch_compile: bool = False  # Enable torch.compile for models
    torch_compile_mode: str = "default"  # Options: "default", "reduce-overhead", "max-autotune"
    torch_compile_backend: str = "inductor"  # Options: "inductor" (fastest), "eager", "aot_eager"


@dataclass
class Config:
    """Full configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            sparsity=SparsityConfig(**raw.get("sparsity", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for W&B logging."""
        from dataclasses import asdict
        return asdict(self)

