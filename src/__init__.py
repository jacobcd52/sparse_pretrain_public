# Weight-sparse transformer pretraining
# Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

from .config import Config, ModelConfig, SparsityConfig, OptimizerConfig, TrainingConfig
from .model import SparseGPT, AbsTopK, create_model
from .sparsity import WeightSparsifier, SharkfinScheduler
from .data import create_dataloader, create_validation_data

# Bridge training components
from .config_bridges import FullBridgesConfig, BridgesConfig, DenseModelConfig
from .bridges import (
    BridgeSet, BridgeEncoder, BridgeDecoder,
    nmse_loss, kl_divergence, compute_hybrid_kl_losses,
    GradientBuffer, HybridKLResult, KLTargetCache,
)

# Hooked model for interpretability (optional - requires transformer_lens)
try:
    from .hooked_model import HookedSparseGPT, HookedSparseGPTConfig
    _HAS_HOOKED = True
except ImportError:
    _HAS_HOOKED = False

__all__ = [
    # Core config
    "Config",
    "ModelConfig", 
    "SparsityConfig",
    "OptimizerConfig",
    "TrainingConfig",
    # Model
    "SparseGPT",
    "AbsTopK",
    "create_model",
    # Sparsity
    "WeightSparsifier",
    "SharkfinScheduler",
    # Data
    "create_dataloader",
    "create_validation_data",
    # Bridges
    "FullBridgesConfig",
    "BridgesConfig",
    "DenseModelConfig",
    "BridgeSet",
    "BridgeEncoder",
    "BridgeDecoder",
    "nmse_loss",
    "kl_divergence",
    "compute_hybrid_kl_losses",
    "GradientBuffer",
    "HybridKLResult",
    "KLTargetCache",
    # Hooked model (if transformer_lens available)
    "HookedSparseGPT",
    "HookedSparseGPTConfig",
]
