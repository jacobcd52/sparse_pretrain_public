"""
Graph/Circuit Pruning for Weight-Sparse Transformers.

Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"
Section 2.2 and Appendix A.5.

This module implements the node-mask-based pruning procedure to isolate minimal
task circuits from weight-sparse models.
"""

from .config import PruningConfig
from .tasks import BinaryTask, DummyQuoteTask, DummyArticleTask
from .node_mask import NodeMaskCollection, HeavisideSTE
from .mean_cache import MeanActivationCache
from .masked_model import MaskedSparseGPT
from .trainer import PruningTrainer
from .discretize import discretize_masks
from .calibrate import calibrate_logits
from .interchange_eval import (
    InterchangeEvalConfig,
    InterchangeResult,
    run_interchange_evaluation,
    run_and_save_interchange_eval,
    # Ablation sweep
    AblationSweepConfig,
    AblationSweepResult,
    run_ablation_sweep_evaluation,
    run_and_save_ablation_sweep,
    # Mask relaxation
    MaskRelaxationConfig,
    MaskRelaxationResult,
    run_mask_relaxation_evaluation,
    run_and_save_mask_relaxation,
)

__all__ = [
    "PruningConfig",
    "BinaryTask",
    "DummyQuoteTask", 
    "DummyArticleTask",
    "NodeMaskCollection",
    "HeavisideSTE",
    "MeanActivationCache",
    "MaskedSparseGPT",
    "PruningTrainer",
    "discretize_masks",
    "calibrate_logits",
    # Interchange evaluation
    "InterchangeEvalConfig",
    "InterchangeResult",
    "run_interchange_evaluation",
    "run_and_save_interchange_eval",
    # Ablation sweep
    "AblationSweepConfig",
    "AblationSweepResult",
    "run_ablation_sweep_evaluation",
    "run_and_save_ablation_sweep",
    # Mask relaxation
    "MaskRelaxationConfig",
    "MaskRelaxationResult",
    "run_mask_relaxation_evaluation",
    "run_and_save_mask_relaxation",
]

