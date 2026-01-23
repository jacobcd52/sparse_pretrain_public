"""
Example script showing how to use the weight ablation evaluation.

This demonstrates the same analysis as the node ablation code, but for weights.
"""

import torch
from pathlib import Path

from .weight_ablation_eval import (
    WeightAblationConfig,
    run_and_save_weight_ablation_sweep,
)
from .masked_model import MaskedSparseGPT
from .tasks import BinaryTask
from .config import PruningConfig


def run_weight_ablation_example(
    masked_model: MaskedSparseGPT,
    base_model: torch.nn.Module,
    task: BinaryTask,
    output_dir: str = "./weight_ablation_results",
):
    """
    Run weight ablation sweep evaluation.

    This compares:
    1. Ablating random weights from the circuit (weights connecting two circuit nodes)
    2. Ablating random weights from all non-zero weights (not in circuit)
    3. Ablating random weights from non-circuit weights with similar magnitude
       (within mean ± std of circuit weights)

    Args:
        masked_model: The pruned model (defines which nodes are in the circuit)
        base_model: The base unpruned model to ablate
        task: The task to evaluate on
        output_dir: Directory to save results
    """

    # Configuration
    config = WeightAblationConfig(
        num_points=11,  # Test 0%, 10%, 20%, ..., 100% of circuit weight count
        num_trials=10,  # Number of random trials per point
        num_batches=10,  # Number of evaluation batches
        batch_size=64,
        seq_length=0,  # Dynamic padding
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run the evaluation
    result = run_and_save_weight_ablation_sweep(
        masked_model=masked_model,
        base_model=base_model,
        task=task,
        output_dir=Path(output_dir),
        config=config,
        pruning_config=masked_model.config,
        title_prefix="",
        show_progress=True,
    )

    print("\n" + "="*60)
    print("Weight Ablation Results Summary")
    print("="*60)
    print(f"\nCircuit contains {result.circuit_weight_count:,} weights out of {result.total_weight_count:,} total")
    print(f"Circuit weight magnitude statistics:")
    print(f"  Mean: {result.circuit_weight_mean:.6f}")
    print(f"  Std:  {result.circuit_weight_std:.6f}")
    print(f"\nLoss comparisons:")
    print(f"  Clean (no ablation): {result.clean_loss:.4f}")
    print(f"  Non-circuit weights ablated: {result.non_circuit_weights_ablated_loss:.4f}")

    # Show a few key points from the sweep
    print(f"\nSweep results (ablating {result.circuit_weight_count:,} weights):")
    for pct in [0.0, 0.5, 1.0]:
        num_weights = int(result.circuit_weight_count * pct)
        if num_weights in result.random_circuit_results:
            circuit_loss, circuit_std = result.random_circuit_results[num_weights]
            all_loss, all_std = result.random_all_results[num_weights]
            size_loss, size_std = result.random_size_matched_results[num_weights]

            print(f"\n  {int(pct*100)}% of circuit ({num_weights:,} weights):")
            print(f"    Circuit weights:        {circuit_loss:.4f} ± {circuit_std:.4f}")
            print(f"    All weights:           {all_loss:.4f} ± {all_std:.4f}")
            print(f"    Size-matched non-circ: {size_loss:.4f} ± {size_std:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - Plot: weight_ablation_sweep.png")
    print(f"  - Data: weight_ablation_sweep_results.json")

    return result


if __name__ == "__main__":
    # Example usage (you'll need to load your actual models and task)
    print("This is an example script. Import and use run_weight_ablation_example()")
    print("with your loaded masked_model, base_model, and task.")
