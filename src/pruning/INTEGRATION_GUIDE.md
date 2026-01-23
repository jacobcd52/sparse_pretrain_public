# Integration Guide: Weight Ablation with Existing Codebase

## Overview

The weight ablation evaluation integrates seamlessly with your existing pruning pipeline. It uses the same models, tasks, and configuration as the node ablation code in `interchange_eval.py`.

## Quick Start

If you already have code that runs node ablation, adding weight ablation is simple:

```python
# Existing node ablation code
from pruning.interchange_eval import run_and_save_interchange_eval

node_result = run_and_save_interchange_eval(
    masked_model=masked_model,
    base_model=base_model,
    task=task,
    output_dir=output_dir / "interchange",
)

# Add weight ablation (NEW)
from pruning.weight_ablation_eval import run_and_save_weight_ablation_sweep

weight_result = run_and_save_weight_ablation_sweep(
    masked_model=masked_model,
    base_model=base_model,
    task=task,
    output_dir=output_dir / "weight_ablation",
)
```

## Complete Example with Pruning Pipeline

Here's how to integrate weight ablation into a complete pruning evaluation:

```python
import torch
from pathlib import Path

from pruning.config import PruningConfig
from pruning.masked_model import MaskedSparseGPT
from pruning.tasks import BinaryTask
from pruning.run_pruning import run_pruning
from pruning.interchange_eval import (
    InterchangeEvalConfig,
    run_and_save_interchange_eval,
)
from pruning.weight_ablation_eval import (
    WeightAblationConfig,
    run_and_save_weight_ablation_sweep,
)


def full_evaluation_with_weight_ablation(
    model_path: str,
    task_name: str,
    output_dir: str,
):
    """
    Complete evaluation including both node and weight ablation.
    """
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model and task
    base_model = load_model(model_path)  # Your model loading function
    task = BinaryTask(task_name)  # Your task

    # 2. Create pruning config
    pruning_config = PruningConfig(
        mask_locations=["mlp_neuron"],  # or ["attn_out", "mlp_neuron"], etc.
        sparsity_lambda=1e-3,
        # ... other config
    )

    # 3. Run pruning to get masked model
    masked_model = MaskedSparseGPT(base_model, pruning_config)
    # ... run pruning training ...
    # (or load pre-trained masks)

    # 4. Evaluate with node ablation (existing)
    print("\n" + "="*60)
    print("Running Node Ablation (Interchange Intervention)")
    print("="*60)

    node_config = InterchangeEvalConfig(
        fractions=[0.0, 0.25, 0.5, 0.75, 1.0],
        num_trials=5,
        num_batches=10,
    )

    node_result = run_and_save_interchange_eval(
        masked_model=masked_model,
        base_model=base_model,
        task=task,
        output_dir=output_dir / "node_ablation",
        config=node_config,
    )

    # 5. Evaluate with weight ablation (NEW)
    print("\n" + "="*60)
    print("Running Weight Ablation")
    print("="*60)

    weight_config = WeightAblationConfig(
        num_points=11,
        num_trials=10,
        num_batches=10,
    )

    weight_result = run_and_save_weight_ablation_sweep(
        masked_model=masked_model,
        base_model=base_model,
        task=task,
        output_dir=output_dir / "weight_ablation",
        config=weight_config,
    )

    # 6. Print comparison
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)

    print(f"\nNode Ablation:")
    print(f"  Active nodes: {node_result.num_active_nodes:,} / {node_result.total_nodes:,}")
    print(f"  Global FVU: {node_result.fvu_global:.4f}")

    print(f"\nWeight Ablation:")
    print(f"  Circuit weights: {weight_result.circuit_weight_count:,} / {weight_result.total_weight_count:,}")
    print(f"  Weight magnitude: {weight_result.circuit_weight_mean:.6f} ± {weight_result.circuit_weight_std:.6f}")

    # Compare baseline losses
    print(f"\nBaseline Losses:")
    print(f"  Unpruned clean: {node_result.unpruned_clean_loss:.4f}")
    print(f"  Clean (weight ablation): {weight_result.clean_loss:.4f}")

    return node_result, weight_result


# Usage
if __name__ == "__main__":
    full_evaluation_with_weight_ablation(
        model_path="path/to/your/model.pt",
        task_name="your_task",
        output_dir="./evaluation_results",
    )
```

## Integration Points

### 1. Shared Dependencies

Weight ablation uses the same imports as node ablation:

```python
from .masked_model import MaskedSparseGPT  # Defines circuit
from .tasks import BinaryTask               # Evaluation task
from .config import PruningConfig          # Configuration
```

### 2. Compatible Configurations

Both use similar config patterns:

```python
# Node ablation config
InterchangeEvalConfig(
    fractions=[0.0, 0.5, 1.0],  # Fractions of nodes
    num_trials=5,
    num_batches=10,
)

# Weight ablation config
WeightAblationConfig(
    num_points=11,              # Points in sweep
    num_trials=10,
    num_batches=10,
)
```

### 3. Same Output Structure

Both produce similar outputs:

```
output_dir/
├── node_ablation/
│   ├── interchange_per_layer.png
│   ├── interchange_all_layers.png
│   ├── interchange_fvu.png
│   └── interchange_results.json
└── weight_ablation/
    ├── weight_ablation_sweep.png
    └── weight_ablation_sweep_results.json
```

## Minimal Integration Example

If you just want to add weight ablation to existing code:

```python
# Add this import
from pruning.weight_ablation_eval import run_and_save_weight_ablation_sweep

# Add this call after your existing evaluations
weight_result = run_and_save_weight_ablation_sweep(
    masked_model=masked_model,
    base_model=base_model,
    task=task,
    output_dir=output_dir / "weight_ablation",
)
```

That's it! The function handles everything else.

## Advanced Usage

### Custom Weight Selection

To modify which weights are considered (e.g., only MLP weights):

```python
from pruning.weight_ablation_eval import get_circuit_weights

# Get all circuit weights
circuit_weights = get_circuit_weights(
    model=base_model,
    masked_model=masked_model,
    mask_locations=pruning_config.mask_locations,
)

# Filter to only MLP weights
mlp_weights = {
    k: v for k, v in circuit_weights.items()
    if "mlp" in k
}

# Use mlp_weights for custom analysis
```

### Custom Size Matching

To change the size matching criterion:

```python
from pruning.weight_ablation_eval import get_weight_statistics

# Get circuit statistics
mean, std, count, total = get_weight_statistics(
    model=base_model,
    circuit_weight_masks=circuit_weights,
    n_layers=masked_model.n_layers,
)

# Use custom bounds (e.g., percentiles instead of mean±std)
custom_lower = np.percentile(circuit_weight_values, 25)
custom_upper = np.percentile(circuit_weight_values, 75)

# Pass to sampling function
sample_random_weights(
    ...,
    circuit_mean=(custom_upper + custom_lower) / 2,
    circuit_std=(custom_upper - custom_lower) / 2,
    ...
)
```

### Combining Results

To analyze both node and weight ablation together:

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Node ablation (left)
xs = sorted(node_result.all_layers_results.keys())
ys = [node_result.all_layers_results[x][0] for x in xs]
ax1.plot(xs, ys)
ax1.set_xlabel("Fraction of Nodes Ablated")
ax1.set_ylabel("Loss")
ax1.set_title("Node Ablation")

# Weight ablation (right)
xs = sorted(weight_result.random_circuit_results.keys())
ys = [weight_result.random_circuit_results[x][0] for x in xs]
ax2.plot(xs, ys)
ax2.set_xlabel("Number of Weights Ablated")
ax2.set_ylabel("Loss")
ax2.set_title("Weight Ablation")

plt.tight_layout()
plt.savefig(output_dir / "combined_ablation.png")
```

## Testing Integration

To verify the integration works:

```python
# Run unit tests
python -m src.pruning.test_weight_ablation

# If successful, try with your actual models
from pruning.example_weight_ablation import run_weight_ablation_example

result = run_weight_ablation_example(
    masked_model=your_masked_model,
    base_model=your_base_model,
    task=your_task,
)
```

## Common Issues

### 1. Weight Shape Mismatch
**Error:** `RuntimeError: shape mismatch`

**Solution:** Ensure mask_locations in pruning_config match the layers you're evaluating

### 2. Out of Memory
**Error:** `CUDA out of memory`

**Solution:** Reduce batch_size or num_batches in WeightAblationConfig

```python
config = WeightAblationConfig(
    batch_size=32,  # Reduce from 64
    num_batches=5,  # Reduce from 10
)
```

### 3. No Circuit Weights Found
**Warning:** `Circuit contains 0 weights`

**Solution:** Check that your masked_model has active nodes (tau >= 0). Run discretization first:

```python
from pruning.discretize import discretize_masks

discretize_masks(masked_model, threshold=0.0)
```

## Performance Tips

1. **Reuse batches**: Both evaluations use fixed batches internally, so they're deterministic

2. **Parallel evaluation**: Run node and weight ablation in parallel if you have multiple GPUs:
   ```python
   # GPU 0: node ablation
   # GPU 1: weight ablation
   ```

3. **Reduce trials for quick tests**:
   ```python
   quick_config = WeightAblationConfig(
       num_points=5,    # Fewer points
       num_trials=3,    # Fewer trials
   )
   ```

## Next Steps

1. Run on your pruned models
2. Compare node vs weight ablation results
3. Analyze if circuit weights are special (beyond just being large)
4. Use insights to improve circuit discovery

## Support

If you encounter issues:
1. Check the unit tests pass: `python -m src.pruning.test_weight_ablation`
2. Review the example: `example_weight_ablation.py`
3. Read the detailed docs: `WEIGHT_ABLATION_README.md`
