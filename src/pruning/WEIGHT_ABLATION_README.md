# Weight Ablation Evaluation

This module implements weight ablation analysis for pruned circuits, analogous to the node ablation code in `interchange_eval.py`.

## Overview

While node ablation tests the importance of individual nodes (activations), weight ablation tests the importance of individual weights (parameters). This is particularly useful for understanding whether the pruned circuit relies on specific weights or if the task can be solved with other weights of similar magnitude.

## What It Does

The weight ablation evaluation compares three types of weight ablations:

1. **Random Circuit Weights**: Randomly ablate (zero out) weights that are part of the circuit
   - A weight is "in the circuit" if it connects two nodes that are both active in the pruned circuit
   - For example, in an MLP layer, weight W[i,j] from c_fc is in the circuit if:
     - Input node i (from mlp_in) is active (tau >= 0)
     - Output node j (from mlp_neuron) is active (tau >= 0)

2. **Random All Weights**: Randomly ablate weights from anywhere (excluding circuit weights)
   - This serves as a baseline showing the effect of ablating arbitrary non-circuit weights

3. **Size-Matched Non-Circuit Weights**: Randomly ablate non-circuit weights with similar magnitude to circuit weights
   - Only samples from weights whose absolute value is within [mean - std, mean + std] of circuit weights
   - This controls for the hypothesis that circuit weights might just be the largest weights

## Circuit Weight Definition

A weight W[i,j] connecting layer A (input) to layer B (output) is considered part of the circuit if:
- Node i in layer A is active in the pruned model (has tau >= 0)
- Node j in layer B is active in the pruned model (has tau >= 0)

For example:
- **Attention c_attn**: Maps attn_in → (Q, K, V). Weight is in circuit if attn_in node AND corresponding Q/K/V node are both active
- **Attention c_proj**: Maps attn_out → residual. Weight is in circuit if attn_out node is active
- **MLP c_fc**: Maps mlp_in → mlp_neuron. Weight is in circuit if both mlp_in and mlp_neuron nodes are active
- **MLP c_proj**: Maps mlp_neuron → mlp_out. Weight is in circuit if both mlp_neuron and mlp_out nodes are active

## Usage

```python
from pruning.weight_ablation_eval import (
    WeightAblationConfig,
    run_and_save_weight_ablation_sweep,
)

# Configure the evaluation
config = WeightAblationConfig(
    num_points=11,      # Test 0%, 10%, ..., 100% of circuit weight count
    num_trials=10,      # Random trials per point for statistics
    num_batches=10,     # Batches for loss evaluation
    batch_size=64,
    device="cuda",
)

# Run the evaluation
result = run_and_save_weight_ablation_sweep(
    masked_model=masked_model,  # Your pruned model
    base_model=base_model,      # Your base unpruned model
    task=task,                  # Your task
    output_dir="./results",
    config=config,
)
```

## Outputs

The evaluation produces:

1. **Plot** (`weight_ablation_sweep.png`): Shows task loss vs number of weights ablated for all three conditions
2. **JSON** (`weight_ablation_sweep_results.json`): Contains all numerical results including:
   - Baseline losses (clean, non-circuit ablated)
   - Sweep results for each condition
   - Circuit weight statistics (count, mean, std)

## Interpretation

If the circuit is truly important:
- Ablating circuit weights should cause **larger** loss increases than ablating random non-circuit weights
- Even after controlling for weight magnitude (size-matched), circuit weights should still be more important

If the effect is mainly due to weight magnitude:
- Circuit weights and size-matched non-circuit weights should show similar loss increases
- This would suggest the circuit just contains the largest weights

## Comparison to Node Ablation

| Aspect | Node Ablation | Weight Ablation |
|--------|---------------|-----------------|
| **Unit** | Activations (nodes) | Parameters (weights) |
| **Method** | Set activations to zero or mean | Set weights to zero |
| **Circuit definition** | Nodes with tau >= 0 | Weights connecting two active nodes |
| **Size matching** | N/A (all activations treated equally) | Weights matched by magnitude |
| **Interchange** | Replace with unpruned activations | N/A (weights don't have "unpruned" values during inference) |

## File Structure

- `weight_ablation_eval.py`: Main implementation
- `example_weight_ablation.py`: Example usage script
- `WEIGHT_ABLATION_README.md`: This file

## Implementation Details

### Efficient Weight Ablation

The implementation temporarily modifies weights during forward pass:
```python
original_weight = layer.weight.data.clone()
layer.weight.data = original_weight * (~ablation_mask).float()
output = layer(input)
layer.weight.data = original_weight  # Restore
```

This is more efficient than creating new model copies for each ablation trial.

### Size Matching

For size-matched sampling:
1. Compute mean and std of absolute values of all circuit weights
2. When sampling non-circuit weights, only consider weights with magnitude in [mean - std, mean + std]
3. This ensures we're comparing circuit weights to non-circuit weights of similar importance (by magnitude)

## Related Work

This analysis is inspired by:
- Node ablation in `interchange_eval.py`
- Edge attribution in circuit discovery (e.g., Conmy et al. 2023, "Towards Automated Circuit Discovery")
- Magnitude-based pruning literature showing weight magnitude correlates with importance
