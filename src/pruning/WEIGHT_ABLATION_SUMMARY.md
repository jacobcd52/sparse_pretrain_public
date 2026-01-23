# Weight Ablation Implementation Summary

## What Was Created

I've implemented a complete weight ablation evaluation system analogous to the existing node ablation code in `interchange_eval.py`. This allows you to test whether the pruned circuit relies on specific weights by comparing the effect of ablating circuit weights vs non-circuit weights.

## Files Created

1. **`weight_ablation_eval.py`** (main implementation, ~850 lines)
   - Core weight ablation functionality
   - Circuit weight identification
   - Random weight sampling (all, circuit-only, size-matched)
   - Evaluation, plotting, and saving functions

2. **`example_weight_ablation.py`** (~100 lines)
   - Example script showing how to use the weight ablation evaluation
   - Can be imported and called with your models

3. **`WEIGHT_ABLATION_README.md`**
   - Comprehensive documentation
   - Explains the approach, usage, and interpretation
   - Comparison to node ablation

4. **`test_weight_ablation.py`** (~250 lines)
   - Unit tests for core functionality
   - Validates weight sampling, size matching, and statistics

## Key Features

### 1. Circuit Weight Definition
A weight W[i,j] is "in the circuit" if it connects two active nodes:
- Input node i is active (tau >= 0 in the pruned model)
- Output node j is active (tau >= 0 in the pruned model)

This applies to all weight matrices:
- Attention: `c_attn` (attn_in → Q/K/V), `c_proj` (attn_out → residual)
- MLP: `c_fc` (mlp_in → mlp_neuron), `c_proj` (mlp_neuron → mlp_out)

### 2. Three Types of Ablation

**Circuit Weights:**
- Random weights that connect two circuit nodes
- Tests whether these specific weights are important

**Random All Weights:**
- Random weights from anywhere (excluding circuit)
- Baseline showing effect of random ablation

**Size-Matched Non-Circuit:**
- Non-circuit weights with similar magnitude to circuit weights
- Only samples from weights in [circuit_mean - std, circuit_mean + std]
- Controls for the hypothesis that circuit weights are just the largest weights

### 3. Complete Evaluation Pipeline

```python
from pruning.weight_ablation_eval import (
    WeightAblationConfig,
    run_and_save_weight_ablation_sweep,
)

config = WeightAblationConfig(
    num_points=11,      # 0%, 10%, ..., 100% of circuit weight count
    num_trials=10,      # Random trials for statistics
    num_batches=10,     # Evaluation batches
    batch_size=64,
)

result = run_and_save_weight_ablation_sweep(
    masked_model=masked_model,  # Defines the circuit
    base_model=base_model,      # Model to ablate
    task=task,
    output_dir="./results",
    config=config,
)
```

### 4. Outputs

**Plot:** `weight_ablation_sweep.png`
- X-axis: Number of weights ablated
- Y-axis: Task loss (log scale)
- Three curves: circuit, random all, size-matched
- Baselines: clean loss, non-circuit ablated

**JSON:** `weight_ablation_sweep_results.json`
```json
{
  "baselines": {
    "clean_loss": 0.123,
    "non_circuit_weights_ablated_loss": 0.456
  },
  "random_all_results": {...},
  "random_circuit_results": {...},
  "random_size_matched_results": {...},
  "metadata": {
    "circuit_weight_count": 12345,
    "total_weight_count": 100000,
    "circuit_weight_mean": 0.0234,
    "circuit_weight_std": 0.0156
  }
}
```

## How It Works

### Weight Identification
```python
def get_circuit_weights(model, masked_model, mask_locations):
    """
    For each weight matrix, create a boolean mask indicating
    which weights connect two active nodes.
    """
    # For MLP c_fc: (d_mlp, d_model)
    # weight[i,j] is in circuit if:
    #   - mlp_neuron[i] is active AND
    #   - mlp_in[j] is active
    weight_mask = neuron_active.unsqueeze(1) & in_active.unsqueeze(0)
```

### Weight Ablation
```python
def run_forward_with_weight_ablation(model, input_ids, ablation_masks):
    """
    Temporarily zero out specified weights during forward pass.
    """
    # For each layer's weight matrix
    original = layer.weight.data.clone()
    layer.weight.data = original * (~ablation_mask).float()
    output = layer(input)
    layer.weight.data = original  # Restore
```

### Size Matching
```python
def sample_random_weights(..., size_matched=True):
    """
    When sampling non-circuit weights, only consider those
    with magnitude in [circuit_mean - std, circuit_mean + std]
    """
    if size_matched and not from_circuit:
        weight_val = weight.abs()[i, j]
        if not (lower_bound <= weight_val <= upper_bound):
            continue  # Skip this weight
```

## Interpretation Guide

### If Circuit Weights Are Important:
- Ablating circuit weights → **large** loss increase
- Ablating random weights → small loss increase
- Even size-matched non-circuit → smaller increase than circuit

**Interpretation:** The circuit has learned to use specific weights that are necessary for the task.

### If Effect Is Due to Weight Magnitude:
- Circuit weights → large loss increase
- Size-matched non-circuit → **similar** large increase
- Random all weights → smaller increase

**Interpretation:** The circuit just contains large weights; any large weights would work.

### If Circuit Is Not Special:
- All three conditions → similar loss increases

**Interpretation:** The specific weights in the circuit are not particularly important.

## Comparison to Node Ablation

| Aspect | Node Ablation | Weight Ablation |
|--------|---------------|-----------------|
| **What's tested** | Importance of activations | Importance of parameters |
| **Ablation method** | Zero/mean ablation of activations | Zero out weights |
| **Circuit definition** | Nodes with tau >= 0 | Edges connecting active nodes |
| **Control** | Interchange with unpruned | Size-matched sampling |
| **Granularity** | Layer outputs | Individual weights |

## Next Steps

To use this with your pruned models:

1. Load your pruned model (with masks) and base model
2. Load your task
3. Run the evaluation:
   ```python
   from pruning.example_weight_ablation import run_weight_ablation_example

   result = run_weight_ablation_example(
       masked_model=your_masked_model,
       base_model=your_base_model,
       task=your_task,
       output_dir="./weight_ablation_results"
   )
   ```

4. Analyze the plot and results to see if circuit weights are special

## Technical Notes

### Efficiency
- Temporarily modifies weights rather than copying models
- Each trial only changes the relevant weight matrices
- Uses same batches across trials for fair comparison

### Correctness
- All tests pass (`test_weight_ablation.py`)
- Weight sampling verified to respect circuit/non-circuit boundaries
- Size matching verified to stay within magnitude bounds

### Extensibility
- Easy to add new sampling strategies
- Can modify which weight matrices to consider
- Can change size matching criterion (e.g., use percentiles instead of mean±std)
