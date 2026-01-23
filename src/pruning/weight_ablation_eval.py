"""
Weight ablation evaluation for pruned circuits.

Compares ablating random weights that are part of the pruned circuit
(connecting two circuit nodes) vs ablating random weights that are not
part of the circuit. Also includes size-matched sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .masked_model import MaskedSparseGPT
from .tasks import BinaryTask
from .config import PruningConfig


@dataclass
class WeightAblationConfig:
    """Configuration for weight ablation evaluation."""
    # Number of weight counts to evaluate
    num_points: int = 11  # e.g., 0, 10%, 20%, ..., 100% of circuit weights

    # Number of random trials per point
    num_trials: int = 10

    # Evaluation settings
    num_batches: int = 10
    batch_size: int = 64
    seq_length: int = 0  # 0 = dynamic padding

    # Device
    device: str = "cuda"


@dataclass
class WeightAblationResult:
    """Results from weight ablation evaluation."""
    # Baseline losses
    clean_loss: float  # Unpruned model, no ablation
    non_circuit_weights_ablated_loss: float  # Model with non-circuit weights ablated

    # Sweep results: num_weights -> (mean_loss, std_loss)
    random_all_results: Dict[int, Tuple[float, float]]  # Random from all weights
    random_circuit_results: Dict[int, Tuple[float, float]]  # Random from circuit weights
    random_size_matched_results: Dict[int, Tuple[float, float]]  # Size-matched non-circuit weights

    # Metadata
    circuit_weight_count: int  # Number of weights in circuit
    total_weight_count: int  # Total non-zero weights
    num_trials: int

    # Circuit weight statistics (for size matching)
    circuit_weight_mean: float
    circuit_weight_std: float


def get_circuit_weights(
    model: nn.Module,
    masked_model: MaskedSparseGPT,
    mask_locations: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Get binary masks indicating which weights are part of the circuit.

    A weight is in the circuit if it connects two circuit nodes (both active).
    For each weight matrix W connecting layer A (input) to layer B (output),
    W[i,j] is in the circuit if node i in A is active AND node j in B is active.

    Returns:
        Dict mapping weight key (e.g., "layer0_attn_c_attn") to boolean tensor
        indicating which weights are in the circuit (True = in circuit)
    """
    circuit_weight_masks = {}
    n_layers = masked_model.n_layers

    # Get active node indices for each location
    active_nodes = {}
    for key, mask in masked_model.masks.masks.items():
        active = (mask.tau >= 0)  # Boolean tensor
        active_nodes[key] = active

    # For each layer, process weight matrices
    for layer_idx in range(n_layers):
        # Attention weights
        # c_attn projects from attn_in to (Q, K, V)
        if "attn_in" in mask_locations:
            # Input: attn_in nodes
            # Output: attn_q, attn_k, attn_v nodes (concatenated)
            in_key = f"layer{layer_idx}_attn_in"

            if in_key in active_nodes:
                in_active = active_nodes[in_key]  # (d_model,)

                # c_attn weight shape: (d_model, 3 * n_heads * d_head)
                weight = model.blocks[layer_idx].attn.c_attn.weight  # (out, in) in PyTorch

                # For each output node, check if both input and output are active
                if "attn_q" in mask_locations or "attn_k" in mask_locations or "attn_v" in mask_locations:
                    # Get active masks for Q, K, V
                    q_key = f"layer{layer_idx}_attn_q"
                    k_key = f"layer{layer_idx}_attn_k"
                    v_key = f"layer{layer_idx}_attn_v"

                    d_qkv = masked_model.n_heads * masked_model.d_head

                    # Build combined output mask
                    out_active = torch.zeros(3 * d_qkv, dtype=torch.bool, device=in_active.device)
                    if q_key in active_nodes:
                        out_active[:d_qkv] = active_nodes[q_key]
                    if k_key in active_nodes:
                        out_active[d_qkv:2*d_qkv] = active_nodes[k_key]
                    if v_key in active_nodes:
                        out_active[2*d_qkv:] = active_nodes[v_key]

                    # Weight mask: out_active[i] AND in_active[j] for weight[i, j]
                    # Broadcasting: (out, 1) AND (1, in) -> (out, in)
                    weight_mask = out_active.unsqueeze(1) & in_active.unsqueeze(0)
                    # Also filter out zero weights
                    nonzero_mask = weight.data != 0
                    weight_mask = weight_mask & nonzero_mask
                    circuit_weight_masks[f"layer{layer_idx}_attn_c_attn"] = weight_mask

        # c_proj projects from attn_out to residual (d_model)
        # Note: there's no mask on the residual stream itself, so we consider
        # a weight in the circuit if the input (attn_out) node is active
        if "attn_out" in mask_locations:
            out_key = f"layer{layer_idx}_attn_out"
            if out_key in active_nodes:
                out_active = active_nodes[out_key]  # (d_model,)

                # c_proj weight shape: (d_model, d_model)
                # Input from attn_out, output to residual (no mask)
                # Weight is in circuit if input node is active
                in_active = out_active
                out_active_full = torch.ones(masked_model.d_model, dtype=torch.bool, device=in_active.device)

                weight = model.blocks[layer_idx].attn.c_proj.weight
                weight_mask = out_active_full.unsqueeze(1) & in_active.unsqueeze(0)
                # Also filter out zero weights
                nonzero_mask = weight.data != 0
                weight_mask = weight_mask & nonzero_mask
                circuit_weight_masks[f"layer{layer_idx}_attn_c_proj"] = weight_mask

        # MLP weights
        # c_fc projects from mlp_in to mlp_neuron
        if "mlp_in" in mask_locations and "mlp_neuron" in mask_locations:
            in_key = f"layer{layer_idx}_mlp_in"
            neuron_key = f"layer{layer_idx}_mlp_neuron"

            if in_key in active_nodes and neuron_key in active_nodes:
                in_active = active_nodes[in_key]  # (d_model,)
                neuron_active = active_nodes[neuron_key]  # (d_mlp,)

                # c_fc weight shape: (d_mlp, d_model)
                weight = model.blocks[layer_idx].mlp.c_fc.weight
                weight_mask = neuron_active.unsqueeze(1) & in_active.unsqueeze(0)
                # Also filter out zero weights
                nonzero_mask = weight.data != 0
                weight_mask = weight_mask & nonzero_mask
                circuit_weight_masks[f"layer{layer_idx}_mlp_c_fc"] = weight_mask

        # c_proj projects from mlp_neuron to mlp_out
        if "mlp_neuron" in mask_locations and "mlp_out" in mask_locations:
            neuron_key = f"layer{layer_idx}_mlp_neuron"
            out_key = f"layer{layer_idx}_mlp_out"

            if neuron_key in active_nodes and out_key in active_nodes:
                neuron_active = active_nodes[neuron_key]  # (d_mlp,)
                out_active = active_nodes[out_key]  # (d_model,)

                # c_proj weight shape: (d_model, d_mlp)
                weight = model.blocks[layer_idx].mlp.c_proj.weight
                weight_mask = out_active.unsqueeze(1) & neuron_active.unsqueeze(0)
                # Also filter out zero weights
                nonzero_mask = weight.data != 0
                weight_mask = weight_mask & nonzero_mask
                circuit_weight_masks[f"layer{layer_idx}_mlp_c_proj"] = weight_mask

    return circuit_weight_masks


def get_weight_statistics(
    model: nn.Module,
    circuit_weight_masks: Dict[str, torch.Tensor],
    n_layers: int,
) -> Tuple[float, float, int, int]:
    """
    Compute statistics about circuit weights.

    Returns:
        (mean, std, circuit_count, total_nonzero_count)
    """
    circuit_weights = []
    total_circuit_count = 0
    total_nonzero_count = 0

    for layer_idx in range(n_layers):
        for weight_name in ["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj"]:
            key = f"layer{layer_idx}_{weight_name}"
            if key not in circuit_weight_masks:
                continue

            # Get the weight matrix
            if "attn_c_attn" in weight_name:
                weight = model.blocks[layer_idx].attn.c_attn.weight
            elif "attn_c_proj" in weight_name:
                weight = model.blocks[layer_idx].attn.c_proj.weight
            elif "mlp_c_fc" in weight_name:
                weight = model.blocks[layer_idx].mlp.c_fc.weight
            elif "mlp_c_proj" in weight_name:
                weight = model.blocks[layer_idx].mlp.c_proj.weight

            circuit_mask = circuit_weight_masks[key]

            # Get circuit weights (absolute values)
            circuit_weight_values = weight[circuit_mask].abs()
            circuit_weights.append(circuit_weight_values)
            total_circuit_count += circuit_mask.sum().item()

            # Count total non-zero weights (for this we consider all weights as potentially non-zero)
            total_nonzero_count += weight.numel()

    if len(circuit_weights) > 0:
        all_circuit_weights = torch.cat(circuit_weights)
        mean = all_circuit_weights.mean().item()
        std = all_circuit_weights.std().item()
    else:
        mean = 0.0
        std = 0.0

    return mean, std, total_circuit_count, total_nonzero_count


def sample_random_weights(
    circuit_weight_masks: Dict[str, torch.Tensor],
    weight_shapes: Dict[str, Tuple[int, int]],
    num_to_sample: int,
    from_circuit: bool,
    size_matched: bool,
    circuit_mean: float,
    circuit_std: float,
    model: nn.Module,
    n_layers: int,
    rng: np.random.Generator,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Sample random weights to ablate.

    Args:
        circuit_weight_masks: Dict of circuit weight masks
        weight_shapes: Dict of weight tensor shapes
        num_to_sample: Number of weights to sample
        from_circuit: If True, sample from circuit weights; else from non-circuit
        size_matched: If True and from_circuit=False, only sample from weights
                     within [circuit_mean - circuit_std, circuit_mean + circuit_std]
        circuit_mean: Mean of circuit weight magnitudes
        circuit_std: Std of circuit weight magnitudes
        model: The model (to access weights for size matching)
        n_layers: Number of layers
        rng: Random number generator
        device: Device for tensors

    Returns:
        Dict mapping weight key to boolean tensor (True = ablate this weight)
    """
    # Build list of (key, valid_indices_array) - don't expand to full list!
    weight_pools = []  # List of (key, indices_array, count) tuples
    total_available = 0

    for layer_idx in range(n_layers):
        for weight_name in ["attn_c_attn", "attn_c_proj", "mlp_c_fc", "mlp_c_proj"]:
            key = f"layer{layer_idx}_{weight_name}"
            if key not in circuit_weight_masks:
                continue

            circuit_mask = circuit_weight_masks[key]

            # Start with circuit/non-circuit filter
            if from_circuit:
                valid_mask = circuit_mask  # Only circuit weights
            else:
                valid_mask = ~circuit_mask  # Only non-circuit weights

            # Apply size matching filter if needed
            if size_matched and not from_circuit:
                if "attn_c_attn" in weight_name:
                    weight = model.blocks[layer_idx].attn.c_attn.weight
                elif "attn_c_proj" in weight_name:
                    weight = model.blocks[layer_idx].attn.c_proj.weight
                elif "mlp_c_fc" in weight_name:
                    weight = model.blocks[layer_idx].mlp.c_fc.weight
                elif "mlp_c_proj" in weight_name:
                    weight = model.blocks[layer_idx].mlp.c_proj.weight

                weight_abs = weight.abs().cpu()
                lower_bound = circuit_mean - circuit_std
                upper_bound = circuit_mean + circuit_std

                size_mask = (weight_abs >= lower_bound) & (weight_abs <= upper_bound)
                valid_mask = valid_mask.cpu() & size_mask

            # Store indices array without expanding
            if valid_mask.any():
                valid_indices = valid_mask.flatten().nonzero(as_tuple=True)[0].cpu().numpy()
                count = len(valid_indices)
                weight_pools.append((key, valid_indices, count))
                total_available += count

    # Sample efficiently using two-step process
    num_to_sample = min(num_to_sample, total_available)
    if num_to_sample == 0:
        # Return empty masks
        ablation_masks = {}
        for key in circuit_weight_masks:
            ablation_masks[key] = torch.zeros(circuit_weight_masks[key].shape, dtype=torch.bool, device=device)
        return ablation_masks

    # Sample which pool each weight comes from (weighted by pool size)
    pool_sizes = np.array([count for _, _, count in weight_pools])
    pool_probs = pool_sizes / total_available
    sampled_pool_indices = rng.choice(len(weight_pools), size=num_to_sample, p=pool_probs, replace=True)

    # For each sampled pool, sample a random weight from it
    sampled_weights = []
    for pool_idx in sampled_pool_indices:
        key, valid_indices, count = weight_pools[pool_idx]
        flat_idx = rng.choice(valid_indices)
        sampled_weights.append((key, int(flat_idx)))

    # Create ablation masks (ensure they're on the right device)
    ablation_masks = {}
    for key in circuit_weight_masks:
        ablation_masks[key] = torch.zeros(circuit_weight_masks[key].shape, dtype=torch.bool, device=device)

    for key, flat_idx in sampled_weights:
        shape = ablation_masks[key].shape
        i = flat_idx // shape[1]
        j = flat_idx % shape[1]
        ablation_masks[key][i, j] = True

    return ablation_masks


def run_forward_with_weight_ablation(
    model: nn.Module,
    input_ids: torch.Tensor,
    weight_ablation_masks: Dict[str, torch.Tensor],
    n_layers: int,
) -> torch.Tensor:
    """
    Run forward pass with specified weights zeroed out.
    OPTIMIZED: Only modifies weights that actually need ablation.

    Args:
        model: The model
        input_ids: Input token IDs (batch, seq)
        weight_ablation_masks: Dict mapping weight keys to boolean tensors (True = ablate)
        n_layers: Number of layers

    Returns:
        Logits tensor
    """
    # If no weights to ablate, just run normal forward
    if not weight_ablation_masks or all(not mask.any() for mask in weight_ablation_masks.values()):
        output = model(input_ids)
        return output[0] if isinstance(output, tuple) else output

    device = input_ids.device

    # Collect modules that need modification
    modules_to_modify = []
    for weight_name, ablation_mask in weight_ablation_masks.items():
        if not ablation_mask.any():  # Skip if no weights to ablate
            continue

        # Parse layer_idx and weight location
        parts = weight_name.split('_')
        layer_idx = int(parts[0].replace('layer', ''))

        # Get the module
        block = model.blocks[layer_idx]
        if 'attn_c_attn' in weight_name:
            module = block.attn.c_attn
        elif 'attn_c_proj' in weight_name:
            module = block.attn.c_proj
        elif 'mlp_c_fc' in weight_name:
            module = block.mlp.c_fc
        elif 'mlp_c_proj' in weight_name:
            module = block.mlp.c_proj
        else:
            continue

        modules_to_modify.append((module, ablation_mask.to(device)))

    # Store original weights and apply ablation
    original_weights = []
    for module, ablation_mask in modules_to_modify:
        original_weights.append(module.weight.data)
        module.weight.data = module.weight.data * (~ablation_mask).float()

    # Run forward pass
    output = model(input_ids)
    logits = output[0] if isinstance(output, tuple) else output

    # Restore weights
    for i, (module, _) in enumerate(modules_to_modify):
        module.weight.data = original_weights[i]

    return logits


def compute_task_loss_from_logits(
    logits: torch.Tensor,
    correct_tokens: torch.Tensor,
    eval_positions: torch.Tensor,
    incorrect_tokens: torch.Tensor = None,
    use_binary_loss: bool = False,
) -> float:
    """
    Compute task cross-entropy loss from logits.
    
    Args:
        logits: (batch, seq, vocab) model output logits
        correct_tokens: (batch,) correct token ids
        eval_positions: (batch,) positions to evaluate
        incorrect_tokens: (batch,) incorrect token ids (needed for binary loss)
        use_binary_loss: If True, compute CE over only [correct, incorrect] logits
    
    Returns:
        Average loss (float)
    """
    batch_size = logits.shape[0]
    batch_indices = torch.arange(batch_size, device=logits.device)
    final_logits = logits[batch_indices, eval_positions, :]
    
    if use_binary_loss and incorrect_tokens is not None:
        # Binary CE: softmax over only [correct, incorrect] logits
        correct_logits = final_logits.gather(1, correct_tokens.unsqueeze(1)).squeeze(1)
        incorrect_logits = final_logits.gather(1, incorrect_tokens.unsqueeze(1)).squeeze(1)
        binary_logits = torch.stack([correct_logits, incorrect_logits], dim=1)
        targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(binary_logits, targets)
    else:
        # Full vocabulary CE
        loss = F.cross_entropy(final_logits, correct_tokens)
    
    return loss.item()


def run_weight_ablation_sweep(
    masked_model: MaskedSparseGPT,
    base_model: nn.Module,
    task: BinaryTask,
    config: WeightAblationConfig,
    pruning_config: PruningConfig,
    show_progress: bool = True,
) -> WeightAblationResult:
    """
    Run weight ablation sweep evaluation.

    Compares:
    1. Clean model (no ablation)
    2. Model with non-circuit weights ablated
    3. Sweep: ablating random weights from:
       - All weights
       - Circuit weights only
       - Size-matched non-circuit weights

    Args:
        masked_model: The pruned/masked model (defines the circuit)
        base_model: The base model to ablate
        task: Binary task for evaluation
        config: Weight ablation config
        pruning_config: Pruning config
        show_progress: Whether to show progress bars

    Returns:
        WeightAblationResult
    """
    device = config.device
    rng = np.random.default_rng(42)
    n_layers = masked_model.n_layers

    # Get circuit weight masks
    print("Identifying circuit weights...")
    circuit_weight_masks = get_circuit_weights(base_model, masked_model, pruning_config.mask_locations)

    # Get weight statistics
    circuit_mean, circuit_std, circuit_count, total_count = get_weight_statistics(
        base_model, circuit_weight_masks, n_layers
    )

    print(f"Circuit weights: {circuit_count:,} / {total_count:,}")
    print(f"Circuit weight magnitude: mean={circuit_mean:.6f}, std={circuit_std:.6f}")

    # Get weight shapes for sampling
    weight_shapes = {key: mask.shape for key, mask in circuit_weight_masks.items()}

    # Generate fixed batches
    print("Pre-generating evaluation batches...")
    fixed_batches = []
    for _ in range(config.num_batches):
        batch = task.generate_batch(
            batch_size=config.batch_size,
            max_length=config.seq_length,
        )
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
        fixed_batches.append({
            "positive_ids": positive_ids.to(device),
            "correct_tokens": correct_tokens.to(device),
            "eval_positions": eval_positions.to(device),
        })

    def compute_loss_with_weight_ablation(ablation_masks):
        """Compute average loss with given weight ablation masks."""
        losses = []
        with torch.no_grad():
            for batch in fixed_batches:
                input_ids = batch["positive_ids"]
                correct_tokens = batch["correct_tokens"]
                eval_positions = batch["eval_positions"]

                logits = run_forward_with_weight_ablation(
                    base_model, input_ids, ablation_masks, n_layers
                )

                loss = compute_task_loss_from_logits(logits, correct_tokens, eval_positions)
                losses.append(loss)

        return np.mean(losses)

    # 1. Clean model (no ablation)
    print("Computing clean loss (no ablation)...")
    empty_masks = {key: torch.zeros_like(mask, dtype=torch.bool, device=device)
                   for key, mask in circuit_weight_masks.items()}
    clean_loss = compute_loss_with_weight_ablation(empty_masks)
    print(f"  Clean loss: {clean_loss:.4f}")

    # 2. Ablate non-circuit weights
    print("Computing loss with non-circuit weights ablated...")
    non_circuit_masks = {key: ~mask for key, mask in circuit_weight_masks.items()}
    non_circuit_ablated_loss = compute_loss_with_weight_ablation(non_circuit_masks)
    num_non_circuit = sum(mask.sum().item() for mask in non_circuit_masks.values())
    print(f"  Non-circuit ablated loss: {non_circuit_ablated_loss:.4f} ({num_non_circuit:,} weights ablated)")

    # 3. Sweep over number of weights to ablate
    weight_counts_to_test = [int(circuit_count * i / (config.num_points - 1))
                             for i in range(config.num_points)]
    weight_counts_to_test[0] = 0
    weight_counts_to_test[-1] = circuit_count
    weight_counts_to_test = list(dict.fromkeys(weight_counts_to_test))

    random_all_results = {}
    random_circuit_results = {}
    random_size_matched_results = {}

    print(f"Running weight ablation sweep ({len(weight_counts_to_test)} points)...")

    for num_ablate in tqdm(weight_counts_to_test, desc="Weight ablation sweep", disable=not show_progress):
        # Random from all weights
        losses_all = []
        for trial in range(config.num_trials):
            ablation_masks = sample_random_weights(
                circuit_weight_masks, weight_shapes, num_ablate,
                from_circuit=False, size_matched=False,
                circuit_mean=circuit_mean, circuit_std=circuit_std,
                model=base_model, n_layers=n_layers, rng=rng, device=device
            )
            loss = compute_loss_with_weight_ablation(ablation_masks)
            losses_all.append(loss)
        random_all_results[num_ablate] = (np.mean(losses_all), np.std(losses_all))

        # Random from circuit weights
        losses_circuit = []
        for trial in range(config.num_trials):
            ablation_masks = sample_random_weights(
                circuit_weight_masks, weight_shapes, num_ablate,
                from_circuit=True, size_matched=False,
                circuit_mean=circuit_mean, circuit_std=circuit_std,
                model=base_model, n_layers=n_layers, rng=rng, device=device
            )
            loss = compute_loss_with_weight_ablation(ablation_masks)
            losses_circuit.append(loss)
        random_circuit_results[num_ablate] = (np.mean(losses_circuit), np.std(losses_circuit))

        # Random from size-matched non-circuit weights
        losses_size_matched = []
        for trial in range(config.num_trials):
            ablation_masks = sample_random_weights(
                circuit_weight_masks, weight_shapes, num_ablate,
                from_circuit=False, size_matched=True,
                circuit_mean=circuit_mean, circuit_std=circuit_std,
                model=base_model, n_layers=n_layers, rng=rng, device=device
            )
            loss = compute_loss_with_weight_ablation(ablation_masks)
            losses_size_matched.append(loss)
        random_size_matched_results[num_ablate] = (np.mean(losses_size_matched), np.std(losses_size_matched))

    return WeightAblationResult(
        clean_loss=clean_loss,
        non_circuit_weights_ablated_loss=non_circuit_ablated_loss,
        random_all_results=random_all_results,
        random_circuit_results=random_circuit_results,
        random_size_matched_results=random_size_matched_results,
        circuit_weight_count=circuit_count,
        total_weight_count=total_count,
        num_trials=config.num_trials,
        circuit_weight_mean=circuit_mean,
        circuit_weight_std=circuit_std,
    )


def plot_weight_ablation_results(
    result: WeightAblationResult,
    output_path: Path,
    title_prefix: str = "",
):
    """
    Create plot for weight ablation results.

    X-axis: Number of weights ablated
    Y-axis: Task loss
    Three lines: random from all, random from circuit, size-matched non-circuit
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Random from all weights
    xs_all = sorted(result.random_all_results.keys())
    means_all = [result.random_all_results[x][0] for x in xs_all]
    stds_all = [result.random_all_results[x][1] for x in xs_all]

    ax.plot(xs_all, means_all, 'b-o', linewidth=2, markersize=6, label='Random from all weights')
    ax.fill_between(xs_all,
                    [m - s for m, s in zip(means_all, stds_all)],
                    [m + s for m, s in zip(means_all, stds_all)],
                    alpha=0.2, color='blue')

    # Random from circuit weights
    xs_circuit = sorted(result.random_circuit_results.keys())
    means_circuit = [result.random_circuit_results[x][0] for x in xs_circuit]
    stds_circuit = [result.random_circuit_results[x][1] for x in xs_circuit]

    ax.plot(xs_circuit, means_circuit, 'r-s', linewidth=2, markersize=6, label='Random from circuit weights')
    ax.fill_between(xs_circuit,
                    [m - s for m, s in zip(means_circuit, stds_circuit)],
                    [m + s for m, s in zip(means_circuit, stds_circuit)],
                    alpha=0.2, color='red')

    # Random from size-matched non-circuit weights
    xs_size = sorted(result.random_size_matched_results.keys())
    means_size = [result.random_size_matched_results[x][0] for x in xs_size]
    stds_size = [result.random_size_matched_results[x][1] for x in xs_size]

    ax.plot(xs_size, means_size, 'g-^', linewidth=2, markersize=6,
            label=f'Size-matched non-circuit (mean±std: {result.circuit_weight_mean:.4f}±{result.circuit_weight_std:.4f})')
    ax.fill_between(xs_size,
                    [m - s for m, s in zip(means_size, stds_size)],
                    [m + s for m, s in zip(means_size, stds_size)],
                    alpha=0.2, color='green')

    # Add horizontal lines for baselines
    ax.axhline(y=result.clean_loss, color='purple', linestyle='--', linewidth=2,
               label=f'Clean (no ablation): {result.clean_loss:.4f}')
    ax.axhline(y=result.non_circuit_weights_ablated_loss, color='orange', linestyle='--', linewidth=2,
               label=f'Non-circuit weights ablated: {result.non_circuit_weights_ablated_loss:.4f}')

    # Add vertical line at circuit weight count
    ax.axvline(x=result.circuit_weight_count, color='gray', linestyle=':', linewidth=1.5,
               label=f'Circuit weight count: {result.circuit_weight_count:,}')

    ax.set_xlabel("Number of Weights Ablated", fontsize=12)
    ax.set_ylabel("Task Loss", fontsize=12)
    ax.set_title(f"{title_prefix}Weight Ablation Sweep\n"
                 f"(Circuit: {result.circuit_weight_count:,} / {result.total_weight_count:,} weights)",
                 fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_weight_ablation_results(
    result: WeightAblationResult,
    output_path: Path,
):
    """Save weight ablation results to JSON."""
    data = {
        "baselines": {
            "clean_loss": result.clean_loss,
            "non_circuit_weights_ablated_loss": result.non_circuit_weights_ablated_loss,
        },
        "random_all_results": {
            str(k): {"mean": v[0], "std": v[1]}
            for k, v in result.random_all_results.items()
        },
        "random_circuit_results": {
            str(k): {"mean": v[0], "std": v[1]}
            for k, v in result.random_circuit_results.items()
        },
        "random_size_matched_results": {
            str(k): {"mean": v[0], "std": v[1]}
            for k, v in result.random_size_matched_results.items()
        },
        "metadata": {
            "circuit_weight_count": result.circuit_weight_count,
            "total_weight_count": result.total_weight_count,
            "num_trials": result.num_trials,
            "circuit_weight_mean": result.circuit_weight_mean,
            "circuit_weight_std": result.circuit_weight_std,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def run_and_save_weight_ablation_sweep(
    masked_model: MaskedSparseGPT,
    base_model: nn.Module,
    task: BinaryTask,
    output_dir: Path,
    config: Optional[WeightAblationConfig] = None,
    pruning_config: Optional[PruningConfig] = None,
    title_prefix: str = "",
    show_progress: bool = True,
) -> WeightAblationResult:
    """
    Convenience function to run weight ablation sweep and save all outputs.

    Args:
        masked_model: The pruned/masked model (defines the circuit)
        base_model: The base model to ablate
        task: Binary task for evaluation
        output_dir: Directory to save outputs
        config: Weight ablation config (uses defaults if None)
        pruning_config: Pruning config (uses masked_model.config if None)
        title_prefix: Prefix for plot titles
        show_progress: Whether to show progress bars

    Returns:
        WeightAblationResult
    """
    if config is None:
        config = WeightAblationConfig()
    if pruning_config is None:
        pruning_config = masked_model.config

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Running Weight Ablation Sweep Evaluation")
    print(f"{'='*60}")
    print(f"Points: {config.num_points}")
    print(f"Trials per point: {config.num_trials}")
    print(f"Batches: {config.num_batches}")
    print(f"Output dir: {output_dir}")
    print()

    result = run_weight_ablation_sweep(
        masked_model=masked_model,
        base_model=base_model,
        task=task,
        config=config,
        pruning_config=pruning_config,
        show_progress=show_progress,
    )

    # Save plot
    plot_path = output_dir / "weight_ablation_sweep.png"
    plot_weight_ablation_results(result, plot_path, title_prefix=title_prefix)
    print(f"Plot saved to {plot_path}")

    # Save JSON
    json_path = output_dir / "weight_ablation_sweep_results.json"
    save_weight_ablation_results(result, json_path)
    print(f"Results saved to {json_path}")

    # Print summary
    print(f"\nWeight Ablation Sweep Summary:")
    print(f"  Circuit weight count: {result.circuit_weight_count:,} / {result.total_weight_count:,}")
    print(f"  Circuit weight magnitude: mean={result.circuit_weight_mean:.6f}, std={result.circuit_weight_std:.6f}")
    print(f"  Clean loss: {result.clean_loss:.4f}")
    print(f"  Non-circuit weights ablated loss: {result.non_circuit_weights_ablated_loss:.4f}")

    return result
