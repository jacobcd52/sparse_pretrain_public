"""
Mask discretization via bisection.

After continuous mask training, find the number of active nodes k that
exactly achieves the target loss.

Based on Appendix A.5 "Mask discretization" of Gao et al. (2025).
"""

import torch
from typing import Dict, Tuple, Optional, Callable
from tqdm import tqdm
import copy

from .masked_model import MaskedSparseGPT
from .tasks import BinaryTask
from .config import PruningConfig


def evaluate_at_k(
    masked_model: MaskedSparseGPT,
    task: BinaryTask,
    k: int,
    config: PruningConfig,
    num_batches: int = 20,
) -> float:
    """
    Evaluate task loss with exactly k active nodes.
    
    Args:
        masked_model: The masked model
        task: Binary task
        k: Number of nodes to keep
        config: Pruning config
        num_batches: Number of batches for evaluation
        
    Returns:
        Average task loss
    """
    device = config.device
    
    # Save original mask state
    original_state = masked_model.get_mask_state()
    
    # Keep top k nodes by tau value
    masked_model.masks.keep_top_k(k)
    
    # Evaluate
    masked_model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_batches):
            positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = task.generate_batch(
                batch_size=config.batch_size,
                max_length=config.seq_length,
            )
            
            positive_ids = positive_ids.to(device)
            negative_ids = negative_ids.to(device)
            correct_tokens = correct_tokens.to(device)
            incorrect_tokens = incorrect_tokens.to(device)
            eval_positions = eval_positions.to(device)
            
            _, metrics = masked_model.compute_task_loss(
                positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions
            )
            total_loss += metrics["task_loss"]
    
    avg_loss = total_loss / num_batches
    
    # Restore original state
    masked_model.load_mask_state(original_state)
    
    return avg_loss


def evaluate_at_k_fixed_batches(
    masked_model: MaskedSparseGPT,
    fixed_batches: list,
    k: int,
    device: str = "cuda",
) -> float:
    """
    Evaluate task loss with exactly k active nodes using pre-generated fixed batches.
    
    This ensures consistent evaluation across different k values by using the same
    data for all evaluations, making the resulting Pareto curve monotonic.
    
    Args:
        masked_model: The masked model
        fixed_batches: List of pre-generated batch tuples 
                       (positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions)
        k: Number of nodes to keep
        device: Device to evaluate on
        
    Returns:
        Average task loss
    """
    # Save original mask state
    original_state = masked_model.get_mask_state()
    
    # Keep top k nodes by tau value
    masked_model.masks.keep_top_k(k)
    
    # Evaluate
    masked_model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in fixed_batches:
            # Handle both tuple and dict formats
            if isinstance(batch, dict):
                positive_ids = batch["positive_ids"].to(device)
                negative_ids = batch["negative_ids"].to(device)
                correct_tokens = batch["correct_tokens"].to(device)
                incorrect_tokens = batch["incorrect_tokens"].to(device)
                eval_positions = batch["eval_positions"].to(device)
            else:
                positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
                positive_ids = positive_ids.to(device)
                negative_ids = negative_ids.to(device)
                correct_tokens = correct_tokens.to(device)
                incorrect_tokens = incorrect_tokens.to(device)
                eval_positions = eval_positions.to(device)
            
            _, metrics = masked_model.compute_task_loss(
                positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions
            )
            total_loss += metrics["task_loss"]
    
    avg_loss = total_loss / len(fixed_batches)
    
    # Restore original state
    masked_model.load_mask_state(original_state)
    
    return avg_loss


def discretize_masks(
    masked_model: MaskedSparseGPT,
    task: BinaryTask,
    config: PruningConfig,
    target_loss: Optional[float] = None,
    num_eval_batches: int = 20,
    show_progress: bool = True,
) -> Tuple[int, float, Dict[str, torch.Tensor]]:
    """
    Discretize masks via bisection to achieve target loss.
    
    Binary search over k (number of active nodes) to find the smallest
    circuit that achieves the target task loss.
    
    Args:
        masked_model: The masked model with trained (continuous) masks
        task: Binary task for evaluation
        config: Pruning configuration
        target_loss: Target task loss (defaults to config.target_loss)
        num_eval_batches: Batches per evaluation
        show_progress: Whether to show progress
        
    Returns:
        Tuple of:
        - k: Number of active nodes
        - loss: Achieved task loss
        - binary_masks: Dictionary of binary mask tensors
    """
    if target_loss is None:
        target_loss = config.target_loss
    
    # Get all tau values and count ACTIVE nodes (tau >= 0)
    # These are the nodes the model learned to use during training
    all_taus = []
    active_taus = []
    tau_info = []
    
    for key, mask in masked_model.masks.masks.items():
        for i in range(mask.num_nodes):
            tau_val = mask.tau[i].item()
            all_taus.append(tau_val)
            tau_info.append((key, i))
            if tau_val >= 0:
                active_taus.append(tau_val)
    
    total_nodes = len(all_taus)
    num_active = len(active_taus)  # Number of nodes active after training
    
    # Binary search bounds (only over ACTIVE nodes)
    k_low = 1
    k_high = num_active  # Only search over nodes that were active in training
    
    # First, check if we can achieve target loss with all active nodes
    # Note: keep_top_k(num_active) keeps all active nodes unchanged
    loss_at_all = evaluate_at_k(masked_model, task, num_active, config, num_eval_batches)
    if loss_at_all > target_loss:
        print(f"Warning: Cannot achieve target loss {target_loss:.4f} even with all {num_active} active nodes (loss={loss_at_all:.4f})")
        # Return all active nodes
        masked_model.masks.keep_top_k(num_active)
        return num_active, loss_at_all, masked_model.get_circuit_mask()
    
    # Check minimum
    loss_at_min = evaluate_at_k(masked_model, task, 1, config, num_eval_batches)
    if loss_at_min <= target_loss:
        print(f"Target achieved with just 1 node! (loss={loss_at_min:.4f})")
        masked_model.masks.keep_top_k(1)
        return 1, loss_at_min, masked_model.get_circuit_mask()
    
    # Binary search
    best_k = num_active
    best_loss = loss_at_all
    
    if show_progress:
        print(f"Bisecting to find k that achieves target loss {target_loss:.4f}...")
        print(f"  Total nodes in model: {total_nodes}")
        print(f"  Active nodes after training: {num_active}")
        print(f"  Loss at k=1: {loss_at_min:.4f}")
        print(f"  Loss at k={num_active} (all active): {loss_at_all:.4f}")
    
    iteration = 0
    max_iters = config.discretization_max_iters
    tolerance = config.discretization_tolerance
    
    while k_high - k_low > 1 and iteration < max_iters:
        k_mid = (k_low + k_high) // 2
        
        loss_mid = evaluate_at_k(masked_model, task, k_mid, config, num_eval_batches)
        
        if show_progress:
            print(f"  Iter {iteration}: k={k_mid}, loss={loss_mid:.4f}")
        
        if loss_mid <= target_loss + tolerance:
            # Can achieve target with k_mid nodes, try fewer
            k_high = k_mid
            best_k = k_mid
            best_loss = loss_mid
        else:
            # Need more nodes
            k_low = k_mid
        
        iteration += 1
    
    # Final check at k_high (the smallest k that works)
    final_loss = evaluate_at_k(masked_model, task, k_high, config, num_eval_batches)
    
    if final_loss <= target_loss + tolerance:
        best_k = k_high
        best_loss = final_loss
    
    # Apply the final k
    masked_model.masks.keep_top_k(best_k)
    
    if show_progress:
        print(f"\nDiscretization complete:")
        print(f"  Final k: {best_k} / {num_active} active ({100*best_k/num_active:.1f}% of active)")
        print(f"  Circuit size: {best_k} / {total_nodes} total ({100*best_k/total_nodes:.1f}% of model)")
        print(f"  Final loss: {best_loss:.4f} (target: {target_loss:.4f})")
    
    return best_k, best_loss, masked_model.get_circuit_mask()


def discretize_to_edge_count(
    masked_model: MaskedSparseGPT,
    task: BinaryTask,
    config: PruningConfig,
    target_edges: int,
    num_eval_batches: int = 20,
    show_progress: bool = True,
) -> Tuple[int, float, Dict[str, torch.Tensor]]:
    """
    Discretize masks to achieve a target number of edges.
    
    Alternative to loss-based discretization: find the smallest circuit
    with at most target_edges edges.
    
    Note: Edge counting requires access to weight sparsity patterns,
    which is more complex. For now, we use node count as a proxy.
    
    Args:
        masked_model: The masked model
        task: Binary task
        config: Pruning config
        target_edges: Target number of edges (approximated by nodes)
        num_eval_batches: Batches per evaluation
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (k, loss, binary_masks)
    """
    # For now, use target_edges as target number of nodes
    # A proper implementation would count actual edges through weight matrices
    
    masked_model.masks.keep_top_k(target_edges)
    
    # Evaluate
    device = config.device
    masked_model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_eval_batches):
            positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = task.generate_batch(
                batch_size=config.batch_size,
                max_length=config.seq_length,
            )
            
            positive_ids = positive_ids.to(device)
            negative_ids = negative_ids.to(device)
            correct_tokens = correct_tokens.to(device)
            incorrect_tokens = incorrect_tokens.to(device)
            eval_positions = eval_positions.to(device)
            
            _, metrics = masked_model.compute_task_loss(
                positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions
            )
            total_loss += metrics["task_loss"]
    
    avg_loss = total_loss / num_eval_batches
    
    if show_progress:
        print(f"Discretized to {target_edges} nodes, loss: {avg_loss:.4f}")
    
    return target_edges, avg_loss, masked_model.get_circuit_mask()

