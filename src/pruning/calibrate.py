"""
Logit scale+shift calibration for discretized circuits.

After mask discretization, fit a scale and shift transformation on the
final logits to compensate for calibration drift.

Based on Appendix A.5 "Mask discretization" of Gao et al. (2025):
"Fit a scale+shift transformation on final logits using 16 steps of LBFGS"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from .masked_model import MaskedSparseGPT
from .tasks import BinaryTask
from .config import PruningConfig


class LogitCalibrator(nn.Module):
    """
    Learnable scale and shift for logit calibration.
    
    Applies: calibrated_logits = logits * scale + shift
    """
    
    def __init__(self, vocab_size: int, device: str = "cuda"):
        super().__init__()
        
        # Initialize scale to 1, shift to 0 (identity transform)
        self.scale = nn.Parameter(torch.ones(1, device=device))
        self.shift = nn.Parameter(torch.zeros(1, device=device))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply scale and shift to logits."""
        return logits * self.scale + self.shift


def calibrate_logits(
    masked_model: MaskedSparseGPT,
    task: BinaryTask,
    config: PruningConfig,
    num_steps: Optional[int] = None,
    num_batches_per_step: int = 4,
    show_progress: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Calibrate logits using LBFGS optimization.
    
    Fits a global scale and shift transformation on the final logits
    to minimize task loss after discretization.
    
    Args:
        masked_model: The masked model with discretized masks
        task: Binary task for evaluation
        config: Pruning configuration
        num_steps: Number of LBFGS steps (defaults to config.calibration_steps)
        num_batches_per_step: Batches to use per optimization step
        show_progress: Whether to show progress
        
    Returns:
        Tuple of:
        - scale: Fitted scale parameter
        - shift: Fitted shift parameter
        - metrics: Dictionary of calibration metrics
    """
    if num_steps is None:
        num_steps = config.calibration_steps
    
    device = config.device
    vocab_size = masked_model.model.config.vocab_size
    
    # Create calibrator
    calibrator = LogitCalibrator(vocab_size, device)
    
    # LBFGS optimizer
    optimizer = LBFGS(
        calibrator.parameters(),
        lr=1.0,
        max_iter=1,  # We control outer iterations
        line_search_fn="strong_wolfe",
    )
    
    # Store batches for consistent evaluation
    batches = []
    for _ in range(num_batches_per_step * 2):  # Extra batches for eval
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = task.generate_batch(
            batch_size=config.batch_size,
            max_length=config.seq_length,
        )
        batches.append((
            positive_ids.to(device),
            negative_ids.to(device),
            correct_tokens.to(device),
            incorrect_tokens.to(device),
            eval_positions.to(device),
        ))
    
    # Helper to get logits at correct positions
    def get_eval_logits(logits: torch.Tensor, eval_pos: torch.Tensor) -> torch.Tensor:
        """Index into logits at the correct position for each batch element."""
        batch_size = logits.shape[0]
        batch_indices = torch.arange(batch_size, device=logits.device)
        return logits[batch_indices, eval_pos, :]  # (batch, vocab)
    
    # Evaluate before calibration
    masked_model.eval()
    with torch.no_grad():
        pre_loss = 0.0
        for batch in batches[:num_batches_per_step]:
            positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
            logits = masked_model.forward(positive_ids)
            final_logits = get_eval_logits(logits, eval_positions)
            loss = F.cross_entropy(final_logits, correct_tokens)
            pre_loss += loss.item()
        pre_loss /= num_batches_per_step
    
    if show_progress:
        print(f"Calibrating logits with LBFGS ({num_steps} steps)...")
        print(f"  Pre-calibration loss: {pre_loss:.4f}")
    
    # LBFGS optimization
    calibrator.train()
    
    def closure():
        optimizer.zero_grad()
        
        total_loss = 0.0
        for batch in batches[:num_batches_per_step]:
            positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
            
            # Get logits from masked model
            with torch.no_grad():
                logits = masked_model.forward(positive_ids)
            
            # Apply calibration at correct positions
            final_logits = get_eval_logits(logits, eval_positions)
            calibrated_logits = calibrator(final_logits)
            
            # Compute loss
            loss = F.cross_entropy(calibrated_logits, correct_tokens)
            total_loss = total_loss + loss
        
        total_loss = total_loss / num_batches_per_step
        total_loss.backward()
        
        return total_loss
    
    # Run LBFGS steps
    for step in range(num_steps):
        optimizer.step(closure)
        
        if show_progress and (step + 1) % 4 == 0:
            with torch.no_grad():
                current_loss = 0.0
                for batch in batches[:num_batches_per_step]:
                    positive_ids, _, correct_tokens, _, eval_positions = batch
                    logits = masked_model.forward(positive_ids)
                    final_logits = get_eval_logits(logits, eval_positions)
                    calibrated = calibrator(final_logits)
                    loss = F.cross_entropy(calibrated, correct_tokens)
                    current_loss += loss.item()
                current_loss /= num_batches_per_step
                print(f"  Step {step+1}: loss={current_loss:.4f}, scale={calibrator.scale.item():.4f}, shift={calibrator.shift.item():.4f}")
    
    # Evaluate after calibration
    calibrator.eval()
    with torch.no_grad():
        post_loss = 0.0
        post_acc = 0.0
        
        for batch in batches[num_batches_per_step:]:  # Use held-out batches
            positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
            logits = masked_model.forward(positive_ids)
            final_logits = get_eval_logits(logits, eval_positions)
            calibrated = calibrator(final_logits)
            
            loss = F.cross_entropy(calibrated, correct_tokens)
            post_loss += loss.item()
            
            preds = calibrated.argmax(dim=-1)
            post_acc += (preds == correct_tokens).float().mean().item()
        
        post_loss /= num_batches_per_step
        post_acc /= num_batches_per_step
    
    scale = calibrator.scale.item()
    shift = calibrator.shift.item()
    
    if show_progress:
        print(f"\nCalibration complete:")
        print(f"  Post-calibration loss: {post_loss:.4f}")
        print(f"  Post-calibration accuracy: {post_acc:.2%}")
        print(f"  Scale: {scale:.4f}")
        print(f"  Shift: {shift:.4f}")
    
    metrics = {
        "pre_calibration_loss": pre_loss,
        "post_calibration_loss": post_loss,
        "post_calibration_accuracy": post_acc,
        "scale": scale,
        "shift": shift,
    }
    
    return scale, shift, metrics


def apply_calibration_to_model(
    masked_model: MaskedSparseGPT,
    scale: float,
    shift: float,
):
    """
    Apply calibration by modifying the model's final layer.
    
    This is an optional step that bakes the calibration into the model
    so it doesn't need to be applied separately.
    
    Note: This modifies the model in-place.
    """
    # Modify the lm_head weights and bias
    # New logits = old_logits * scale + shift
    # = (x @ W + b) * scale + shift
    # = x @ (W * scale) + (b * scale + shift)
    
    with torch.no_grad():
        masked_model.model.lm_head.weight.mul_(scale)
        
        if masked_model.model.lm_head.bias is not None:
            masked_model.model.lm_head.bias.mul_(scale)
            masked_model.model.lm_head.bias.add_(shift)
        else:
            # Create bias if it doesn't exist
            vocab_size = masked_model.model.lm_head.weight.shape[0]
            device = masked_model.model.lm_head.weight.device
            masked_model.model.lm_head.bias = nn.Parameter(
                torch.full((vocab_size,), shift, device=device)
            )

