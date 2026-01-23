"""
Pruning trainer for circuit discovery.

Implements the mask optimization loop with:
- AdamW optimizer with gradient clipping
- Linear LR decay
- Task loss + sparsity penalty

Based on Appendix A.5 of Gao et al. (2025).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import Dict, Optional, Callable, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from .config import PruningConfig
from .masked_model import MaskedSparseGPT
from .tasks import BinaryTask

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingState:
    """Current training state."""
    step: int = 0
    best_loss: float = float('inf')
    best_step: int = 0


class PruningTrainer:
    """
    Trainer for mask optimization in circuit pruning.
    
    Implements the pruning procedure from Appendix A.5:
    1. Initialize masks with Gaussian noise
    2. Optimize mask parameters with AdamW
    3. Use task_loss + k_coef * num_active_nodes as objective
    4. Clamp parameters to [-1, 1] after each step
    5. Linear LR decay
    """
    
    def __init__(
        self,
        masked_model: MaskedSparseGPT,
        task: BinaryTask,
        config: PruningConfig,
        val_task: Optional[BinaryTask] = None,
        use_wandb: bool = False,
        wandb_project: str = "circuit-pruning",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ):
        self.masked_model = masked_model
        self.task = task
        self.val_task = val_task  # Optional validation task with different templates
        self.config = config
        self.device = config.device
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Move model to device
        self.masked_model.to(self.device)
        
        # Setup optimizer (only for mask parameters)
        # Include both node masks and token mask (if enabled)
        mask_params = list(self.masked_model.masks.parameters())
        if self.masked_model.token_mask is not None:
            mask_params.extend(list(self.masked_model.token_mask.parameters()))
        
        self.optimizer = AdamW(
            mask_params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        
        # Training state
        self.state = TrainingState()
        
        # Logging
        self.history = []
        
        # Pareto history for both train and val
        self._pareto_history = []
        self._pareto_history_val = []
        
        # Initialize wandb if requested
        if self.use_wandb:
            self._init_wandb(wandb_project, wandb_run_name, wandb_config)
    
    def _init_wandb(
        self,
        project: str,
        run_name: Optional[str],
        extra_config: Optional[Dict],
    ):
        """Initialize wandb logging."""
        # Build comprehensive config
        config_dict = {
            # Hyperparameters
            "hparams/k_coef": self.config.k_coef,
            "hparams/init_noise_scale": self.config.init_noise_scale,
            "hparams/init_noise_bias": self.config.init_noise_bias,
            "hparams/heaviside_temp": self.config.heaviside_temp,
            "hparams/lr": self.config.lr,
            "hparams/weight_decay": self.config.weight_decay,
            "hparams/beta1": self.config.beta1,
            "hparams/beta2": self.config.beta2,
            "hparams/grad_clip_norm": self.config.grad_clip_norm,
            "hparams/lr_warmup_frac": self.config.lr_warmup_frac,
            
            # Training settings
            "training/num_steps": self.config.num_steps,
            "training/batch_size": self.config.batch_size,
            "training/seq_length": self.config.seq_length,
            
            # Model architecture info
            "model/n_layers": self.masked_model.n_layers,
            "model/d_model": self.masked_model.model.config.d_model,
            "model/total_nodes": self.masked_model.masks.get_total_nodes(),
            "model/mask_locations": self.config.mask_locations,
            
            # Task info
            "task/name": self.task.__class__.__name__,
        }
        
        if extra_config:
            config_dict.update(extra_config)
        
        wandb.init(
            project=project,
            name=run_name,
            config=config_dict,
            reinit=True,
        )
        
        # Log initial tau histogram
        self._log_tau_histogram(prefix="initial")
    
    def _get_lr_multiplier(self, step: int) -> float:
        """Get learning rate multiplier for current step."""
        total_steps = self.config.num_steps
        warmup_steps = int(total_steps * self.config.lr_warmup_frac)
        
        if step < warmup_steps:
            # Linear warmup
            return (step + 1) / warmup_steps
        else:
            # Linear decay from 1 to 0
            decay_steps = total_steps - warmup_steps
            progress = (step - warmup_steps) / decay_steps
            return 1.0 - progress
    
    def _log_tau_histogram(self, prefix: str = ""):
        """Log tau value histogram to wandb."""
        if not self.use_wandb:
            return
        
        all_taus = self.masked_model.masks.get_all_tau_values().detach().cpu().numpy()
        
        # Create histogram with fixed bins for consistency
        hist_name = f"{prefix}/tau_histogram" if prefix else "tau/histogram"
        wandb.log({
            hist_name: wandb.Histogram(all_taus, num_bins=50),
        }, commit=False)
    
    def _compute_tau_statistics(self) -> Dict[str, float]:
        """Compute detailed tau statistics."""
        all_taus = self.masked_model.masks.get_all_tau_values().detach().cpu().numpy()
        
        stats = {
            # Global tau statistics
            "tau/mean": float(np.mean(all_taus)),
            "tau/std": float(np.std(all_taus)),
            "tau/min": float(np.min(all_taus)),
            "tau/max": float(np.max(all_taus)),
            "tau/median": float(np.median(all_taus)),
            
            # Percentiles
            "tau/p10": float(np.percentile(all_taus, 10)),
            "tau/p25": float(np.percentile(all_taus, 25)),
            "tau/p75": float(np.percentile(all_taus, 75)),
            "tau/p90": float(np.percentile(all_taus, 90)),
            
            # Fraction in different regions
            "tau/frac_negative": float(np.mean(all_taus < 0)),
            "tau/frac_positive": float(np.mean(all_taus >= 0)),
            "tau/frac_near_minus1": float(np.mean(all_taus < -0.9)),
            "tau/frac_near_plus1": float(np.mean(all_taus > 0.9)),
            "tau/frac_in_middle": float(np.mean((all_taus > -0.5) & (all_taus < 0.5))),
        }
        
        return stats
    
    def _compute_per_location_stats(self) -> Dict[str, float]:
        """Compute statistics broken down by location type."""
        stats = {}
        
        # Group by location type
        location_taus = {}
        location_active = {}
        
        for key, mask in self.masked_model.masks.masks.items():
            # Extract location type (e.g., "layer0_attn_in" -> "attn_in")
            loc_type = key.split("_", 1)[1]
            
            if loc_type not in location_taus:
                location_taus[loc_type] = []
                location_active[loc_type] = 0
            
            taus = mask.tau.detach().cpu().numpy()
            location_taus[loc_type].extend(taus.tolist())
            location_active[loc_type] += mask.get_num_active()
        
        # Compute stats per location type
        for loc_type, taus in location_taus.items():
            taus = np.array(taus)
            stats[f"location/{loc_type}/mean_tau"] = float(np.mean(taus))
            stats[f"location/{loc_type}/std_tau"] = float(np.std(taus))
            stats[f"location/{loc_type}/active_nodes"] = location_active[loc_type]
            stats[f"location/{loc_type}/total_nodes"] = len(taus)
            stats[f"location/{loc_type}/frac_active"] = location_active[loc_type] / len(taus)
        
        return stats
    
    def _compute_per_layer_stats(self) -> Dict[str, float]:
        """Compute statistics broken down by layer."""
        stats = {}
        
        layer_active = {}
        layer_total = {}
        
        for key, mask in self.masked_model.masks.masks.items():
            # Extract layer number (e.g., "layer0_attn_in" -> 0)
            layer_num = int(key.split("_")[0].replace("layer", ""))
            
            if layer_num not in layer_active:
                layer_active[layer_num] = 0
                layer_total[layer_num] = 0
            
            layer_active[layer_num] += mask.get_num_active()
            layer_total[layer_num] += mask.num_nodes
        
        for layer_num in sorted(layer_active.keys()):
            stats[f"layer/{layer_num}/active_nodes"] = layer_active[layer_num]
            stats[f"layer/{layer_num}/frac_active"] = layer_active[layer_num] / layer_total[layer_num]
        
        return stats
    
    def _compute_gradient_stats(self) -> Dict[str, float]:
        """Compute gradient statistics."""
        stats = {}
        
        # Collect all mask parameters (node masks + token mask if enabled)
        all_mask_params = list(self.masked_model.masks.parameters())
        if self.masked_model.token_mask is not None:
            all_mask_params.extend(list(self.masked_model.token_mask.parameters()))
        
        all_grads = []
        for param in all_mask_params:
            if param.grad is not None:
                all_grads.append(param.grad.detach().cpu().numpy().flatten())
        
        if all_grads:
            all_grads = np.concatenate(all_grads)
            stats["grad/mean"] = float(np.mean(all_grads))
            stats["grad/std"] = float(np.std(all_grads))
            stats["grad/min"] = float(np.min(all_grads))
            stats["grad/max"] = float(np.max(all_grads))
            stats["grad/abs_mean"] = float(np.mean(np.abs(all_grads)))
            stats["grad/rms"] = float(np.sqrt(np.mean(all_grads ** 2)))
            
            # Gradient sparsity (near-zero gradients)
            stats["grad/frac_near_zero"] = float(np.mean(np.abs(all_grads) < 1e-6))
        
        return stats
    
    def _compute_threshold_analysis(self) -> Dict[str, float]:
        """Analyze what circuit sizes different thresholds would give."""
        all_taus = self.masked_model.masks.get_all_tau_values().detach().cpu().numpy()
        
        stats = {}
        
        # Circuit size at various thresholds
        thresholds = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]
        for thresh in thresholds:
            circuit_size = int(np.sum(all_taus >= thresh))
            thresh_name = f"p{int(thresh*100)}" if thresh >= 0 else f"m{int(-thresh*100)}"
            stats[f"threshold/{thresh_name}/circuit_size"] = circuit_size
        
        # What threshold gives specific circuit sizes
        sorted_taus = np.sort(all_taus)[::-1]
        target_sizes = [100, 500, 1000, 2000, 5000, 10000]
        for size in target_sizes:
            if size <= len(sorted_taus):
                stats[f"circuit_size/{size}/threshold"] = float(sorted_taus[size - 1])
        
        return stats
    
    def _compute_bimodality_metrics(self) -> Dict[str, float]:
        """
        Compute metrics about the bimodality of the tau distribution.
        
        A perfectly trained mask should be bimodal: most taus near -1 (inactive)
        and a small number near +1 (active), with few in between.
        """
        all_taus = self.masked_model.masks.get_all_tau_values().detach().cpu().numpy()
        
        stats = {}
        
        # Split into "active" (tau > 0) and "inactive" (tau <= 0)
        active_taus = all_taus[all_taus > 0]
        inactive_taus = all_taus[all_taus <= 0]
        
        if len(active_taus) > 0 and len(inactive_taus) > 0:
            # Gap between clusters (higher = more separated)
            active_min = np.min(active_taus)
            inactive_max = np.max(inactive_taus)
            stats["bimodal/gap"] = float(active_min - inactive_max)
            
            # Mean of each cluster
            stats["bimodal/active_mean"] = float(np.mean(active_taus))
            stats["bimodal/inactive_mean"] = float(np.mean(inactive_taus))
            
            # Separation score: (mean_active - mean_inactive) / pooled_std
            pooled_std = np.std(all_taus)
            if pooled_std > 1e-8:
                separation = (np.mean(active_taus) - np.mean(inactive_taus)) / pooled_std
                stats["bimodal/separation_score"] = float(separation)
        
        # "Middle zone" nodes - these are uncertain
        middle_zone = np.sum((all_taus > -0.5) & (all_taus < 0.5))
        stats["bimodal/middle_zone_count"] = int(middle_zone)
        stats["bimodal/middle_zone_frac"] = float(middle_zone / len(all_taus))
        
        # Entropy of the soft mask (higher = more uncertain)
        # Treat sigmoid(tau) as probabilities
        probs = 1 / (1 + np.exp(-all_taus))  # sigmoid
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        entropy = -np.mean(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
        stats["bimodal/entropy"] = float(entropy)
        
        # "Decisiveness" - fraction of nodes near the boundaries
        near_boundary = np.mean((all_taus < -0.8) | (all_taus > 0.8))
        stats["bimodal/decisiveness"] = float(near_boundary)
        
        return stats
    
    def _compute_tau_velocity(self) -> Dict[str, float]:
        """
        Track how fast tau values are changing.
        Requires storing previous tau values.
        """
        all_taus = self.masked_model.masks.get_all_tau_values().detach().cpu().numpy()
        
        stats = {}
        
        if hasattr(self, '_prev_taus') and self._prev_taus is not None:
            delta = all_taus - self._prev_taus
            stats["velocity/mean_abs_delta"] = float(np.mean(np.abs(delta)))
            stats["velocity/max_abs_delta"] = float(np.max(np.abs(delta)))
            stats["velocity/rms_delta"] = float(np.sqrt(np.mean(delta ** 2)))
            
            # How many nodes changed sign (crossed threshold)
            sign_flips = np.sum((self._prev_taus > 0) != (all_taus > 0))
            stats["velocity/sign_flips"] = int(sign_flips)
            
            # Correlation between previous and current rankings
            prev_ranks = np.argsort(np.argsort(-self._prev_taus))  # Rank by tau (descending)
            curr_ranks = np.argsort(np.argsort(-all_taus))
            rank_corr = np.corrcoef(prev_ranks, curr_ranks)[0, 1]
            stats["velocity/rank_correlation"] = float(rank_corr)
        
        # Store for next call
        self._prev_taus = all_taus.copy()
        
        return stats
    
    def _compute_topk_stability(self, k_values: list = [50, 100, 200, 500]) -> Dict[str, float]:
        """
        Track which nodes are in the top-k and how stable this is.
        """
        all_taus = self.masked_model.masks.get_all_tau_values().detach().cpu().numpy()
        sorted_indices = np.argsort(-all_taus)  # Indices sorted by tau (descending)
        
        stats = {}
        
        for k in k_values:
            if k > len(all_taus):
                continue
            
            current_topk = set(sorted_indices[:k].tolist())
            
            # Check overlap with previous top-k
            prev_key = f'_prev_topk_{k}'
            if hasattr(self, prev_key) and getattr(self, prev_key) is not None:
                prev_topk = getattr(self, prev_key)
                overlap = len(current_topk & prev_topk)
                stats[f"stability/top{k}_overlap"] = int(overlap)
                stats[f"stability/top{k}_overlap_frac"] = float(overlap / k)
            
            # Store for next call
            setattr(self, prev_key, current_topk)
            
            # What's the tau value at position k (the "cut-off")?
            stats[f"stability/top{k}_cutoff_tau"] = float(all_taus[sorted_indices[k - 1]])
        
        return stats
    
    def _run_pareto_probes(self, k_values: list = None, include_val: bool = True) -> Dict[str, any]:
        """
        Evaluate task loss at multiple k values to track Pareto curve evolution.
        
        This is expensive, so only run periodically!
        
        Accumulates curves over training and plots them all on the same axes,
        with color indicating training step (darker = earlier, lighter = later).
        
        Args:
            k_values: List of k values to probe (defaults to strategic points)
            include_val: Whether to also compute validation Pareto curve
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from .discretize import evaluate_at_k
        
        num_active_non_embed = self.masked_model.masks.get_total_active_nodes()
        # Get active token count for total nodes on x-axis
        num_active_embed = 0
        if self.masked_model.token_mask is not None:
            num_active_embed = self.masked_model.token_mask.get_num_active()
        
        if num_active_non_embed == 0:
            return {"pareto/num_active": 0}
        
        if k_values is None:
            # Default: probe at a few strategic points based on non-embed nodes
            # More fine-grained probing around the active region
            k_values = [
                max(1, num_active_non_embed // 20),      # 5% of active
                max(1, num_active_non_embed // 10),      # 10% of active
                max(1, num_active_non_embed // 5),       # 20% of active
                max(1, num_active_non_embed // 4),       # 25% of active
                max(1, num_active_non_embed // 3),       # 33% of active
                max(1, num_active_non_embed // 2),       # 50% of active
                max(1, int(num_active_non_embed * 0.67)), # 67% of active
                max(1, int(num_active_non_embed * 0.75)), # 75% of active
                max(1, int(num_active_non_embed * 0.85)), # 85% of active
                max(1, int(num_active_non_embed * 0.9)),  # 90% of active
                max(1, int(num_active_non_embed * 0.95)), # 95% of active
                num_active_non_embed,                     # 100% of active
            ]
            # Remove duplicates and sort
            k_values = sorted(set(k_values))
        
        # Save current state
        original_state = self.masked_model.get_mask_state()
        
        # Collect (total_nodes, loss) pairs for TRAIN
        # x-axis is total nodes = k (non-embed) + active embed tokens
        train_curve_data = []
        for k in k_values:
            loss = evaluate_at_k(
                self.masked_model, 
                self.task, 
                k, 
                self.config, 
                num_batches=3  # Quick probe
            )
            total_k = k + num_active_embed  # Add embed tokens for total nodes
            train_curve_data.append((total_k, float(loss)))
            # Restore state after each evaluation
            self.masked_model.load_mask_state(original_state)
        
        # Store train curve in history
        num_active_total = num_active_non_embed + num_active_embed
        self._pareto_history.append({
            'step': self.state.step,
            'num_active': num_active_total,
            'curve': train_curve_data,
        })
        
        # Collect (total_nodes, loss) pairs for VAL if we have a val task
        val_curve_data = None
        if include_val and self.val_task is not None:
            val_curve_data = []
            for k in k_values:
                loss = evaluate_at_k(
                    self.masked_model, 
                    self.val_task, 
                    k, 
                    self.config, 
                    num_batches=3
                )
                total_k = k + num_active_embed  # Add embed tokens for total nodes
                val_curve_data.append((total_k, float(loss)))
                self.masked_model.load_mask_state(original_state)
            
            self._pareto_history_val.append({
                'step': self.state.step,
                'num_active': num_active_total,
                'curve': val_curve_data,
            })
        
        # Restore state
        self.masked_model.load_mask_state(original_state)
        
        # Create matplotlib figure with ALL curves (train and val side by side)
        has_val = val_curve_data is not None
        fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(16 if has_val else 10, 6))
        if not has_val:
            axes = [axes]
        
        # Plot TRAIN curves
        ax = axes[0]
        n_curves = len(self._pareto_history)
        colors = cm.viridis(np.linspace(0.1, 0.9, max(1, n_curves)))
        
        for i, entry in enumerate(self._pareto_history):
            k_vals = [k for k, loss in entry['curve']]
            losses = [loss for k, loss in entry['curve']]
            
            alpha = 0.3 + 0.7 * (i / max(1, n_curves - 1)) if n_curves > 1 else 1.0
            linewidth = 1.0 + 1.5 * (i / max(1, n_curves - 1)) if n_curves > 1 else 2.0
            
            ax.plot(k_vals, losses, 'o-', 
                   color=colors[i], 
                   alpha=alpha,
                   linewidth=linewidth,
                   markersize=4,
                   label=f"Step {entry['step']} (n={entry['num_active']})")
        
        ax.set_xlabel("Total Nodes (non-embed + embed)", fontsize=12)
        ax.set_ylabel("Task Loss", fontsize=12)
        ax.set_title("Train Pareto Curve Evolution", fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        if n_curves <= 6:
            ax.legend(loc='upper right', fontsize=8)
        else:
            handles, labels = ax.get_legend_handles_labels()
            indices = [0, n_curves // 3, 2 * n_curves // 3, n_curves - 1]
            indices = sorted(set(indices))
            ax.legend([handles[i] for i in indices], 
                     [labels[i] for i in indices],
                     loc='upper right', fontsize=8)
        
        # Plot VAL curves
        if has_val:
            ax = axes[1]
            n_curves_val = len(self._pareto_history_val)
            colors_val = cm.plasma(np.linspace(0.1, 0.9, max(1, n_curves_val)))
            
            for i, entry in enumerate(self._pareto_history_val):
                k_vals = [k for k, loss in entry['curve']]
                losses = [loss for k, loss in entry['curve']]
                
                alpha = 0.3 + 0.7 * (i / max(1, n_curves_val - 1)) if n_curves_val > 1 else 1.0
                linewidth = 1.0 + 1.5 * (i / max(1, n_curves_val - 1)) if n_curves_val > 1 else 2.0
                
                ax.plot(k_vals, losses, 'o-', 
                       color=colors_val[i], 
                       alpha=alpha,
                       linewidth=linewidth,
                       markersize=4,
                       label=f"Step {entry['step']} (n={entry['num_active']})")
            
            ax.set_xlabel("Total Nodes (non-embed + embed)", fontsize=12)
            ax.set_ylabel("Task Loss", fontsize=12)
            ax.set_title("Val Pareto Curve Evolution", fontsize=14)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            if n_curves_val <= 6:
                ax.legend(loc='upper right', fontsize=8)
            else:
                handles, labels = ax.get_legend_handles_labels()
                indices = [0, n_curves_val // 3, 2 * n_curves_val // 3, n_curves_val - 1]
                indices = sorted(set(indices))
                ax.legend([handles[i] for i in indices], 
                         [labels[i] for i in indices],
                         loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        # Compute stats
        train_min_loss = min(loss for k, loss in train_curve_data)
        train_k_for_min = min((k for k, loss in train_curve_data if loss == train_min_loss))
        
        stats = {
            "pareto/evolution": wandb.Image(fig),
            "pareto/train_min_loss": train_min_loss,
            "pareto/train_k_for_min_loss": train_k_for_min,
            "pareto/num_active_total": num_active_total,
            "pareto/num_active_non_embed": num_active_non_embed,
            "pareto/num_active_embed": num_active_embed,
            "pareto/num_curves": n_curves,
        }
        
        if val_curve_data is not None:
            val_min_loss = min(loss for k, loss in val_curve_data)
            val_k_for_min = min((k for k, loss in val_curve_data if loss == val_min_loss))
            stats["pareto/val_min_loss"] = val_min_loss
            stats["pareto/val_k_for_min_loss"] = val_k_for_min
        
        plt.close(fig)
        
        return stats
    
    def _compute_val_metrics(self, num_batches: int = 5) -> Dict[str, float]:
        """Compute validation metrics using val_task."""
        if self.val_task is None:
            return {}
        
        self.masked_model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_logit_diff = 0.0
        
        with torch.no_grad():
            for _ in range(num_batches):
                positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = self.val_task.generate_batch(
                    batch_size=self.config.batch_size,
                    max_length=self.config.seq_length,
                )
                
                positive_ids = positive_ids.to(self.device)
                negative_ids = negative_ids.to(self.device)
                correct_tokens = correct_tokens.to(self.device)
                incorrect_tokens = incorrect_tokens.to(self.device)
                eval_positions = eval_positions.to(self.device)
                
                _, batch_metrics = self.masked_model.compute_task_loss(
                    positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions
                )
                
                total_loss += batch_metrics["task_loss"]
                total_acc += batch_metrics["accuracy"]
                total_logit_diff += batch_metrics["logit_diff"]
        
        self.masked_model.train()
        
        return {
            "val/task_loss": total_loss / num_batches,
            "val/accuracy": total_acc / num_batches,
            "val/logit_diff": total_logit_diff / num_batches,
        }
    
    def _log_wandb_metrics(self, metrics: Dict[str, float], step: int, detailed: bool = False, pareto_probe: bool = False):
        """Log metrics to wandb."""
        if not self.use_wandb:
            return
        
        # Compute total nodes including token mask
        num_active_non_embed = metrics["num_active_nodes"]
        total_non_embed = metrics["total_nodes"]
        num_active_embed = metrics.get("num_active_tokens", 0)
        total_embed = metrics.get("total_tokens", 0)
        num_active_total = num_active_non_embed + num_active_embed
        total_nodes = total_non_embed + total_embed
        
        log_dict = {
            # Core metrics
            "train/task_loss": metrics["task_loss"],
            "train/accuracy": metrics["accuracy"],
            "train/logit_diff": metrics["logit_diff"],
            "train/total_loss": metrics["total_loss"],
            "train/sparsity_loss": metrics["sparsity_loss"],
            
            # Circuit size - total (non-embed + embed)
            "circuit/active_nodes_total": num_active_total,
            "circuit/total_nodes": total_nodes,
            "circuit/frac_active_total": num_active_total / total_nodes if total_nodes > 0 else 0,
            
            # Circuit size - non-embed only
            "circuit/active_nodes_non_embed": num_active_non_embed,
            "circuit/total_nodes_non_embed": total_non_embed,
            "circuit/frac_active_non_embed": num_active_non_embed / total_non_embed if total_non_embed > 0 else 0,
            
            # Circuit size - embed only (if token mask enabled)
            "circuit/active_nodes_embed": num_active_embed,
            "circuit/total_nodes_embed": total_embed,
            "circuit/frac_active_embed": num_active_embed / total_embed if total_embed > 0 else 0,
            
            # Learning rate
            "optim/lr": metrics["lr"],
            "optim/lr_multiplier": self._get_lr_multiplier(step),
        }
        
        if detailed:
            # Validation metrics
            val_metrics = self._compute_val_metrics(num_batches=5)
            log_dict.update(val_metrics)
            
            # Tau statistics
            tau_stats = self._compute_tau_statistics()
            log_dict.update(tau_stats)
            
            # Per-location breakdown
            loc_stats = self._compute_per_location_stats()
            log_dict.update(loc_stats)
            
            # Per-layer breakdown
            layer_stats = self._compute_per_layer_stats()
            log_dict.update(layer_stats)
            
            # Threshold analysis
            threshold_stats = self._compute_threshold_analysis()
            log_dict.update(threshold_stats)
            
            # Bimodality metrics - key for understanding circuit size convergence
            bimodal_stats = self._compute_bimodality_metrics()
            log_dict.update(bimodal_stats)
            
            # Tau velocity - how fast is learning?
            velocity_stats = self._compute_tau_velocity()
            log_dict.update(velocity_stats)
            
            # Top-k stability - are important nodes stable?
            stability_stats = self._compute_topk_stability()
            log_dict.update(stability_stats)
            
            # Gradient stats (after backward, before optimizer step)
            grad_stats = self._compute_gradient_stats()
            log_dict.update(grad_stats)
        
        if pareto_probe:
            # Expensive! Only run occasionally
            pareto_stats = self._run_pareto_probes(include_val=self.val_task is not None)
            log_dict.update(pareto_stats)
        
        wandb.log(log_dict, step=step)
    
    def _update_lr(self, step: int):
        """Update learning rate based on schedule."""
        multiplier = self._get_lr_multiplier(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr * multiplier
    
    def _clip_gradients(self):
        """Clip gradients by RMS norm."""
        # Collect all mask parameters (node masks + token mask if enabled)
        all_mask_params = list(self.masked_model.masks.parameters())
        if self.masked_model.token_mask is not None:
            all_mask_params.extend(list(self.masked_model.token_mask.parameters()))
        
        # Compute RMS of all gradients
        total_norm_sq = 0.0
        num_params = 0
        
        for param in all_mask_params:
            if param.grad is not None:
                total_norm_sq += param.grad.pow(2).sum().item()
                num_params += param.grad.numel()
        
        if num_params > 0:
            rms_norm = (total_norm_sq / num_params) ** 0.5
            
            if rms_norm > self.config.grad_clip_norm:
                clip_coef = self.config.grad_clip_norm / (rms_norm + 1e-6)
                for param in all_mask_params:
                    if param.grad is not None:
                        param.grad.mul_(clip_coef)
    
    def train_step(self, log_detailed: bool = False) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            log_detailed: Whether to compute and log detailed diagnostics
        
        Returns:
            Dictionary of metrics
        """
        self.masked_model.train()
        
        # Generate batch
        positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = self.task.generate_batch(
            batch_size=self.config.batch_size,
            max_length=self.config.seq_length,
        )
        
        # Move to device
        positive_ids = positive_ids.to(self.device)
        negative_ids = negative_ids.to(self.device)
        correct_tokens = correct_tokens.to(self.device)
        incorrect_tokens = incorrect_tokens.to(self.device)
        eval_positions = eval_positions.to(self.device)
        
        # Update learning rate
        self._update_lr(self.state.step)
        
        # Forward pass and compute loss
        self.optimizer.zero_grad()
        
        loss, metrics = self.masked_model.compute_full_loss(
            positive_ids,
            negative_ids,
            correct_tokens,
            incorrect_tokens,
            k_coef=self.config.k_coef,
            eval_positions=eval_positions,
        )
        
        # Backward pass
        loss.backward()
        
        # Clip gradients (compute gradient stats before clipping for diagnostics)
        grad_stats_pre_clip = {}
        if log_detailed and self.use_wandb:
            grad_stats_pre_clip = self._compute_gradient_stats()
            # Add pre-clip prefix
            grad_stats_pre_clip = {k.replace("grad/", "grad_pre_clip/"): v 
                                   for k, v in grad_stats_pre_clip.items()}
        
        self._clip_gradients()
        
        # Optimizer step
        self.optimizer.step()
        
        # Clamp parameters to [-1, 1]
        self.masked_model.clamp_mask_parameters()
        
        # Update state
        self.state.step += 1
        
        # Track best loss
        task_loss = metrics["task_loss"]
        if task_loss < self.state.best_loss:
            self.state.best_loss = task_loss
            self.state.best_step = self.state.step
        
        # Add LR to metrics
        metrics["lr"] = self.optimizer.param_groups[0]['lr']
        metrics["step"] = self.state.step
        
        # Store pre-clip gradient stats for logging
        if grad_stats_pre_clip:
            metrics["_grad_stats_pre_clip"] = grad_stats_pre_clip
        
        return metrics
    
    def train(
        self,
        num_steps: Optional[int] = None,
        log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
        show_progress: bool = True,
        histogram_every: int = 100,
        detailed_log_every: Optional[int] = None,
        pareto_probe_every: int = 0,
    ) -> Dict[str, float]:
        """
        Run the full training loop.
        
        Args:
            num_steps: Number of steps (defaults to config.num_steps)
            log_fn: Optional logging function called with metrics
            show_progress: Whether to show progress bar
            histogram_every: Log tau histograms every N steps (0 to disable)
            detailed_log_every: Log detailed diagnostics every N steps (defaults to log_every)
            pareto_probe_every: Evaluate loss at multiple k values every N steps (0 to disable)
                               This is expensive, so use sparingly (e.g., every 500-1000 steps)
            
        Returns:
            Final metrics
        """
        if num_steps is None:
            num_steps = self.config.num_steps
        
        if detailed_log_every is None:
            detailed_log_every = self.config.log_every
        
        # Initialize velocity tracking
        self._prev_taus = None
        
        # Log step 0 metrics BEFORE any training (evaluation only)
        if self.use_wandb and self.state.step == 0:
            step0_metrics = self._evaluate_step0()
            self._log_wandb_metrics(step0_metrics, step=0, detailed=True, pareto_probe=False)
        
        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Pruning")
        
        for step in iterator:
            # Determine if this is a detailed logging step
            should_log_detailed = (self.state.step + 1) % detailed_log_every == 0
            should_pareto_probe = pareto_probe_every > 0 and (self.state.step + 1) % pareto_probe_every == 0
            
            metrics = self.train_step(log_detailed=should_log_detailed)
            
            # Store history (without internal diagnostic fields)
            clean_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
            self.history.append(clean_metrics)
            
            # Log to wandb
            if self.use_wandb:
                self._log_wandb_metrics(
                    metrics, 
                    self.state.step, 
                    detailed=should_log_detailed,
                    pareto_probe=should_pareto_probe
                )
                
                # Log pre-clip gradient stats if available
                if "_grad_stats_pre_clip" in metrics:
                    wandb.log(metrics["_grad_stats_pre_clip"], step=self.state.step, commit=False)
                
                # Log histogram periodically
                if histogram_every > 0 and self.state.step % histogram_every == 0:
                    self._log_tau_histogram(prefix="train")
                    wandb.log({}, step=self.state.step)  # Commit the histogram
            
            # Standard logging
            if self.state.step % self.config.log_every == 0:
                if log_fn is not None:
                    log_fn(metrics)
                
                if show_progress:
                    iterator.set_postfix({
                        "loss": f"{metrics['task_loss']:.4f}",
                        "active": metrics['num_active_nodes'],
                        "acc": f"{metrics['accuracy']:.2%}",
                    })
        
        # Log final histogram and do final Pareto probe
        if self.use_wandb:
            self._log_tau_histogram(prefix="final")
            
            # Final detailed stats
            final_bimodal = self._compute_bimodality_metrics()
            final_stability = self._compute_topk_stability()
            
            # Final Pareto probe
            final_pareto = self._run_pareto_probes()
            
            final_log = {
                "final/task_loss": metrics["task_loss"],
                "final/accuracy": metrics["accuracy"],
                "final/active_nodes": metrics["num_active_nodes"],
                "final/best_loss": self.state.best_loss,
                "final/best_step": self.state.best_step,
            }
            final_log.update({f"final_{k}": v for k, v in final_bimodal.items()})
            final_log.update({f"final_{k}": v for k, v in final_pareto.items()})
            
            wandb.log(final_log, step=self.state.step)
        
        return metrics
    
    def _evaluate_step0(self, num_batches: int = 10) -> Dict[str, float]:
        """
        Evaluate metrics at step 0 before any training.
        This helps diagnose initialization issues.
        """
        self.masked_model.eval()
        
        total_task_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_logit_diff = 0.0
        
        with torch.no_grad():
            for _ in range(num_batches):
                positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = self.task.generate_batch(
                    batch_size=self.config.batch_size,
                    max_length=self.config.seq_length,
                )
                
                positive_ids = positive_ids.to(self.device)
                correct_tokens = correct_tokens.to(self.device)
                incorrect_tokens = incorrect_tokens.to(self.device)
                eval_positions = eval_positions.to(self.device)
                
                # Forward pass
                logits = self.masked_model(positive_ids)
                
                # Get logits at eval positions
                batch_indices = torch.arange(logits.size(0), device=self.device)
                eval_logits = logits[batch_indices, eval_positions]
                
                # Compute task loss
                correct_logits = eval_logits[batch_indices, correct_tokens]
                incorrect_logits = eval_logits[batch_indices, incorrect_tokens]
                logit_diff = correct_logits - incorrect_logits
                task_loss = torch.nn.functional.softplus(-logit_diff).mean()
                
                total_task_loss += task_loss.item()
                total_correct += (logit_diff > 0).sum().item()
                total_samples += logit_diff.size(0)
                total_logit_diff += logit_diff.mean().item()
        
        self.masked_model.train()
        
        num_active = self.masked_model.masks.get_total_active_nodes()
        total_nodes = self.masked_model.masks.get_total_nodes()
        
        return {
            "task_loss": total_task_loss / num_batches,
            "accuracy": 100 * total_correct / total_samples,
            "logit_diff": total_logit_diff / num_batches,
            "total_loss": total_task_loss / num_batches + self.config.k_coef * num_active,
            "sparsity_loss": self.config.k_coef * num_active,
            "num_active_nodes": num_active,
            "total_nodes": total_nodes,
            "lr": self.config.lr,
        }
    
    def evaluate(self, num_batches: int = 10) -> Dict[str, float]:
        """
        Evaluate current masks on task.
        
        Args:
            num_batches: Number of batches to evaluate
            
        Returns:
            Average metrics
        """
        self.masked_model.eval()
        
        total_metrics = {}
        
        with torch.no_grad():
            for _ in range(num_batches):
                # Generate batch
                positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = self.task.generate_batch(
                    batch_size=self.config.batch_size,
                    max_length=self.config.seq_length,
                )
                
                # Move to device
                positive_ids = positive_ids.to(self.device)
                negative_ids = negative_ids.to(self.device)
                correct_tokens = correct_tokens.to(self.device)
                incorrect_tokens = incorrect_tokens.to(self.device)
                eval_positions = eval_positions.to(self.device)
                
                # Compute metrics
                _, metrics = self.masked_model.compute_full_loss(
                    positive_ids,
                    negative_ids,
                    correct_tokens,
                    incorrect_tokens,
                    k_coef=self.config.k_coef,
                    eval_positions=eval_positions,
                )
                
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
        
        # Average
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        return total_metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "mask_state": self.masked_model.get_mask_state(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_state": {
                "step": self.state.step,
                "best_loss": self.state.best_loss,
                "best_step": self.state.best_step,
            },
            "config": self.config.to_dict(),
            "history": self.history,
        }
        
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.masked_model.load_mask_state(checkpoint["mask_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        ts = checkpoint["training_state"]
        self.state.step = ts["step"]
        self.state.best_loss = ts["best_loss"]
        self.state.best_step = ts["best_step"]
        
        self.history = checkpoint.get("history", [])


def run_pruning(
    model: nn.Module,
    task: BinaryTask,
    config: PruningConfig,
    data_iterator: Optional[Iterator[torch.Tensor]] = None,
    log_fn: Optional[Callable[[Dict[str, float]], None]] = None,
    show_progress: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "circuit-pruning",
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    histogram_every: int = 100,
) -> Tuple[MaskedSparseGPT, Dict[str, float], "PruningTrainer"]:
    """
    Run the full pruning procedure.
    
    Args:
        model: The SparseGPT model to prune
        task: Binary task for evaluation
        config: Pruning configuration
        data_iterator: Iterator for computing mean activations (optional)
        log_fn: Logging function
        show_progress: Whether to show progress bars
        use_wandb: Whether to log to Weights & Biases
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name
        wandb_config: Additional wandb config
        histogram_every: Log tau histogram every N steps
        
    Returns:
        Tuple of (masked_model, final_metrics, trainer)
    """
    # Create masked model
    masked_model = MaskedSparseGPT(model, config)
    
    # Compute mean activations if data provided
    if data_iterator is not None:
        print("Computing mean activation cache...")
        mean_cache = masked_model.compute_mean_cache(
            data_iterator,
            num_batches=config.mean_cache_num_batches,
            show_progress=show_progress,
        )
        masked_model.set_means_from_dict(mean_cache)
        print(f"Mean cache computed for {len(mean_cache)} locations")
    
    # Create trainer with optional wandb logging
    trainer = PruningTrainer(
        masked_model,
        task,
        config,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config,
    )
    
    # Train
    print(f"Starting pruning optimization for {config.num_steps} steps...")
    final_metrics = trainer.train(
        num_steps=config.num_steps,
        log_fn=log_fn,
        show_progress=show_progress,
        histogram_every=histogram_every,
    )
    
    # Evaluate
    print("Evaluating final masks...")
    eval_metrics = trainer.evaluate(num_batches=20)
    
    # Log final evaluation to wandb
    if trainer.use_wandb:
        wandb.log({
            "eval/task_loss": eval_metrics["task_loss"],
            "eval/accuracy": eval_metrics["accuracy"],
            "eval/logit_diff": eval_metrics["logit_diff"],
            "eval/active_nodes": eval_metrics["num_active_nodes"],
        })
        wandb.finish()
    
    print(f"\nFinal results:")
    print(f"  Task loss: {eval_metrics['task_loss']:.4f}")
    print(f"  Accuracy: {eval_metrics['accuracy']:.2%}")
    print(f"  Active nodes: {eval_metrics['num_active_nodes']} / {eval_metrics['total_nodes']}")
    
    return masked_model, eval_metrics, trainer

