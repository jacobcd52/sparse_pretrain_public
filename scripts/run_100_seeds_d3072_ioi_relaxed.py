#!/usr/bin/env python3
"""
Run 100 seeds for IOI Relaxed task on D3072 model, filter by target loss, 
then evaluate same-gender vs opposite-gender performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from contextlib import nullcontext
from tqdm import tqdm

from transformers import AutoTokenizer

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.trainer import PruningTrainer
from sparse_pretrain.src.pruning.tasks import get_task, IOIRelaxedTask, TaskExample
from sparse_pretrain.src.pruning.run_pruning import load_model
from sparse_pretrain.src.pruning.discretize import evaluate_at_k


def run_single_seed_quick(
    seed: int,
    hparams: Dict[str, float],
    model,
    tokenizer,
    mean_cache: Optional[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run a single pruning experiment with a specific seed (minimal output)."""
    device = config["device"]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    pruning_config = PruningConfig(
        k_coef=hparams["k_coef"],
        init_noise_scale=config["init_noise_scale"],
        init_noise_bias=config["init_noise_bias"],
        weight_decay=hparams["weight_decay"],
        lr=hparams["lr"],
        beta2=hparams["beta2"],
        lr_warmup_frac=config["lr_warmup_frac"],
        heaviside_temp=hparams["heaviside_temp"],
        num_steps=config["num_steps"],
        batch_size=config["batch_size"],
        seq_length=config.get("task_max_length", 0),
        device=device,
        log_every=500,
        target_loss=config["target_loss"],
        ablation_type=config["ablation_type"],
        mask_token_embeds=config["mask_token_embeds"],
    )
    
    train_task = get_task(config["task_name"], tokenizer, seed=seed, split="train")
    val_task = get_task(config["task_name"], tokenizer, seed=seed, split="val")
    
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if config["ablation_type"] != "zero" and mean_cache is not None:
        masked_model.set_means_from_dict(mean_cache)
    
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        val_task=val_task,
        config=pruning_config,
        use_wandb=False,
    )
    
    if config.get("use_autocast", True):
        dtype = torch.bfloat16 if config.get("autocast_dtype", "bfloat16") == "bfloat16" else torch.float16
        autocast_ctx = torch.autocast('cuda', dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    
    with autocast_ctx:
        trainer.train(
            num_steps=config["num_steps"],
            show_progress=False,
            histogram_every=0,
            pareto_probe_every=0,
        )
    
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    original_state = masked_model.get_mask_state()
    
    eval_batches = config.get("bisection_eval_batches", 5)
    with autocast_ctx:
        loss_at_all = evaluate_at_k(masked_model, val_task, num_active, pruning_config, num_batches=eval_batches)
    
    target_achieved = loss_at_all <= config["target_loss"]
    
    if target_achieved:
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(original_state, seed_dir / "masks.pt")
        
        summary = {
            "seed": seed,
            "loss_at_all_active": float(loss_at_all),
            "num_active": num_active,
            "total_nodes": total_nodes,
            "target_achieved": target_achieved,
        }
        with open(seed_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    return {
        "seed": seed,
        "loss_at_all_active": float(loss_at_all),
        "num_active": num_active,
        "total_nodes": total_nodes,
        "target_achieved": target_achieved,
        "mask_state": original_state if target_achieved else None,
    }


class IOIRelaxedSameGenderTask(IOIRelaxedTask):
    """IOI task where both names are same gender."""
    
    @property
    def name(self) -> str:
        return f"ioi_relaxed_same_gender_{self.split}"
    
    def generate_example(self):
        use_male = self.rng.random() > 0.5
        
        if use_male:
            name1 = self.rng.choice(self.MALE_NAMES)
            name2 = self.rng.choice([n for n in self.MALE_NAMES if n != name1])
            correct_token = self.him_token
            incorrect_token = self.her_token
        else:
            name1 = self.rng.choice(self.FEMALE_NAMES)
            name2 = self.rng.choice([n for n in self.FEMALE_NAMES if n != name1])
            correct_token = self.her_token
            incorrect_token = self.him_token
        
        template = self.rng.choice(self.TEMPLATES)
        context = template.format(name1=name1, name2=name2)
        
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        context_tensor = torch.tensor(context_ids, dtype=torch.long)
        
        return TaskExample(
            positive_ids=context_tensor,
            negative_ids=context_tensor.clone(),
            correct_token=correct_token,
            incorrect_token=incorrect_token,
        )


def evaluate_gender_performance(masked_model, tokenizer, config, device):
    """Evaluate correct answer probability for same-gender and opposite-gender prompts."""
    opposite_gender_task = get_task(config["task_name"], tokenizer, seed=12345, split="val")
    same_gender_task = IOIRelaxedSameGenderTask(tokenizer, seed=12345, split="val")
    
    if config.get("use_autocast", True):
        dtype = torch.bfloat16 if config.get("autocast_dtype", "bfloat16") == "bfloat16" else torch.float16
        autocast_ctx = torch.autocast('cuda', dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    
    batch_size = config.get("batch_size", 64)
    num_batches = 20
    
    results = {}
    
    for task_name, task in [("opposite_gender", opposite_gender_task), ("same_gender", same_gender_task)]:
        all_probs = []
        
        for _ in range(num_batches):
            positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = task.generate_batch(
                batch_size=batch_size,
                max_length=config.get("task_max_length", 0),
            )
            
            positive_ids = positive_ids.to(device)
            correct_tokens = correct_tokens.to(device)
            incorrect_tokens = incorrect_tokens.to(device)
            eval_positions = eval_positions.to(device)
            
            with torch.no_grad(), autocast_ctx:
                logits = masked_model.forward(positive_ids, return_logits_only=True)
                
                batch_indices = torch.arange(positive_ids.shape[0], device=device)
                eval_logits = logits[batch_indices, eval_positions, :]
                
                probs = torch.softmax(eval_logits, dim=-1)
                correct_probs = probs[batch_indices, correct_tokens]
                incorrect_probs = probs[batch_indices, incorrect_tokens]
                
                binary_probs = correct_probs / (correct_probs + incorrect_probs + 1e-10)
                all_probs.extend(binary_probs.cpu().tolist())
        
        results[f"{task_name}_mean_prob"] = float(np.mean(all_probs))
        results[f"{task_name}_std_prob"] = float(np.std(all_probs))
        results[f"{task_name}_all_probs"] = all_probs
    
    return results


def main():
    CHECKPOINT_DIR = Path("outputs/carbs_results_ioi_relaxed/ss_bridges_d3072_f0.005_zero_noembed")
    NUM_SEEDS = 100
    DEVICE = "cuda"
    
    with open(CHECKPOINT_DIR / "sweep_config.json") as f:
        config = json.load(f)
    
    with open(CHECKPOINT_DIR / "best_checkpoint" / "hparams.json") as f:
        hparams = json.load(f)
    hparams = {k: v for k, v in hparams.items() if k != "suggestion_uuid"}
    
    config["device"] = DEVICE
    
    print("=" * 70)
    print("Running 100 Seeds for IOI Relaxed - D3072 Model")
    print("=" * 70)
    print(f"Model: {config['model_path']}")
    print(f"Task: {config['task_name']}")
    print(f"Target loss: {config['target_loss']}")
    print("=" * 70)
    
    output_dir = CHECKPOINT_DIR / "repetitions_100"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading model...")
    model, _ = load_model(config["model_path"], DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    mean_cache = None
    if config["ablation_type"] != "zero":
        mean_cache_path = CHECKPOINT_DIR / "mean_cache.pt"
        if mean_cache_path.exists():
            mean_cache = torch.load(mean_cache_path, map_location=DEVICE, weights_only=True)
            mean_cache = {k: v.to(DEVICE) for k, v in mean_cache.items()}
    
    print(f"\nRunning {NUM_SEEDS} seeds...")
    all_results = []
    successful_seeds = []
    failed_seeds = []
    
    for seed in tqdm(range(NUM_SEEDS), desc="Seeds"):
        try:
            result = run_single_seed_quick(
                seed=seed,
                hparams=hparams,
                model=model,
                tokenizer=tokenizer,
                mean_cache=mean_cache,
                config=config,
                output_dir=output_dir,
            )
            all_results.append(result)
            
            if result["target_achieved"]:
                successful_seeds.append(seed)
                print(f"  Seed {seed}: PASS (loss={result['loss_at_all_active']:.4f}, nodes={result['num_active']})")
            else:
                failed_seeds.append(seed)
                print(f"  Seed {seed}: FAIL (loss={result['loss_at_all_active']:.4f})")
                
        except Exception as e:
            print(f"  Seed {seed}: ERROR - {e}")
            failed_seeds.append(seed)
        
        if seed % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n{'='*70}")
    print(f"Phase 1 Complete: {len(successful_seeds)}/{NUM_SEEDS} seeds achieved target loss")
    print(f"{'='*70}")
    
    phase1_summary = {
        "total_seeds": NUM_SEEDS,
        "successful_seeds": successful_seeds,
        "failed_seeds": failed_seeds,
        "success_rate": len(successful_seeds) / NUM_SEEDS,
        "config": config,
        "hparams": hparams,
    }
    with open(output_dir / "phase1_summary.json", "w") as f:
        json.dump(phase1_summary, f, indent=2)
    
    if len(successful_seeds) == 0:
        print("No seeds achieved target loss! Exiting.")
        return
    
    print(f"\nPhase 2: Evaluating same-gender vs opposite-gender performance...")
    
    pruning_config = PruningConfig(
        batch_size=config["batch_size"],
        seq_length=config.get("task_max_length", 0),
        device=DEVICE,
        ablation_type=config["ablation_type"],
        mask_token_embeds=config["mask_token_embeds"],
    )
    
    gender_results = []
    
    for seed in tqdm(successful_seeds, desc="Evaluating gender performance"):
        seed_dir = output_dir / f"seed{seed}"
        masks_path = seed_dir / "masks.pt"
        
        if not masks_path.exists():
            continue
        
        masked_model = MaskedSparseGPT(model, pruning_config)
        masked_model.to(DEVICE)
        if config["ablation_type"] != "zero" and mean_cache is not None:
            masked_model.set_means_from_dict(mean_cache)
        
        masks = torch.load(masks_path, map_location=DEVICE, weights_only=True)
        masked_model.load_mask_state(masks)
        
        gender_perf = evaluate_gender_performance(masked_model, tokenizer, config, DEVICE)
        gender_perf["seed"] = seed
        gender_results.append(gender_perf)
        
        with open(seed_dir / "gender_eval.json", "w") as f:
            json.dump({
                "seed": seed,
                "opposite_gender_mean_prob": gender_perf["opposite_gender_mean_prob"],
                "opposite_gender_std_prob": gender_perf["opposite_gender_std_prob"],
                "same_gender_mean_prob": gender_perf["same_gender_mean_prob"],
                "same_gender_std_prob": gender_perf["same_gender_std_prob"],
            }, f, indent=2)
    
    print("\nCreating histograms...")
    
    opposite_means = [r["opposite_gender_mean_prob"] for r in gender_results]
    same_means = [r["same_gender_mean_prob"] for r in gender_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(opposite_means, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=np.mean(opposite_means), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(opposite_means):.3f}')
    ax1.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Chance (0.5)')
    ax1.set_xlabel("P(correct) / (P(correct) + P(incorrect))", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Opposite-Gender Prompts\n(n={len(opposite_means)} seeds)", fontsize=14)
    ax1.legend()
    ax1.set_xlim(0, 1)
    
    ax2 = axes[1]
    ax2.hist(same_means, bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(x=np.mean(same_means), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(same_means):.3f}')
    ax2.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, label='Chance (0.5)')
    ax2.set_xlabel("P(correct) / (P(correct) + P(incorrect))", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title(f"Same-Gender Prompts\n(n={len(same_means)} seeds)", fontsize=14)
    ax2.legend()
    ax2.set_xlim(0, 1)
    
    plt.suptitle("IOI Relaxed: Correct Answer Probability by Gender Configuration\n(D3072 Model, Zero Ablation)", 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "gender_performance_histograms.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create scatter plot
    # Load circuit sizes
    circuit_sizes = []
    same_gender_probs = []
    seeds_list = []
    
    for seed in successful_seeds:
        seed_dir = output_dir / f"seed{seed}"
        summary_path = seed_dir / "summary.json"
        gender_path = seed_dir / "gender_eval.json"
        
        if not summary_path.exists() or not gender_path.exists():
            continue
        
        with open(summary_path) as f:
            summary = json.load(f)
        with open(gender_path) as f:
            gender = json.load(f)
        
        circuit_sizes.append(summary["num_active"])
        same_gender_probs.append(gender["same_gender_mean_prob"])
        seeds_list.append(seed)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(circuit_sizes, same_gender_probs, 
                         c=same_gender_probs, cmap='RdYlGn', 
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('P(correct) on Same-Gender', fontsize=11)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance (0.5)')
    ax.axhline(y=np.mean(same_gender_probs), color='red', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'Mean: {np.mean(same_gender_probs):.3f}')
    corr = np.corrcoef(circuit_sizes, same_gender_probs)[0, 1]
    ax.set_xlabel("Circuit Size (# active nodes)", fontsize=12)
    ax.set_ylabel("P(correct) on Same-Gender Prompts", fontsize=12)
    ax.set_title(f"Circuit Size vs Same-Gender Generalization (D3072)\n(n={len(circuit_sizes)} seeds, r={corr:.3f})", fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "circuit_size_vs_same_gender.png", dpi=150)
    plt.close()
    
    final_summary = {
        "total_seeds": NUM_SEEDS,
        "successful_seeds": len(successful_seeds),
        "failed_seeds": len(failed_seeds),
        "success_rate": len(successful_seeds) / NUM_SEEDS,
        "opposite_gender": {
            "mean": float(np.mean(opposite_means)),
            "std": float(np.std(opposite_means)),
            "min": float(np.min(opposite_means)),
            "max": float(np.max(opposite_means)),
        },
        "same_gender": {
            "mean": float(np.mean(same_means)),
            "std": float(np.std(same_means)),
            "min": float(np.min(same_means)),
            "max": float(np.max(same_means)),
        },
    }
    with open(output_dir / "final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Final Results - D3072 Model")
    print(f"{'='*70}")
    print(f"Seeds that achieved target loss: {len(successful_seeds)}/{NUM_SEEDS}")
    print(f"\nOpposite-Gender Performance:")
    print(f"  Mean P(correct): {np.mean(opposite_means):.4f} ± {np.std(opposite_means):.4f}")
    print(f"\nSame-Gender Performance:")
    print(f"  Mean P(correct): {np.mean(same_means):.4f} ± {np.std(same_means):.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
