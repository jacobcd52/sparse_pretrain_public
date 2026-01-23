#!/usr/bin/env python
"""
Re-run pruning for d1024 zero_noembed model to get a new working mask.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import torch
import json
from contextlib import nullcontext

from my_sparse_pretrain.src.pruning.run_pruning import load_model
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.trainer import PruningTrainer
from my_sparse_pretrain.src.pruning.discretize import evaluate_at_k, discretize_masks
from transformers import AutoTokenizer


def main():
    device = "cuda"
    model_dir = Path("my_sparse_pretrain/outputs/carbs_results_pronoun/ss_bridges_d1024_f0.015625_zero_noembed")
    
    # Load configs
    with open(model_dir / "sweep_config.json") as f:
        config = json.load(f)
    
    with open(model_dir / "best_checkpoint" / "hparams.json") as f:
        hparams = json.load(f)
    
    print("=" * 70)
    print("Re-running pruning for d1024 zero_noembed")
    print("=" * 70)
    print(f"Model: {config['model_path']}")
    print(f"Task: {config['task_name']}")
    print(f"Ablation: {config['ablation_type']}")
    print(f"Hyperparameters:")
    for k, v in hparams.items():
        if k != "suggestion_uuid":
            print(f"  {k}: {v:.6e}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, _ = load_model(config["model_path"], device=device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create pruning config
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
        target_loss=config["target_loss"],
        ablation_type=config["ablation_type"],
        mask_token_embeds=config["mask_token_embeds"],
        use_binary_loss=config.get("use_binary_loss", False),
    )
    
    # Create tasks
    train_task = get_task(config["task_name"], tokenizer, seed=42, split="train")
    val_task = get_task(config["task_name"], tokenizer, seed=42, split="val")
    superval_task = get_task(config["task_name"], tokenizer, seed=42, split="superval")
    print(f"Train task: {train_task.name} ({len(train_task.templates)} templates)")
    print(f"Val task: {val_task.name} ({len(val_task.templates)} templates)")
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    # For zero ablation, no mean cache needed
    
    # Create trainer
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        val_task=val_task,
        config=pruning_config,
        use_wandb=False,
    )
    
    # Autocast context
    if config.get("use_autocast", True):
        dtype = torch.bfloat16 if config.get("autocast_dtype", "bfloat16") == "bfloat16" else torch.float16
        autocast_ctx = torch.autocast('cuda', dtype=dtype)
    else:
        autocast_ctx = nullcontext()
    
    print("\nStarting training...")
    with autocast_ctx:
        trainer.train(
            num_steps=config["num_steps"],
            show_progress=True,
            histogram_every=0,
            pareto_probe_every=0,
        )
    
    # Get active nodes
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    print(f"\nTraining complete. Active nodes: {num_active}/{total_nodes}")
    
    # Save mask state
    original_state = masked_model.get_mask_state()
    
    # Evaluate at full capacity
    eval_batches = config.get("bisection_eval_batches", 5)
    with autocast_ctx:
        loss_at_all = evaluate_at_k(masked_model, val_task, num_active, pruning_config, num_batches=eval_batches)
    masked_model.load_mask_state(original_state)
    print(f"Loss at all active ({num_active}): {loss_at_all:.6f}")
    
    # Bisection search for smallest circuit
    target_loss = config["target_loss"]
    best_k, best_loss = num_active, loss_at_all
    
    if loss_at_all <= target_loss:
        print(f"\nRunning bisection to find smallest circuit achieving target loss {target_loss}...")
        low, high = 1, num_active
        max_iters = config.get("bisection_max_iters", 15)
        for i in range(max_iters):
            if low > high:
                break
            mid = (low + high) // 2
            with autocast_ctx:
                loss = evaluate_at_k(masked_model, val_task, mid, pruning_config, num_batches=eval_batches)
            masked_model.load_mask_state(original_state)
            
            print(f"  Bisection {i+1}: k={mid}, loss={loss:.4f}")
            
            if loss <= target_loss:
                best_k, best_loss = mid, loss
                high = mid - 1
            else:
                low = mid + 1
    
    print(f"\nBest circuit: k={best_k}, loss={best_loss:.6f}")
    
    # Get the binary circuit mask
    masked_model.masks.keep_top_k(best_k)
    circuit_mask = masked_model.get_circuit_mask()
    
    # Restore to original state
    masked_model.load_mask_state(original_state)
    
    # Save to best_checkpoint
    output_dir = model_dir / "best_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(original_state, output_dir / "masks.pt")
    torch.save(circuit_mask, output_dir / "circuit_mask.pt")
    
    summary = {
        "circuit_size": best_k,
        "achieved_loss_val": best_loss,
        "loss_at_all_active": loss_at_all,
        "num_active_after_training": num_active,
        "total_nodes": total_nodes,
        "frac_active": num_active / total_nodes,
        "frac_circuit": best_k / total_nodes,
        "target_achieved": best_loss <= target_loss,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved to {output_dir}")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    
    # Verify on superval
    print("\nVerifying on superval set...")
    masked_model.masks.keep_top_k(best_k)
    
    batch = superval_task.generate_batch(batch_size=64, max_length=0)
    positive_ids = batch[0].to(device)
    correct_tokens = batch[2].to(device)
    eval_positions = batch[4].to(device)
    batch_indices = torch.arange(positive_ids.shape[0], device=device)
    
    with torch.no_grad(), autocast_ctx:
        logits = masked_model(positive_ids)
        final_logits = logits[batch_indices, eval_positions, :]
        superval_loss = torch.nn.functional.cross_entropy(final_logits, correct_tokens).item()
    
    print(f"Superval loss at k={best_k}: {superval_loss:.6f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()


