#!/usr/bin/env python3
"""
Test script for tense task pruning with binary cross-entropy loss.

Runs a quick pruning test to verify everything works.
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from my_sparse_pretrain.src.pruning.run_pruning import load_model
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.trainer import PruningTrainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model_path = "jacobcd52/ss_bridges_d1024_f0.015625"
    tokenizer_name = "SimpleStories/SimpleStories-1.25M"
    
    print(f"Loading model: {model_path}")
    model, config = load_model(model_path, device=device)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create task
    train_task = get_task("dummy_tense", tokenizer, split="train")
    val_task = get_task("dummy_tense", tokenizer, split="val")
    
    print(f"Train task: {train_task.name}")
    print(f"Val task: {val_task.name}")
    print(f"Train templates: {len(train_task.templates)}")
    print(f"Val templates: {len(val_task.templates)}")
    
    # Create pruning config with binary loss enabled
    # Using ablation_type="zero" which doesn't need mean cache
    pruning_config = PruningConfig(
        target_loss=0.15,
        k_coef=1e-4,
        lr=1e-2,
        weight_decay=1e-3,
        num_steps=100,  # Short run for testing
        batch_size=32,
        seq_length=0,  # Dynamic padding
        ablation_type="zero",  # Simple ablation - no mean cache needed
        use_binary_loss=True,  # Use binary CE loss
        device=device,
    )
    
    print(f"\nPruning config:")
    print(f"  use_binary_loss: {pruning_config.use_binary_loss}")
    print(f"  target_loss: {pruning_config.target_loss}")
    print(f"  num_steps: {pruning_config.num_steps}")
    print(f"  ablation_type: {pruning_config.ablation_type}")
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    
    # Create trainer
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        config=pruning_config,
        use_wandb=False,
    )
    
    # Run training
    print("\nRunning pruning...")
    for step in range(pruning_config.num_steps):
        metrics = trainer.train_step()
        if step % 20 == 0:
            print(f"  Step {step}: loss={metrics['task_loss']:.4f}, "
                  f"binary_acc={metrics.get('binary_accuracy', 'N/A')}, "
                  f"active_nodes={metrics['num_active_nodes']}")
    
    # Final evaluation
    print("\nFinal evaluation on val set...")
    masked_model.eval()
    
    total_loss = 0
    total_binary_acc = 0
    n_batches = 10
    
    with torch.no_grad():
        for _ in range(n_batches):
            pos, neg, correct, incorrect, eval_pos = val_task.generate_batch(batch_size=32)
            pos, neg = pos.to(device), neg.to(device)
            correct, incorrect = correct.to(device), incorrect.to(device)
            eval_pos = eval_pos.to(device)
            
            _, metrics = masked_model.compute_task_loss(
                pos, neg, correct, incorrect, eval_pos
            )
            total_loss += metrics["task_loss"]
            total_binary_acc += metrics.get("binary_accuracy", 0)
    
    avg_loss = total_loss / n_batches
    avg_binary_acc = total_binary_acc / n_batches
    
    print(f"\nVal results:")
    print(f"  Avg binary CE loss: {avg_loss:.4f}")
    print(f"  Avg binary accuracy: {avg_binary_acc:.4f}")
    print(f"  Active nodes: {masked_model.masks.get_total_active_nodes()}")
    
    # Save results
    output_dir = Path("my_sparse_pretrain/outputs/carbs_results_tense_binary/test_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_path": model_path,
        "task": "dummy_tense",
        "use_binary_loss": True,
        "num_steps": pruning_config.num_steps,
        "final_loss": avg_loss,
        "final_binary_accuracy": avg_binary_acc,
        "active_nodes": masked_model.masks.get_total_active_nodes(),
        "total_nodes": masked_model.masks.get_total_nodes(),
    }
    
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print("\nTest complete! Binary loss pruning is working.")


if __name__ == "__main__":
    main()

