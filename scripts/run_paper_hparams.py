"""
Run pruning experiments with paper's default hyperparameters at different step counts.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.trainer import PruningTrainer
from my_sparse_pretrain.src.pruning.sweep_target_loss import sweep_target_losses, plot_loss_vs_circuit_size


def run_experiment(
    model_path: str,
    num_steps: int,
    output_dir: Path,
    device: str = "cuda",
    mean_cache: dict = None,
    tokenizer_name: str = None,
):
    """Run a single experiment with paper hyperparameters."""
    
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {num_steps} steps")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, model_config_dict = load_model(model_path, device)
    
    if tokenizer_name is None:
        tokenizer_name = model_config_dict.get("training_config", {}).get("tokenizer_name")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    task = get_task("dummy_pronoun", tokenizer, seed=42)
    
    # Paper's exact hyperparameters
    config = PruningConfig(
        # From paper Table 2
        k_coef=1e-4,
        init_noise_scale=1e-2,
        init_noise_bias=1e-1,
        weight_decay=1e-3,
        lr=3e-3,
        beta2=0.95,  # inv_beta2 = 0.05
        lr_warmup_frac=0.05,  # Paper Table 2 value
        heaviside_temp=1.0,
        # Training settings
        num_steps=num_steps,
        batch_size=64,
        seq_length=512,
        device=device,
    )
    
    # Save config
    config.to_yaml(output_dir / "config.yaml")
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)
    
    # Compute or reuse mean cache
    if mean_cache is not None:
        masked_model.set_means_from_dict({k: v.to(device) for k, v in mean_cache.items()})
    else:
        print("Computing mean cache...")
        data_iter = create_data_iterator(
            tokenizer_name=tokenizer_name,
            batch_size=64,
            seq_length=512,
            num_batches=100,
            seed=42,
        )
        mean_cache = masked_model.compute_mean_cache(
            data_iter,
            num_batches=100,
            show_progress=True,
        )
        masked_model.set_means_from_dict(mean_cache)
        torch.save(mean_cache, output_dir / "mean_cache.pt")
    
    # Train
    print(f"\nTraining for {num_steps} steps...")
    trainer = PruningTrainer(masked_model, task, config)
    
    def log_fn(metrics):
        print(f"Step {metrics['step']}: loss={metrics['task_loss']:.4f}, "
              f"active={metrics['num_active_nodes']}, acc={metrics['accuracy']:.2%}")
    
    final_metrics = trainer.train(log_fn=log_fn, show_progress=True)
    
    # Save checkpoint
    trainer.save_checkpoint(output_dir / "checkpoint.pt")
    
    # Get tau distribution stats
    all_taus = []
    for mask in masked_model.masks.masks.values():
        all_taus.extend(mask.tau.detach().cpu().numpy().tolist())
    all_taus = np.array(all_taus)
    
    tau_stats = {
        "mean": float(np.mean(all_taus)),
        "std": float(np.std(all_taus)),
        "min": float(np.min(all_taus)),
        "max": float(np.max(all_taus)),
        "near_zero": int(np.sum(np.abs(all_taus) < 0.1)),
        "near_boundary": int(np.sum((all_taus > -0.5) & (all_taus < 0.5))),
    }
    
    # Sweep target losses
    print("\nSweeping target losses...")
    target_losses = np.logspace(np.log10(0.001), np.log10(0.5), 8).tolist()
    
    sweep_results = sweep_target_losses(
        masked_model=masked_model,
        task=task,
        config=config,
        target_losses=target_losses,
        num_eval_batches=20,
        show_progress=True,
    )
    
    total_nodes = masked_model.masks.get_total_nodes()
    
    # Save results
    results = {
        "num_steps": num_steps,
        "final_metrics": final_metrics,
        "tau_stats": tau_stats,
        "total_nodes": total_nodes,
        "sweep_results": [
            {"target_loss": t, "circuit_size": s, "achieved_loss": a}
            for t, s, a in sweep_results
        ],
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    plot_loss_vs_circuit_size(
        results=sweep_results,
        total_nodes=total_nodes,
        output_path=str(output_dir / "loss_vs_circuit_size.png"),
        title=f"Loss vs Circuit Size ({num_steps} steps)",
    )
    
    # Also save tau histogram
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_taus, bins=50, range=(-1, 1), alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    ax.set_xlabel('tau value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Tau Distribution ({num_steps} steps)\n'
                 f'Active: {final_metrics["num_active_nodes"]}/{total_nodes}, '
                 f'Loss: {final_metrics["task_loss"]:.4f}', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tau_distribution.png", dpi=150)
    plt.close()
    
    print(f"\nExperiment complete!")
    print(f"  Final loss: {final_metrics['task_loss']:.4f}")
    print(f"  Active nodes: {final_metrics['num_active_nodes']}/{total_nodes}")
    print(f"  Results saved to: {output_dir}")
    
    # Clean up
    del masked_model, model, trainer
    torch.cuda.empty_cache()
    
    return mean_cache, tokenizer_name, results


def main():
    model_path = "jacobcd52/ss_bridges_d1024_f0.015625"
    base_output_dir = Path("my_sparse_pretrain/outputs/paper_hparams")
    
    step_counts = [500, 2000, 20000]
    
    mean_cache = None
    tokenizer_name = None
    all_results = {}
    
    for num_steps in step_counts:
        output_dir = base_output_dir / f"steps_{num_steps}"
        
        mean_cache, tokenizer_name, results = run_experiment(
            model_path=model_path,
            num_steps=num_steps,
            output_dir=output_dir,
            mean_cache=mean_cache,
            tokenizer_name=tokenizer_name,
        )
        
        all_results[num_steps] = results
    
    # Save combined summary
    with open(base_output_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"\n{'Steps':<10} {'Loss':>10} {'Active':>10} {'Circuit @0.01':>15}")
    print("-"*50)
    
    for steps, res in all_results.items():
        circuit_at_001 = "N/A"
        for sr in res["sweep_results"]:
            if sr["target_loss"] <= 0.015:  # Close to 0.01
                circuit_at_001 = sr["circuit_size"]
                break
        print(f"{steps:<10} {res['final_metrics']['task_loss']:>10.4f} "
              f"{res['final_metrics']['num_active_nodes']:>10} {circuit_at_001:>15}")
    
    print(f"\nAll results saved to: {base_output_dir}")


if __name__ == "__main__":
    main()

