#!/usr/bin/env python3
"""
Run full evaluation pipeline for a pruned model.

This script:
1. Runs CARBS hyperparameter tuning (optional, if not already done)
2. Generates circuit HTML visualization for the best checkpoint
3. Runs interchange intervention evaluation
4. Runs Pareto curve evaluation on superval set

All model-specific outputs are saved to the best_checkpoint folder of the CARBS output.

Usage:
    # Run full pipeline including CARBS
    python scripts/run_full_evaluation.py --model MODEL_PATH
    
    # Skip CARBS (use existing best checkpoint)
    python scripts/run_full_evaluation.py --model MODEL_PATH --skip-carbs
    
    # Only run specific evaluations on existing checkpoint
    python scripts/run_full_evaluation.py --model MODEL_PATH --skip-carbs --only interchange
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
from datetime import datetime
from contextlib import nullcontext
from typing import Optional, Dict, Any

from transformers import AutoTokenizer

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator
from sparse_pretrain.src.pruning.interchange_eval import (
    InterchangeEvalConfig,
    run_and_save_interchange_eval,
    AblationSweepConfig,
    run_and_save_ablation_sweep,
)

# Import CARBS sweep
from sparse_pretrain.scripts.run_carbs_clean import (
    CleanSweepConfig, 
    run_carbs_sweep,
    create_pareto_plot_superval,
)

# Import HTML generation
from sparse_pretrain.scripts.run_single_pruning import generate_html_with_dashboards


def load_best_checkpoint(
    result_dir: Path,
    device: str = "cuda",
) -> tuple:
    """
    Load the best checkpoint from a CARBS result directory.
    
    Returns:
        Tuple of (masked_model, base_model, tokenizer, task, sweep_config, hparams, mean_cache)
    """
    # Load sweep config
    with open(result_dir / "sweep_config.json") as f:
        sweep_config = json.load(f)
    
    # Load best checkpoint info
    best_checkpoint_dir = result_dir / "best_checkpoint"
    if not best_checkpoint_dir.exists():
        raise FileNotFoundError(f"No best_checkpoint found in {result_dir}")
    
    with open(best_checkpoint_dir / "summary.json") as f:
        summary = json.load(f)
    with open(best_checkpoint_dir / "hparams.json") as f:
        hparams = json.load(f)
    
    # Load model
    model_path = sweep_config["model_path"]
    tokenizer_name = sweep_config["tokenizer_name"]
    task_name = sweep_config["task_name"]
    
    print(f"Loading model: {model_path}")
    model, model_config = load_model(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load mean cache if exists
    mean_cache_path = result_dir / "mean_cache.pt"
    if mean_cache_path.exists():
        mean_cache = torch.load(mean_cache_path, map_location=device, weights_only=True)
    else:
        mean_cache = {}
    
    # Create pruning config
    pruning_config = PruningConfig(
        k_coef=hparams["k_coef"],
        init_noise_scale=sweep_config.get("init_noise_scale", 0.01),
        init_noise_bias=sweep_config.get("init_noise_bias", 0.1),
        weight_decay=hparams["weight_decay"],
        lr=hparams["lr"],
        beta2=hparams["beta2"],
        lr_warmup_frac=sweep_config.get("lr_warmup_frac", 0.0),
        heaviside_temp=hparams["heaviside_temp"],
        num_steps=sweep_config.get("num_steps", 500),
        batch_size=sweep_config.get("batch_size", 64),
        seq_length=sweep_config.get("task_max_length", 0),
        device=device,
        ablation_type=sweep_config.get("ablation_type", "mean_pretrain"),
        mask_token_embeds=sweep_config.get("mask_token_embeds", False),
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if sweep_config.get("ablation_type", "mean_pretrain") != "zero" and mean_cache:
        masked_model.set_means_from_dict(mean_cache)
    
    # Load masks
    masks_path = best_checkpoint_dir / "masks.pt"
    mask_state = torch.load(masks_path, map_location=device, weights_only=True)
    masked_model.load_mask_state(mask_state)
    
    # Load task
    task = get_task(task_name, tokenizer, seed=42, split="superval")
    
    return masked_model, model, tokenizer, task, sweep_config, hparams, mean_cache, pruning_config


def run_circuit_html_generation(
    masked_model: MaskedSparseGPT,
    tokenizer,
    task,
    output_dir: Path,
    target_loss: float = 0.15,
    max_nodes: int = 500,
    n_dashboard_samples: int = 200,
    device: str = "cuda",
) -> Path:
    """Generate circuit HTML visualization."""
    print("\n" + "="*60)
    print("Generating Circuit HTML Visualization")
    print("="*60)
    
    output_path = output_dir / "circuit.html"
    
    html_path = generate_html_with_dashboards(
        masked_model=masked_model,
        tokenizer=tokenizer,
        output_path=output_path,
        max_nodes=max_nodes,
        n_dashboard_samples=n_dashboard_samples,
        n_top_examples=10,
        device=device,
        task=task,
        target_loss=target_loss,
    )
    
    print(f"Circuit HTML saved to: {html_path}")
    size_mb = html_path.stat().st_size / (1024 * 1024)
    print(f"HTML file size: {size_mb:.2f} MB")
    
    return html_path


def run_interchange_evaluation_full(
    masked_model: MaskedSparseGPT,
    base_model,
    task,
    output_dir: Path,
    pruning_config: PruningConfig,
    model_name: str = "",
    num_fractions: int = 3,  # Default to 3 for testing (0%, 50%, 100%)
    num_trials: int = 3,
    num_batches: int = 10,
    device: str = "cuda",
):
    """Run interchange intervention evaluation."""
    print("\n" + "="*60)
    print("Running Interchange Intervention Evaluation")
    print("="*60)
    
    # Create fractions list
    fractions = [i / (num_fractions - 1) for i in range(num_fractions)]
    
    config = InterchangeEvalConfig(
        fractions=fractions,
        num_trials=num_trials,
        num_batches=num_batches,
        batch_size=pruning_config.batch_size,
        seq_length=pruning_config.seq_length,
        device=device,
    )
    
    # Create title prefix with model name and ablation strategy
    title_prefix = f"{model_name} (zero_ablate)\n" if model_name else ""
    
    result = run_and_save_interchange_eval(
        masked_model=masked_model,
        base_model=base_model,
        task=task,
        output_dir=output_dir,
        config=config,
        pruning_config=pruning_config,
        title_prefix=title_prefix,
        show_progress=True,
    )
    
    return result


def run_full_pipeline(
    model_path: str,
    output_base_dir: str = "outputs/carbs_results_pronoun",
    skip_carbs: bool = False,
    only_eval: Optional[str] = None,
    carbs_num_runs: int = 32,
    carbs_steps: int = 500,
    target_loss: float = 0.15,
    interchange_fractions: int = 3,
    interchange_trials: int = 3,
    interchange_batches: int = 10,
    ablation_points: int = 11,
    ablation_trials: int = 10,
    ablation_batches: int = 10,
    device: str = "cuda",
    use_wandb: bool = True,
    task_name: str = "dummy_pronoun",
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline for a model.
    
    Args:
        model_path: Path to the model (HuggingFace or local)
        output_base_dir: Base directory for outputs
        skip_carbs: Skip CARBS tuning (use existing checkpoint)
        only_eval: Only run specific evaluation ("interchange", "html", "pareto", or None for all)
        carbs_num_runs: Number of CARBS runs
        carbs_steps: Training steps per CARBS run
        target_loss: Target loss for evaluation
        interchange_fractions: Number of fraction points for interchange eval
        interchange_trials: Number of trials per fraction
        interchange_batches: Number of batches per trial
        device: Device to use
        use_wandb: Whether to log to wandb
        task_name: Name of the task
        tokenizer_name: Name of the tokenizer
        
    Returns:
        Dictionary with all results
    """
    model_short_name = model_path.split("/")[-1]
    result_dir = Path(output_base_dir) / model_short_name
    best_checkpoint_dir = result_dir / "best_checkpoint"
    
    results = {"model": model_path}
    
    # Step 1: CARBS hyperparameter tuning
    if not skip_carbs:
        print("\n" + "="*70)
        print("STEP 1: CARBS Hyperparameter Tuning")
        print("="*70)
        
        carbs_config = CleanSweepConfig(
            model_path=model_path,
            tokenizer_name=tokenizer_name,
            task_name=task_name,
            num_runs=carbs_num_runs,
            num_steps=carbs_steps,
            target_loss=target_loss,
            output_base_dir=output_base_dir,
            device=device,
            use_wandb=use_wandb,
        )
        
        carbs_results_pronoun = run_carbs_sweep(carbs_config)
        results["carbs"] = carbs_results_pronoun
        
        if carbs_results_pronoun.get("best_result") is None:
            print("\nERROR: CARBS did not find any successful runs!")
            return results
    else:
        print(f"\nSkipping CARBS - using existing checkpoint at {result_dir}")
        if not best_checkpoint_dir.exists():
            raise FileNotFoundError(f"No best_checkpoint found at {best_checkpoint_dir}")
    
    # Load the best checkpoint
    print("\n" + "="*70)
    print("Loading Best Checkpoint")
    print("="*70)
    
    (
        masked_model, base_model, tokenizer, task, 
        sweep_config, hparams, mean_cache, pruning_config
    ) = load_best_checkpoint(result_dir, device)
    
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    print(f"Loaded checkpoint: {num_active:,} / {total_nodes:,} active nodes")
    
    # Use best_checkpoint_dir for all model-specific outputs
    output_dir = best_checkpoint_dir
    
    # Step 2: Circuit HTML visualization
    if only_eval is None or only_eval == "html":
        print("\n" + "="*70)
        print("STEP 2: Circuit HTML Visualization")
        print("="*70)
        
        try:
            html_path = run_circuit_html_generation(
                masked_model=masked_model,
                tokenizer=tokenizer,
                task=task,
                output_dir=output_dir,
                target_loss=target_loss,
                device=device,
            )
            results["html_path"] = str(html_path)
        except Exception as e:
            print(f"WARNING: Circuit HTML generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3: Interchange intervention evaluation
    if only_eval is None or only_eval == "interchange":
        print("\n" + "="*70)
        print("STEP 3: Interchange Intervention Evaluation")
        print("="*70)
        
        try:
            with torch.autocast(device, dtype=torch.bfloat16):
                interchange_result = run_interchange_evaluation_full(
                    masked_model=masked_model,
                    base_model=base_model,
                    task=task,
                    output_dir=output_dir,
                    pruning_config=pruning_config,
                    model_name=model_short_name,
                    num_fractions=interchange_fractions,
                    num_trials=interchange_trials,
                    num_batches=interchange_batches,
                    device=device,
                )
            results["interchange"] = {
                "unpruned_clean_loss": interchange_result.unpruned_clean_loss,
                "fvu_global": interchange_result.fvu_global,
                "fvu_per_layer": {
                    str(k): v for k, v in interchange_result.fvu_per_layer.items()
                },
                "all_layers_results": {
                    str(k): {"mean": v[0], "std": v[1]} 
                    for k, v in interchange_result.all_layers_results.items()
                },
            }
        except Exception as e:
            print(f"WARNING: Interchange evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 3b: Ablation Sweep Evaluation
    if only_eval is None or only_eval == "ablation":
        print("\n" + "="*70)
        print("STEP 3b: Ablation Sweep Evaluation")
        print("="*70)
        
        try:
            ablation_config = AblationSweepConfig(
                num_points=ablation_points,
                num_trials=ablation_trials,
                num_batches=ablation_batches,
                batch_size=pruning_config.batch_size,
                seq_length=pruning_config.seq_length,
                device=device,
            )
            
            title_prefix = f"{model_short_name} (zero_ablate)\n" if model_short_name else ""
            
            with torch.autocast(device, dtype=torch.bfloat16):
                ablation_result = run_and_save_ablation_sweep(
                    masked_model=masked_model,
                    base_model=base_model,
                    task=task,
                    output_dir=output_dir,
                    config=ablation_config,
                    pruning_config=pruning_config,
                    mean_cache=mean_cache,
                    title_prefix=title_prefix,
                    show_progress=True,
                )
            results["ablation_sweep"] = {
                "clean_loss": ablation_result.clean_loss,
                "circuit_ablated_loss": ablation_result.circuit_ablated_loss,
                "circuit_size": ablation_result.circuit_size,
            }
        except Exception as e:
            print(f"WARNING: Ablation sweep evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 4: Pareto curve on superval (if not already done by CARBS)
    if only_eval is None or only_eval == "pareto":
        pareto_file = result_dir / "pareto_superval_data.json"
        if not pareto_file.exists():
            print("\n" + "="*70)
            print("STEP 4: Pareto Curve on Superval Set")
            print("="*70)
            
            try:
                # Need to recreate best_result dict for the function
                with open(best_checkpoint_dir / "summary.json") as f:
                    summary = json.load(f)
                best_result = {
                    "mask_state": masked_model.get_mask_state(),
                    "hparams": hparams,
                    "circuit_size": summary.get("circuit_size", num_active),
                }
                
                pareto_data = create_pareto_plot_superval(
                    best_result=best_result,
                    model=base_model,
                    tokenizer=tokenizer,
                    mean_cache=mean_cache,
                    config=CleanSweepConfig(
                        model_path=model_path,
                        tokenizer_name=tokenizer_name,
                        task_name=task_name,
                        target_loss=target_loss,
                        device=device,
                    ),
                    output_dir=result_dir,
                )
                results["pareto_superval"] = pareto_data
            except Exception as e:
                print(f"WARNING: Pareto evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nPareto data already exists at {pareto_file}")
    
    # Save combined results
    results_path = output_dir / "full_evaluation_results.json"
    
    # Filter out non-serializable items
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items() 
                    if not k.startswith("mask_state")}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_results = make_serializable(results)
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - full_evaluation_results.json")
    if "html_path" in results:
        print(f"  - circuit.html")
    if "interchange" in results:
        print(f"  - interchange_*.png")
        print(f"  - interchange_results.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run full evaluation pipeline for a pruned model"
    )
    
    # Required
    parser.add_argument("--model", type=str, required=True,
                       help="Model path (HuggingFace repo or local path)")
    
    # Pipeline control
    parser.add_argument("--skip-carbs", action="store_true",
                       help="Skip CARBS tuning (use existing checkpoint)")
    parser.add_argument("--only", type=str, default=None,
                       choices=["interchange", "ablation", "html", "pareto"],
                       help="Only run specific evaluation")
    
    # CARBS settings
    parser.add_argument("--carbs-runs", type=int, default=32,
                       help="Number of CARBS runs (default: 32)")
    parser.add_argument("--carbs-steps", type=int, default=500,
                       help="Training steps per CARBS run (default: 500)")
    
    # Task settings
    parser.add_argument("--task", type=str, default="dummy_pronoun",
                       help="Task name (default: dummy_pronoun)")
    parser.add_argument("--target-loss", type=float, default=0.15,
                       help="Target loss (default: 0.15)")
    
    # Interchange settings
    parser.add_argument("--interchange-fractions", type=int, default=3,
                       help="Number of fraction points for interchange (default: 3)")
    parser.add_argument("--interchange-trials", type=int, default=3,
                       help="Trials per fraction (default: 3)")
    parser.add_argument("--interchange-batches", type=int, default=10,
                       help="Batches per trial (default: 10)")
    
    # Ablation sweep settings
    parser.add_argument("--ablation-points", type=int, default=11,
                       help="Number of points in ablation sweep (default: 11)")
    parser.add_argument("--ablation-trials", type=int, default=10,
                       help="Trials per point (default: 10)")
    parser.add_argument("--ablation-batches", type=int, default=10,
                       help="Batches per trial (default: 10)")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, 
                       default="outputs/carbs_results_pronoun",
                       help="Base output directory")
    
    # Device and logging
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (default: cuda)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        model_path=args.model,
        output_base_dir=args.output_dir,
        skip_carbs=args.skip_carbs,
        only_eval=args.only,
        carbs_num_runs=args.carbs_runs,
        carbs_steps=args.carbs_steps,
        target_loss=args.target_loss,
        interchange_fractions=args.interchange_fractions,
        interchange_trials=args.interchange_trials,
        interchange_batches=args.interchange_batches,
        ablation_points=args.ablation_points,
        ablation_trials=args.ablation_trials,
        ablation_batches=args.ablation_batches,
        device=args.device,
        use_wandb=not args.no_wandb,
        task_name=args.task,
    )


if __name__ == "__main__":
    main()

