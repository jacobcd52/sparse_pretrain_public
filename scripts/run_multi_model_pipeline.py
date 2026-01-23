#!/usr/bin/env python3
"""
Run the full evaluation pipeline for multiple models in the background.

This script orchestrates running the full evaluation pipeline
(CARBS + HTML + Interchange + Pareto) for a list of models.

Usage:
    # Run for default model list
    python scripts/run_multi_model_pipeline.py
    
    # Run for specific models
    python scripts/run_multi_model_pipeline.py \
        --models model1 model2 model3
    
    # Skip CARBS for all models (use existing checkpoints)
    python scripts/run_multi_model_pipeline.py --skip-carbs
    
    # Only run interchange evaluation for all models
    python scripts/run_multi_model_pipeline.py --skip-carbs --only interchange
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import subprocess
import time
from datetime import datetime
from typing import List, Optional


# Default list of models to evaluate
DEFAULT_MODELS = [
    "jacobcd52/ss_bridges_d1024_f0.015625",
    "jacobcd52/ss_bridges_d3072_f0.005",
    "jacobcd52/ss_bridges_d4096_f0.002",
]


def run_single_model(
    model_path: str,
    output_dir: str,
    skip_carbs: bool = False,
    only_eval: Optional[str] = None,
    carbs_runs: int = 32,
    carbs_steps: int = 500,
    target_loss: float = 0.15,
    interchange_fractions: int = 3,
    interchange_trials: int = 3,
    interchange_batches: int = 10,
    device: str = "cuda",
    no_wandb: bool = False,
    task_name: str = "dummy_pronoun",
    background: bool = False,
) -> Optional[subprocess.Popen]:
    """
    Run the full evaluation pipeline for a single model.
    
    If background=True, returns the subprocess.Popen object.
    Otherwise, waits for completion and returns None.
    """
    cmd = [
        sys.executable,
        "scripts/run_full_evaluation.py",
        "--model", model_path,
        "--output-dir", output_dir,
        "--carbs-runs", str(carbs_runs),
        "--carbs-steps", str(carbs_steps),
        "--target-loss", str(target_loss),
        "--interchange-fractions", str(interchange_fractions),
        "--interchange-trials", str(interchange_trials),
        "--interchange-batches", str(interchange_batches),
        "--device", device,
        "--task", task_name,
    ]
    
    if skip_carbs:
        cmd.append("--skip-carbs")
    if only_eval:
        cmd.extend(["--only", only_eval])
    if no_wandb:
        cmd.append("--no-wandb")
    
    model_short_name = model_path.split("/")[-1]
    log_dir = Path(output_dir) / model_short_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"\n{'='*70}")
    print(f"Starting pipeline for: {model_short_name}")
    print(f"Log file: {log_file}")
    print(f"{'='*70}")
    
    if background:
        # Run in background with output to log file
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.parent.parent),
            )
        print(f"Started process {process.pid}")
        return process
    else:
        # Run and wait for completion
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
        )
        
        # Save output to log file
        with open(log_file, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
        
        if result.returncode != 0:
            print(f"ERROR: Pipeline failed for {model_short_name}")
            print(f"See log file: {log_file}")
        else:
            print(f"SUCCESS: Pipeline completed for {model_short_name}")
        
        return None


def run_all_models(
    models: List[str],
    output_dir: str,
    skip_carbs: bool = False,
    only_eval: Optional[str] = None,
    carbs_runs: int = 32,
    carbs_steps: int = 500,
    target_loss: float = 0.15,
    interchange_fractions: int = 3,
    interchange_trials: int = 3,
    interchange_batches: int = 10,
    device: str = "cuda",
    no_wandb: bool = False,
    task_name: str = "dummy_pronoun",
    parallel: bool = False,
    max_parallel: int = 1,
):
    """
    Run the full evaluation pipeline for multiple models.
    
    Args:
        models: List of model paths
        output_dir: Base output directory
        parallel: If True, run models in parallel (up to max_parallel)
        max_parallel: Maximum number of parallel processes
        ... other args passed to run_single_model
    """
    print("\n" + "="*70)
    print("MULTI-MODEL EVALUATION PIPELINE")
    print("="*70)
    print(f"Models to process: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print(f"Output directory: {output_dir}")
    print(f"Skip CARBS: {skip_carbs}")
    print(f"Only eval: {only_eval or 'all'}")
    print(f"Parallel: {parallel} (max {max_parallel})")
    print("="*70)
    
    results = {}
    start_time = time.time()
    
    if parallel:
        # Run models in parallel
        active_processes = {}  # model_path -> Popen
        completed = set()
        
        model_queue = list(models)
        
        while model_queue or active_processes:
            # Start new processes up to max_parallel
            while len(active_processes) < max_parallel and model_queue:
                model_path = model_queue.pop(0)
                process = run_single_model(
                    model_path=model_path,
                    output_dir=output_dir,
                    skip_carbs=skip_carbs,
                    only_eval=only_eval,
                    carbs_runs=carbs_runs,
                    carbs_steps=carbs_steps,
                    target_loss=target_loss,
                    interchange_fractions=interchange_fractions,
                    interchange_trials=interchange_trials,
                    interchange_batches=interchange_batches,
                    device=device,
                    no_wandb=no_wandb,
                    task_name=task_name,
                    background=True,
                )
                active_processes[model_path] = process
            
            # Check for completed processes
            completed_this_round = []
            for model_path, process in active_processes.items():
                if process.poll() is not None:  # Process has finished
                    completed_this_round.append(model_path)
                    model_short_name = model_path.split("/")[-1]
                    if process.returncode == 0:
                        print(f"\n✓ Completed: {model_short_name}")
                        results[model_path] = "success"
                    else:
                        print(f"\n✗ Failed: {model_short_name} (exit code {process.returncode})")
                        results[model_path] = f"failed (exit {process.returncode})"
            
            for model_path in completed_this_round:
                del active_processes[model_path]
                completed.add(model_path)
            
            if active_processes:
                time.sleep(10)  # Check every 10 seconds
    else:
        # Run models sequentially
        for model_path in models:
            model_start = time.time()
            try:
                run_single_model(
                    model_path=model_path,
                    output_dir=output_dir,
                    skip_carbs=skip_carbs,
                    only_eval=only_eval,
                    carbs_runs=carbs_runs,
                    carbs_steps=carbs_steps,
                    target_loss=target_loss,
                    interchange_fractions=interchange_fractions,
                    interchange_trials=interchange_trials,
                    interchange_batches=interchange_batches,
                    device=device,
                    no_wandb=no_wandb,
                    task_name=task_name,
                    background=False,
                )
                results[model_path] = "success"
            except Exception as e:
                results[model_path] = f"error: {e}"
                print(f"ERROR: {e}")
            
            model_duration = time.time() - model_start
            print(f"Duration: {model_duration/60:.1f} minutes")
    
    total_duration = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"Total duration: {total_duration/60:.1f} minutes")
    print(f"\nResults:")
    for model_path, result in results.items():
        model_short = model_path.split("/")[-1]
        status = "✓" if result == "success" else "✗"
        print(f"  {status} {model_short}: {result}")
    
    # Save summary
    summary_path = Path(output_dir) / "multi_model_summary.json"
    summary = {
        "models": models,
        "results": results,
        "total_duration_minutes": total_duration / 60,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "skip_carbs": skip_carbs,
            "only_eval": only_eval,
            "carbs_runs": carbs_runs,
            "carbs_steps": carbs_steps,
            "target_loss": target_loss,
        }
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation pipeline for multiple models"
    )
    
    # Model selection
    parser.add_argument("--models", nargs="+", default=None,
                       help="List of model paths (default: use DEFAULT_MODELS)")
    
    # Pipeline control
    parser.add_argument("--skip-carbs", action="store_true",
                       help="Skip CARBS tuning for all models")
    parser.add_argument("--only", type=str, default=None,
                       choices=["interchange", "html", "pareto"],
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
    
    # Output settings
    parser.add_argument("--output-dir", type=str,
                       default="outputs/carbs_results_pronoun",
                       help="Base output directory")
    
    # Execution control
    parser.add_argument("--parallel", action="store_true",
                       help="Run models in parallel")
    parser.add_argument("--max-parallel", type=int, default=1,
                       help="Maximum parallel processes (default: 1)")
    
    # Device and logging
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (default: cuda)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable wandb logging")
    
    args = parser.parse_args()
    
    models = args.models if args.models else DEFAULT_MODELS
    
    run_all_models(
        models=models,
        output_dir=args.output_dir,
        skip_carbs=args.skip_carbs,
        only_eval=args.only,
        carbs_runs=args.carbs_runs,
        carbs_steps=args.carbs_steps,
        target_loss=args.target_loss,
        interchange_fractions=args.interchange_fractions,
        interchange_trials=args.interchange_trials,
        interchange_batches=args.interchange_batches,
        device=args.device,
        no_wandb=args.no_wandb,
        task_name=args.task,
        parallel=args.parallel,
        max_parallel=args.max_parallel,
    )


if __name__ == "__main__":
    main()

