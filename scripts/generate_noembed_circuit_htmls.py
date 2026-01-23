#!/usr/bin/env python3
"""
Generate circuit HTML visualizations for all no_embed models in carbs_results_pronoun.

This script regenerates the circuit HTMLs for models where embedding nodes
were NOT pruned (mask_token_embeds=False), using the new logic that:
1. Gets tokens from the task dataset
2. Shows only residual channels in the intersection of upstream and downstream connections
3. Shows only embedding nodes that connect to these residual channels

Usage:
    python scripts/generate_noembed_circuit_htmls.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
from tqdm import tqdm

from transformers import AutoTokenizer

from sparse_pretrain.src.pruning.config import PruningConfig
from sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.run_pruning import load_model

# Import the HTML generation function from run_single_pruning
from sparse_pretrain.scripts.run_single_pruning import generate_html_with_dashboards


def generate_circuit_html_for_model(
    result_dir: Path,
    device: str = "cuda",
    max_nodes: int = 500,
    n_dashboard_samples: int = 200,
    use_bisection: bool = True,
) -> Path:
    """Generate circuit HTML for a single model from CARBS results."""
    
    print(f"\n{'='*60}")
    print(f"Processing: {result_dir.name}")
    print(f"{'='*60}")
    
    # Load sweep config
    with open(result_dir / "sweep_config.json") as f:
        sweep_config = json.load(f)
    
    # Load best checkpoint info
    best_checkpoint_dir = result_dir / "best_checkpoint"
    with open(best_checkpoint_dir / "summary.json") as f:
        summary = json.load(f)
    with open(best_checkpoint_dir / "hparams.json") as f:
        hparams = json.load(f)
    
    # Load model
    model_path = sweep_config["model_path"]
    tokenizer_name = sweep_config["tokenizer_name"]
    task_name = sweep_config["task_name"]
    target_loss = sweep_config.get("target_loss", 0.15)
    
    print(f"Loading model: {model_path}")
    model, model_config = load_model(model_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load mean cache if exists
    mean_cache_path = result_dir / "mean_cache.pt"
    if mean_cache_path.exists():
        mean_cache = torch.load(mean_cache_path, map_location=device, weights_only=True)
    else:
        mean_cache = {}
    
    # Load task for target tokens
    task = get_task(task_name, tokenizer, seed=42, split="val")
    print(f"Task: {task.name}")
    
    # Create pruning config - note mask_token_embeds should be False for noembed models
    mask_token_embeds = sweep_config.get("mask_token_embeds", True)
    print(f"mask_token_embeds: {mask_token_embeds}")
    
    pruning_config = PruningConfig(
        k_coef=hparams["k_coef"],
        init_noise_scale=sweep_config.get("init_noise_scale", 0.01),
        init_noise_bias=sweep_config.get("init_noise_bias", 0.1),
        weight_decay=hparams["weight_decay"],
        lr=hparams["lr"],
        beta2=hparams["beta2"],
        lr_warmup_frac=sweep_config.get("lr_warmup_frac", 0.0),
        heaviside_temp=hparams["heaviside_temp"],
        num_steps=sweep_config.get("num_steps", 1000),
        batch_size=sweep_config.get("batch_size", 64),
        seq_length=sweep_config.get("task_max_length", 0),
        device=device,
        ablation_type=sweep_config.get("ablation_type", "zero"),
        mask_token_embeds=mask_token_embeds,
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    if sweep_config.get("ablation_type", "zero") != "zero" and mean_cache:
        masked_model.set_means_from_dict(mean_cache)
    
    # Load masks
    masks_path = best_checkpoint_dir / "masks.pt"
    mask_state = torch.load(masks_path, map_location=device, weights_only=True)
    masked_model.load_mask_state(mask_state)
    
    # Get node counts
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    
    print(f"Total nodes: {total_nodes:,}")
    print(f"Active after training: {num_active:,}")
    print(f"Token mask present: {masked_model.token_mask is not None}")
    
    # Generate HTML
    output_path = result_dir / "circuit.html"
    print(f"Generating circuit HTML...")
    
    generate_html_with_dashboards(
        masked_model=masked_model,
        tokenizer=tokenizer,
        output_path=output_path,
        max_nodes=max_nodes,
        n_dashboard_samples=n_dashboard_samples,
        n_top_examples=10,
        device=device,
        task=task,
        target_loss=target_loss if use_bisection else None,
    )
    
    print(f"Circuit HTML saved to: {output_path}")
    
    # Clean up
    del masked_model, model
    torch.cuda.empty_cache()
    
    return output_path


def main():
    results_base = Path("outputs/carbs_results_pronoun")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find all directories with "noembed" in the name
    noembed_dirs = sorted([d for d in results_base.iterdir() 
                           if d.is_dir() and "noembed" in d.name.lower()])
    
    print(f"Found {len(noembed_dirs)} noembed model directories:")
    for d in noembed_dirs:
        print(f"  - {d.name}")
    
    # Process each model
    generated_htmls = []
    for result_dir in noembed_dirs:
        try:
            html_path = generate_circuit_html_for_model(
                result_dir, 
                device=device,
                max_nodes=500,
                n_dashboard_samples=200,
                use_bisection=True,  # Use bisection to find minimum circuit at target loss
            )
            generated_htmls.append(html_path)
        except Exception as e:
            print(f"ERROR processing {result_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Generated {len(generated_htmls)} circuit HTMLs:")
    for path in generated_htmls:
        print(f"  - {path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

