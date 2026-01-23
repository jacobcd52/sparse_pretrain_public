#!/usr/bin/env python3
"""
Generate circuit HTMLs for all models in carbs_results_pronoun.
Saves to each model's folder with an informative name.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
from tqdm import tqdm

from transformers import AutoTokenizer

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model

from my_sparse_pretrain.scripts.run_single_pruning import generate_html_with_dashboards


def get_informative_name(model_name: str) -> str:
    """Generate an informative filename from the model directory name."""
    parts = model_name.split('_')
    
    is_bridges = 'bridges' in model_name
    is_noembed = 'noembed' in model_name
    
    # Find d_model
    d_model = None
    for p in parts:
        if p.startswith('d') and p[1:].isdigit():
            d_model = p[1:]
            break
    
    ablation = 'mean' if 'mean' in model_name else 'zero'
    model_type = 'bridges' if is_bridges else 'base'
    embed_type = 'noembed' if is_noembed else 'embed'
    
    return f"circuit_{model_type}_d{d_model}_{ablation}_{embed_type}.html"


def generate_circuit_html_for_model(
    result_dir: Path,
    device: str = "cuda",
    max_nodes: int = 500,
    n_dashboard_samples: int = 100,
) -> Path:
    """Generate circuit HTML for a single model."""
    
    model_name = result_dir.name
    output_name = get_informative_name(model_name)
    output_path = result_dir / output_name
    
    # Load configs
    with open(result_dir / "sweep_config.json") as f:
        sweep_config = json.load(f)
    
    best_checkpoint_dir = result_dir / "best_checkpoint"
    with open(best_checkpoint_dir / "hparams.json") as f:
        hparams = json.load(f)
    
    # Check if noembed mode
    is_noembed = "noembed" in model_name
    
    # Load model
    model, _ = load_model(sweep_config["model_path"], device=device)
    tokenizer = AutoTokenizer.from_pretrained(sweep_config["tokenizer_name"])
    task = get_task(sweep_config["task_name"], tokenizer, seed=42, split="val")
    
    # Create pruning config
    pruning_config = PruningConfig(
        k_coef=hparams["k_coef"],
        weight_decay=hparams["weight_decay"],
        lr=hparams["lr"],
        beta2=hparams["beta2"],
        heaviside_temp=hparams["heaviside_temp"],
        device=device,
        ablation_type=sweep_config.get("ablation_type", "zero"),
        mask_token_embeds=not is_noembed,
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, pruning_config)
    masked_model.to(device)
    
    # Load trained masks
    mask_state = torch.load(
        best_checkpoint_dir / "masks.pt",
        map_location=device,
        weights_only=True
    )
    masked_model.load_mask_state(mask_state)
    
    # Generate HTML
    # Note: task is always passed to extract target tokens for unembedding nodes
    # No bisection - use all training nodes as-is
    generate_html_with_dashboards(
        masked_model=masked_model,
        tokenizer=tokenizer,
        output_path=output_path,
        max_nodes=max_nodes,
        n_dashboard_samples=n_dashboard_samples,
        n_top_examples=10,
        device=device,
        task=task,
        target_loss=None,
    )
    
    return output_path


def main():
    results_base = Path("my_sparse_pretrain/outputs/carbs_results_pronoun")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Find all model directories
    model_dirs = sorted([
        d for d in results_base.iterdir() 
        if d.is_dir() and d.name.startswith("ss_") 
        and (d / "best_checkpoint" / "masks.pt").exists()
    ])
    
    print(f"Found {len(model_dirs)} model directories with checkpoints:")
    for d in model_dirs:
        print(f"  - {d.name}")
    print()
    
    generated_htmls = []
    for result_dir in tqdm(model_dirs, desc="Generating HTMLs"):
        try:
            html_path = generate_circuit_html_for_model(
                result_dir,
                device=device,
                max_nodes=500,
                n_dashboard_samples=100,
            )
            generated_htmls.append(html_path)
            print(f"  ✓ {html_path.name}")
        except Exception as e:
            print(f"\n  ✗ ERROR processing {result_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Generated {len(generated_htmls)} circuit HTMLs:")
    for path in sorted(generated_htmls):
        print(f"  {path.parent.name}/{path.name}")


if __name__ == "__main__":
    main()
