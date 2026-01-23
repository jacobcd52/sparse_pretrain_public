#!/usr/bin/env python3
"""
Regenerate circuit HTMLs for all models in carbs_results_ioi_relaxed.
This applies the updated dashboard generation with:
- Distinct contexts for top examples
- Larger window size (24 tokens)
- Fixed EOS token and hashtag display
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

from sparse_pretrain.scripts.run_single_pruning import generate_html_with_dashboards


def generate_circuit_html_for_model(
    result_dir: Path,
    device: str = "cuda",
    max_nodes: int = 500,
    n_dashboard_samples: int = 200,
) -> Path:
    """Generate circuit HTML for a single model."""
    
    model_name = result_dir.name
    output_path = result_dir / "circuit.html"
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")
    
    # Load configs
    with open(result_dir / "sweep_config.json") as f:
        sweep_config = json.load(f)
    
    best_checkpoint_dir = result_dir / "best_checkpoint"
    with open(best_checkpoint_dir / "hparams.json") as f:
        hparams = json.load(f)
    
    # Check if noembed mode
    is_noembed = "noembed" in model_name
    
    # Load model
    print(f"  Loading model: {sweep_config['model_path']}")
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
    
    # Get node counts
    num_active = masked_model.masks.get_total_active_nodes()
    print(f"  Active nodes: {num_active:,}")
    
    # Generate HTML
    print(f"  Generating circuit HTML with {n_dashboard_samples} dashboard samples...")
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
    
    # Clean up
    del masked_model, model
    torch.cuda.empty_cache()
    
    print(f"  ✓ Saved to: {output_path}")
    return output_path


def main():
    results_base = Path("outputs/carbs_results_ioi_relaxed")
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
    for result_dir in model_dirs:
        try:
            html_path = generate_circuit_html_for_model(
                result_dir,
                device=device,
                max_nodes=500,
                n_dashboard_samples=200,
            )
            generated_htmls.append(html_path)
        except Exception as e:
            print(f"\n  ✗ ERROR processing {result_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Generated {len(generated_htmls)} circuit HTMLs:")
    for path in sorted(generated_htmls):
        print(f"  {path.parent.name}/{path.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()


