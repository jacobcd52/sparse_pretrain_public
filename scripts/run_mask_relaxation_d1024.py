#!/usr/bin/env python3
"""
Run mask relaxation evaluation for d1024 models with log-spaced fractions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from my_sparse_pretrain.scripts.run_all_evals import load_model_checkpoint
from my_sparse_pretrain.src.pruning.interchange_eval import (
    MaskRelaxationConfig,
    run_mask_relaxation_evaluation,
    plot_mask_relaxation_results,
    save_mask_relaxation_results,
    get_masked_node_indices,
    get_masked_token_indices,
    run_masked_forward_with_relaxation,
    create_relaxation_mask,
    compute_task_loss_from_logits,
    MaskRelaxationResult,
)
from my_sparse_pretrain.src.pruning.tasks import get_task

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Get d1024 model directories
carbs_dir = Path('my_sparse_pretrain/outputs/carbs_results_pronoun')
model_dirs = [d for d in carbs_dir.iterdir() 
              if d.is_dir() and (d / 'best_checkpoint').exists() and 'd1024' in d.name]
model_dirs = sorted(model_dirs)

print(f'Found {len(model_dirs)} d1024 models:')
for d in model_dirs:
    print(f'  - {d.name}')
print()

# Create log-spaced fractions: 0, then log-spaced from 0.01 to 1.0
internal_fractions = np.logspace(-2, 0, 19).tolist()  # 19 points from 0.01 to 1.0
fractions = [0.0] + internal_fractions[:-1] + [1.0]  # 20 points total
print(f'Using {len(fractions)} log-spaced fractions')
print(f'Fractions: {[round(f, 4) for f in fractions]}')
print()


def run_eval_for_model(model_dir):
    print(f'\n{"="*70}')
    print(f'Processing: {model_dir.name}')
    print(f'{"="*70}')
    
    evals_dir = model_dir / 'best_checkpoint' / 'evals'
    evals_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        print('  Loading model...')
        (
            masked_model, base_model, tokenizer,
            sweep_config, hparams, mean_cache, pruning_config
        ) = load_model_checkpoint(model_dir, device)
        
        # Get task
        task_name = sweep_config['task_name']
        task = get_task(task_name, tokenizer, seed=42, split='superval')
        
        masked_model.eval()
        rng = np.random.default_rng(42)
        
        # Get masked node/token indices
        masked_indices = get_masked_node_indices(masked_model)
        num_masked_nodes = sum(len(v) for v in masked_indices.values())
        
        masked_token_indices = get_masked_token_indices(masked_model)
        num_masked_tokens = len(masked_token_indices) if masked_token_indices is not None else 0
        vocab_size = masked_model.vocab_size if masked_model.token_mask is not None else None
        
        num_masked = num_masked_nodes + num_masked_tokens
        
        # Get circuit size
        circuit_size = masked_model.masks.get_total_active_nodes()
        if masked_model.token_mask is not None:
            circuit_size += masked_model.token_mask.get_num_active()
        
        node_counts = {key: mask.tau.shape[0] for key, mask in masked_model.masks.masks.items()}
        total_nodes = sum(node_counts.values())
        if vocab_size is not None:
            total_nodes += vocab_size
        
        print(f'  Circuit: {circuit_size:,} active, {num_masked:,} masked ({num_masked_nodes} nodes + {num_masked_tokens} tokens)')
        
        # Generate fixed batches
        config = MaskRelaxationConfig(
            num_points=len(fractions),
            num_trials=10,
            num_batches=10,
            batch_size=pruning_config.batch_size,
            seq_length=pruning_config.seq_length,
            device=device,
        )
        
        fixed_batches = []
        for _ in range(config.num_batches):
            batch = task.generate_batch(
                batch_size=config.batch_size,
                max_length=config.seq_length,
            )
            positive_ids, negative_ids, correct_tokens, incorrect_tokens, eval_positions = batch
            fixed_batches.append({
                'positive_ids': positive_ids.to(device),
                'correct_tokens': correct_tokens.to(device),
                'eval_positions': eval_positions.to(device),
            })
        
        def compute_loss(relaxation_mask, token_relaxation_mask=None):
            losses = []
            with torch.no_grad():
                for batch in fixed_batches:
                    input_ids = batch['positive_ids']
                    correct_tokens = batch['correct_tokens']
                    eval_positions = batch['eval_positions']
                    
                    logits = run_masked_forward_with_relaxation(
                        masked_model, input_ids, relaxation_mask, token_relaxation_mask
                    )
                    
                    loss = compute_task_loss_from_logits(logits, correct_tokens, eval_positions)
                    losses.append(loss)
            return np.mean(losses)
        
        # Compute baselines
        print('  Computing baselines...')
        empty_node = {key: torch.zeros(count, dtype=torch.bool, device=device) for key, count in node_counts.items()}
        empty_token = torch.zeros(vocab_size, dtype=torch.bool, device=device) if vocab_size else None
        
        with torch.autocast(device, dtype=torch.bfloat16):
            circuit_only_loss = compute_loss(empty_node, empty_token)
        
        full_node = {key: torch.zeros(count, dtype=torch.bool, device=device) for key, count in node_counts.items()}
        for key in masked_indices:
            full_node[key][masked_indices[key]] = True
        full_token = None
        if vocab_size is not None and masked_token_indices is not None and len(masked_token_indices) > 0:
            full_token = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            full_token[masked_token_indices] = True
        
        with torch.autocast(device, dtype=torch.bfloat16):
            all_active_loss = compute_loss(full_node, full_token)
        
        print(f'  Circuit-only: {circuit_only_loss:.4f}, All-active: {all_active_loss:.4f}')
        
        # Sweep
        relaxation_results = {}
        print(f'  Running sweep ({len(fractions)} points, {config.num_trials} trials)...')
        
        with torch.autocast(device, dtype=torch.bfloat16):
            for frac in tqdm(fractions, desc='  Sweep', leave=False):
                losses = []
                for trial in range(config.num_trials):
                    node_mask, token_mask = create_relaxation_mask(
                        masked_indices, node_counts, frac, rng, device,
                        masked_token_indices=masked_token_indices,
                        vocab_size=vocab_size,
                    )
                    loss = compute_loss(node_mask, token_mask)
                    losses.append(loss)
                relaxation_results[frac] = (np.mean(losses), np.std(losses))
        
        result = MaskRelaxationResult(
            circuit_only_loss=circuit_only_loss,
            all_active_loss=all_active_loss,
            relaxation_results=relaxation_results,
            circuit_size=circuit_size,
            num_masked=num_masked,
            total_nodes=total_nodes,
            num_trials=config.num_trials,
            fractions=fractions,
        )
        
        # Save
        ablation_type = sweep_config.get('ablation_type', 'mean_pretrain')
        ablation_str = 'zero' if ablation_type == 'zero' else 'mean'
        title_prefix = f'{model_dir.name} ({ablation_str}_ablate)\n'
        
        plot_path = evals_dir / 'mask_relaxation_sweep.png'
        plot_mask_relaxation_results(result, plot_path, title_prefix=title_prefix)
        
        json_path = evals_dir / 'mask_relaxation_results.json'
        save_mask_relaxation_results(result, json_path)
        
        print(f'  Saved to {evals_dir}')
        
        # Free memory
        del masked_model, base_model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    successes = 0
    for model_dir in model_dirs:
        if run_eval_for_model(model_dir):
            successes += 1
    
    print(f'\n{"="*70}')
    print(f'COMPLETE: {successes}/{len(model_dirs)} models processed')
    print(f'{"="*70}')


