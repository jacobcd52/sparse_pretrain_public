#!/usr/bin/env python3
"""
Run a single pruning experiment and generate a lightweight HTML visualization.

Usage:
    python my_sparse_pretrain/scripts/run_single_pruning.py --model MODEL --steps 500
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
from datetime import datetime
from contextlib import nullcontext

from transformers import AutoTokenizer

from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT
from my_sparse_pretrain.src.pruning.trainer import PruningTrainer
from my_sparse_pretrain.src.pruning.tasks import get_task
from my_sparse_pretrain.src.pruning.run_pruning import load_model, create_data_iterator


def compute_mean_cache_from_task(masked_model, task, num_batches=100, device="cuda"):
    """
    Compute mean activation cache from task data.
    
    Uses the positive examples from the task (same data used for training).
    """
    from tqdm import tqdm
    
    # Generate batches from the task
    def task_data_iterator():
        for _ in range(num_batches):
            positive_ids, _, _, _, _ = task.generate_batch(batch_size=64, max_length=0)
            yield positive_ids
    
    data_iter = task_data_iterator()
    mean_cache = masked_model.compute_mean_cache(data_iter, num_batches=num_batches, show_progress=True)
    return mean_cache


def main():
    parser = argparse.ArgumentParser(description="Run single pruning experiment")
    parser.add_argument("--model", type=str, default="jacobcd52/ss_bridges_d1024_f0.015625")
    parser.add_argument("--task", type=str, default="dummy_pronoun")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--k-coef", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--noise-bias", type=float, default=0.1)
    parser.add_argument("--heaviside-temp", type=float, default=1.0)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--target-loss", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None,
                       help="Custom wandb run name (default: auto-generated)")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    parser.add_argument("--max-html-nodes", type=int, default=500, 
                       help="Max nodes to include in HTML (for size reduction)")
    parser.add_argument("--pareto-every", type=int, default=100,
                       help="Log Pareto curve every N steps (0 to disable)")
    parser.add_argument("--ablation", type=str, default="mean_pretrain",
                       choices=["zero", "mean_pretrain", "mean_task"],
                       help="Ablation type: zero, mean_pretrain (SimpleStories), or mean_task")
    parser.add_argument("--mask-token-embeds", action="store_true",
                       help="Also learn a mask over vocabulary (token embeddings)")
    parser.add_argument("--binary-loss", action="store_true",
                       help="Use binary CE loss (for tense task)")
    parser.add_argument("--freeze-layernorm-scale", action="store_true",
                       help="Freeze layer norm outputs to unpruned values during forward pass")
    
    args = parser.parse_args()
    
    device = args.device
    model_name = args.model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"my_sparse_pretrain/outputs/single_run/{model_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Single Pruning Run")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Steps: {args.steps}")
    print(f"k_coef: {args.k_coef}")
    print(f"lr: {args.lr}")
    print(f"Ablation: {args.ablation}")
    print(f"Mask token embeds: {args.mask_token_embeds}")
    print(f"Freeze layer norm scale: {args.freeze_layernorm_scale}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Save config
    config_dict = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Load model
    print("\nLoading model...")
    model, _ = load_model(args.model, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "SimpleStories/SimpleStories-1.25M", 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create tasks
    train_task = get_task(args.task, tokenizer, seed=42, split="train")
    val_task = get_task(args.task, tokenizer, seed=42, split="val")
    print(f"Train task: {train_task.name} ({len(train_task.templates)} templates)")
    print(f"Val task: {val_task.name} ({len(val_task.templates)} templates)")
    
    # Create pruning config
    config = PruningConfig(
        k_coef=args.k_coef,
        init_noise_scale=args.noise_scale,
        init_noise_bias=args.noise_bias,
        weight_decay=args.weight_decay,
        lr=args.lr,
        beta2=args.beta2,
        lr_warmup_frac=0.0,
        heaviside_temp=args.heaviside_temp,
        num_steps=args.steps,
        batch_size=64,
        seq_length=0,  # Dynamic padding
        device=device,
        log_every=50,
        target_loss=args.target_loss,
        ablation_type=args.ablation,
        mask_token_embeds=args.mask_token_embeds,
        use_binary_loss=args.binary_loss,
        freeze_layernorm_scale=args.freeze_layernorm_scale,
    )
    
    # Create masked model
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)
    
    # Handle ablation type
    if config.ablation_type == "zero":
        print("\nUsing ZERO ablation (masked nodes → 0)")
    elif config.ablation_type == "mean_pretrain":
        print("\nComputing mean activation cache from SimpleStories...")
        data_iter = create_data_iterator(
            tokenizer_name="SimpleStories/SimpleStories-1.25M",
            batch_size=64,
            seq_length=256,
            num_batches=100,
            seed=42,
        )
        mean_cache = masked_model.compute_mean_cache(data_iter, num_batches=100, show_progress=True)
        masked_model.set_means_from_dict(mean_cache)
        print("Using MEAN_PRETRAIN ablation (masked nodes → mean over SimpleStories)")
    elif config.ablation_type == "mean_task":
        print("\nComputing mean activation cache from task data...")
        mean_cache = compute_mean_cache_from_task(masked_model, train_task, num_batches=100, device=device)
        masked_model.set_means_from_dict(mean_cache)
        print("Using MEAN_TASK ablation (masked nodes → mean over task data)")
    else:
        raise ValueError(f"Unknown ablation_type: {config.ablation_type}")
    
    # Create trainer
    trainer = PruningTrainer(
        masked_model=masked_model,
        task=train_task,
        val_task=val_task,
        config=config,
        use_wandb=not args.no_wandb,
        wandb_run_name=args.wandb_name or f"single_{model_name}_{timestamp}",
    )
    
    # Train
    print(f"\nTraining for {args.steps} steps...")
    with torch.autocast('cuda', dtype=torch.bfloat16):
        trainer.train(
            num_steps=args.steps,
            show_progress=True,
            histogram_every=100,
            pareto_probe_every=args.pareto_every,
        )
    
    # Get results
    num_active_non_embed = masked_model.masks.get_total_active_nodes()
    total_non_embed = masked_model.masks.get_total_nodes()
    num_active_embed = 0
    total_embed = 0
    if masked_model.token_mask is not None:
        num_active_embed = masked_model.token_mask.get_num_active()
        total_embed = masked_model.vocab_size
    num_active_total = num_active_non_embed + num_active_embed
    total_nodes = total_non_embed + total_embed
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Active nodes (total): {num_active_total:,} / {total_nodes:,} ({100*num_active_total/total_nodes:.2f}%)")
    print(f"  Non-embed nodes: {num_active_non_embed:,} / {total_non_embed:,} ({100*num_active_non_embed/total_non_embed:.2f}%)")
    if masked_model.token_mask is not None:
        print(f"  Token embed nodes: {num_active_embed:,} / {total_embed:,} ({100*num_active_embed/total_embed:.2f}%)")
    
    # Save checkpoint
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save masks
    mask_state = masked_model.get_mask_state()
    torch.save(mask_state, checkpoint_dir / "masks.pt")
    
    # Save mean cache (only if computed)
    if config.ablation_type != "zero":
        torch.save(mean_cache, checkpoint_dir / "mean_cache.pt")
    
    # Save config
    with open(checkpoint_dir / "pruning_config.json", "w") as f:
        json.dump({
            "model_path": args.model,
            "task": args.task,
            "k_coef": args.k_coef,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "noise_scale": args.noise_scale,
            "noise_bias": args.noise_bias,
            "heaviside_temp": args.heaviside_temp,
            "beta2": args.beta2,
            "steps": args.steps,
            "num_active_total": num_active_total,
            "num_active_non_embed": num_active_non_embed,
            "num_active_embed": num_active_embed,
            "total_nodes": total_nodes,
            "total_non_embed": total_non_embed,
            "total_embed": total_embed,
        }, f, indent=2)
    
    print(f"\nCheckpoint saved to: {checkpoint_dir}")
    
    # Generate HTML visualization
    if not args.no_html:
        print("\nGenerating HTML visualization with dashboards...")
        html_path = generate_html_with_dashboards(
            masked_model=masked_model,
            tokenizer=tokenizer,
            output_path=output_dir / "circuit.html",
            max_nodes=args.max_html_nodes,
            n_dashboard_samples=200,
            n_top_examples=10,
            device=device,
        )
        print(f"HTML saved to: {html_path}")
        
        # Print file size
        size_mb = html_path.stat().st_size / (1024 * 1024)
        print(f"HTML file size: {size_mb:.2f} MB")
    
    print(f"\nDone! Output directory: {output_dir}")


def find_bias_only_nodes(masked_model: MaskedSparseGPT, task=None, tokenizer=None):
    """
    Find nodes that have no upstream weight connections (only contribute via bias).
    
    A node is "bias-only" if:
    - It's an attn_in or mlp_in node at d_model index X, and no active attn_out, mlp_out, 
      or embedding writes to channel X
    - It's an attn_q, attn_k, or attn_v node and no active attn_in node has a nonzero 
      weight connecting to it
    - It's an attn_out node and no active attn_v has nonzero weight to it
    - It's an mlp_neuron node and no active mlp_in has nonzero weight to it
    - It's an mlp_out node and no active mlp_neuron has nonzero weight to it
    
    Args:
        masked_model: The masked model
        task: Optional task object for no_embed mode
        tokenizer: Optional tokenizer for no_embed mode
        
    Returns:
        Set of node_ids that are bias-only
    """
    import torch
    
    model = masked_model.model
    n_layers = model.config.n_layer
    d_model = model.config.d_model
    d_vocab = model.config.vocab_size
    
    mask_state = masked_model.get_mask_state()
    bias_only_nodes = set()
    
    # Find channels written to by active nodes
    written_to = set()
    
    # Active embeddings write to their nonzero channels
    wte = model.wte.weight.data
    if masked_model.token_mask is not None:
        # Embed mode: use mask to find active tokens
        token_binary_mask = masked_model.token_mask.get_binary_mask()
        active_token_ids = (token_binary_mask > 0.5).nonzero().squeeze(-1).tolist()
        if isinstance(active_token_ids, int):
            active_token_ids = [active_token_ids]
        for tid in active_token_ids:
            nonzero = (wte[tid] != 0).nonzero().squeeze(-1).tolist()
            if isinstance(nonzero, int):
                nonzero = [nonzero]
            written_to.update(nonzero)
    elif task is not None:
        # No-embed mode: get tokens from task dataset
        task_tokens = set()
        for _ in range(500):
            example = task.generate_example()
            for tok in example.positive_ids.tolist():
                task_tokens.add(tok)
            for tok in example.negative_ids.tolist():
                task_tokens.add(tok)
        # Filter out padding tokens
        pad_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
        task_tokens.discard(pad_id)
        
        for tid in task_tokens:
            if tid < d_vocab:
                nonzero = (wte[tid] != 0).nonzero().squeeze(-1).tolist()
                if isinstance(nonzero, int):
                    nonzero = [nonzero]
                written_to.update(nonzero)
    
    # attn_out and mlp_out write to their index
    for layer in range(n_layers):
        for loc in ['attn_out', 'mlp_out']:
            key = f'layer{layer}_{loc}'
            if key in mask_state:
                tau = mask_state[key]
                active_indices = (tau > 0).nonzero().squeeze(-1).tolist()
                if isinstance(active_indices, int):
                    active_indices = [active_indices]
                written_to.update(active_indices)
    
    # Check attn_in and mlp_in nodes
    for layer in range(n_layers):
        for loc in ['attn_in', 'mlp_in']:
            key = f'layer{layer}_{loc}'
            if key not in mask_state:
                continue
            tau = mask_state[key]
            active_indices = (tau > 0).nonzero().squeeze(-1).tolist()
            if isinstance(active_indices, int):
                active_indices = [active_indices]
            for idx in active_indices:
                if idx not in written_to:
                    bias_only_nodes.add(f'{loc}_L{layer}_I{idx}')
    
    # Check Q/K/V nodes
    for layer in range(n_layers):
        attn_in_key = f'layer{layer}_attn_in'
        if attn_in_key not in mask_state:
            continue
        attn_in_tau = mask_state[attn_in_key]
        active_attn_in = (attn_in_tau > 0).nonzero().squeeze(-1).tolist()
        if isinstance(active_attn_in, int):
            active_attn_in = [active_attn_in]
        active_attn_in = set(active_attn_in)
        
        W_qkv = model.blocks[layer].attn.c_attn.weight.data
        
        for loc, offset in [('attn_q', 0), ('attn_k', d_model), ('attn_v', 2*d_model)]:
            key = f'layer{layer}_{loc}'
            if key not in mask_state:
                continue
            tau = mask_state[key]
            active_indices = (tau > 0).nonzero().squeeze(-1).tolist()
            if isinstance(active_indices, int):
                active_indices = [active_indices]
            
            for idx in active_indices:
                has_upstream = False
                for attn_in_idx in active_attn_in:
                    w = W_qkv[idx + offset, attn_in_idx].item()
                    if w != 0:
                        has_upstream = True
                        break
                if not has_upstream:
                    bias_only_nodes.add(f'{loc}_L{layer}_I{idx}')
    
    # Check attn_out nodes
    for layer in range(n_layers):
        attn_v_key = f'layer{layer}_attn_v'
        if attn_v_key not in mask_state:
            continue
        attn_v_tau = mask_state[attn_v_key]
        active_attn_v = (attn_v_tau > 0).nonzero().squeeze(-1).tolist()
        if isinstance(active_attn_v, int):
            active_attn_v = [active_attn_v]
        active_attn_v = set(active_attn_v)
        
        W_attn_out = model.blocks[layer].attn.c_proj.weight.data
        
        key = f'layer{layer}_attn_out'
        if key not in mask_state:
            continue
        tau = mask_state[key]
        active_indices = (tau > 0).nonzero().squeeze(-1).tolist()
        if isinstance(active_indices, int):
            active_indices = [active_indices]
        
        for idx in active_indices:
            has_upstream = False
            for v_idx in active_attn_v:
                w = W_attn_out[idx, v_idx].item()
                if w != 0:
                    has_upstream = True
                    break
            if not has_upstream:
                bias_only_nodes.add(f'attn_out_L{layer}_I{idx}')
    
    # Check mlp_neuron nodes
    for layer in range(n_layers):
        mlp_in_key = f'layer{layer}_mlp_in'
        if mlp_in_key not in mask_state:
            continue
        mlp_in_tau = mask_state[mlp_in_key]
        active_mlp_in = (mlp_in_tau > 0).nonzero().squeeze(-1).tolist()
        if isinstance(active_mlp_in, int):
            active_mlp_in = [active_mlp_in]
        active_mlp_in = set(active_mlp_in)
        
        W_mlp_up = model.blocks[layer].mlp.c_fc.weight.data
        
        key = f'layer{layer}_mlp_neuron'
        if key not in mask_state:
            continue
        tau = mask_state[key]
        active_indices = (tau > 0).nonzero().squeeze(-1).tolist()
        if isinstance(active_indices, int):
            active_indices = [active_indices]
        
        for idx in active_indices:
            has_upstream = False
            for in_idx in active_mlp_in:
                w = W_mlp_up[idx, in_idx].item()
                if w != 0:
                    has_upstream = True
                    break
            if not has_upstream:
                bias_only_nodes.add(f'mlp_neuron_L{layer}_I{idx}')
    
    # Check mlp_out nodes
    for layer in range(n_layers):
        mlp_neuron_key = f'layer{layer}_mlp_neuron'
        if mlp_neuron_key not in mask_state:
            continue
        mlp_neuron_tau = mask_state[mlp_neuron_key]
        active_mlp_neuron = (mlp_neuron_tau > 0).nonzero().squeeze(-1).tolist()
        if isinstance(active_mlp_neuron, int):
            active_mlp_neuron = [active_mlp_neuron]
        active_mlp_neuron = set(active_mlp_neuron)
        
        W_mlp_down = model.blocks[layer].mlp.c_proj.weight.data
        
        key = f'layer{layer}_mlp_out'
        if key not in mask_state:
            continue
        tau = mask_state[key]
        active_indices = (tau > 0).nonzero().squeeze(-1).tolist()
        if isinstance(active_indices, int):
            active_indices = [active_indices]
        
        for idx in active_indices:
            has_upstream = False
            for neuron_idx in active_mlp_neuron:
                w = W_mlp_down[idx, neuron_idx].item()
                if w != 0:
                    has_upstream = True
                    break
            if not has_upstream:
                bias_only_nodes.add(f'mlp_out_L{layer}_I{idx}')
    
    return bias_only_nodes


def generate_html_with_dashboards(
    masked_model: MaskedSparseGPT,
    tokenizer,
    output_path: Path,
    max_nodes: int = 500,
    n_dashboard_samples: int = 200,
    n_top_examples: int = 10,
    device: str = "cuda",
    task=None,
    target_loss: float = None,
) -> Path:
    """
    Generate HTML visualization with dashboards for active nodes only.
    
    Args:
        masked_model: The pruned masked model
        tokenizer: Tokenizer for decoding
        output_path: Where to save the HTML
        max_nodes: Max nodes to include
        n_dashboard_samples: Number of samples for dashboard computation
        n_top_examples: Top/bottom examples per node
        device: Device for computation
        task: Optional task object - if provided, only unembed nodes for target tokens are shown
        target_loss: Optional target loss - if provided, bisect to find minimum k that achieves this loss
    """
    import json
    import re
    from datasets import load_dataset
    from tqdm import tqdm
    
    # If target_loss is provided, run bisection first
    if target_loss is not None and task is not None:
        from my_sparse_pretrain.src.pruning.discretize import discretize_masks
        from my_sparse_pretrain.src.pruning.config import PruningConfig
        
        print(f"  Bisecting to find mask achieving target loss {target_loss}...")
        config = PruningConfig(device=device, ablation_type='zero')
        k, achieved_loss, _ = discretize_masks(
            masked_model=masked_model,
            task=task,
            config=config,
            target_loss=target_loss,
            num_eval_batches=20,
            show_progress=True,
        )
        print(f"  Bisection complete: k={k}, achieved_loss={achieved_loss:.4f}")
    
    # Find bias-only nodes (nodes with no upstream weight connections)
    print("  Finding bias-only nodes...")
    bias_only_node_ids = find_bias_only_nodes(masked_model, task=task, tokenizer=tokenizer)
    if bias_only_node_ids:
        print(f"  Found {len(bias_only_node_ids)} bias-only nodes: {sorted(bias_only_node_ids)}")
    else:
        print("  No bias-only nodes found")
    
    print("  Extracting active nodes...")
    
    # Get active mask values
    mask_state = masked_model.get_mask_state()
    
    # Get model metadata early for head calculations
    base_config = masked_model.model.config
    n_heads_tmp = getattr(base_config, 'n_heads', 8)
    d_model_tmp = getattr(base_config, 'd_model', 1024)
    d_head_tmp = getattr(base_config, 'd_head', d_model_tmp // n_heads_tmp)
    
    # Collect all active nodes
    all_nodes = []
    model = masked_model.model
    for loc_name, tau in mask_state.items():
        match = re.match(r'layer(\d+)_(.+)', loc_name)
        if not match:
            continue
        layer = int(match.group(1))
        location = match.group(2)
        
        tau_flat = tau.flatten()
        for idx in range(tau_flat.shape[0]):
            tau_val = tau_flat[idx].item()
            if tau_val > 0:
                node_id = f"{location}_L{layer}_I{idx}"
                
                # Get bias value for this node if applicable
                bias_val = None
                block = model.blocks[layer]
                if location == 'attn_out' and block.attn.c_proj.bias is not None:
                    bias_val = block.attn.c_proj.bias[idx].item()
                elif location == 'mlp_out' and block.mlp.c_proj.bias is not None:
                    bias_val = block.mlp.c_proj.bias[idx].item()
                elif location == 'mlp_neuron' and block.mlp.c_fc.bias is not None:
                    bias_val = block.mlp.c_fc.bias[idx].item()
                elif location in ['attn_q', 'attn_k', 'attn_v'] and block.attn.c_attn.bias is not None:
                    offset = {'attn_q': 0, 'attn_k': d_model_tmp, 'attn_v': 2*d_model_tmp}[location]
                    bias_val = block.attn.c_attn.bias[idx + offset].item()
                
                node = {
                    "node_id": node_id,
                    "layer": layer,
                    "location": location,
                    "index": idx,
                    "tau": tau_val,
                    "head_idx": None,
                    "head_dim_idx": None,
                    "bias_only": node_id in bias_only_node_ids,
                    "bias": bias_val,
                }
                # For Q/K/V nodes, compute head_idx and head_dim_idx
                if location in ['attn_q', 'attn_k', 'attn_v']:
                    node["head_idx"] = idx // d_head_tmp
                    node["head_dim_idx"] = idx % d_head_tmp
                all_nodes.append(node)
    
    all_nodes.sort(key=lambda x: -x["tau"])
    selected_nodes = all_nodes[:max_nodes]
    print(f"  Selected {len(selected_nodes)} / {len(all_nodes)} active nodes")
    
    # Get model metadata
    base_config = masked_model.model.config
    n_layers = getattr(base_config, 'n_layer', 8)
    d_model = getattr(base_config, 'd_model', 1024)
    d_mlp = getattr(base_config, 'd_mlp', d_model * 4)
    n_heads = getattr(base_config, 'n_heads', 8)
    d_head = getattr(base_config, 'd_head', d_model // n_heads)
    
    # === Add residual, embedding, and unembedding nodes ===
    print("  Adding residual, embedding, and unembedding nodes...")
    
    # Extract target token IDs from task if provided
    target_token_ids = None
    if task is not None:
        print("  Extracting target tokens from task...")
        target_token_ids = set()
        # Generate a bunch of examples to get all possible target tokens
        for _ in range(1000):
            example = task.generate_example()
            target_token_ids.add(example.correct_token)
            target_token_ids.add(example.incorrect_token)
        print(f"  Found {len(target_token_ids)} unique target tokens: {[tokenizer.decode([t]) for t in target_token_ids]}")
    
    all_nodes_with_resid, resid_edges = add_residual_and_io_nodes(
        masked_model, selected_nodes, tokenizer, device, target_token_ids, task=task
    )
    print(f"  Added {len(all_nodes_with_resid) - len(selected_nodes)} residual/embed/unembed nodes")
    
    # Build node ID set for fast lookup
    node_ids = {n["node_id"] for n in all_nodes_with_resid}
    
    # === Compute edges between active nodes ===
    print("  Computing edges between active nodes...")
    edges = compute_edges_for_active_nodes(masked_model, selected_nodes, device)
    edges.extend(resid_edges)
    print(f"  Found {len(edges)} edges")
    
    # === Compute dashboards ===
    # Pretrain Original: unpruned model on pretraining data
    print(f"  Computing pretrain-original dashboards ({n_dashboard_samples} samples)...")
    pretrain_original_dashboards = compute_dashboards_for_nodes(
        masked_model, tokenizer, selected_nodes, 
        n_samples=n_dashboard_samples, 
        n_top=n_top_examples,
        device=device
    )
    
    # Generate task prompts once for both task dashboards (ensures same samples are used)
    task_prompts = None
    task_original_dashboards = {}
    task_pruned_dashboards = {}
    if task is not None:
        print(f"  Generating {n_dashboard_samples} task prompts...")
        task_prompts = []
        for _ in range(n_dashboard_samples):
            example = task.generate_example()
            if hasattr(example, 'positive_ids'):
                prompt_ids = example.positive_ids.tolist() if isinstance(example.positive_ids, torch.Tensor) else list(example.positive_ids)
                if prompt_ids:
                    task_prompts.append(prompt_ids)
        print(f"    Generated {len(task_prompts)} valid prompts")
        
        # Task Original: unpruned model on task data
        print(f"  Computing task-original dashboards...")
        task_original_dashboards = compute_task_original_dashboards_for_nodes(
            masked_model, tokenizer, selected_nodes,
            task=task,
            n_samples=n_dashboard_samples,
            n_top=n_top_examples,
            device=device,
            task_prompts=task_prompts
        )
        
        # Task Pruned: pruned model on task data (same prompts!)
        print(f"  Computing task-pruned dashboards...")
        task_pruned_dashboards = compute_task_dashboards_for_nodes(
            masked_model, tokenizer, selected_nodes,
            task=task,
            n_samples=n_dashboard_samples,
            n_top=n_top_examples,
            device=device,
            task_prompts=task_prompts
        )
    
    # Create graph data
    def get_node_dict(n):
        loc = n["location"]
        # Only Q/K/V nodes have per-head per-dim indices
        is_qkv = loc in ["attn_q", "attn_k", "attn_v"]
        result = {
            "node_id": n["node_id"],
            "layer": n["layer"],
            "location": loc,
            "index": n["index"],
            "head_idx": n.get("head_idx") if is_qkv else None,
            "head_dim_idx": n.get("head_dim_idx") if is_qkv else None,
            "bias_only": n.get("bias_only", False),
            "bias": n.get("bias"),
        }
        # Add token string for embed/unembed nodes
        if "token_str" in n:
            result["token_str"] = n["token_str"]
        return result
    
    graph_data = {
        "nodes": [get_node_dict(n) for n in all_nodes_with_resid],
        "edges": edges,
        "metadata": {
            "n_layers": n_layers,
            "d_model": d_model,
            "d_mlp": d_mlp,
            "n_heads": n_heads,
            "d_head": d_head,
            "num_nodes_total": len(all_nodes),
            "num_nodes_shown": len(all_nodes_with_resid),
            "num_mask_site_nodes": len(selected_nodes),
        }
    }
    
    # Generate HTML using our local lightweight template with two-column layout
    from my_sparse_pretrain.src.visualization.circuit_viewer import _generate_dashboard_html
    from my_sparse_pretrain.src.visualization.dashboard import NodeDashboardData
    
    # Convert pretrain-original dashboards to HTML
    pretrain_original_html = {}
    for node_id, dash_data in pretrain_original_dashboards.items():
        pretrain_original_html[node_id] = _generate_dashboard_html(dash_data)
    
    # Convert task-original dashboards to HTML
    task_original_html = {}
    for node_id, dash_data in task_original_dashboards.items():
        task_original_html[node_id] = _generate_dashboard_html(dash_data)
    
    # Convert task-pruned dashboards to HTML
    task_pruned_html = {}
    for node_id, dash_data in task_pruned_dashboards.items():
        task_pruned_html[node_id] = _generate_dashboard_html(dash_data)
    
    html = _get_lightweight_html_template()
    html = html.replace("__GRAPH_DATA__", json.dumps(graph_data))
    html = html.replace("__PRETRAIN_ORIGINAL_DATA__", json.dumps(pretrain_original_html))
    html = html.replace("__TASK_ORIGINAL_DATA__", json.dumps(task_original_html))
    html = html.replace("__TASK_PRUNED_DATA__", json.dumps(task_pruned_html))
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    
    return output_path


def add_residual_and_io_nodes(masked_model, mask_site_nodes, tokenizer=None, device="cuda", 
                               target_token_ids=None, task=None):
    """
    Add residual stream channels, embedding nodes, and unembedding nodes.
    
    Strategy:
    1. Find d_model dimensions WRITTEN TO by active embeddings or mask site nodes (attn_out, mlp_out)
    2. Find d_model dimensions READ FROM by mask site nodes (attn_in, mlp_in) or unembedding of target tokens
    3. Only show residual channels in the INTERSECTION of these two sets
    
    Residual channels are represented as vertical gray lines.
    Active embedding nodes are shown if their written dimensions are in the intersection.
    
    For no_embed mode (token_mask is None):
    - Get all tokens that appear in the task dataset
    - Only show embedding nodes that connect to residual channels that have both upstream and downstream connections
    
    Args:
        masked_model: The masked model
        mask_site_nodes: List of mask site nodes
        tokenizer: Optional tokenizer for decoding token strings
        device: Device for computation
        target_token_ids: Token IDs for unembedding (used to determine read-from set)
        task: Optional task object - used to get all tokens in task dataset for no_embed mode
    
    Returns:
        all_nodes: list of all nodes (mask site + embed + residual channels + unembed)
        extra_edges: list of edges connecting them
    """
    import torch
    
    model = masked_model.model
    n_layers = model.config.n_layer
    d_model = model.config.d_model
    d_vocab = model.config.vocab_size
    
    # Get embedding and unembedding matrices
    W_E = model.wte.weight.data  # [vocab_size, d_model]
    W_U = model.lm_head.weight.data  # [vocab_size, d_model]
    ln_f_weight = model.ln_f.weight.data  # [d_model]
    
    all_nodes = list(mask_site_nodes)
    extra_edges = []
    
    def get_token_str(token_id):
        """Get a display string for a token."""
        if tokenizer is None:
            return f"T{token_id}"
        try:
            s = tokenizer.decode([token_id])
            s = s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
            if len(s) > 12:
                s = s[:10] + "..."
            return repr(s) if s.strip() == '' else s
        except:
            return f"T{token_id}"
    
    # Group mask site nodes by layer and location
    nodes_by_layer_loc = {}
    for n in mask_site_nodes:
        key = (n["layer"], n["location"])
        if key not in nodes_by_layer_loc:
            nodes_by_layer_loc[key] = set()
        nodes_by_layer_loc[key].add(n["index"])
    
    # === Get active token embeddings ===
    active_embed_tokens = []
    embed_writes_to = {}  # token_id -> set of d_model indices it writes to
    
    if masked_model.token_mask is not None:
        # Case 1: Embedding nodes were pruned - use the mask to find active tokens
        token_binary_mask = masked_model.token_mask.get_binary_mask()
        active_embed_tokens = (token_binary_mask > 0.5).nonzero().squeeze(-1).tolist()
        if isinstance(active_embed_tokens, int):
            active_embed_tokens = [active_embed_tokens]
        
        # For each active token, find which d_model dimensions it writes to
        for token_id in active_embed_tokens:
            token_embed = W_E[token_id]
            nonzero_indices = (token_embed != 0).nonzero().squeeze(-1).tolist()
            if isinstance(nonzero_indices, int):
                nonzero_indices = [nonzero_indices]
            embed_writes_to[token_id] = set(nonzero_indices)
    elif task is not None:
        # Case 2: No embedding pruning (no_embed mode) - get tokens from task dataset
        # Collect all unique tokens that appear in the task dataset
        task_tokens = set()
        print("    Collecting tokens from task dataset for no_embed mode...")
        for _ in range(500):  # Sample 500 examples to get a good coverage of tokens
            example = task.generate_example()
            # Add all tokens from positive and negative sequences
            for tok in example.positive_ids.tolist():
                task_tokens.add(tok)
            for tok in example.negative_ids.tolist():
                task_tokens.add(tok)
        
        # Filter out padding tokens
        pad_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
        task_tokens.discard(pad_id)
        
        print(f"    Found {len(task_tokens)} unique tokens in task dataset")
        active_embed_tokens = list(task_tokens)
        
        # For each token, find which d_model dimensions it writes to
        for token_id in active_embed_tokens:
            if token_id < d_vocab:
                token_embed = W_E[token_id]
                nonzero_indices = (token_embed != 0).nonzero().squeeze(-1).tolist()
                if isinstance(nonzero_indices, int):
                    nonzero_indices = [nonzero_indices]
                embed_writes_to[token_id] = set(nonzero_indices)
    
    # === Step 1: Find dimensions READ FROM ===
    # attn_in reads from resid_pre (same d_model index)
    # mlp_in reads from resid_mid (same d_model index)
    # Unembedding of target tokens reads from final resid_post
    
    # Track channels read by mask site nodes separately (for embedding filtering)
    read_by_mask_sites = set()
    for layer in range(n_layers):
        for idx in nodes_by_layer_loc.get((layer, "attn_in"), set()):
            read_by_mask_sites.add(idx)
        for idx in nodes_by_layer_loc.get((layer, "mlp_in"), set()):
            read_by_mask_sites.add(idx)
    
    # read_from includes both mask sites and unembed
    read_from = set(read_by_mask_sites)
    
    # Add dimensions read by unembedding of target tokens
    if target_token_ids is not None:
        for token_id in target_token_ids:
            if token_id < d_vocab:
                token_unembed = W_U[token_id]  # [d_model]
                for idx in range(d_model):
                    if token_unembed[idx].item() != 0:
                        read_from.add(idx)
    
    # === Step 2: Determine which embeddings will be shown ===
    # An embedding is shown only if it writes to at least one channel read by a mask site node
    shown_embed_tokens = []
    for token_id in active_embed_tokens:
        token_writes = embed_writes_to.get(token_id, set())
        if token_writes & read_by_mask_sites:
            shown_embed_tokens.append(token_id)
    
    # === Step 3: Find dimensions WRITTEN TO by visible nodes ===
    # Only count embeddings that will actually be shown, plus mask site nodes
    written_to = set()
    
    # Add dimensions written by SHOWN embeddings only
    for token_id in shown_embed_tokens:
        written_to.update(embed_writes_to[token_id])
    
    # Track channels written by mask site nodes separately
    written_by_mask_sites = set()
    for layer in range(n_layers):
        for idx in nodes_by_layer_loc.get((layer, "attn_out"), set()):
            written_to.add(idx)
            written_by_mask_sites.add(idx)
        for idx in nodes_by_layer_loc.get((layer, "mlp_out"), set()):
            written_to.add(idx)
            written_by_mask_sites.add(idx)
    
    # === Step 4: Get intersection ===
    # Residual channels are shown if they have BOTH:
    # 1. Written to by a VISIBLE node (shown embeddings or mask site nodes)
    # 2. Read from by something (mask site nodes OR unembed)
    # 3. AND connect to at least one mask site node (not just embed→unembed)
    active_resid_channels = written_to & read_from
    
    # Filter out channels that only connect embed↔unembed (no mask site connections)
    channels_with_mask_site = read_by_mask_sites | written_by_mask_sites
    active_resid_channels = active_resid_channels & channels_with_mask_site
    
    # Print debug info for no_embed mode
    if masked_model.token_mask is None and task is not None:
        print(f"    No-embed mode: {len(shown_embed_tokens)}/{len(active_embed_tokens)} embeddings shown (connect to mask sites)")
        print(f"    {len(written_to)} channels with visible upstream, {len(read_from)} channels with downstream")
        print(f"    Intersection: {len(active_resid_channels)} active residual channels")
    
    # === Create residual channel "nodes" ===
    # Each channel is a single node that represents the vertical flow through all layers
    # Location is "resid_channel" to distinguish from per-layer nodes
    for idx in sorted(active_resid_channels):
        all_nodes.append({
            "node_id": f"resid_channel_I{idx}",
            "layer": -0.5,  # Special layer value for channels (between embed and layer 0)
            "location": "resid_channel",
            "index": idx,
        })
    
    # === Add edges from mask site nodes to/from residual channels ===
    for layer in range(n_layers):
        block = model.blocks[layer]
        ln_1_weight = block.ln_1.weight.data
        ln_2_weight = block.ln_2.weight.data
        
        # resid_channel -> attn_in (through ln_1)
        for idx in nodes_by_layer_loc.get((layer, "attn_in"), set()):
            if idx in active_resid_channels:
                w = ln_1_weight[idx].item()
                extra_edges.append({
                    "source_id": f"resid_channel_I{idx}",
                    "target_id": f"attn_in_L{layer}_I{idx}",
                    "weight": w,
                    "edge_type": "ln_1",
                    "target_layer": layer,
                    "resid_position": "pre",
                })
        
        # attn_out -> resid_channel
        for idx in nodes_by_layer_loc.get((layer, "attn_out"), set()):
            if idx in active_resid_channels:
                extra_edges.append({
                    "source_id": f"attn_out_L{layer}_I{idx}",
                    "target_id": f"resid_channel_I{idx}",
                    "weight": 1.0,
                    "edge_type": "resid_add",
                    "source_layer": layer,
                    "resid_position": "mid",
                })
        
        # resid_channel -> mlp_in (through ln_2)
        for idx in nodes_by_layer_loc.get((layer, "mlp_in"), set()):
            if idx in active_resid_channels:
                w = ln_2_weight[idx].item()
                extra_edges.append({
                    "source_id": f"resid_channel_I{idx}",
                    "target_id": f"mlp_in_L{layer}_I{idx}",
                    "weight": w,
                    "edge_type": "ln_2",
                    "target_layer": layer,
                    "resid_position": "mid",
                })
        
        # mlp_out -> resid_channel
        for idx in nodes_by_layer_loc.get((layer, "mlp_out"), set()):
            if idx in active_resid_channels:
                extra_edges.append({
                    "source_id": f"mlp_out_L{layer}_I{idx}",
                    "target_id": f"resid_channel_I{idx}",
                    "weight": 1.0,
                    "edge_type": "resid_add",
                    "source_layer": layer,
                    "resid_position": "post",
                })
    
    # === Add embedding nodes (for shown embeddings only) ===
    # shown_embed_tokens was already filtered to only include embeddings that connect to mask sites
    for token_id in shown_embed_tokens:
        token_writes = embed_writes_to.get(token_id, set())
        
        # Add the embedding node
        all_nodes.append({
            "node_id": f"embed_T{token_id}",
            "layer": -1,  # Below layer 0
            "location": "embed",
            "index": token_id,
            "token_str": get_token_str(token_id),
        })
        
        # Add edges from embed to all active channels it writes to
        active_writes = token_writes & active_resid_channels
        for idx in active_writes:
            w = W_E[token_id, idx].item()
            extra_edges.append({
                "source_id": f"embed_T{token_id}",
                "target_id": f"resid_channel_I{idx}",
                "weight": w,
                "edge_type": "embed",
                "source_layer": 0,
                "resid_position": "pre",
            })
    
    # === Add unembedding nodes (only for target tokens) ===
    if target_token_ids is not None:
        for token_id in target_token_ids:
            if token_id >= d_vocab:
                continue
            token_unembed = W_U[token_id]  # [d_model]
            token_edges = []
            
            for idx in active_resid_channels:
                w = token_unembed[idx].item() / ln_f_weight[idx].item()
                if w != 0:
                    token_edges.append({
                        "source_id": f"resid_channel_I{idx}",
                        "target_id": f"unembed_T{token_id}",
                        "weight": w,
                        "edge_type": "unembed",
                    })
            
            if token_edges:
                all_nodes.append({
                    "node_id": f"unembed_T{token_id}",
                    "layer": n_layers,
                    "location": "unembed",
                    "index": token_id,
                    "token_str": get_token_str(token_id),
                })
                extra_edges.extend(token_edges)
    
    return all_nodes, extra_edges


def compute_edges_for_active_nodes(masked_model, nodes, device="cuda"):
    """Compute edges between active nodes using model weights."""
    import torch
    
    edges = []
    
    # Group nodes by layer and location for efficient lookup
    nodes_by_layer_loc = {}
    for n in nodes:
        key = (n["layer"], n["location"])
        if key not in nodes_by_layer_loc:
            nodes_by_layer_loc[key] = []
        nodes_by_layer_loc[key].append(n)
    
    model = masked_model.model
    n_layers = model.config.n_layer
    
    # For each layer, compute edges between consecutive locations
    # Flow: resid -> attn_in -> attn_q/k/v -> attn_out -> resid_mid -> mlp_in -> mlp_neuron -> mlp_out -> resid
    
    for layer_idx in range(n_layers):
        block = model.blocks[layer_idx]
        
        # Get weight matrices (note: weight shapes are [out_features, in_features])
        W_qkv = block.attn.c_attn.weight.data  # [3*d_model, d_model]
        W_attn_out = block.attn.c_proj.weight.data  # [d_model, d_model]
        W_mlp_up = block.mlp.c_fc.weight.data  # [d_mlp, d_model]
        W_mlp_down = block.mlp.c_proj.weight.data  # [d_model, d_mlp]
        
        d_model = W_qkv.shape[1]  # Input dimension
        
        # attn_in -> attn_q/k/v edges (through W_qkv)
        attn_in_nodes = nodes_by_layer_loc.get((layer_idx, "attn_in"), [])
        for out_loc, offset in [("attn_q", 0), ("attn_k", d_model), ("attn_v", 2*d_model)]:
            out_nodes = nodes_by_layer_loc.get((layer_idx, out_loc), [])
            for src in attn_in_nodes:
                for tgt in out_nodes:
                    # W_qkv[output_idx, input_idx]
                    w = W_qkv[tgt["index"] + offset, src["index"]].item()
                    if abs(w) > 0.01:
                        edges.append({
                            "source_id": src["node_id"],
                            "target_id": tgt["node_id"],
                            "weight": w,
                            "edge_type": f"W_{out_loc}",
                        })
        
        # Q-K edges: Q and K in same head and same d_head channel connect to each other
        attn_q_nodes = nodes_by_layer_loc.get((layer_idx, "attn_q"), [])
        attn_k_nodes = nodes_by_layer_loc.get((layer_idx, "attn_k"), [])
        
        # Build lookup by (head_idx, head_dim_idx) for K nodes
        k_by_head_dim = {}
        for k_node in attn_k_nodes:
            key = (k_node.get("head_idx"), k_node.get("head_dim_idx"))
            if key[0] is not None:
                k_by_head_dim[key] = k_node
        
        # Connect Q to K if same head and same d_head dimension
        for q_node in attn_q_nodes:
            key = (q_node.get("head_idx"), q_node.get("head_dim_idx"))
            if key in k_by_head_dim:
                k_node = k_by_head_dim[key]
                edges.append({
                    "source_id": q_node["node_id"],
                    "target_id": k_node["node_id"],
                    "weight": 1.0,
                    "edge_type": "qk_match",
                })
        
        # attn_out edges
        attn_out_nodes = nodes_by_layer_loc.get((layer_idx, "attn_out"), [])
        mlp_in_nodes = nodes_by_layer_loc.get((layer_idx, "mlp_in"), [])
        
        # mlp_in -> mlp_neuron (through W_mlp_up)
        mlp_neuron_nodes = nodes_by_layer_loc.get((layer_idx, "mlp_neuron"), [])
        for src in mlp_in_nodes:
            for tgt in mlp_neuron_nodes:
                # W_mlp_up[output_idx, input_idx]
                w = W_mlp_up[tgt["index"], src["index"]].item()
                if abs(w) > 0.01:
                    edges.append({
                        "source_id": src["node_id"],
                        "target_id": tgt["node_id"],
                        "weight": w,
                        "edge_type": "W_up",
                    })
        
        # mlp_neuron -> mlp_out (through W_mlp_down)
        mlp_out_nodes = nodes_by_layer_loc.get((layer_idx, "mlp_out"), [])
        for src in mlp_neuron_nodes:
            for tgt in mlp_out_nodes:
                # W_mlp_down[output_idx, input_idx]
                w = W_mlp_down[tgt["index"], src["index"]].item()
                if abs(w) > 0.01:
                    edges.append({
                        "source_id": src["node_id"],
                        "target_id": tgt["node_id"],
                        "weight": w,
                        "edge_type": "W_down",
                    })
        
        # attn_v -> attn_out (through W_O projection)
        attn_v_nodes = nodes_by_layer_loc.get((layer_idx, "attn_v"), [])
        for src in attn_v_nodes:
            for tgt in attn_out_nodes:
                # W_attn_out[output_idx, input_idx]
                w = W_attn_out[tgt["index"], src["index"]].item()
                if abs(w) > 0.01:
                    edges.append({
                        "source_id": src["node_id"],
                        "target_id": tgt["node_id"],
                        "weight": w,
                        "edge_type": "W_O",
                    })
    
    return edges


def compute_dashboards_for_nodes(masked_model, tokenizer, nodes, n_samples=200, n_top=5, device="cuda"):
    """
    Compute activation dashboards for a small set of nodes.
    
    Uses sequence packing (like pretraining): texts are concatenated with EOS tokens
    between them, then chunked into fixed-length sequences. NO PADDING.
    """
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from my_sparse_pretrain.src.visualization.dashboard import NodeDashboardData, ActivationExample
    
    # Create sequence-packed batches (matching pretraining strategy)
    print(f"    Loading {n_samples} text samples with sequence packing...")
    dataset = load_dataset("SimpleStories/SimpleStories", split="train", streaming=True, trust_remote_code=True)
    
    eos_token_id = tokenizer.eos_token_id
    token_buffer = []
    batch_size = 16
    seq_length = 128
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Build all batches using sequence packing
    all_batches = []
    for example in dataset:
        if len(all_batches) >= n_batches:
            break
        
        text = example.get("story", "")
        if not text:
            continue
        
        # Tokenize without special tokens (matching pretraining)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            continue
        
        # Add tokens to buffer
        token_buffer.extend(tokens)
        
        # Add EOS between examples (matching pretraining)
        if eos_token_id is not None:
            token_buffer.append(eos_token_id)
        
        # Extract complete batches when we have enough tokens
        while len(token_buffer) >= batch_size * seq_length and len(all_batches) < n_batches:
            batch = []
            for _ in range(batch_size):
                chunk = token_buffer[:seq_length]
                token_buffer = token_buffer[seq_length:]
                batch.append(torch.tensor(chunk, dtype=torch.long))
            all_batches.append(torch.stack(batch))
    
    # Group nodes by layer and location
    nodes_by_loc = {}
    for n in nodes:
        key = f"layer{n['layer']}_{n['location']}"
        if key not in nodes_by_loc:
            nodes_by_loc[key] = []
        nodes_by_loc[key].append(n)
    
    # Collect activations for each node
    node_activations = {n["node_id"]: [] for n in nodes}
    node_tokens = {n["node_id"]: [] for n in nodes}
    
    print(f"    Running forward passes...")
    model = masked_model.model
    model.eval()
    
    for input_ids in tqdm(all_batches, desc="    Dashboard batches"):
        input_ids = input_ids.to(device)
        
        B, T = input_ids.shape
        
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            # Manual forward to capture activations
            x = model.wte(input_ids)
            if model.wpe is not None:
                pos = torch.arange(T, device=device)
                x = x + model.wpe(pos)
            x = model.drop(x)
            
            for layer_idx, block in enumerate(model.blocks):
                # attn_in
                normed = block.ln_1(x)
                key = f"layer{layer_idx}_attn_in"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = normed[:, :, n["index"]].float().cpu().numpy()  # [B, T]
                        for b in range(B):
                            node_activations[n["node_id"]].extend(acts[b].tolist())
                            node_tokens[n["node_id"]].extend(input_ids[b].cpu().tolist())
                
                # Q, K, V
                qkv = block.attn.c_attn(normed)
                d_head = block.attn.d_head
                n_heads = block.attn.n_heads
                q, k, v = qkv.split(n_heads * d_head, dim=-1)
                
                for loc, tensor in [("attn_q", q), ("attn_k", k), ("attn_v", v)]:
                    key = f"layer{layer_idx}_{loc}"
                    if key in nodes_by_loc:
                        for n in nodes_by_loc[key]:
                            acts = tensor[:, :, n["index"]].float().cpu().numpy()
                            for b in range(B):
                                node_activations[n["node_id"]].extend(acts[b].tolist())
                                node_tokens[n["node_id"]].extend(input_ids[b].cpu().tolist())
                
                # Attention output
                attn_out = block.attn(normed, None)
                key = f"layer{layer_idx}_attn_out"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = attn_out[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            node_activations[n["node_id"]].extend(acts[b].tolist())
                            node_tokens[n["node_id"]].extend(input_ids[b].cpu().tolist())
                
                x = x + attn_out
                
                # MLP
                normed = block.ln_2(x)
                key = f"layer{layer_idx}_mlp_in"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = normed[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            node_activations[n["node_id"]].extend(acts[b].tolist())
                            node_tokens[n["node_id"]].extend(input_ids[b].cpu().tolist())
                
                mlp_hidden = block.mlp.c_fc(normed)
                mlp_hidden = block.mlp.act_fn(mlp_hidden)
                key = f"layer{layer_idx}_mlp_neuron"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = mlp_hidden[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            node_activations[n["node_id"]].extend(acts[b].tolist())
                            node_tokens[n["node_id"]].extend(input_ids[b].cpu().tolist())
                
                mlp_out = block.mlp.c_proj(mlp_hidden)
                mlp_out = block.mlp.dropout(mlp_out)
                key = f"layer{layer_idx}_mlp_out"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = mlp_out[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            node_activations[n["node_id"]].extend(acts[b].tolist())
                            node_tokens[n["node_id"]].extend(input_ids[b].cpu().tolist())
                
                x = x + mlp_out
    
    # Build dashboards
    print(f"    Building dashboard data...")
    dashboards = {}
    
    import numpy as np
    
    # Track which context (batch, sample) each position came from
    # Each batch has batch_size=16 samples, each sample has seq_length=512 tokens
    def get_context_id(position):
        """Get a unique context ID for a given position in the flattened activation array."""
        # Each sample is seq_length tokens, each batch is batch_size samples
        sample_idx = position // seq_length
        return sample_idx  # Return the sample index as context ID
    
    for n in nodes:
        node_id = n["node_id"]
        acts = np.array(node_activations[node_id])
        tokens = node_tokens[node_id]
        
        if len(acts) == 0:
            continue
        
        # Find top and bottom activating positions
        sorted_indices = np.argsort(acts)
        
        def find_transition_window(idx, acts, tokens, window_size=24):
            """
            Find a window that captures the transition from low to high absolute activation.
            Instead of just centering on the max, try to include tokens where
            the absolute activation is much smaller, to show the boundary/transition.
            
            Searches both backwards AND forwards from the max position to find
            the nearest token with small absolute activation.
            """
            max_abs_val = abs(acts[idx])
            threshold = max_abs_val * 0.5  # Look for tokens with |activation| < 50% of |max|
            
            search_range = window_size * 2  # How far to search
            boundary_idx = None
            
            # Search BACKWARDS from idx for tokens with small absolute value
            backward_boundary = None
            for i in range(idx - 1, max(0, idx - search_range) - 1, -1):
                if abs(acts[i]) < threshold:
                    backward_boundary = i
                    break
            
            # Search FORWARDS from idx for tokens with small absolute value
            forward_boundary = None
            for i in range(idx + 1, min(len(acts), idx + search_range + 1)):
                if abs(acts[i]) < threshold:
                    forward_boundary = i
                    break
            
            # Choose the closer boundary, preferring backward for context
            if backward_boundary is not None and forward_boundary is not None:
                # Both found - use backward (shows lead-up to activation)
                boundary_idx = backward_boundary
            elif backward_boundary is not None:
                boundary_idx = backward_boundary
            elif forward_boundary is not None:
                boundary_idx = forward_boundary
            else:
                # No boundary found - just center on max
                boundary_idx = idx
            
            # Position window to include both the boundary and the max position
            if boundary_idx < idx:
                # Boundary is before max - show: boundary context -> max
                tokens_before_boundary = window_size // 4
                start = max(0, boundary_idx - tokens_before_boundary)
                end = min(len(tokens), start + window_size * 2)
                # Ensure max is included
                if idx >= end:
                    end = min(len(tokens), idx + window_size // 4)
                    start = max(0, end - window_size * 2)
            else:
                # Boundary is after max or same as max - show: max -> boundary context
                start = max(0, idx - window_size // 4)
                end = min(len(tokens), boundary_idx + window_size // 4)
                # Ensure we have enough context
                if end - start < window_size:
                    end = min(len(tokens), start + window_size * 2)
            
            return start, end
        
        # Build examples - show context around max activation
        # Select top n_top examples from DISTINCT contexts
        top_examples = []
        seen_contexts = set()
        for idx in sorted_indices[::-1]:  # Iterate from highest to lowest
            context_id = get_context_id(idx)
            if context_id in seen_contexts:
                continue
            seen_contexts.add(context_id)
            
            # Get FULL context boundaries (the complete seq_length sample)
            full_start = context_id * seq_length
            full_end = min(len(tokens), (context_id + 1) * seq_length)
            
            # Get full context tokens and activations
            full_tokens = tokens[full_start:full_end]
            full_acts = acts[full_start:full_end].tolist()
            
            # Position within the full context
            pos_in_context = idx - full_start
            
            top_examples.append(ActivationExample(
                tokens=[tokenizer.decode([t]) for t in full_tokens],
                token_ids=full_tokens,
                activations=full_acts,
                max_activation=float(acts[idx]),
                max_position=pos_in_context,
            ))
            
            if len(top_examples) >= n_top:
                break
        
        # Same for bottom examples (most negative activations)
        bottom_examples = []
        seen_contexts = set()
        for idx in sorted_indices:  # Iterate from lowest to highest
            context_id = get_context_id(idx)
            if context_id in seen_contexts:
                continue
            seen_contexts.add(context_id)
            
            # Get FULL context boundaries (the complete seq_length sample)
            full_start = context_id * seq_length
            full_end = min(len(tokens), (context_id + 1) * seq_length)
            
            # Get full context tokens and activations
            full_tokens = tokens[full_start:full_end]
            full_acts = acts[full_start:full_end].tolist()
            
            # Position within the full context
            pos_in_context = idx - full_start
            
            bottom_examples.append(ActivationExample(
                tokens=[tokenizer.decode([t]) for t in full_tokens],
                token_ids=full_tokens,
                activations=full_acts,
                max_activation=float(acts[idx]),
                max_position=pos_in_context,
                min_activation=float(acts[idx]),
                min_position=pos_in_context,
            ))
            
            if len(bottom_examples) >= n_top:
                break
        
        dashboards[node_id] = NodeDashboardData(
            node_id=node_id,
            layer=n["layer"],
            location=n["location"],
            index=n["index"],
            top_examples=top_examples,
            bottom_examples=bottom_examples,
            mean_activation=float(np.mean(acts)),
            std_activation=float(np.std(acts)),
            frequency=float(np.mean(acts != 0)),
        )
    
    return dashboards


def compute_task_dashboards_for_nodes(masked_model, tokenizer, nodes, task, n_samples=200, n_top=5, device="cuda", task_prompts=None):
    """
    Compute activation dashboards for nodes using the MASKED (pruned) model on the TASK dataset.
    
    This differs from compute_dashboards_for_nodes which uses the unpruned model on pretraining data.
    
    Args:
        task_prompts: Optional list of pre-generated prompts to use. If provided, uses these
                      instead of generating new ones (for consistency with other dashboard functions).
    """
    import torch
    import numpy as np
    from tqdm import tqdm
    from my_sparse_pretrain.src.visualization.dashboard import NodeDashboardData, ActivationExample
    
    if task is None and task_prompts is None:
        print("    No task provided, skipping task dashboards")
        return {}
    
    # Use provided prompts or generate new ones
    if task_prompts is not None:
        print(f"    Using {len(task_prompts)} pre-generated task prompts...")
        all_prompts = task_prompts
    else:
        print(f"    Generating task examples for {n_samples} samples...")
        
        # Generate examples from the task
        all_prompts = []
        for _ in range(n_samples):
            example = task.generate_example()
            # Use positive_ids (the sequence where the correct token should be predicted)
            if hasattr(example, 'positive_ids'):
                prompt_ids = example.positive_ids.tolist() if isinstance(example.positive_ids, torch.Tensor) else list(example.positive_ids)
                if prompt_ids:
                    all_prompts.append(prompt_ids)
    
    if not all_prompts:
        print("    No valid prompts generated for task dashboards")
        return {}
    
    batch_size = 16
    
    # Create batches with RIGHT padding to max length in each batch
    # Also track which positions are real tokens (not padding)
    all_batches = []
    all_seq_lengths = []  # Track actual sequence lengths for each batch
    
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        if len(batch_prompts) < 2:  # Skip very small batches
            continue
        
        # Find max length in this batch
        max_len = max(len(p) for p in batch_prompts)
        
        # Pad each sequence to max_len with RIGHT padding (EOS tokens at the end)
        padded_batch = []
        seq_lengths = []
        for prompt in batch_prompts:
            seq_len = len(prompt)
            seq_lengths.append(seq_len)
            if seq_len < max_len:
                # Right padding with EOS
                padded = prompt + [tokenizer.eos_token_id] * (max_len - seq_len)
            else:
                padded = prompt
            padded_batch.append(torch.tensor(padded, dtype=torch.long))
        
        all_batches.append(torch.stack(padded_batch))
        all_seq_lengths.append(seq_lengths)
    
    if not all_batches:
        print("    No valid batches created for task dashboards")
        return {}
    
    # Group nodes by layer and location
    nodes_by_loc = {}
    for n in nodes:
        key = f"layer{n['layer']}_{n['location']}"
        if key not in nodes_by_loc:
            nodes_by_loc[key] = []
        nodes_by_loc[key].append(n)
    
    # Collect activations using the MASKED model
    # Store as list of (tokens, activations) tuples per node - one per example
    node_examples = {n["node_id"]: [] for n in nodes}
    
    print(f"    Running forward passes with masked model...")
    masked_model.eval()
    
    for batch_idx, input_ids in enumerate(tqdm(all_batches, desc="    Task dashboard batches")):
        input_ids = input_ids.to(device)
        B, T = input_ids.shape
        seq_lengths = all_seq_lengths[batch_idx]  # Actual lengths (without padding)
        
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            # Forward through masked model to capture activations
            # Use mask_fn from masked_model to properly handle AbsTopK + node masks
            model = masked_model.model
            masks = masked_model.masks
            mask_fn = masked_model._get_masking_fn()
            
            x = model.wte(input_ids)
            if model.wpe is not None:
                pos = torch.arange(T, device=device)
                x = x + model.wpe(pos)
            x = model.drop(x)
            
            for layer_idx, block in enumerate(model.blocks):
                masked_model._current_layer = layer_idx
                
                # attn_in (after ln_1)
                normed = block.ln_1(x)
                
                # Apply AbsTopK + node mask via mask_fn
                normed = mask_fn(normed, "attn_in")
                
                key = f"layer{layer_idx}_attn_in"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = normed[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            # Only include non-padding tokens
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                # Q, K, V
                attn = block.attn
                qkv = attn.c_attn(normed)
                d_head = attn.d_head
                n_heads = attn.n_heads
                q, k, v = qkv.split(n_heads * d_head, dim=-1)
                
                # Apply AbsTopK + node masks via mask_fn
                q = mask_fn(q, "attn_q")
                k = mask_fn(k, "attn_k")
                v = mask_fn(v, "attn_v")
                
                # Collect Q/K/V activations AFTER masks are applied
                for loc in ["attn_q", "attn_k", "attn_v"]:
                    key = f"layer{layer_idx}_{loc}"
                    if key in nodes_by_loc:
                        curr_tensor = {"attn_q": q, "attn_k": k, "attn_v": v}[loc]
                        for n in nodes_by_loc[key]:
                            acts = curr_tensor[:, :, n["index"]].float().cpu().numpy()
                            for b in range(B):
                                seq_len = seq_lengths[b]
                                node_examples[n["node_id"]].append((
                                    input_ids[b, :seq_len].cpu().tolist(),
                                    acts[b, :seq_len].tolist()
                                ))
                
                # Compute attention output using MASKED Q, K, V
                # Use _forward_attention_with_masks logic exactly to match
                q_reshaped = q.view(B, T, n_heads, d_head).transpose(1, 2)
                k_reshaped = k.view(B, T, n_heads, d_head).transpose(1, 2)
                v_reshaped = v.view(B, T, n_heads, d_head).transpose(1, 2)
                
                scale = 1.0 / (d_head ** 0.5)
                
                # Match _forward_attention_with_masks exactly
                if attn.use_flash and attn.attn_fn is not None:
                    # Use attention with sinks (custom attn_fn)
                    attn_output = attn.attn_fn(
                        q_reshaped, k_reshaped, v_reshaped,
                        dropout_p=0.0,
                        is_causal=True,
                        scale=scale,
                    )
                elif attn.use_flash:
                    # Standard flash attention
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        q_reshaped, k_reshaped, v_reshaped,
                        dropout_p=0.0,
                        is_causal=True,
                        scale=scale,
                    )
                else:
                    # Manual attention
                    att_scores = (q_reshaped @ k_reshaped.transpose(-2, -1)) * scale
                    att_scores = att_scores.masked_fill(
                        attn.causal_mask[:, :, :T, :T] == 0,
                        float('-inf')
                    )
                    att_probs = torch.nn.functional.softmax(att_scores, dim=-1)
                    att_probs = attn.attn_dropout(att_probs)
                    attn_output = att_probs @ v_reshaped
                
                # Reshape back and project
                attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, n_heads * d_head)
                attn_out = attn.c_proj(attn_output)
                attn_out = attn.resid_dropout(attn_out)
                attn_out = mask_fn(attn_out, "attn_out")
                
                key = f"layer{layer_idx}_attn_out"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = attn_out[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                x = x + attn_out
                
                # MLP
                normed = block.ln_2(x)
                normed = mask_fn(normed, "mlp_in")
                
                key = f"layer{layer_idx}_mlp_in"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = normed[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                # MLP hidden
                mlp_hidden = block.mlp.c_fc(normed)
                mlp_hidden = block.mlp.act_fn(mlp_hidden)
                mlp_hidden = mask_fn(mlp_hidden, "mlp_neuron")
                
                key = f"layer{layer_idx}_mlp_neuron"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = mlp_hidden[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                mlp_out = block.mlp.c_proj(mlp_hidden)
                mlp_out = block.mlp.dropout(mlp_out)
                mlp_out = mask_fn(mlp_out, "mlp_out")
                
                key = f"layer{layer_idx}_mlp_out"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = mlp_out[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                x = x + mlp_out
    
    # Build dashboard data structures
    print("    Building task dashboard data...")
    dashboards = {}
    
    # Use a fixed random seed for reproducibility
    rng = np.random.default_rng(42)
    
    for n in nodes:
        node_id = n["node_id"]
        examples = node_examples[node_id]
        
        if len(examples) == 0:
            continue
        
        # Collect all activations for stats
        all_acts = []
        for tok_ids, acts in examples:
            all_acts.extend(acts)
        all_acts = np.array(all_acts)
        
        if len(all_acts) == 0:
            continue
        
        # Select random samples (up to n_top * 2 = 20 by default)
        n_random = min(n_top * 2, len(examples))
        random_indices = rng.choice(len(examples), size=n_random, replace=False)
        
        # Build random examples (stored in top_examples, bottom_examples left empty)
        random_examples = []
        for ex_idx in random_indices:
            tok_ids, acts = examples[ex_idx]
            if acts:
                max_act = max(acts)
                max_pos = acts.index(max_act)
                random_examples.append(ActivationExample(
                    tokens=[tokenizer.decode([t]) for t in tok_ids],
                    token_ids=tok_ids,
                    activations=acts,
                    max_activation=max_act,
                    max_position=max_pos,
                ))
        
        dashboards[node_id] = NodeDashboardData(
            node_id=node_id,
            layer=n["layer"],
            location=n["location"],
            index=n["index"],
            top_examples=random_examples,  # Random samples stored in top_examples
            bottom_examples=[],  # Empty for task dashboards
            mean_activation=float(np.mean(all_acts)) if len(all_acts) > 0 else 0.0,
            std_activation=float(np.std(all_acts)) if len(all_acts) > 0 else 0.0,
            frequency=float(np.mean(all_acts != 0)) if len(all_acts) > 0 else 0.0,
        )
    
    return dashboards


def compute_task_original_dashboards_for_nodes(masked_model, tokenizer, nodes, task, n_samples=200, n_top=5, device="cuda", task_prompts=None):
    """
    Compute activation dashboards for nodes using the UNPRUNED (original) model on the TASK dataset.
    
    This is the third dashboard type - showing how the original model activates on task examples.
    
    Args:
        task_prompts: Optional list of pre-generated prompts to use. If provided, uses these
                      instead of generating new ones (for consistency with other dashboard functions).
    """
    import torch
    import numpy as np
    from tqdm import tqdm
    from my_sparse_pretrain.src.visualization.dashboard import NodeDashboardData, ActivationExample
    
    if task is None and task_prompts is None:
        print("    No task provided, skipping task original dashboards")
        return {}
    
    # Use provided prompts or generate new ones
    if task_prompts is not None:
        print(f"    Using {len(task_prompts)} pre-generated task prompts...")
        all_prompts = task_prompts
    else:
        print(f"    Generating task examples for {n_samples} samples...")
        
        # Generate examples from the task
        all_prompts = []
        for _ in range(n_samples):
            example = task.generate_example()
            # Use positive_ids (the sequence where the correct token should be predicted)
            if hasattr(example, 'positive_ids'):
                prompt_ids = example.positive_ids.tolist() if isinstance(example.positive_ids, torch.Tensor) else list(example.positive_ids)
                if prompt_ids:
                    all_prompts.append(prompt_ids)
    
    if not all_prompts:
        print("    No valid prompts generated for task original dashboards")
        return {}
    
    batch_size = 16
    
    # Create batches with RIGHT padding to max length in each batch
    all_batches = []
    all_seq_lengths = []
    
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i+batch_size]
        if len(batch_prompts) < 2:
            continue
        
        max_len = max(len(p) for p in batch_prompts)
        
        padded_batch = []
        seq_lengths = []
        for prompt in batch_prompts:
            seq_len = len(prompt)
            seq_lengths.append(seq_len)
            if seq_len < max_len:
                padded = prompt + [tokenizer.eos_token_id] * (max_len - seq_len)
            else:
                padded = prompt
            padded_batch.append(torch.tensor(padded, dtype=torch.long))
        
        all_batches.append(torch.stack(padded_batch))
        all_seq_lengths.append(seq_lengths)
    
    if not all_batches:
        print("    No valid batches created for task original dashboards")
        return {}
    
    # Group nodes by layer and location
    nodes_by_loc = {}
    for n in nodes:
        key = f"layer{n['layer']}_{n['location']}"
        if key not in nodes_by_loc:
            nodes_by_loc[key] = []
        nodes_by_loc[key].append(n)
    
    # Collect activations using the UNPRUNED model (no masks applied)
    node_examples = {n["node_id"]: [] for n in nodes}
    
    print(f"    Running forward passes with unpruned model...")
    model = masked_model.model  # Use the base model directly
    model.eval()
    
    for batch_idx, input_ids in enumerate(tqdm(all_batches, desc="    Task original batches")):
        input_ids = input_ids.to(device)
        B, T = input_ids.shape
        seq_lengths = all_seq_lengths[batch_idx]
        
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            # Forward through UNPRUNED model (no masks)
            x = model.wte(input_ids)
            if model.wpe is not None:
                pos = torch.arange(T, device=device)
                x = x + model.wpe(pos)
            x = model.drop(x)
            
            for layer_idx, block in enumerate(model.blocks):
                # attn_in (after ln_1)
                normed = block.ln_1(x)
                
                key = f"layer{layer_idx}_attn_in"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = normed[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                # Q, K, V
                qkv = block.attn.c_attn(normed)
                d_head = block.attn.d_head
                n_heads = block.attn.n_heads
                q, k, v = qkv.split(n_heads * d_head, dim=-1)
                
                for loc, tensor in [("attn_q", q), ("attn_k", k), ("attn_v", v)]:
                    key = f"layer{layer_idx}_{loc}"
                    if key in nodes_by_loc:
                        for n in nodes_by_loc[key]:
                            acts = tensor[:, :, n["index"]].float().cpu().numpy()
                            for b in range(B):
                                seq_len = seq_lengths[b]
                                node_examples[n["node_id"]].append((
                                    input_ids[b, :seq_len].cpu().tolist(),
                                    acts[b, :seq_len].tolist()
                                ))
                
                # Attention output
                attn_out = block.attn(normed, None)
                key = f"layer{layer_idx}_attn_out"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = attn_out[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                x = x + attn_out
                
                # MLP
                normed = block.ln_2(x)
                key = f"layer{layer_idx}_mlp_in"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = normed[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                mlp_hidden = block.mlp.c_fc(normed)
                mlp_hidden = block.mlp.act_fn(mlp_hidden)
                key = f"layer{layer_idx}_mlp_neuron"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = mlp_hidden[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                mlp_out = block.mlp.c_proj(mlp_hidden)
                mlp_out = block.mlp.dropout(mlp_out)
                key = f"layer{layer_idx}_mlp_out"
                if key in nodes_by_loc:
                    for n in nodes_by_loc[key]:
                        acts = mlp_out[:, :, n["index"]].float().cpu().numpy()
                        for b in range(B):
                            seq_len = seq_lengths[b]
                            node_examples[n["node_id"]].append((
                                input_ids[b, :seq_len].cpu().tolist(),
                                acts[b, :seq_len].tolist()
                            ))
                
                x = x + mlp_out
    
    # Build dashboard data structures
    print("    Building task original dashboard data...")
    dashboards = {}
    
    # Use a fixed random seed for reproducibility
    rng = np.random.default_rng(42)
    
    for n in nodes:
        node_id = n["node_id"]
        examples = node_examples[node_id]
        
        if len(examples) == 0:
            continue
        
        # Collect all activations for stats
        all_acts = []
        for tok_ids, acts in examples:
            all_acts.extend(acts)
        all_acts = np.array(all_acts)
        
        if len(all_acts) == 0:
            continue
        
        # Select random samples (up to n_top * 2 = 20 by default)
        n_random = min(n_top * 2, len(examples))
        random_indices = rng.choice(len(examples), size=n_random, replace=False)
        
        # Build random examples (stored in top_examples, bottom_examples left empty)
        random_examples = []
        for ex_idx in random_indices:
            tok_ids, acts = examples[ex_idx]
            if acts:
                max_act = max(acts)
                max_pos = acts.index(max_act)
                random_examples.append(ActivationExample(
                    tokens=[tokenizer.decode([t]) for t in tok_ids],
                    token_ids=tok_ids,
                    activations=acts,
                    max_activation=max_act,
                    max_position=max_pos,
                ))
        
        dashboards[node_id] = NodeDashboardData(
            node_id=node_id,
            layer=n["layer"],
            location=n["location"],
            index=n["index"],
            top_examples=random_examples,  # Random samples stored in top_examples
            bottom_examples=[],  # Empty for task dashboards
            mean_activation=float(np.mean(all_acts)) if len(all_acts) > 0 else 0.0,
            std_activation=float(np.std(all_acts)) if len(all_acts) > 0 else 0.0,
            frequency=float(np.mean(all_acts != 0)) if len(all_acts) > 0 else 0.0,
        )
    
    return dashboards


def _get_lightweight_html_template() -> str:
    """Get the HTML template for the circuit viewer with dashboards."""
    return '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Circuit Viewer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
        }
        #container { display: flex; height: 100vh; }
        #graph-container { flex: 1; position: relative; overflow: hidden; background: #1e1e2f; }
        .header { padding: 10px 15px; background: #0f3460; border-bottom: 1px solid #333; font-size: 14px; }
        svg { width: 100%; height: 100%; }
        
        .node { cursor: pointer; }
        .node circle, .node rect, .node polygon { stroke: #555; stroke-width: 1.5px; }
        .node.hovered circle, .node.hovered rect, .node.hovered polygon { stroke: #fff; stroke-width: 3px; }
        .node.selected circle, .node.selected rect, .node.selected polygon { stroke: #ff0; stroke-width: 3px; }
        .node.connected circle, .node.connected rect, .node.connected polygon { stroke: #4a90d9; stroke-width: 2.5px; opacity: 1; }
        .node.connected-upstream circle, .node.connected-upstream rect, .node.connected-upstream polygon { stroke: #00b894; stroke-width: 2.5px; }
        .node.connected-downstream circle, .node.connected-downstream rect, .node.connected-downstream polygon { stroke: #e17055; stroke-width: 2.5px; }
        .node text { font-size: 8px; fill: #888; pointer-events: none; }
        
        /* Attention nodes: orange/yellow/red palette */
        .node-attn_in circle { fill: #e17055; }
        .node-attn_q circle { fill: #ff7675; }
        .node-attn_k circle { fill: #fdcb6e; }
        .node-attn_v circle { fill: #f39c12; }
        .node-attn_out circle { fill: #d63031; }
        /* MLP nodes: blue palette */
        .node-mlp_in circle { fill: #74b9ff; }
        .node-mlp_neuron circle { fill: #0984e3; }
        .node-mlp_out circle { fill: #6c5ce7; }
        .node-embed polygon { fill: #81ecec; }
        .node-unembed polygon { fill: #fab1a0; }
        
        /* Bias-only nodes (no upstream weight connections) */
        .node.bias-only circle, .node.bias-only rect, .node.bias-only polygon {
            stroke: #ff9f43 !important;
            stroke-width: 2px !important;
            stroke-dasharray: 4,2;
        }
        .node.bias-only text { fill: #ff9f43; }
        
        /* Residual channels as vertical lines */
        .resid-channel { stroke: #666; stroke-width: 2px; opacity: 0.5; }
        .resid-channel-highlight { stroke: #aaa; stroke-width: 3px; opacity: 1; display: none; }
        .resid-channel-label { font-size: 9px; fill: #888; }
        
        .link { fill: none; stroke-opacity: 0.3; }
        .link.positive { stroke: #4a90d9; }
        .link.negative { stroke: #e94560; }
        .link.highlighted { stroke-opacity: 1; stroke-width: 2px !important; }
        .link.qk { stroke: #ff9f43; stroke-dasharray: 3,2; }
        .link.selected-edge { stroke-opacity: 1; stroke-width: 3px !important; filter: drop-shadow(0 0 4px currentColor); }
        .link-hitarea { stroke: transparent; stroke-width: 12px; cursor: pointer; fill: none; }
        
        /* Edge product display */
        #edge-product-display {
            position: fixed; bottom: 20px; left: 20px; background: rgba(22, 33, 62, 0.95);
            border: 1px solid #4a90d9; border-radius: 8px; padding: 12px 16px; z-index: 1000;
            display: none; box-shadow: 0 4px 20px rgba(0,0,0,0.4); max-width: 400px;
        }
        #edge-product-display.visible { display: block; }
        #edge-product-display .product-title { color: #4a90d9; font-weight: bold; margin-bottom: 8px; font-size: 12px; }
        #edge-product-display .product-value { font-family: monospace; font-size: 14px; margin-bottom: 8px; }
        #edge-product-display .edge-list { font-size: 10px; color: #888; max-height: 150px; overflow-y: auto; }
        #edge-product-display .edge-item { margin: 2px 0; }
        #edge-product-display .clear-btn { background: #e94560; color: #fff; border: none; border-radius: 4px; padding: 4px 12px; cursor: pointer; font-size: 11px; margin-top: 8px; }
        #edge-product-display .clear-btn:hover { background: #ff6b81; }
        
        .node-info { background: #222; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        
        .zoom-controls { position: absolute; bottom: 20px; right: 20px; display: flex; gap: 5px; }
        .zoom-btn { width: 30px; height: 30px; background: #333; border: 1px solid #555; color: #fff; border-radius: 4px; cursor: pointer; }
        .zoom-btn:hover { background: #444; }
        
        /* Dashboard panel on right side - 50% of screen */
        #dashboard-tooltip {
            position: fixed; background: #16213e; border-left: 1px solid #444;
            padding: 15px; width: 50vw; height: calc(100vh - 50px); overflow-y: auto; z-index: 1000;
            display: none; box-shadow: -4px 0 20px rgba(0,0,0,0.3); font-size: 11px;
            right: 0; top: 50px;
        }
        #dashboard-tooltip .tooltip-header { font-size: 14px; }
        #dashboard-tooltip .token-span { font-size: 10px; padding: 2px 3px; }
        #dashboard-tooltip.visible { display: block; }
        #dashboard-tooltip.pinned { border-left: 3px solid #4a90d9; }
        #dashboard-tooltip .close-btn { position: absolute; top: 8px; right: 10px; cursor: pointer; color: #888; font-size: 16px; }
        #dashboard-tooltip .close-btn:hover { color: #fff; }
        .tooltip-header { font-weight: bold; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #444; color: #4a90d9; }
        .no-data { color: #888; font-style: italic; }
        .token-span:hover { outline: 2px solid #fff; cursor: pointer; }
        .expand-all-btn:hover { background: #3a5f8f !important; }
        .dash-type-btn:hover { opacity: 0.8; }
        .dash-type-btn.active { font-weight: bold; }
        
        /* Edge tooltip */
        #edge-tooltip {
            position: fixed; background: rgba(22, 33, 62, 0.95); border: 1px solid #444;
            padding: 8px 12px; border-radius: 4px; font-size: 11px; z-index: 2000;
            pointer-events: none; display: none; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        #edge-tooltip.visible { display: block; }
        
        /* Edge weights section in dashboard */
        .edge-weights-section { margin: 10px 0; padding: 8px; background: #1a1a2e; border-radius: 4px; }
        .edge-weights-section h4 { font-size: 11px; color: #4a90d9; margin-bottom: 6px; }
        .edge-weight-item { font-size: 10px; margin: 3px 0; display: flex; justify-content: space-between; }
        .edge-weight-item .node-name { color: #aaa; }
        .edge-weight-item .weight-value { font-family: monospace; }
        .edge-weight-item .weight-positive { color: #4a90d9; }
        .edge-weight-item .weight-negative { color: #e94560; }
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <div class="header">
                <strong>Circuit Graph</strong> - <span id="graph-summary"></span>
            </div>
            <svg id="graph-svg"></svg>
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomIn()">+</button>
                <button class="zoom-btn" onclick="zoomOut()">−</button>
                <button class="zoom-btn" onclick="resetZoom()">⟲</button>
            </div>
        </div>
    </div>
    <div id="dashboard-tooltip"></div>
    <div id="edge-tooltip"></div>
    <div id="edge-product-display">
        <div class="product-title">Edge Product</div>
        <div class="product-value" id="product-value"></div>
        <div class="edge-list" id="edge-list"></div>
        <button class="clear-btn" onclick="clearSelectedEdges()">Clear Selection</button>
    </div>
    
    <script>
        const graphData = __GRAPH_DATA__;
        const pretrainOriginalData = __PRETRAIN_ORIGINAL_DATA__;
        const taskOriginalData = __TASK_ORIGINAL_DATA__;
        const taskPrunedData = __TASK_PRUNED_DATA__;
        
        // Dashboard type toggle state
        // Types: 'pretrain_original', 'task_original', 'task_pruned'
        let currentDashboardType = 'pretrain_original';
        let currentDashboardNode = null;
        
        function setDashboardType(type) {
            currentDashboardType = type;
            // Update button states
            document.querySelectorAll('.dashboard-type-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            const activeBtn = document.querySelector(`.dashboard-type-btn[data-type="${type}"]`);
            if (activeBtn) activeBtn.classList.add('active');
            
            if (currentDashboardNode) {
                showDashboard(currentDashboardNode, true);
            }
        }
        
        function getDashboardData(nodeId) {
            if (currentDashboardType === 'pretrain_original') {
                return pretrainOriginalData[nodeId];
            } else if (currentDashboardType === 'task_original') {
                return taskOriginalData[nodeId];
            } else {
                return taskPrunedData[nodeId];
            }
        }
        
        function hasDashboardData(nodeId) {
            return pretrainOriginalData[nodeId] || taskOriginalData[nodeId] || taskPrunedData[nodeId];
        }
        
        // Layout constants
        const NODE_SPACING = 14;
        const HEAD_GAP = 30;
        const SECTION_GAP = 40;
        const LAYER_GAP = 15;  // Reduced space between layers
        const COLUMN_GAP = 50;  // Gap between mask nodes and resid channels
        const EMBED_ZONE_HEIGHT = 30;  // Minimal space for embed/unembed nodes
        const EMBED_NODE_SPACING = 35;  // Horizontal spacing between embed nodes
        const MARGIN = { top: 30, right: 50, bottom: 30, left: 80 };
        
        const svg = d3.select("#graph-svg");
        const container = d3.select("#graph-container");
        const tooltip = d3.select("#dashboard-tooltip");
        
        let width = container.node().clientWidth;
        let height = container.node().clientHeight - 40;
        
        const g = svg.append("g");
        
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => g.attr("transform", event.transform));
        svg.call(zoom);
        
        function zoomIn() { svg.transition().call(zoom.scaleBy, 1.3); }
        function zoomOut() { svg.transition().call(zoom.scaleBy, 0.7); }
        function resetZoom() { svg.transition().call(zoom.transform, d3.zoomIdentity); }
        
        const nodes = graphData.nodes;
        const edges = graphData.edges || [];
        const metadata = graphData.metadata;
        const d_head = metadata.d_head || 64;
        
        document.getElementById("graph-summary").textContent = 
            `${nodes.length} nodes, ${edges.length} edges, ${metadata.n_layers} layers`;
        
        // Categorize nodes
        const embedNodes = nodes.filter(n => n.location === 'embed');
        const unembedNodes = nodes.filter(n => n.location === 'unembed');
        const residChannelNodes = nodes.filter(n => n.location === 'resid_channel');
        const maskSiteNodes = nodes.filter(n => 
            !n.location.startsWith('resid') && n.location !== 'embed' && n.location !== 'unembed');
        
        // Group by layer
        function groupByLayer(nodeList) {
            const groups = {};
            nodeList.forEach(n => {
                if (!groups[n.layer]) groups[n.layer] = [];
                groups[n.layer].push(n);
            });
            return groups;
        }
        
        const maskSiteByLayer = groupByLayer(maskSiteNodes);
        
        // For attention nodes, group by head
        function getActiveHeads(layer) {
            const heads = new Set();
            (maskSiteByLayer[layer] || []).forEach(n => {
                if (['attn_q', 'attn_k', 'attn_v'].includes(n.location) && n.head_idx !== null) {
                    heads.add(n.head_idx);
                }
            });
            return Array.from(heads).sort((a, b) => a - b);
        }
        
        // Calculate layer heights dynamically
        function calcLayerHeight(layer) {
            const heads = getActiveHeads(layer);
            const attnHeight = Math.max(heads.length * (NODE_SPACING * 3 + HEAD_GAP), 50);
            const mlpNodes = (maskSiteByLayer[layer] || []).filter(n => n.location.startsWith('mlp'));
            const mlpHeight = mlpNodes.length > 0 ? 80 : 0;
            return attnHeight + mlpHeight + SECTION_GAP * 3;
        }
        
        // Calculate total height
        let totalLayerHeight = 0;
        const layerYStarts = {};
        for (let layer = 0; layer < metadata.n_layers; layer++) {
            layerYStarts[layer] = totalLayerHeight;
            totalLayerHeight += calcLayerHeight(layer) + LAYER_GAP;
        }
        
        const totalHeight = totalLayerHeight + MARGIN.top + MARGIN.bottom + 2 * EMBED_ZONE_HEIGHT;
        
        // Calculate widths
        let maxLeftWidth = 200;
        let maxRightWidth = 100;
        
        for (let layer = 0; layer < metadata.n_layers; layer++) {
            const heads = getActiveHeads(layer);
            const leftW = heads.length * (NODE_SPACING * 4 + HEAD_GAP) + 100;
            maxLeftWidth = Math.max(maxLeftWidth, leftW);
        }
        // Right column width based on residual channels and unembed nodes
        maxRightWidth = Math.max(
            residChannelNodes.length * NODE_SPACING + 30,
            unembedNodes.length * NODE_SPACING + 30,
            100
        );
        
        const totalWidth = MARGIN.left + maxLeftWidth + COLUMN_GAP + maxRightWidth + MARGIN.right;
        const leftColumnX = MARGIN.left + maxLeftWidth / 2;
        const rightColumnX = MARGIN.left + maxLeftWidth + COLUMN_GAP + maxRightWidth / 2;
        
        // Create node lookup
        const nodeById = new Map(nodes.map(n => [n.node_id, n]));
        
        // Track Y positions for residual alignment
        const layerYPositions = {};  // layer -> {resid_pre_y, resid_mid_y, resid_post_y}
        
        // Position mask site nodes (left column) - organized by layer sections
        // Within each layer (bottom to top): attn_in -> Q/K/V by head -> attn_out -> mlp_in -> mlp_neuron -> mlp_out
        for (let layer = 0; layer < metadata.n_layers; layer++) {
            const layerTop = totalHeight - MARGIN.bottom - EMBED_ZONE_HEIGHT - layerYStarts[layer] - calcLayerHeight(layer);
            const layerNodes = maskSiteByLayer[layer] || [];
            const heads = getActiveHeads(layer);
            
            let currentY = layerTop + calcLayerHeight(layer);  // Start from bottom
            
            // attn_in at bottom of attention section - THIS IS WHERE resid_pre ALIGNS
            const attnInY = currentY - 15;
            const attnInNodes = layerNodes.filter(n => n.location === 'attn_in');
            attnInNodes.forEach((n, idx) => {
                n.y = attnInY;
                const groupW = attnInNodes.length * NODE_SPACING;
                n.x = leftColumnX - groupW / 2 + idx * NODE_SPACING + NODE_SPACING / 2;
            });
            currentY -= 35;
            
            // Q, K, V organized by head (Q and K side by side, V to right)
            heads.forEach((headIdx, hi) => {
                const headX = leftColumnX - (heads.length - 1) * (NODE_SPACING * 3 + HEAD_GAP) / 2 + hi * (NODE_SPACING * 3 + HEAD_GAP);
                
                const qNodes = layerNodes.filter(n => n.location === 'attn_q' && n.head_idx === headIdx);
                const kNodes = layerNodes.filter(n => n.location === 'attn_k' && n.head_idx === headIdx);
                const vNodes = layerNodes.filter(n => n.location === 'attn_v' && n.head_idx === headIdx);
                
                // Q nodes (left)
                qNodes.forEach((n, idx) => {
                    n.x = headX - NODE_SPACING;
                    n.y = currentY - idx * NODE_SPACING;
                });
                
                // K nodes (right of Q, same y)
                kNodes.forEach((n, idx) => {
                    n.x = headX;
                    n.y = currentY - idx * NODE_SPACING;
                });
                
                // V nodes (right of K)
                vNodes.forEach((n, idx) => {
                    n.x = headX + NODE_SPACING;
                    n.y = currentY - idx * NODE_SPACING;
                });
            });
            
            const maxQKVNodes = Math.max(
                ...heads.map(h => layerNodes.filter(n => ['attn_q', 'attn_k', 'attn_v'].includes(n.location) && n.head_idx === h).length / 3)
            , 1);
            currentY -= maxQKVNodes * NODE_SPACING + 25;
            
            // attn_out above Q/K/V
            const attnOutNodes = layerNodes.filter(n => n.location === 'attn_out');
            attnOutNodes.forEach((n, idx) => {
                n.y = currentY;
                const groupW = attnOutNodes.length * NODE_SPACING;
                n.x = leftColumnX - groupW / 2 + idx * NODE_SPACING + NODE_SPACING / 2;
            });
            currentY -= SECTION_GAP / 2;
            
            // THIS IS WHERE resid_mid ALIGNS (between attention and MLP)
            const residMidY = currentY;
            currentY -= SECTION_GAP / 2;
            
            // MLP section: mlp_in -> mlp_neuron -> mlp_out (bottom to top)
            const mlpInNodes = layerNodes.filter(n => n.location === 'mlp_in');
            mlpInNodes.forEach((n, idx) => {
                n.y = currentY;
                const groupW = mlpInNodes.length * NODE_SPACING;
                n.x = leftColumnX - groupW / 2 + idx * NODE_SPACING + NODE_SPACING / 2;
            });
            currentY -= 25;
            
            const mlpNeuronNodes = layerNodes.filter(n => n.location === 'mlp_neuron');
            mlpNeuronNodes.forEach((n, idx) => {
                n.y = currentY;
                const groupW = mlpNeuronNodes.length * NODE_SPACING;
                n.x = leftColumnX - groupW / 2 + idx * NODE_SPACING + NODE_SPACING / 2;
            });
            currentY -= 25;
            
            const mlpOutNodes = layerNodes.filter(n => n.location === 'mlp_out');
            const mlpOutY = currentY;
            mlpOutNodes.forEach((n, idx) => {
                n.y = mlpOutY;
                const groupW = mlpOutNodes.length * NODE_SPACING;
                n.x = leftColumnX - groupW / 2 + idx * NODE_SPACING + NODE_SPACING / 2;
            });
            
            // Store Y positions for residual nodes
            // Offset so that reading nodes are clearly ABOVE their residual source
            const RESID_OFFSET = 45;  // Larger vertical offset for steeper slant
            layerYPositions[layer] = {
                resid_pre_y: attnInY + RESID_OFFSET,      // Below attn_in (attn_in reads from resid_pre)
                resid_mid_y: mlpInNodes.length > 0 ? mlpInNodes[0].y + RESID_OFFSET : residMidY,  // Below mlp_in
                resid_post_y: mlpOutY - RESID_OFFSET      // Above mlp_out (mlp_out writes to resid_post)
            };
        }
        
        // Position residual channels as vertical lines (right column)
        // Each channel spans from bottom of layer 0 to top of last layer
        const channelBottomY = totalHeight - MARGIN.bottom - EMBED_ZONE_HEIGHT + 20;
        // Use the topmost layer's resid_post_y as channel top (where mlp_out writes to)
        const topLayer = metadata.n_layers - 1;
        const channelTopY = layerYPositions[topLayer] ? layerYPositions[topLayer].resid_post_y - 10 : MARGIN.top;
        
        // Initial x positioning for residual channels
        residChannelNodes.forEach((n, idx) => {
            const groupW = residChannelNodes.length * NODE_SPACING;
            n.x = rightColumnX - groupW / 2 + idx * NODE_SPACING + NODE_SPACING / 2;
            // Initialize with full range, will be trimmed below
            n.y1 = channelBottomY;  // Bottom of channel (larger Y)
            n.y2 = channelTopY;     // Top of channel (smaller Y)
        });
        
        // Helper to get Y coordinate for edges to/from resid_channel
        // Defined early so we can use it to compute channel bounds
        function getChannelY(edge, isSource) {
            const node = nodeById.get(isSource ? edge.source_id : edge.target_id);
            if (!node || node.location !== 'resid_channel') {
                return node?.y || 0;
            }
            // Handle embed/unembed edges FIRST - they connect at channel ends
            // (even though they have layer info, we want them at the very ends)
            if (edge.edge_type === 'embed') {
                return channelBottomY;  // Embed connects at the very bottom
            }
            if (edge.edge_type === 'unembed') {
                return channelTopY;  // Unembed connects at the very top
            }
            // For mask site edges, use the layer position of the mask site node
            const layer = isSource ? edge.target_layer : edge.source_layer;
            const position = edge.resid_position;
            if (layer !== undefined && layerYPositions[layer]) {
                if (position === 'pre') return layerYPositions[layer].resid_pre_y;
                if (position === 'mid') return layerYPositions[layer].resid_mid_y;
                if (position === 'post') return layerYPositions[layer].resid_post_y;
            }
            return node.y;
        }
        
        // Position embed nodes - their Y should be just below the lowest channel bottom they connect to
        embedNodes.forEach((n, idx) => {
            const groupW = embedNodes.length * EMBED_NODE_SPACING;
            n.x = rightColumnX - groupW / 2 + idx * EMBED_NODE_SPACING + EMBED_NODE_SPACING / 2;
            n.y = channelBottomY + 25;  // Below all channels
        });
        
        // Position unembed nodes - their Y should be just above the highest channel top they connect to
        unembedNodes.forEach((n, idx) => {
            const groupW = unembedNodes.length * EMBED_NODE_SPACING;
            n.x = rightColumnX - groupW / 2 + idx * EMBED_NODE_SPACING + EMBED_NODE_SPACING / 2;
            n.y = channelTopY - 20;  // Above all channels
        });
        
        // Draw background labels
        const layerBg = g.append("g").attr("class", "layer-backgrounds");
        
        layerBg.append("text").attr("x", leftColumnX).attr("y", MARGIN.top - EMBED_ZONE_HEIGHT - 20)
            .attr("fill", "#888").attr("font-size", "13px").attr("text-anchor", "middle").attr("font-weight", "bold")
            .text("Mask Site Nodes");
        layerBg.append("text").attr("x", rightColumnX).attr("y", MARGIN.top - EMBED_ZONE_HEIGHT - 20)
            .attr("fill", "#888").attr("font-size", "13px").attr("text-anchor", "middle").attr("font-weight", "bold")
            .text("Residual Stream");
        
        for (let layer = 0; layer < metadata.n_layers; layer++) {
            const layerTop = totalHeight - MARGIN.bottom - EMBED_ZONE_HEIGHT - layerYStarts[layer] - calcLayerHeight(layer);
            layerBg.append("text").attr("x", 20).attr("y", layerTop + calcLayerHeight(layer) / 2)
                .attr("fill", "#666").attr("font-size", "12px").attr("font-weight", "bold")
                .attr("dominant-baseline", "middle").text(`L${layer}`);
        }
        
        if (embedNodes.length > 0) {
            layerBg.append("text").attr("x", rightColumnX).attr("y", channelBottomY + 50)
                .attr("fill", "#81ecec").attr("font-size", "11px").attr("text-anchor", "middle").text("Embedding");
        }
        if (unembedNodes.length > 0) {
            layerBg.append("text").attr("x", rightColumnX).attr("y", channelTopY - 40)
                .attr("fill", "#fab1a0").attr("font-size", "11px").attr("text-anchor", "middle").text("Unembedding");
        }
        
        // Draw residual channel vertical lines first (behind other elements)
        // Compute actual Y bounds from edges for each channel (ensures exact match with edge endpoints)
        const channelBounds = new Map();
        residChannelNodes.forEach(n => {
            let minY = Infinity, maxY = -Infinity;
            edges.forEach(e => {
                let y = null;
                if (e.source_id === n.node_id || e.target_id === n.node_id) {
                    // Use getChannelY to get exact same Y as edge endpoints
                    if (e.source_id === n.node_id) {
                        y = getChannelY(e, true);  // Channel is source
                    } else {
                        y = getChannelY(e, false); // Channel is target
                    }
                }
                if (y !== null && y !== undefined) {
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            });
            if (minY !== Infinity && maxY !== -Infinity) {
                channelBounds.set(n.node_id, { minY, maxY });
                n.y2 = minY;  // Top (smaller Y)
                n.y1 = maxY;  // Bottom (larger Y)
            }
        });
        
        const channelGroup = g.append("g").attr("class", "resid-channels");
        residChannelNodes.forEach(n => {
            // Base channel line (always visible, dim)
            channelGroup.append("line")
                .attr("class", "resid-channel")
                .attr("data-node-id", n.node_id)
                .attr("x1", n.x).attr("y1", n.y1)
                .attr("x2", n.x).attr("y2", n.y2);
            // Highlight overlay line (initially hidden, shows only the active segment)
            channelGroup.append("line")
                .attr("class", "resid-channel-highlight")
                .attr("data-node-id", n.node_id)
                .attr("x1", n.x).attr("y1", n.y1)
                .attr("x2", n.x).attr("y2", n.y2)
                .style("display", "none");
        });
        
        // Draw edges
        const edgeTooltip = d3.select("#edge-tooltip");
        const edgeProductDisplay = d3.select("#edge-product-display");
        const selectedEdges = new Map();
        
        // Create link group
        const linksGroup = g.append("g").attr("class", "links");
        
        // Draw invisible hit areas first (behind visible edges) for easier clicking
        const linkHitAreas = linksGroup.selectAll("line.link-hitarea").data(edges).join("line")
            .attr("class", "link-hitarea")
            .attr("x1", d => nodeById.get(d.source_id)?.x || 0)
            .attr("y1", d => getChannelY(d, true))
            .attr("x2", d => nodeById.get(d.target_id)?.x || 0)
            .attr("y2", d => getChannelY(d, false))
            .style("display", d => (nodeById.get(d.source_id) && nodeById.get(d.target_id)) ? "block" : "none")
            .on("mouseenter", (event, d) => showEdgeTooltip(event, d))
            .on("mousemove", (event) => moveEdgeTooltip(event))
            .on("mouseleave", () => hideEdgeTooltip())
            .on("click", (event, d) => toggleEdgeSelection(d));
        
        // Draw visible edges
        const linkElements = linksGroup.selectAll("line.link").data(edges).join("line")
            .attr("class", d => `link ${d.weight >= 0 ? 'positive' : 'negative'} ${d.edge_type === 'qk_match' ? 'qk' : ''}`)
            .attr("stroke-width", d => Math.max(0.5, Math.min(3, Math.abs(d.weight) * 2)))
            .attr("x1", d => nodeById.get(d.source_id)?.x || 0)
            .attr("y1", d => getChannelY(d, true))
            .attr("x2", d => nodeById.get(d.target_id)?.x || 0)
            .attr("y2", d => getChannelY(d, false))
            .style("display", d => (nodeById.get(d.source_id) && nodeById.get(d.target_id)) ? "block" : "none")
            .style("pointer-events", "none");  // Let hit areas handle events
        
        function showEdgeTooltip(event, d) {
            const weightColor = d.weight >= 0 ? '#4a90d9' : '#e94560';
            const srcNode = nodeById.get(d.source_id);
            const tgtNode = nodeById.get(d.target_id);
            const srcName = srcNode?.token_str || srcNode?.node_id || d.source_id;
            const tgtName = tgtNode?.token_str || tgtNode?.node_id || d.target_id;
            edgeTooltip.html(`
                <div><strong>Edge Weight:</strong> <span style="color:${weightColor}">${d.weight.toFixed(4)}</span></div>
                <div style="color:#888;font-size:10px;margin-top:4px;">${srcName} → ${tgtName}</div>
                <div style="color:#666;font-size:9px;">Type: ${d.edge_type || 'weight'}</div>
                <div style="color:#4a90d9;font-size:9px;margin-top:4px;">Click to add to product</div>
            `);
            edgeTooltip.classed("visible", true)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }
        
        function moveEdgeTooltip(event) {
            edgeTooltip.style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }
        
        function hideEdgeTooltip() {
            edgeTooltip.classed("visible", false);
        }
        
        function toggleEdgeSelection(d) {
            // Create a unique key for this edge
            const edgeKey = `${d.source_id}|${d.target_id}`;
            
            if (selectedEdges.has(edgeKey)) {
                selectedEdges.delete(edgeKey);
            } else {
                selectedEdges.set(edgeKey, d);
            }
            
            updateEdgeSelectionDisplay();
        }
        
        function updateEdgeSelectionDisplay() {
            // Update visual highlighting
            linkElements.classed("selected-edge", d => selectedEdges.has(`${d.source_id}|${d.target_id}`));
            
            if (selectedEdges.size === 0) {
                edgeProductDisplay.classed("visible", false);
                return;
            }
            
            // Calculate product
            let product = 1;
            const edgeListHtml = [];
            selectedEdges.forEach((edge, key) => {
                product *= edge.weight;
                const srcNode = nodeById.get(edge.source_id);
                const tgtNode = nodeById.get(edge.target_id);
                const srcName = srcNode?.token_str || srcNode?.node_id || edge.source_id;
                const tgtName = tgtNode?.token_str || tgtNode?.node_id || edge.target_id;
                const weightColor = edge.weight >= 0 ? '#4a90d9' : '#e94560';
                edgeListHtml.push(`<div class="edge-item"><span style="color:${weightColor}">${edge.weight.toFixed(4)}</span> (${srcName} → ${tgtName})</div>`);
            });
            
            const productColor = product >= 0 ? '#4a90d9' : '#e94560';
            d3.select("#product-value").html(`<span style="color:${productColor}">${product.toExponential(4)}</span> <span style="color:#888;font-size:11px;">(${selectedEdges.size} edge${selectedEdges.size > 1 ? 's' : ''})</span>`);
            d3.select("#edge-list").html(edgeListHtml.join(''));
            edgeProductDisplay.classed("visible", true);
        }
        
        window.clearSelectedEdges = function() {
            selectedEdges.clear();
            updateEdgeSelectionDisplay();
        };
        
        // Draw nodes (excluding resid_channel which are drawn as lines)
        const renderableNodes = nodes.filter(n => n.location !== 'resid_channel');
        const nodeElements = g.append("g").attr("class", "nodes").selectAll("g").data(renderableNodes).join("g")
            .attr("class", d => `node node-${d.location}${d.bias_only ? ' bias-only' : ''}`)
            .attr("transform", d => `translate(${d.x},${d.y})`)
            .on("click", (event, d) => { selectNode(d); showDashboard(d, true); })
            .on("mouseenter", (event, d) => { hoverNode(d); if (!tooltipPinned) showDashboard(d, false); })
            .on("mouseleave", () => { unhoverNode(); if (!tooltipPinned) tooltip.classed("visible", false); });
        
        nodeElements.each(function(d) {
            const el = d3.select(this);
            if (d.location === 'embed') {
                el.append("polygon").attr("points", "0,-6 6,0 0,6 -6,0");
                el.append("text").attr("x", 0).attr("y", 14)
                    .attr("text-anchor", "middle").attr("fill", "#81ecec")
                    .attr("font-size", "9px").text(d.token_str || `T${d.index}`);
            } else if (d.location === 'unembed') {
                el.append("polygon").attr("points", "0,-6 6,0 0,6 -6,0");
                el.append("text").attr("x", 0).attr("y", -10)
                    .attr("text-anchor", "middle").attr("fill", "#fab1a0")
                    .attr("font-size", "9px").text(d.token_str || `T${d.index}`);
            } else {
                el.append("circle").attr("r", 4);
            }
        });
        
        let selectedNode = null;
        let tooltipPinned = false;
        
        // Helper: find all nodes connected through residual channels
        function findConnectedNodes(node) {
            const upstreamNodes = new Set();
            const downstreamNodes = new Set();
            const connectedChannels = new Set();
            const allConnectedEdges = new Set();
            // Track Y range for each channel: { channelId: { minY, maxY } }
            const channelYRanges = new Map();
            
            function updateChannelYRange(channelId, y) {
                if (!channelYRanges.has(channelId)) {
                    channelYRanges.set(channelId, { minY: y, maxY: y });
                } else {
                    const range = channelYRanges.get(channelId);
                    range.minY = Math.min(range.minY, y);
                    range.maxY = Math.max(range.maxY, y);
                }
            }
            
            // Find channels this node connects to
            edges.forEach(e => {
                if (e.source_id === node.node_id) {
                    allConnectedEdges.add(e);
                    const targetNode = nodeById.get(e.target_id);
                    if (targetNode?.location === 'resid_channel') {
                        // Node writes to this channel - find all readers (downstream)
                        connectedChannels.add(e.target_id);
                        // Get Y where this node connects to the channel
                        const sourceY = getChannelY(e, false);
                        updateChannelYRange(e.target_id, sourceY);
                        
                        edges.forEach(e2 => {
                            if (e2.source_id === e.target_id && e2.target_id !== node.node_id) {
                                downstreamNodes.add(e2.target_id);
                                allConnectedEdges.add(e2);
                                // Get Y where downstream node connects to the channel
                                const targetY = getChannelY(e2, true);
                                updateChannelYRange(e.target_id, targetY);
                            }
                        });
                    } else {
                        downstreamNodes.add(e.target_id);
                    }
                }
                if (e.target_id === node.node_id) {
                    allConnectedEdges.add(e);
                    const sourceNode = nodeById.get(e.source_id);
                    if (sourceNode?.location === 'resid_channel') {
                        // Node reads from this channel - find all writers (upstream)
                        connectedChannels.add(e.source_id);
                        // Get Y where this node connects to the channel
                        const targetY = getChannelY(e, true);
                        updateChannelYRange(e.source_id, targetY);
                        
                        edges.forEach(e2 => {
                            if (e2.target_id === e.source_id && e2.source_id !== node.node_id) {
                                upstreamNodes.add(e2.source_id);
                                allConnectedEdges.add(e2);
                                // Get Y where upstream node connects to the channel
                                const sourceY = getChannelY(e2, false);
                                updateChannelYRange(e.source_id, sourceY);
                            }
                        });
                    } else {
                        upstreamNodes.add(e.source_id);
                    }
                }
            });
            
            return { upstreamNodes, downstreamNodes, connectedChannels, allConnectedEdges, channelYRanges };
        }
        
        function selectNode(node) {
            selectedNode = node;
            const { upstreamNodes, downstreamNodes, connectedChannels, allConnectedEdges, channelYRanges } = findConnectedNodes(node);
            
            nodeElements
                .classed("selected", d => d === node)
                .classed("connected-upstream", d => upstreamNodes.has(d.node_id))
                .classed("connected-downstream", d => downstreamNodes.has(d.node_id));
            
            linkElements.classed("highlighted", l => allConnectedEdges.has(l));
            
            // Update channel highlight overlays to show only the active segment
            d3.selectAll(".resid-channel-highlight").each(function() {
                const el = d3.select(this);
                const channelId = el.attr("data-node-id");
                if (connectedChannels.has(channelId) && channelYRanges.has(channelId)) {
                    const range = channelYRanges.get(channelId);
                    el.attr("y1", range.maxY).attr("y2", range.minY)
                      .style("display", "block");
                } else {
                    el.style("display", "none");
                }
            });
        }
        
        function hoverNode(node) {
            const { upstreamNodes, downstreamNodes, connectedChannels, allConnectedEdges, channelYRanges } = findConnectedNodes(node);
            
            nodeElements
                .classed("hovered", d => d === node)
                .classed("connected-upstream", d => upstreamNodes.has(d.node_id))
                .classed("connected-downstream", d => downstreamNodes.has(d.node_id));
            
            linkElements.classed("highlighted", l => allConnectedEdges.has(l));
            
            // Update channel highlight overlays to show only the active segment
            d3.selectAll(".resid-channel-highlight").each(function() {
                const el = d3.select(this);
                const channelId = el.attr("data-node-id");
                if (connectedChannels.has(channelId) && channelYRanges.has(channelId)) {
                    const range = channelYRanges.get(channelId);
                    el.attr("y1", range.maxY).attr("y2", range.minY)
                      .style("display", "block");
                } else {
                    el.style("display", "none");
                }
            });
        }
        
        function unhoverNode() {
            nodeElements.classed("hovered", false);
            if (selectedNode) {
                // Restore the selected node's highlighting
                const { upstreamNodes, downstreamNodes, connectedChannels, allConnectedEdges, channelYRanges } = findConnectedNodes(selectedNode);
                nodeElements
                    .classed("connected-upstream", d => upstreamNodes.has(d.node_id))
                    .classed("connected-downstream", d => downstreamNodes.has(d.node_id));
                linkElements.classed("highlighted", l => allConnectedEdges.has(l));
                d3.selectAll(".resid-channel-highlight").each(function() {
                    const el = d3.select(this);
                    const channelId = el.attr("data-node-id");
                    if (connectedChannels.has(channelId) && channelYRanges.has(channelId)) {
                        const range = channelYRanges.get(channelId);
                        el.attr("y1", range.maxY).attr("y2", range.minY).style("display", "block");
                    } else {
                        el.style("display", "none");
                    }
                });
            } else {
                nodeElements.classed("connected-upstream", false).classed("connected-downstream", false);
                d3.selectAll(".resid-channel-highlight").style("display", "none");
                linkElements.classed("highlighted", false);
            }
        }
        
        function showDashboard(node, pinned) {
            currentDashboardNode = node;
            const dashData = getDashboardData(node.node_id);
            const hasTaskOriginal = taskOriginalData[node.node_id] !== undefined;
            const hasTaskPruned = taskPrunedData[node.node_id] !== undefined;
            const hasAnyTaskData = hasTaskOriginal || hasTaskPruned;
            let html = pinned ? '<span class="close-btn" onclick="closeDashboard()">✕</span>' : '';
            
            // Node info header
            if (node.location === 'embed' || node.location === 'unembed') {
                html += `<div class="tooltip-header">${node.location} - ${node.token_str || 'T' + node.index}</div>`;
            } else if (node.location === 'resid_channel') {
                html += `<div class="tooltip-header">Residual Channel Index ${node.index}</div>`;
            } else {
                html += `<div class="tooltip-header">${node.location} @ Layer ${node.layer}, Index ${node.index}</div>`;
                if (node.head_idx !== null && node.head_idx !== undefined) {
                    html += `<div style="color:#888;font-size:11px;">Head ${node.head_idx}, Dim ${node.head_dim_idx}</div>`;
                }
            }
            
            // Dashboard type toggle - three options
            html += `<div style="margin:8px 0;font-size:10px;font-weight:bold;color:#aaa;">Dashboard Type:</div>`;
            html += `<div class="dashboard-type-toggle" style="margin:5px 0 8px 0;display:flex;gap:4px;flex-wrap:wrap;">`;
            
            // Pretrain Original button
            const pretrainActive = currentDashboardType === 'pretrain_original' ? 'active' : '';
            html += `<button class="dashboard-type-btn ${pretrainActive}" data-type="pretrain_original" onclick="setDashboardType('pretrain_original')" style="flex:1;min-width:90px;padding:6px 8px;font-size:9px;border:1px solid #4a90d9;border-radius:4px;cursor:pointer;background:${currentDashboardType==='pretrain_original'?'#4a90d9':'#1a1a2e'};color:#fff;">Pretrain<br/>Original</button>`;
            
            // Task Original button
            const taskOrigActive = currentDashboardType === 'task_original' ? 'active' : '';
            const taskOrigDisabled = !hasTaskOriginal ? 'opacity:0.4;cursor:not-allowed;' : '';
            html += `<button class="dashboard-type-btn ${taskOrigActive}" data-type="task_original" onclick="${hasTaskOriginal ? "setDashboardType('task_original')" : ''}" style="flex:1;min-width:90px;padding:6px 8px;font-size:9px;border:1px solid #43aa8b;border-radius:4px;cursor:pointer;background:${currentDashboardType==='task_original'?'#43aa8b':'#1a1a2e'};color:#fff;${taskOrigDisabled}">Task<br/>Original</button>`;
            
            // Task Pruned button
            const taskPrunedActive = currentDashboardType === 'task_pruned' ? 'active' : '';
            const taskPrunedDisabled = !hasTaskPruned ? 'opacity:0.4;cursor:not-allowed;' : '';
            html += `<button class="dashboard-type-btn ${taskPrunedActive}" data-type="task_pruned" onclick="${hasTaskPruned ? "setDashboardType('task_pruned')" : ''}" style="flex:1;min-width:90px;padding:6px 8px;font-size:9px;border:1px solid #e94560;border-radius:4px;cursor:pointer;background:${currentDashboardType==='task_pruned'?'#e94560':'#1a1a2e'};color:#fff;${taskPrunedDisabled}">Task<br/>Pruned</button>`;
            
            html += `</div>`;
            
            // Description of current dashboard type
            const typeDescriptions = {
                'pretrain_original': 'Original model on pretraining data (SimpleStories)',
                'task_original': 'Original model on task data',
                'task_pruned': 'Pruned model on task data'
            };
            html += `<div style="margin-bottom:8px;font-size:9px;color:#888;">${typeDescriptions[currentDashboardType]}</div>`;
            
            // Bias info
            if (node.bias !== null && node.bias !== undefined) {
                const biasColor = node.bias >= 0 ? '#4a90d9' : '#e94560';
                html += `<div style="margin:5px 0;"><strong>Bias:</strong> <span style="color:${biasColor}">${node.bias.toFixed(4)}</span></div>`;
            }
            if (node.bias_only) {
                html += `<div style="color:#ff9f43;margin-bottom:8px;"><strong>⚡ Bias-only</strong> (no weight inputs)</div>`;
            }
            
            // Connections summary (no top weights listing)
            const incoming = edges.filter(e => e.target_id === node.node_id);
            const outgoing = edges.filter(e => e.source_id === node.node_id);
            if (incoming.length > 0 || outgoing.length > 0) {
                html += `<div style="color:#888;font-size:10px;margin-bottom:8px;border-bottom:1px solid #333;padding-bottom:5px;">`;
                html += `${incoming.length} incoming, ${outgoing.length} outgoing connections`;
                html += `</div>`;
            }
            
            // Dashboard data
            html += dashData ? dashData : '<div class="no-data">No dashboard data</div>';
            if (pinned) html += '<div style="margin-top:10px;color:#666;font-size:10px;">Press Escape or click ✕ to close</div>';
            tooltip.html(html).classed("visible", true).classed("pinned", pinned);
            tooltipPinned = pinned;
        }
        
        window.closeDashboard = function() {
            tooltip.classed("visible", false).classed("pinned", false);
            tooltipPinned = false;
            selectedNode = null;
            currentDashboardNode = null;
            nodeElements.classed("selected", false).classed("connected-upstream", false).classed("connected-downstream", false);
            linkElements.classed("highlighted", false);
            d3.selectAll(".resid-channel-highlight").style("display", "none");
        };
        document.addEventListener("keydown", e => { if (e.key === "Escape" && tooltipPinned) closeDashboard(); });
        
        // Token activation tooltip - shows activation value on hover
        const tokenTooltip = d3.select("body")
            .append("div")
            .attr("id", "token-act-tooltip")
            .style("position", "fixed")
            .style("background", "#2a2a40")
            .style("border", "1px solid #666")
            .style("border-radius", "4px")
            .style("padding", "4px 8px")
            .style("font-size", "12px")
            .style("font-family", "monospace")
            .style("color", "#fff")
            .style("pointer-events", "none")
            .style("z-index", "2001")
            .style("display", "none");
        
        document.addEventListener("mouseover", (e) => {
            if (e.target.classList.contains("token-span")) {
                const act = e.target.dataset.act;
                if (act !== undefined && act !== null && act !== "") {
                    tokenTooltip
                        .style("display", "block")
                        .style("left", (e.clientX + 10) + "px")
                        .style("top", (e.clientY - 25) + "px")
                        .html(`activation: <b>${parseFloat(act).toFixed(4)}</b>`);
                }
            }
        });
        
        document.addEventListener("mouseout", (e) => {
            if (e.target.classList.contains("token-span")) {
                tokenTooltip.style("display", "none");
            }
        });
        
        document.addEventListener("mousemove", (e) => {
            if (e.target.classList.contains("token-span")) {
                tokenTooltip
                    .style("left", (e.clientX + 10) + "px")
                    .style("top", (e.clientY - 25) + "px");
            }
        });
        
        // Toggle between windowed and full context views
        window.toggleExpandAll = function(btn) {
            const dashboard = btn.closest('#dashboard-tooltip');
            if (!dashboard) return;
            
            const isExpanded = btn.dataset.expanded === 'true';
            const windowedViews = dashboard.querySelectorAll('.windowed-view');
            const fullViews = dashboard.querySelectorAll('.full-view');
            
            if (isExpanded) {
                // Collapse: show windowed, hide full
                windowedViews.forEach(el => el.style.display = 'inline');
                fullViews.forEach(el => el.style.display = 'none');
                btn.textContent = 'Expand All';
                btn.dataset.expanded = 'false';
            } else {
                // Expand: hide windowed, show full
                windowedViews.forEach(el => el.style.display = 'none');
                fullViews.forEach(el => el.style.display = 'inline');
                btn.textContent = 'Collapse All';
                btn.dataset.expanded = 'true';
            }
        };
        
        // Initial zoom - position graph in left half of screen (dashboard takes right 50%)
        const leftHalfWidth = width / 2;
        const scale = Math.min((leftHalfWidth - 100) / totalWidth, (height - 100) / totalHeight, 0.9);
        // Center the graph within the left half of the viewport
        const translateX = (leftHalfWidth - totalWidth * scale) / 2;
        const translateY = (height - totalHeight * scale) / 2;
        svg.call(zoom.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));
    </script>
</body>
</html>'''


if __name__ == "__main__":
    main()

