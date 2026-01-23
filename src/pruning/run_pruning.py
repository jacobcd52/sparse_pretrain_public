"""
CLI entry point for running circuit pruning.

Usage:
    python -m src.pruning.run_pruning \
        --model_path path/to/model \
        --task dummy_quote \
        --output_dir outputs/pruning

Or as a module:
    from sparse_pretrain.src.pruning import run_pruning
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from .config import PruningConfig
from .tasks import get_task, TASK_REGISTRY
from .masked_model import MaskedSparseGPT
from .trainer import run_pruning, PruningTrainer
from .discretize import discretize_masks
from .calibrate import calibrate_logits


def load_model(model_path: str, device: str = "cuda"):
    """
    Load a SparseGPT model from path.
    
    Supports:
    - HuggingFace Hub repo IDs
    - Local standalone model checkpoints (with pytorch_model.bin)
    - Local bridge model checkpoints (with sparse_model.bin) - loads only the sparse model
    
    Returns:
        Tuple of (model, config_dict) where config_dict contains the full config
        including training_config with tokenizer_name.
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from sparse_pretrain.src.model import SparseGPT
    from sparse_pretrain.src.config import ModelConfig, SparsityConfig
    
    # Check if it's a local path or HF repo
    if os.path.exists(model_path):
        # Local checkpoint
        config_path = os.path.join(model_path, "config.json")
        
        # Check which type of checkpoint this is
        standalone_model_file = os.path.join(model_path, "pytorch_model.bin")
        bridge_model_file = os.path.join(model_path, "sparse_model.bin")
        
        # Also check for checkpoint subdirectories (e.g., checkpoint-30000/)
        if not os.path.exists(config_path):
            # Maybe it's a checkpoint dir inside a training run
            parent_config = os.path.join(os.path.dirname(model_path), "config.json")
            if os.path.exists(parent_config):
                config_path = parent_config
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find config.json at {model_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        model_config = ModelConfig(**config_dict["model_config"])
        sparsity_config = SparsityConfig(**config_dict.get("sparsity_config", {}))
        
        model = SparseGPT(model_config, sparsity_config)
        
        # Determine which model file to load
        if os.path.exists(standalone_model_file):
            # Standalone model
            model_file = standalone_model_file
            print(f"Loading standalone model from {model_file}")
        elif os.path.exists(bridge_model_file):
            # Bridge model - load only sparse model, ignore bridges
            model_file = bridge_model_file
            print(f"Loading sparse model from bridge checkpoint: {model_file}")
            print("Note: Bridge weights are ignored for pruning")
        else:
            raise FileNotFoundError(
                f"Could not find model weights. Expected either:\n"
                f"  - {standalone_model_file} (standalone model)\n"
                f"  - {bridge_model_file} (bridge model)"
            )
        
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # Try HuggingFace Hub
        from huggingface_hub import hf_hub_download
        
        # First, try to download config.json
        try:
            config_file = hf_hub_download(repo_id=model_path, filename="config.json")
        except Exception as e:
            raise ValueError(
                f"Could not load model from '{model_path}'.\n"
                f"Failed to download config.json from HuggingFace Hub: {e}"
            )
        
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        model_config = ModelConfig(**config_dict["model_config"])
        sparsity_config = SparsityConfig(**config_dict.get("sparsity_config", {}))
        
        model = SparseGPT(model_config, sparsity_config)
        
        # Try to download model weights - first standalone, then bridge
        model_file = None
        is_bridge = False
        
        try:
            model_file = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
            print(f"Loading standalone model from HuggingFace Hub: {model_path}")
        except Exception:
            # Try bridge model format
            try:
                model_file = hf_hub_download(repo_id=model_path, filename="sparse_model.bin")
                is_bridge = True
                print(f"Loading sparse model from HuggingFace Hub bridge checkpoint: {model_path}")
                print("Note: Bridge weights are ignored for pruning")
            except Exception as e:
                raise ValueError(
                    f"Could not load model from '{model_path}'.\n"
                    f"Neither pytorch_model.bin nor sparse_model.bin found on HuggingFace Hub: {e}"
                )
        
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    return model, config_dict


def create_data_iterator(
    tokenizer_name: str,
    dataset_name: str = "SimpleStories/SimpleStories",
    text_column: str = "story",
    batch_size: int = 32,
    seq_length: int = 256,
    num_batches: int = 100,
    seed: int = 42,
):
    """
    Create a data iterator for mean cache computation.
    
    Uses sequence packing (like pretraining): texts are concatenated with EOS tokens
    between them, then chunked into seq_length pieces. NO PADDING.
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    
    def batch_generator():
        # Token buffer for sequence packing
        token_buffer = []
        batches_yielded = 0
        
        for example in dataset:
            if batches_yielded >= num_batches:
                break
                
            text = example.get(text_column, "")
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
            
            # Yield complete batches when we have enough tokens
            while len(token_buffer) >= batch_size * seq_length:
                # Extract batch_size chunks of seq_length
                batch = []
                for _ in range(batch_size):
                    chunk = token_buffer[:seq_length]
                    token_buffer = token_buffer[seq_length:]
                    batch.append(torch.tensor(chunk, dtype=torch.long))
                
                yield torch.stack(batch)
                batches_yielded += 1
                
                if batches_yielded >= num_batches:
                    break
    
    return batch_generator()


def main():
    parser = argparse.ArgumentParser(description="Run circuit pruning on a SparseGPT model")
    
    # Model and task
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint or HuggingFace repo ID")
    parser.add_argument("--task", type=str, default="dummy_quote",
                        choices=list(TASK_REGISTRY.keys()),
                        help="Task to use for pruning")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name (defaults to model's tokenizer)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="pruning_outputs",
                        help="Output directory for results")
    
    # Pruning hyperparameters
    parser.add_argument("--target_loss", type=float, default=0.15,
                        help="Target task loss")
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate")
    parser.add_argument("--k_coef", type=float, default=1e-4,
                        help="Sparsity penalty coefficient")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length")
    
    # Mean cache
    parser.add_argument("--mean_cache_batches", type=int, default=100,
                        help="Number of batches for mean cache computation")
    parser.add_argument("--skip_mean_cache", action="store_true",
                        help="Skip mean cache computation (use zero ablation)")
    
    # Dataset for mean cache
    parser.add_argument("--dataset", type=str, default="SimpleStories/SimpleStories",
                        help="Dataset for mean cache computation")
    parser.add_argument("--text_column", type=str, default="story",
                        help="Text column in dataset")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip_calibration", action="store_true",
                        help="Skip logit calibration")
    parser.add_argument("--skip_discretization", action="store_true",
                        help="Skip mask discretization")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, model_config_dict = load_model(args.model_path, args.device)
    print(f"Model loaded: {model.config.n_layer} layers, d_model={model.config.d_model}")
    
    # Get tokenizer
    tokenizer_name = args.tokenizer
    if tokenizer_name is None:
        # Get from model config
        tokenizer_name = model_config_dict.get("training_config", {}).get("tokenizer_name")
    
    if tokenizer_name is None:
        raise ValueError(
            "No tokenizer found. Either:\n"
            "  1. Specify --tokenizer argument\n"
            "  2. Ensure the model config.json has training_config.tokenizer_name"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create task
    print(f"Creating task: {args.task}")
    task = get_task(args.task, tokenizer, seed=args.seed)
    
    # Create config
    config = PruningConfig(
        target_loss=args.target_loss,
        num_steps=args.num_steps,
        lr=args.lr,
        k_coef=args.k_coef,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        mean_cache_num_batches=args.mean_cache_batches,
        device=args.device,
        output_dir=str(output_dir),
    )
    
    # Save config
    config.to_yaml(output_dir / "config.yaml")
    
    # Create masked model
    print("Creating masked model...")
    masked_model = MaskedSparseGPT(model, config)
    
    # Compute mean cache
    if not args.skip_mean_cache:
        print("Computing mean activation cache...")
        data_iter = create_data_iterator(
            tokenizer_name=tokenizer_name,
            dataset_name=args.dataset,
            text_column=args.text_column,
            batch_size=config.mean_cache_batch_size,
            seq_length=config.seq_length,
            num_batches=config.mean_cache_num_batches,
            seed=args.seed,
        )
        mean_cache = masked_model.compute_mean_cache(
            data_iter,
            num_batches=config.mean_cache_num_batches,
            show_progress=True,
        )
        masked_model.set_means_from_dict(mean_cache)
        
        # Save mean cache
        torch.save(mean_cache, output_dir / "mean_cache.pt")
        print(f"Mean cache saved to {output_dir / 'mean_cache.pt'}")
    
    # Create trainer and run
    print(f"\nStarting pruning optimization ({config.num_steps} steps)...")
    trainer = PruningTrainer(masked_model, task, config)
    
    # Training loop with logging
    def log_fn(metrics):
        print(f"Step {metrics['step']}: loss={metrics['task_loss']:.4f}, "
              f"active={metrics['num_active_nodes']}, acc={metrics['accuracy']:.2%}")
    
    final_metrics = trainer.train(log_fn=log_fn, show_progress=True)
    
    # Save checkpoint
    trainer.save_checkpoint(output_dir / "checkpoint.pt")
    print(f"Checkpoint saved to {output_dir / 'checkpoint.pt'}")
    
    # Discretization
    if not args.skip_discretization:
        print("\nDiscretizing masks...")
        k, loss, binary_masks = discretize_masks(
            masked_model, task, config,
            target_loss=args.target_loss,
            show_progress=True,
        )
        
        # Save binary masks
        torch.save(binary_masks, output_dir / "binary_masks.pt")
        print(f"Binary masks saved to {output_dir / 'binary_masks.pt'}")
    
    # Calibration
    if not args.skip_calibration:
        print("\nCalibrating logits...")
        scale, shift, cal_metrics = calibrate_logits(
            masked_model, task, config,
            show_progress=True,
        )
        
        # Save calibration
        cal_data = {"scale": scale, "shift": shift, **cal_metrics}
        with open(output_dir / "calibration.json", "w") as f:
            json.dump(cal_data, f, indent=2)
        print(f"Calibration saved to {output_dir / 'calibration.json'}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    eval_metrics = trainer.evaluate(num_batches=50)
    
    # Save final results
    results = {
        "task": args.task,
        "model_path": args.model_path,
        "target_loss": args.target_loss,
        "final_metrics": eval_metrics,
        "num_active_nodes": masked_model.masks.get_total_active_nodes(),
        "total_nodes": masked_model.masks.get_total_nodes(),
        "mask_summary": masked_model.masks.get_mask_summary(),
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("PRUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Target loss: {args.target_loss}")
    print(f"Final task loss: {eval_metrics['task_loss']:.4f}")
    print(f"Final accuracy: {eval_metrics['accuracy']:.2%}")
    print(f"Active nodes: {results['num_active_nodes']} / {results['total_nodes']} "
          f"({100*results['num_active_nodes']/results['total_nodes']:.1f}%)")
    print(f"\nResults saved to: {output_dir}")
    

if __name__ == "__main__":
    main()

