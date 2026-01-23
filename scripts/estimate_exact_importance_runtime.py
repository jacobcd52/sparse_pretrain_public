#!/usr/bin/env python3
"""
Estimate runtime for exact node importance computation.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sparse_pretrain.src.pruning.run_pruning import load_model
from sparse_pretrain.src.pruning.tasks import get_task
from sparse_pretrain.src.pruning.exact_node_importance import estimate_runtime
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_duration", type=float, default=60.0)
    parser.add_argument("--use_binary_loss", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, model_config_dict = load_model(args.model_path, args.device)
    model.eval()
    print(f"Model: {model.config.n_layer} layers, d_model={model.config.d_model}")
    
    # Get tokenizer
    tokenizer_name = args.tokenizer or model_config_dict.get("training_config", {}).get("tokenizer_name")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create task
    print(f"Creating task: {args.task_name}")
    task = get_task(args.task_name, tokenizer, seed=42)
    
    # Estimate runtime
    print(f"\nEstimating runtime with batch_size={args.batch_size}...")
    results = estimate_runtime(
        model=model,
        task=task,
        batch_size=args.batch_size,
        use_binary_loss=args.use_binary_loss,
        device=args.device,
        test_duration=args.test_duration,
    )


if __name__ == "__main__":
    main()

