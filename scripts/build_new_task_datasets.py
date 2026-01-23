#!/usr/bin/env python3
"""
Build and test the task datasets:
1. IOITask - Indirect Object Identification (predicting pronoun for first name)
2. PronounDistractorTask - pronoun prediction with distractor name

This script:
1. Loads the SimpleStories model
2. For each task, filters examples where P(correct) >= 0.5
3. Prints the first 20 examples from each dataset

Usage:
    python my_sparse_pretrain/scripts/build_new_task_datasets.py
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from my_sparse_pretrain.src.model import SparseGPT
from my_sparse_pretrain.src.pruning.tasks import (
    IOITask, PronounDistractorTask
)
from my_sparse_pretrain.src.pruning.run_pruning import load_model


def load_simplestories_model(device: str = "cuda"):
    """Load the SimpleStories model for testing predictions."""
    model_path = "jacobcd52/ss_bridges_d1024_f0.015625"
    tokenizer_name = "SimpleStories/SimpleStories-1.25M"
    
    print(f"Loading model: {model_path}")
    model, config = load_model(model_path, device=device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def compute_next_token_probs(model, tokenizer, context: str, device: str = "cuda"):
    """
    Compute probability distribution over next token given context.
    
    Returns:
        probs: Tensor of shape (vocab_size,) with probabilities
        logits: Tensor of shape (vocab_size,) with raw logits
    """
    input_ids = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        logits = logits[:, -1, :]  # (1, vocab_size)
        probs = F.softmax(logits, dim=-1)
    
    return probs[0], logits[0]  # (vocab_size,)


def filter_and_collect_examples(
    model, tokenizer, task, n_target: int = 20, 
    min_prob: float = 0.5, max_attempts: int = 10000, device: str = "cuda"
):
    """
    Filter examples from a task where P(correct) >= min_prob.
    
    Returns:
        List of dicts with context, correct_token, prob, etc.
    """
    examples = []
    attempts = 0
    
    print(f"\nFiltering {task.name} examples (min_prob={min_prob})...")
    
    pbar = tqdm(total=n_target)
    while len(examples) < n_target and attempts < max_attempts:
        attempts += 1
        
        # Generate example
        ex = task.generate_example()
        context = tokenizer.decode(ex.positive_ids.tolist())
        correct_token = ex.correct_token
        incorrect_token = ex.incorrect_token
        
        # Get model probability
        probs, logits = compute_next_token_probs(model, tokenizer, context, device)
        correct_prob = probs[correct_token].item()
        incorrect_prob = probs[incorrect_token].item()
        
        if correct_prob >= min_prob:
            correct_str = tokenizer.decode([correct_token])
            incorrect_str = tokenizer.decode([incorrect_token])
            
            examples.append({
                "context": context,
                "correct_token": correct_str,
                "incorrect_token": incorrect_str,
                "correct_prob": correct_prob,
                "incorrect_prob": incorrect_prob,
            })
            pbar.update(1)
    
    pbar.close()
    
    print(f"Found {len(examples)} valid examples after {attempts} attempts")
    if attempts >= max_attempts and len(examples) < n_target:
        print(f"WARNING: Could only find {len(examples)} examples (target was {n_target})")
    
    return examples


def print_examples(examples, task_name: str, n: int = 30):
    """Print examples in a nice format."""
    print(f"\n{'='*80}")
    print(f"FIRST {min(n, len(examples))} EXAMPLES FROM {task_name.upper()}")
    print(f"{'='*80}\n")
    
    for i, ex in enumerate(examples[:n]):
        print(f"Example {i+1}:")
        print(f"  Context: \"{ex['context']}\"")
        print(f"  Correct: \"{ex['correct_token']}\" (P={ex['correct_prob']:.4f})")
        print(f"  Incorrect: \"{ex['incorrect_token']}\" (P={ex['incorrect_prob']:.4f})")
        print()


def compute_task_stats(model, tokenizer, task, n_samples: int = 500, device: str = "cuda"):
    """Compute statistics for a task across many samples."""
    correct_probs = []
    incorrect_probs = []
    
    print(f"\nComputing stats for {task.name} ({n_samples} samples)...")
    
    for _ in tqdm(range(n_samples)):
        ex = task.generate_example()
        context = tokenizer.decode(ex.positive_ids.tolist())
        
        probs, _ = compute_next_token_probs(model, tokenizer, context, device)
        correct_probs.append(probs[ex.correct_token].item())
        incorrect_probs.append(probs[ex.incorrect_token].item())
    
    correct_probs = np.array(correct_probs)
    incorrect_probs = np.array(incorrect_probs)
    
    print(f"\n{task.name} Statistics:")
    print(f"  Correct token probability:")
    print(f"    Mean: {np.mean(correct_probs):.4f}")
    print(f"    Median: {np.median(correct_probs):.4f}")
    print(f"    Std: {np.std(correct_probs):.4f}")
    print(f"    Min: {np.min(correct_probs):.4f}")
    print(f"    Max: {np.max(correct_probs):.4f}")
    print(f"    % >= 0.5%: {100 * np.mean(correct_probs >= 0.005):.1f}%")
    print(f"    % >= 1%: {100 * np.mean(correct_probs >= 0.01):.1f}%")
    print(f"    % >= 5%: {100 * np.mean(correct_probs >= 0.05):.1f}%")
    print(f"    % >= 10%: {100 * np.mean(correct_probs >= 0.1):.1f}%")
    
    # Accuracy (correct > incorrect)
    accuracy = np.mean(correct_probs > incorrect_probs)
    print(f"\n  Accuracy (P(correct) > P(incorrect)): {100 * accuracy:.1f}%")
    
    return correct_probs, incorrect_probs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_simplestories_model(device)
    
    # Create tasks
    ioi_task = IOITask(tokenizer, seed=42, split="train")
    pronoun_distractor_task = PronounDistractorTask(tokenizer, seed=42, split="train")
    
    tasks = [
        ("IOI", ioi_task),
        ("Pronoun Distractor", pronoun_distractor_task),
    ]
    
    # First, compute stats for each task
    print("\n" + "="*80)
    print("COMPUTING TASK STATISTICS")
    print("="*80)
    
    for task_name, task in tasks:
        compute_task_stats(model, tokenizer, task, n_samples=500, device=device)
    
    # Then filter and print examples
    print("\n" + "="*80)
    print("FILTERING AND COLLECTING EXAMPLES")
    print("="*80)
    
    all_examples = {}
    for task_name, task in tasks:
        examples = filter_and_collect_examples(
            model, tokenizer, task, 
            n_target=20, min_prob=0.5, max_attempts=10000, 
            device=device
        )
        all_examples[task_name] = examples
        print_examples(examples, task_name, n=20)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for task_name, examples in all_examples.items():
        if examples:
            probs = [ex["correct_prob"] for ex in examples]
            print(f"\n{task_name}:")
            print(f"  Examples collected: {len(examples)}")
            print(f"  Mean correct prob (filtered): {np.mean(probs):.4f}")
            print(f"  Median correct prob (filtered): {np.median(probs):.4f}")
        else:
            print(f"\n{task_name}: NO EXAMPLES FOUND")


if __name__ == "__main__":
    main()

