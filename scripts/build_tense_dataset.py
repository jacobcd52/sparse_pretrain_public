#!/usr/bin/env python3
"""
Build and test the tense prediction dataset with BINARY cross-entropy loss.

This script:
1. Loads the SimpleStories model
2. For each template structure, finds the best verb pair where model prob > 0.1
3. Uses binary CE loss: softmax over just [correct_verb, incorrect_verb] logits
4. Outputs filtered templates for the DummyTenseTask class

Usage:
    python scripts/build_tense_dataset.py
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_pretrain.src.model import SparseGPT
from sparse_pretrain.src.pruning.tasks import DummyTenseTask
from sparse_pretrain.src.pruning.run_pruning import load_model


def load_simplestories_model(device: str = "cuda"):
    """Load the SimpleStories model for testing tense predictions."""
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


def compute_binary_ce_loss(logits: torch.Tensor, correct_token: int, incorrect_token: int) -> float:
    """
    Compute binary cross-entropy loss over just two tokens.
    
    Takes softmax over [correct_logit, incorrect_logit] and computes CE loss
    for predicting the correct token.
    
    Returns:
        Binary CE loss (float)
    """
    binary_logits = torch.stack([logits[correct_token], logits[incorrect_token]])
    binary_probs = F.softmax(binary_logits, dim=0)
    # CE loss for predicting correct (index 0)
    loss = -torch.log(binary_probs[0] + 1e-10)
    return loss.item()


def find_best_verb_for_template(
    model, tokenizer, template_present: str, template_past: str,
    verb_pairs: list, names: list, name_to_pronoun: dict,
    min_prob: float = 0.1, device: str = "cuda"
):
    """
    Find the best verb pair for a template that has probability >= min_prob.
    
    For each verb pair, tests both present and past tense contexts and
    returns the verb pair with highest average probability across names.
    
    Returns:
        List of (context_template, present_verb, past_verb, is_present_tense, avg_prob)
        or empty list if no verb passes threshold.
    """
    results = []
    
    for present_verb, past_verb in verb_pairs:
        # Test with present tense context
        present_probs = []
        for name in names:
            pron = name_to_pronoun[name]
            context = template_present.format(name=name, pron=pron)
            correct_verb = " " + present_verb
            
            correct_id = tokenizer.encode(correct_verb, add_special_tokens=False)
            if len(correct_id) == 0:
                continue
            correct_token = correct_id[0]
            
            probs, _ = compute_next_token_probs(model, tokenizer, context, device)
            present_probs.append(probs[correct_token].item())
        
        # Test with past tense context
        past_probs = []
        for name in names:
            pron = name_to_pronoun[name]
            context = template_past.format(name=name, pron=pron)
            correct_verb = " " + past_verb
            
            correct_id = tokenizer.encode(correct_verb, add_special_tokens=False)
            if len(correct_id) == 0:
                continue
            correct_token = correct_id[0]
            
            probs, _ = compute_next_token_probs(model, tokenizer, context, device)
            past_probs.append(probs[correct_token].item())
        
        if present_probs and past_probs:
            avg_present = np.mean(present_probs)
            avg_past = np.mean(past_probs)
            
            # Add if either passes threshold
            if avg_present >= min_prob:
                results.append((template_present, present_verb, past_verb, True, avg_present))
            if avg_past >= min_prob:
                results.append((template_past, present_verb, past_verb, False, avg_past))
    
    return results


def filter_templates_new_strategy(model, tokenizer, min_prob: float = 0.1, device: str = "cuda"):
    """
    New filtering strategy: for each template, find verb pairs with prob >= min_prob.
    
    Returns:
        List of (context_template, present_verb, past_verb, is_present_tense) tuples.
    """
    template_structures = DummyTenseTask.TEMPLATE_STRUCTURES
    verb_pairs = DummyTenseTask.VERB_PAIRS
    names = DummyTenseTask.ALL_NAMES
    name_to_pronoun = DummyTenseTask.NAME_TO_PRONOUN
    
    print(f"\nFiltering templates with new strategy...")
    print(f"  Template structures: {len(template_structures)}")
    print(f"  Verb pairs: {len(verb_pairs)}")
    print(f"  Names: {len(names)}")
    print(f"  Min probability threshold: {min_prob}")
    
    valid_templates = []
    
    for template_present, template_past in tqdm(template_structures):
        results = find_best_verb_for_template(
            model, tokenizer, template_present, template_past,
            verb_pairs, names, name_to_pronoun, min_prob, device
        )
        
        for ctx, pres_v, past_v, is_present, prob in results:
            valid_templates.append((ctx, pres_v, past_v, is_present))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_templates = []
    for t in valid_templates:
        if t not in seen:
            seen.add(t)
            unique_templates.append(t)
    
    print(f"\n{len(unique_templates)} unique templates passed threshold")
    
    return unique_templates


def compute_binary_ce_stats(model, tokenizer, templates, n_samples=500, device="cuda"):
    """
    Compute binary cross-entropy loss statistics for the dataset.
    
    For each example:
    1. Get logits for correct and incorrect verb tokens
    2. Compute softmax over just those two logits
    3. Compute CE loss for predicting correct token
    """
    names = DummyTenseTask.ALL_NAMES
    name_to_pronoun = DummyTenseTask.NAME_TO_PRONOUN
    
    losses = []
    examples = []
    
    rng = np.random.RandomState(42)
    
    print(f"\nComputing binary CE loss for {n_samples} samples...")
    
    for _ in tqdm(range(n_samples)):
        # Pick random template and name
        template = templates[rng.randint(len(templates))]
        context_template, present_verb, past_verb, is_present_tense = template
        name = names[rng.randint(len(names))]
        pronoun = name_to_pronoun[name]
        
        context = context_template.format(name=name, pron=pronoun)
        
        # Determine correct and incorrect verbs
        if is_present_tense:
            correct_verb = " " + present_verb
            incorrect_verb = " " + past_verb
        else:
            correct_verb = " " + past_verb
            incorrect_verb = " " + present_verb
        
        # Get token IDs
        correct_id = tokenizer.encode(correct_verb, add_special_tokens=False)
        incorrect_id = tokenizer.encode(incorrect_verb, add_special_tokens=False)
        if len(correct_id) == 0 or len(incorrect_id) == 0:
            continue
        correct_token = correct_id[0]
        incorrect_token = incorrect_id[0]
        
        # Compute logits and binary CE loss
        probs, logits = compute_next_token_probs(model, tokenizer, context, device)
        
        # Binary CE loss
        binary_loss = compute_binary_ce_loss(logits, correct_token, incorrect_token)
        
        # Also compute binary probability
        binary_logits = torch.stack([logits[correct_token], logits[incorrect_token]])
        binary_probs = F.softmax(binary_logits, dim=0)
        binary_prob = binary_probs[0].item()
        
        # Full vocab probability for comparison
        full_prob = probs[correct_token].item()
        
        losses.append(binary_loss)
        examples.append({
            "context": context,
            "correct_verb": correct_verb.strip(),
            "incorrect_verb": incorrect_verb.strip(),
            "binary_ce_loss": binary_loss,
            "binary_prob": binary_prob,
            "full_vocab_prob": full_prob,
        })
    
    return losses, examples


def split_templates(valid_templates, train_frac=0.5, val_frac=0.25, seed=42):
    """Split templates into train/val/superval sets."""
    rng = np.random.RandomState(seed)
    indices = list(range(len(valid_templates)))
    rng.shuffle(indices)
    
    n_train = int(len(indices) * train_frac)
    n_val = int(len(indices) * val_frac)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    superval_indices = indices[n_train + n_val:]
    
    train_templates = [valid_templates[i] for i in sorted(train_indices)]
    val_templates = [valid_templates[i] for i in sorted(val_indices)]
    superval_templates = [valid_templates[i] for i in sorted(superval_indices)]
    
    return train_templates, val_templates, superval_templates


def format_templates_for_class(templates, name="TRAIN_TEMPLATES"):
    """Format templates as Python code for the task class."""
    lines = [f"    {name} = ["]
    for context_template, present_verb, past_verb, is_present_tense in templates:
        lines.append(f'        ("{context_template}", "{present_verb}", "{past_verb}", {is_present_tense}),')
    lines.append("    ]")
    return "\n".join(lines)


def plot_binary_loss_histogram(losses, save_path=None):
    """Plot histogram of binary cross-entropy losses."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(losses, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(losses), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(losses):.3f}')
    plt.axvline(np.median(losses), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(losses):.3f}')
    
    plt.xlabel('Binary Cross-Entropy Loss', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Binary CE Loss Distribution for Tense Task\n(softmax over correct/incorrect verb only)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram to {save_path}")
    
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_simplestories_model(device)
    
    # Filter templates with new strategy
    valid_templates = filter_templates_new_strategy(
        model, tokenizer, min_prob=0.1, device=device
    )
    
    if len(valid_templates) < 10:
        print(f"\nWARNING: Only {len(valid_templates)} templates passed threshold 0.1")
        print("Lowering threshold to 0.05...")
        valid_templates = filter_templates_new_strategy(
            model, tokenizer, min_prob=0.05, device=device
        )
    
    # Split into train/val/superval
    train_templates, val_templates, superval_templates = split_templates(valid_templates)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_templates)} templates")
    print(f"  Val: {len(val_templates)} templates")
    print(f"  Superval: {len(superval_templates)} templates")
    
    # Print formatted templates for updating the class
    print("\n" + "="*80)
    print("COPY THE FOLLOWING INTO DummyTenseTask class in tasks.py:")
    print("="*80 + "\n")
    
    print(format_templates_for_class(train_templates, "TRAIN_TEMPLATES"))
    print()
    print(format_templates_for_class(val_templates, "VAL_TEMPLATES"))
    print()
    print(format_templates_for_class(superval_templates, "SUPERVAL_TEMPLATES"))
    
    # Compute binary CE loss statistics
    print("\n" + "="*80)
    print("BINARY CROSS-ENTROPY LOSS VALIDATION")
    print("="*80)
    
    all_templates = train_templates + val_templates + superval_templates
    losses, examples = compute_binary_ce_stats(model, tokenizer, all_templates, n_samples=500, device=device)
    
    print(f"\nBinary Cross-Entropy Loss Statistics:")
    print(f"  Mean: {np.mean(losses):.4f}")
    print(f"  Median: {np.median(losses):.4f}")
    print(f"  Std: {np.std(losses):.4f}")
    print(f"  Min: {np.min(losses):.4f}")
    print(f"  Max: {np.max(losses):.4f}")
    
    # Binary probability statistics
    binary_probs = [ex["binary_prob"] for ex in examples]
    print(f"\nBinary Probability (P(correct) in 2-way softmax):")
    print(f"  Mean: {np.mean(binary_probs):.4f}")
    print(f"  Median: {np.median(binary_probs):.4f}")
    print(f"  % >= 0.9: {100 * np.mean(np.array(binary_probs) >= 0.9):.1f}%")
    print(f"  % >= 0.7: {100 * np.mean(np.array(binary_probs) >= 0.7):.1f}%")
    print(f"  % >= 0.5: {100 * np.mean(np.array(binary_probs) >= 0.5):.1f}%")
    
    # Show some examples
    examples_sorted = sorted(examples, key=lambda x: x["binary_ce_loss"])
    print(f"\nSample examples (sorted by binary CE loss):")
    print("\nBest (lowest loss):")
    for ex in examples_sorted[:5]:
        print(f"  Loss: {ex['binary_ce_loss']:.3f}, Binary P: {ex['binary_prob']:.3f}, Full P: {ex['full_vocab_prob']:.3f}")
        print(f"    '{ex['context']}' → correct: '{ex['correct_verb']}' vs incorrect: '{ex['incorrect_verb']}'")
    
    print("\nWorst (highest loss):")
    for ex in examples_sorted[-5:]:
        print(f"  Loss: {ex['binary_ce_loss']:.3f}, Binary P: {ex['binary_prob']:.3f}, Full P: {ex['full_vocab_prob']:.3f}")
        print(f"    '{ex['context']}' → correct: '{ex['correct_verb']}' vs incorrect: '{ex['incorrect_verb']}'")
    
    # Plot histogram
    save_path = Path("outputs/carbs_results_tense_binary/binary_loss_histogram.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_binary_loss_histogram(losses, save_path=save_path)
    
    # Save results to JSON
    results = {
        "threshold": 0.1,
        "loss_type": "binary_ce",
        "n_valid": len(valid_templates),
        "n_train": len(train_templates),
        "n_val": len(val_templates),
        "n_superval": len(superval_templates),
        "binary_ce_loss_mean": float(np.mean(losses)),
        "binary_ce_loss_median": float(np.median(losses)),
        "binary_ce_loss_std": float(np.std(losses)),
        "binary_prob_mean": float(np.mean(binary_probs)),
        "train_templates": train_templates,
        "val_templates": val_templates,
        "superval_templates": superval_templates,
    }
    
    results_path = Path("outputs/carbs_results_tense_binary/dataset_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
