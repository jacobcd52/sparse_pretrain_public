"""
Evaluate cross-entropy loss of SimpleStories models on the SimpleStories dataset.

Usage:
    python eval_simplestories.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def create_eval_batches(
    dataset_name: str = "SimpleStories/SimpleStories",
    tokenizer_name: str = "SimpleStories/SimpleStories-1.25M",
    split: str = "test",
    text_column: str = "story",
    ctx_len: int = 512,
    max_tokens: int = 500_000,  # Evaluate on ~500k tokens
    seed: int = 42,
):
    """
    Create evaluation batches by concatenating text samples with EOT tokens.
    
    Returns:
        List of token tensors, each of shape (ctx_len,)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    eot_token = tokenizer.eos_token_id
    
    # Load dataset
    print(f"Loading dataset {dataset_name} ({split})...")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    ds = ds.shuffle(seed=seed)
    
    # Tokenize and concatenate
    token_buffer = []
    batches = []
    
    print("Tokenizing and creating batches...")
    for example in tqdm(ds, desc="Processing"):
        text = example.get(text_column)
        if text is None or len(text) == 0:
            continue
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(tokens)
        
        if eot_token is not None:
            token_buffer.append(eot_token)
        
        # Create complete chunks
        while len(token_buffer) >= ctx_len:
            chunk = token_buffer[:ctx_len]
            token_buffer = token_buffer[ctx_len:]
            batches.append(torch.tensor(chunk, dtype=torch.long))
            
            # Stop if we have enough tokens
            if len(batches) * ctx_len >= max_tokens:
                break
        
        if len(batches) * ctx_len >= max_tokens:
            break
    
    print(f"Created {len(batches)} batches ({len(batches) * ctx_len:,} tokens)")
    return batches, tokenizer


@torch.no_grad()
def evaluate_model(
    model_name: str,
    batches: list,
    batch_size: int = 16,
    device: str = "cuda",
):
    """
    Evaluate a model's cross-entropy loss on the given batches.
    
    Returns:
        Average cross-entropy loss (in nats)
    """
    print(f"\nLoading model: {model_name}")
    
    # Load model in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    total_loss = 0.0
    total_tokens = 0
    
    # Process in batches
    for i in tqdm(range(0, len(batches), batch_size), desc="Evaluating"):
        batch_tensors = batches[i:i+batch_size]
        input_ids = torch.stack(batch_tensors).to(device)  # (B, ctx_len)
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits  # (B, ctx_len, vocab_size)
        
        # Compute loss: predict next token
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute per-token cross-entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='sum',
        )
        
        total_loss += loss.item()
        total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return avg_loss, perplexity


def main():
    # Models to evaluate
    model_names = [
        "SimpleStories/SimpleStories-1.25M",
        "SimpleStories/SimpleStories-5M",
        "SimpleStories/SimpleStories-11M",
        "SimpleStories/SimpleStories-35M",
    ]
    
    ctx_len = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print(f"Context length: {ctx_len}")
    print("=" * 60)
    
    # Create evaluation batches (use the smallest model's tokenizer - they should all be the same)
    batches, tokenizer = create_eval_batches(
        dataset_name="SimpleStories/SimpleStories",
        tokenizer_name="SimpleStories/SimpleStories-1.25M",
        split="test",
        text_column="story",
        ctx_len=ctx_len,
        max_tokens=500_000,
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    results = []
    for model_name in model_names:
        avg_loss, perplexity = evaluate_model(
            model_name=model_name,
            batches=batches,
            batch_size=16,
            device=device,
        )
        results.append({
            "model": model_name,
            "ce_loss": avg_loss,
            "perplexity": perplexity,
        })
        print(f"  {model_name}:")
        print(f"    CE Loss: {avg_loss:.4f}")
        print(f"    Perplexity: {perplexity:.2f}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<40} {'CE Loss':>10} {'Perplexity':>12}")
    print("-" * 62)
    for r in results:
        model_short = r['model'].split('/')[-1]
        print(f"{model_short:<40} {r['ce_loss']:>10.4f} {r['perplexity']:>12.2f}")


if __name__ == "__main__":
    main()

