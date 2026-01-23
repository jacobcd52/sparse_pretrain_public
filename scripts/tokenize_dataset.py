#!/usr/bin/env python3
"""
Tokenize the SimpleStories dataset and upload to HuggingFace.

This creates a pre-tokenized version of the dataset that can be loaded
much faster than tokenizing on-the-fly during training.

Usage:
    python -m my_sparse_pretrain.scripts.tokenize_dataset \
        --hf_repo jacobcd52/simplestories-tokenized \
        --push
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


def tokenize_split(
    dataset_name: str,
    tokenizer,
    split: str,
    text_column: str = "story",
    max_examples: int = None,
) -> list[list[int]]:
    """
    Tokenize a dataset split.
    
    Returns a list of token ID lists (one per document).
    """
    print(f"Loading {split} split...")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    
    print(f"Tokenizing {len(ds)} examples...")
    
    all_tokens = []
    eot_token = tokenizer.eos_token_id
    
    for example in tqdm(ds, desc=f"Tokenizing {split}"):
        text = example[text_column]
        if text is None or len(text) == 0:
            continue
        
        # Tokenize without special tokens (we add EOT manually)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Add EOT token at the end
        if eot_token is not None:
            tokens.append(eot_token)
        
        all_tokens.append(tokens)
    
    return all_tokens


def pack_tokens_into_chunks(
    all_tokens: list[list[int]],
    chunk_size: int = 2048,
) -> list[list[int]]:
    """
    Pack variable-length token sequences into fixed-size chunks.
    
    Documents are concatenated and split into chunks of exactly chunk_size.
    This is more efficient for training as we don't waste padding.
    """
    print(f"Packing into chunks of {chunk_size} tokens...")
    
    # Concatenate all tokens
    flat_tokens = []
    for tokens in tqdm(all_tokens, desc="Concatenating"):
        flat_tokens.extend(tokens)
    
    print(f"Total tokens: {len(flat_tokens):,}")
    
    # Split into chunks
    chunks = []
    for i in range(0, len(flat_tokens) - chunk_size + 1, chunk_size):
        chunks.append(flat_tokens[i:i + chunk_size])
    
    print(f"Created {len(chunks):,} chunks")
    
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Tokenize SimpleStories dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SimpleStories/SimpleStories",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="SimpleStories/SimpleStories-1.25M",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="story",
        help="Name of text column in dataset",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="Size of each token chunk (use larger than training seq_len for flexibility)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/simplestories-tokenized",
        help="Local output directory",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repo to push to (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to HuggingFace Hub",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Max examples per split (for testing)",
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # Process each split
    splits_data = {}
    
    for split in ["train", "test"]:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print('='*60)
        
        # Tokenize
        all_tokens = tokenize_split(
            args.dataset_name,
            tokenizer,
            split,
            args.text_column,
            args.max_examples,
        )
        
        # Pack into chunks
        chunks = pack_tokens_into_chunks(all_tokens, args.chunk_size)
        
        # Create dataset
        splits_data[split] = Dataset.from_dict({
            "input_ids": chunks,
        })
        
        print(f"{split}: {len(chunks):,} chunks, {len(chunks) * args.chunk_size:,} total tokens")
    
    # Create DatasetDict
    dataset_dict = DatasetDict(splits_data)
    
    # Save locally
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_dir}...")
    dataset_dict.save_to_disk(str(output_dir))
    print("Saved!")
    
    # Push to HuggingFace
    if args.push and args.hf_repo:
        print(f"\nPushing to HuggingFace: {args.hf_repo}")
        dataset_dict.push_to_hub(
            args.hf_repo,
            private=False,
        )
        print("Pushed!")
        
        # Also save tokenizer info
        print("Saving tokenizer info...")
        tokenizer.push_to_hub(args.hf_repo)
    
    print("\nDone!")
    print(f"Dataset info:")
    print(f"  Tokenizer: {args.tokenizer_name}")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Train chunks: {len(splits_data['train']):,}")
    print(f"  Test chunks: {len(splits_data['test']):,}")


if __name__ == "__main__":
    main()

