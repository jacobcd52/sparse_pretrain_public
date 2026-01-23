#!/usr/bin/env python3
"""
Run 10 random seeds for d3072 IOI Mixed zero model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

import subprocess
import os

MODEL_DIR = "my_sparse_pretrain/outputs/carbs_results_ioi_mixed/ss_bridges_d3072_f0.005_zero_noembed"
NUM_SEEDS = 10

def main():
    print("=" * 70)
    print("Running 10 random seeds for d3072 IOI Mixed zero model")
    print("=" * 70)
    
    # Run seed repetitions
    print("\n1. Running seed repetitions...")
    result = subprocess.run([
        "python", "my_sparse_pretrain/scripts/run_seed_repetitions.py",
        "--checkpoint-dir", MODEL_DIR,
        "--num-seeds", str(NUM_SEEDS),
        "--device", "cuda"
    ], cwd="/root/global_circuits")
    
    if result.returncode != 0:
        print("Seed repetitions failed!")
        return
    
    print("\n2. Analyzing seed repetitions...")
    result = subprocess.run([
        "python", "my_sparse_pretrain/scripts/analyze_seed_repetitions.py",
        "--checkpoint-dir", MODEL_DIR,
    ], cwd="/root/global_circuits")
    
    if result.returncode != 0:
        print("Analysis failed!")
    
    print("\n3. Generating circuit HTMLs for each seed...")
    rep_dir = Path(MODEL_DIR) / "repetitions"
    
    from my_sparse_pretrain.scripts.generate_circuit_htmls import generate_circuit_html_for_model
    
    for seed_dir in sorted(rep_dir.glob("seed*")):
        if seed_dir.is_dir():
            print(f"  Processing {seed_dir.name}...")
            try:
                generate_circuit_html_for_model(seed_dir)
                print(f"    ✓ HTML generated")
            except Exception as e:
                print(f"    ✗ Failed: {e}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"\nResults in: {rep_dir}")

if __name__ == "__main__":
    main()
