#!/bin/bash
# Run CARBS sweep for tense task on all four models with zero ablation, no embedding mask
# Then run all evaluations for each model

set -e

cd /root/global_circuits

OUTPUT_DIR="my_sparse_pretrain/outputs/carbs_results_tense_binary"
LOG_FILE="${OUTPUT_DIR}/full_run.log"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "Starting tense task CARBS sweep for all models at $(date)" | tee -a "${LOG_FILE}"
echo "Results will be saved to: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"

# Array of models
MODELS=(
    "jacobcd52/ss_d128_f1"
    "jacobcd52/ss_bridges_d1024_f0.015625"
    "jacobcd52/ss_bridges_d3072_f0.005"
    "jacobcd52/ss_bridges_d4096_f0.002"
)

# Run CARBS for each model
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "${MODEL}")
    echo "" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    echo "Starting CARBS for ${MODEL_NAME} at $(date)" | tee -a "${LOG_FILE}"
    echo "============================================================" | tee -a "${LOG_FILE}"
    
    python my_sparse_pretrain/scripts/run_carbs_clean.py \
        --model "${MODEL}" \
        --task dummy_tense \
        --binary-loss \
        --ablation zero \
        --num-runs 32 \
        --steps 1000 \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee -a "${LOG_FILE}"
    
    echo "CARBS completed for ${MODEL_NAME} at $(date)" | tee -a "${LOG_FILE}"
done

# Now run all evaluations for the tense results
echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Running all evaluations for tense models at $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

# Temporarily change the CARBS_RESULTS_DIR in run_all_evals.py
# by using a Python command directly
python << 'EOF' 2>&1 | tee -a "${LOG_FILE}"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import json
import torch
import numpy as np
from tqdm import tqdm

from my_sparse_pretrain.scripts.run_all_evals import (
    run_all_evals_for_model,
    create_comparison_pareto_plot,
)

CARBS_RESULTS_DIR = Path("my_sparse_pretrain/outputs/carbs_results_tense_binary")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Get all model directories in tense results
model_dirs = []
for d in CARBS_RESULTS_DIR.iterdir():
    if d.is_dir() and (d / "best_checkpoint").exists():
        if "ignore" not in d.name.lower():
            model_dirs.append(d)
model_dirs = sorted(model_dirs)

print(f"\nFound {len(model_dirs)} model directories:")
for d in model_dirs:
    print(f"  - {d.name}")

# Run evaluations for each model
all_results = {}
for model_dir in model_dirs:
    results = run_all_evals_for_model(model_dir, device)
    all_results[model_dir.name] = results

# Create comparison plots
print("\nCreating comparison pareto plots...")
all_zero_noembed = [d for d in model_dirs if d.name.endswith("_zero_noembed")]
if len(all_zero_noembed) > 0:
    create_comparison_pareto_plot(
        all_zero_noembed,
        CARBS_RESULTS_DIR / "pareto_comparison_zero_noembed.png",
        "Tense Task: Pareto Comparison (Zero Ablation, No Embed Mask)"
    )

print("\nAll evaluations complete!")
EOF

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "All models completed at $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
