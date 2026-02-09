#!/bin/bash
# Setup script for sparse_pretrain

set -e

echo "=== sparse_pretrain Setup ==="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install CARBS for hyperparameter optimization
echo "Installing CARBS..."
pip install "carbs @ git+https://github.com/imbue-ai/carbs.git"

# Install transformer-lens for interpretability
echo "Installing transformer-lens..."
pip install transformer-lens>=1.0.0

# Install package in editable mode
echo ""
echo "Installing sparse_pretrain in editable mode..."
pip install -e .

# Download pre-tokenized dataset
echo ""
echo "Downloading pre-tokenized SimpleStories dataset..."
python -c "
from datasets import load_dataset
from pathlib import Path

print('Downloading dataset...')
ds = load_dataset('jacobcd52/simplestories-tokenized', trust_remote_code=True)
print(f'Dataset: {ds}')

# Save to local data directory for faster loading
local_path = Path('data/simplestories-tokenized')
if not local_path.exists():
    print(f'Saving to {local_path}...')
    local_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(local_path))
    print('Done!')
else:
    print(f'Local copy already exists at {local_path}')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  # Train a sparse model with bridges"
echo "  python -m src.train_bridges --config configs_bridges/ss128/d1024.yaml"
echo ""
echo "  # Run pruning/circuit discovery"
echo "  python -m src.pruning.run_pruning --model PATH_TO_MODEL --task pronoun"
echo ""






