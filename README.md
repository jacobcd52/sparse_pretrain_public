# sparse_pretrain

Weight-sparse transformer pretraining with interpretable circuits.

Based on Gao et al. (2025) "Weight-sparse transformers have interpretable circuits"

## Installation

```bash
# Clone the repo
git clone https://github.com/jacobcd52/sparse_pretrain_public.git
cd sparse_pretrain_public

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .

# Optional: Install CARBS for hyperparameter sweeps
pip install "carbs @ git+https://github.com/imbue-ai/carbs.git"

# Optional: Install transformer-lens for interpretability
pip install transformer-lens
```

Or use the setup script:
```bash
./setup.sh
```

## Usage

### Training a Sparse Model with Bridges

```bash
python -m src.train_bridges --config configs_bridges/ss128/d1024.yaml
```

### Running Circuit Discovery (Pruning)

```bash
python -m src.pruning.run_pruning --model PATH_TO_MODEL --task pronoun
```

### Environment Variables

For HuggingFace and W&B integration, set:
```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
```

## Project Structure

```
sparse_pretrain/
├── src/                    # Core source code
│   ├── model.py           # SparseGPT model
│   ├── bridges.py         # Bridge training components
│   ├── train_bridges.py   # Main training script
│   ├── pruning/           # Circuit discovery via pruning
│   └── visualization/     # Circuit visualization
├── scripts/               # Experiment scripts
├── configs_bridges/       # Training configs
├── conversion_utils/      # Model conversion utilities
├── notebooks/             # Jupyter notebooks
└── tests/                 # Unit tests
```

## License

MIT
