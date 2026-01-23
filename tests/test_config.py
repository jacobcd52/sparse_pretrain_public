"""
Tests for configuration system.
"""

import pytest
import tempfile
import os

from sparse_pretrain.src.config import (
    Config,
    ModelConfig,
    SparsityConfig,
    OptimizerConfig,
    TrainingConfig,
)


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_default_values(self):
        """Test default configuration values match authors' code."""
        config = ModelConfig()
        
        assert config.n_layer == 8
        assert config.d_model == 1024  # From authors' code (paper mentions 2048 for some experiments)
        assert config.n_ctx == 256
        assert config.d_head == 16
        assert config.use_rms_norm == True
        assert config.tie_embeddings == False
        assert config.use_positional_embeddings == False
        assert config.use_bigram_table == True
        assert config.use_attention_sinks == True
    
    def test_n_heads_calculation(self):
        """Test number of heads calculation."""
        config = ModelConfig(d_model=2048, d_head=16)
        assert config.n_heads == 128
        
        config = ModelConfig(d_model=512, d_head=64)
        assert config.n_heads == 8
    
    def test_d_mlp_default(self):
        """Test d_mlp defaults to 4 * d_model."""
        config = ModelConfig(d_model=512)
        assert config.d_mlp == 2048
    
    def test_d_mlp_override(self):
        """Test d_mlp can be overridden."""
        config = ModelConfig(d_model=512, d_mlp=1024)
        assert config.d_mlp == 1024


class TestSparsityConfig:
    """Tests for SparsityConfig."""
    
    def test_default_values(self):
        """Test default sparsity configuration values match authors' code."""
        config = SparsityConfig()
        
        assert config.enable_weight_sparsity == True
        assert config.target_l0_fraction == 0.015625  # 1/64 from authors' code
        assert config.sparsity_anneal_start_fraction == 0.01
        assert config.sparsity_anneal_end_fraction == 0.5
        assert config.min_weights_per_neuron == 4
        assert config.enable_activation_sparsity == True
        assert config.activation_topk_fraction == 0.25


class TestOptimizerConfig:
    """Tests for OptimizerConfig."""
    
    def test_default_values(self):
        """Test default optimizer configuration values match paper."""
        config = OptimizerConfig()
        
        assert config.beta1 == 0.9
        assert config.beta2 == 0.95
        assert config.weight_decay == 0.1
        assert config.eps == 0.1  # Unusually large!
        assert config.grad_clip_rms == 1.0
        assert config.warmup_fraction == 0.01
        assert config.use_sharkfin_schedule == False  # Authors don't use this in simple interface


class TestConfig:
    """Tests for full Config."""
    
    def test_default_config(self):
        """Test creating default config."""
        config = Config()
        
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.sparsity, SparsityConfig)
        assert isinstance(config.optimizer, OptimizerConfig)
        assert isinstance(config.training, TrainingConfig)
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        d = config.to_dict()
        
        assert "model" in d
        assert "sparsity" in d
        assert "optimizer" in d
        assert "training" in d
        
        assert d["model"]["n_layer"] == 8
        assert d["optimizer"]["eps"] == 0.1
    
    def test_yaml_round_trip(self):
        """Test saving and loading config from YAML."""
        config = Config(
            model=ModelConfig(n_layer=4, d_model=512),
            sparsity=SparsityConfig(target_l0_fraction=0.05),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            
            # Save
            config.to_yaml(path)
            
            # Load
            loaded = Config.from_yaml(path)
            
            assert loaded.model.n_layer == 4
            assert loaded.model.d_model == 512
            assert loaded.sparsity.target_l0_fraction == 0.05
    
    def test_from_yaml_with_partial_config(self):
        """Test loading config with only some values specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            
            # Write minimal config
            with open(path, "w") as f:
                f.write("""
model:
  n_layer: 6
training:
  batch_size: 32
""")
            
            config = Config.from_yaml(path)
            
            # Specified values
            assert config.model.n_layer == 6
            assert config.training.batch_size == 32
            
            # Defaults for unspecified
            assert config.model.d_model == 1024  # Default from authors' code
            assert config.optimizer.eps == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

