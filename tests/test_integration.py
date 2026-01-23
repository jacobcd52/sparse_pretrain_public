"""
Integration tests for the full training pipeline.
"""

import pytest
import torch
import tempfile
import os

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from my_sparse_pretrain.src.config import (
    Config,
    ModelConfig,
    SparsityConfig,
    OptimizerConfig,
    TrainingConfig,
)
from my_sparse_pretrain.src.model import SparseGPT, create_model
from my_sparse_pretrain.src.sparsity import WeightSparsifier, SharkfinScheduler, clip_grad_rms_


class TestTrainingIntegration:
    """Integration tests for training components working together."""
    
    @pytest.fixture
    def tiny_config(self):
        """Create a tiny config for fast testing."""
        return Config(
            model=ModelConfig(
                n_layer=2,
                d_model=64,
                n_ctx=32,
                d_head=16,
                vocab_size=100,
                use_rms_norm=True,
                tie_embeddings=False,
                use_positional_embeddings=False,
                use_bigram_table=True,
                use_attention_sinks=True,
            ),
            sparsity=SparsityConfig(
                enable_weight_sparsity=True,
                target_l0_fraction=0.1,
                sparsity_anneal_start_fraction=0.01,
                sparsity_anneal_end_fraction=0.5,
                min_weights_per_neuron=2,
                enable_activation_sparsity=True,
                activation_topk_fraction=0.25,
            ),
            optimizer=OptimizerConfig(
                learning_rate=0.001,
                eps=0.1,
                grad_clip_rms=1.0,
                warmup_fraction=0.1,
                use_sharkfin_schedule=True,
            ),
        )
    
    def test_training_step(self, tiny_config):
        """Test a single training step with all components."""
        # Create model
        model = create_model(tiny_config.model, tiny_config.sparsity)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tiny_config.optimizer.learning_rate,
            betas=(tiny_config.optimizer.beta1, tiny_config.optimizer.beta2),
            eps=tiny_config.optimizer.eps,
            weight_decay=tiny_config.optimizer.weight_decay,
        )
        
        # Create sparsifier
        total_steps = 100
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=tiny_config.sparsity.target_l0_fraction,
            anneal_start_fraction=tiny_config.sparsity.sparsity_anneal_start_fraction,
            anneal_end_fraction=tiny_config.sparsity.sparsity_anneal_end_fraction,
            min_weights_per_neuron=tiny_config.sparsity.min_weights_per_neuron,
            total_steps=total_steps,
        )
        
        # Create scheduler
        scheduler = SharkfinScheduler(
            optimizer=optimizer,
            base_lr=tiny_config.optimizer.learning_rate,
            total_steps=total_steps,
            warmup_fraction=tiny_config.optimizer.warmup_fraction,
            sparsifier=sparsifier,
            use_sharkfin=tiny_config.optimizer.use_sharkfin_schedule,
        )
        
        # Create fake batch
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, tiny_config.model.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, tiny_config.model.vocab_size, (batch_size, seq_len))
        
        # Training step
        model.train()
        
        # Forward
        logits, loss, _ = model(input_ids, labels=labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_rms = clip_grad_rms_(model.parameters(), max_rms=tiny_config.optimizer.grad_clip_rms)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Sparsifier step
        sparsifier.step()
        
        # Scheduler step
        scheduler.step()
        
        # Verify everything worked
        assert loss.item() > 0
        assert grad_rms > 0
        assert sparsifier.state.current_step == 1
        assert scheduler.current_step == 1
    
    def test_multiple_training_steps(self, tiny_config):
        """Test multiple training steps with sparsity annealing."""
        model = create_model(tiny_config.model, tiny_config.sparsity)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tiny_config.optimizer.learning_rate,
            eps=tiny_config.optimizer.eps,
        )
        
        total_steps = 20
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=tiny_config.sparsity.target_l0_fraction,
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.5,  # Anneal over 10 steps
            min_weights_per_neuron=tiny_config.sparsity.min_weights_per_neuron,
            total_steps=total_steps,
        )
        
        scheduler = SharkfinScheduler(
            optimizer=optimizer,
            base_lr=tiny_config.optimizer.learning_rate,
            total_steps=total_steps,
            warmup_fraction=0.1,  # 2 steps warmup
            sparsifier=sparsifier,
            use_sharkfin=True,
        )
        
        batch_size = 4
        seq_len = 16
        
        losses = []
        l0_fractions = []
        learning_rates = []
        
        model.train()
        
        for step in range(total_steps):
            input_ids = torch.randint(0, tiny_config.model.vocab_size, (batch_size, seq_len))
            labels = torch.randint(0, tiny_config.model.vocab_size, (batch_size, seq_len))
            
            logits, loss, _ = model(input_ids, labels=labels)
            loss.backward()
            
            clip_grad_rms_(model.parameters())
            optimizer.step()
            optimizer.zero_grad()
            
            sparsifier.step()
            scheduler.step()
            
            losses.append(loss.item())
            l0_fractions.append(sparsifier.get_current_l0_fraction())
            learning_rates.append(scheduler.get_lr())
        
        # Verify L0 decreased
        assert l0_fractions[0] > l0_fractions[-1]
        
        # Verify L0 reached target after annealing
        assert abs(l0_fractions[-1] - tiny_config.sparsity.target_l0_fraction) < 0.01
        
        # Verify LR schedule worked (should increase initially due to warmup, then decrease)
        assert learning_rates[0] < learning_rates[5]  # Warmup
        assert learning_rates[-1] < learning_rates[10]  # Decay
    
    def test_sparsity_actually_zeros_weights(self, tiny_config):
        """Test that sparsifier actually zeros out weights."""
        model = create_model(tiny_config.model, tiny_config.sparsity)
        
        # Count initial nonzero weights (all should be nonzero after init)
        initial_nonzero = sum(
            (p != 0).sum().item()
            for n, p in model.named_parameters()
            if p.requires_grad and "norm" not in n.lower() and "bigram" not in n.lower()
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.1,
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.0,  # Immediate sparsity
            min_weights_per_neuron=2,
            total_steps=10,
        )
        
        # Do one training step
        input_ids = torch.randint(0, tiny_config.model.vocab_size, (4, 16))
        labels = torch.randint(0, tiny_config.model.vocab_size, (4, 16))
        
        logits, loss, _ = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Apply sparsity
        sparsifier.step()
        
        # Count final nonzero weights
        final_nonzero = sum(
            (p != 0).sum().item()
            for n, p in model.named_parameters()
            if p.requires_grad and "norm" not in n.lower() and "bigram" not in n.lower()
        )
        
        # Should have significantly fewer nonzero weights
        assert final_nonzero < initial_nonzero * 0.5
    
    def test_model_still_learns_with_sparsity(self, tiny_config):
        """Test that model can still learn with sparsity applied."""
        model = create_model(tiny_config.model, tiny_config.sparsity)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.5,  # Moderate sparsity
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.0,  # Immediate sparsity
            min_weights_per_neuron=4,
            total_steps=100,
        )
        
        # Use same batch repeatedly to overfit
        input_ids = torch.randint(0, tiny_config.model.vocab_size, (4, 16))
        labels = input_ids.clone()  # Predict same tokens
        
        initial_loss = None
        final_loss = None
        
        model.train()
        for step in range(50):
            logits, loss, _ = model(input_ids, labels=labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            sparsifier.step()
            
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()
        
        # Loss should decrease (model is learning)
        assert final_loss < initial_loss


class TestConfigIntegration:
    """Test config loading and usage integration."""
    
    def test_save_and_load_config_creates_valid_model(self):
        """Test that saved and loaded config creates a working model."""
        original_config = Config(
            model=ModelConfig(n_layer=2, d_model=128, d_head=16, vocab_size=50),
            sparsity=SparsityConfig(target_l0_fraction=0.2),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            original_config.to_yaml(config_path)
            
            loaded_config = Config.from_yaml(config_path)
            
            # Create model from loaded config
            model = create_model(loaded_config.model, loaded_config.sparsity)
            
            # Verify model works
            input_ids = torch.randint(0, 50, (2, 8))
            logits, _, _ = model(input_ids)
            
            assert logits.shape == (2, 8, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

