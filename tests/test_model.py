"""
Tests for the SparseGPT model.
"""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from my_sparse_pretrain.src.config import ModelConfig, SparsityConfig
from my_sparse_pretrain.src.model import AbsTopK, SDPAWithSink, SparseGPT, create_model


class TestAbsTopK:
    """Tests for AbsTopK activation function."""
    
    def test_basic_functionality(self):
        """Test that AbsTopK keeps the k largest magnitude values."""
        layer = AbsTopK(k=3)
        x = torch.tensor([[1.0, -5.0, 3.0, -2.0, 4.0]])
        
        result = layer(x)
        
        # Should keep -5, 4, 3 (largest by magnitude)
        expected_nonzero = 3
        assert (result != 0).sum().item() == expected_nonzero
        
        # Check the correct values are kept
        assert result[0, 1].item() == -5.0  # -5 should be kept
        assert result[0, 4].item() == 4.0   # 4 should be kept
        assert result[0, 2].item() == 3.0   # 3 should be kept
        assert result[0, 0].item() == 0.0   # 1 should be zeroed
        assert result[0, 3].item() == 0.0   # -2 should be zeroed
    
    def test_batch_dimension(self):
        """Test that AbsTopK works with batched input."""
        layer = AbsTopK(k=2)
        x = torch.randn(4, 8, 16)  # (batch, seq, features)
        
        result = layer(x)
        
        assert result.shape == x.shape
        # Each position should have exactly k nonzero values
        nonzero_per_position = (result != 0).sum(dim=-1)
        assert (nonzero_per_position == 2).all()
    
    def test_k_larger_than_dim(self):
        """Test that if k >= dim, all values are kept."""
        layer = AbsTopK(k=10)
        x = torch.randn(2, 5)
        
        result = layer(x)
        
        assert torch.allclose(result, x)
    
    def test_gradient_flow(self):
        """Test that gradients flow through AbsTopK."""
        layer = AbsTopK(k=3)
        x = torch.randn(2, 5, requires_grad=True)
        
        result = layer(x)
        loss = result.sum()
        loss.backward()
        
        assert x.grad is not None
        # Gradients should only flow through the kept values
        nonzero_grad = (x.grad != 0).sum().item()
        assert nonzero_grad <= 6  # 2 batches * 3 values each


class TestSDPAWithSink:
    """Tests for attention with sinks."""
    
    def test_output_shape(self):
        """Test that output shape matches input query shape."""
        n_heads = 4
        sink = SDPAWithSink(n_heads)
        
        B, H, L, D = 2, 4, 16, 32
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        
        result = sink(q, k, v, is_causal=True)
        
        assert result.shape == (B, H, L, D)
    
    def test_learnable_sink_logit(self):
        """Test that sink logit is learnable."""
        n_heads = 4
        sink = SDPAWithSink(n_heads, init_logit=1.0)
        
        assert sink.sink_logit.requires_grad
        assert sink.sink_logit.shape == (n_heads,)
        assert (sink.sink_logit == 1.0).all()
    
    def test_gradient_flow(self):
        """Test that gradients flow to sink logit."""
        n_heads = 4
        sink = SDPAWithSink(n_heads)
        
        B, H, L, D = 2, 4, 8, 16
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        
        result = sink(q, k, v, is_causal=True)
        loss = result.sum()
        loss.backward()
        
        assert sink.sink_logit.grad is not None


class TestSparseGPT:
    """Tests for the full SparseGPT model."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        return ModelConfig(
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
            dropout=0.0,
        )
    
    @pytest.fixture
    def small_sparsity_config(self):
        """Create sparsity config for testing."""
        return SparsityConfig(
            enable_weight_sparsity=True,
            target_l0_fraction=0.1,
            enable_activation_sparsity=True,
            activation_topk_fraction=0.25,
        )
    
    def test_model_creation(self, small_config, small_sparsity_config):
        """Test that model can be created."""
        model = create_model(small_config, small_sparsity_config)
        
        assert isinstance(model, SparseGPT)
        assert len(model.blocks) == small_config.n_layer
    
    def test_forward_pass(self, small_config, small_sparsity_config):
        """Test forward pass produces correct output shapes."""
        model = create_model(small_config, small_sparsity_config)
        
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        logits, loss, hidden_states = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert loss is None  # No labels provided
        assert hidden_states is None  # return_hidden_states=False
    
    def test_forward_with_labels(self, small_config, small_sparsity_config):
        """Test forward pass with labels computes loss."""
        model = create_model(small_config, small_sparsity_config)
        
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        logits, loss, _ = model(input_ids, labels=labels)
        
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert loss is not None
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive
    
    def test_forward_with_hidden_states(self, small_config, small_sparsity_config):
        """Test forward pass returns hidden states when requested."""
        model = create_model(small_config, small_sparsity_config)
        
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        logits, loss, hidden_states = model(input_ids, return_hidden_states=True)
        
        assert hidden_states is not None
        # Should have n_layer + 1 states (initial + after each block)
        assert len(hidden_states) == small_config.n_layer + 1
        for hs in hidden_states:
            assert hs.shape == (batch_size, seq_len, small_config.d_model)
    
    def test_bigram_table(self, small_config, small_sparsity_config):
        """Test that bigram table is applied correctly."""
        model = create_model(small_config, small_sparsity_config)
        
        assert model.bigram_table is not None
        assert model.bigram_table.shape == (small_config.vocab_size, small_config.vocab_size)
        
        # Modify bigram table and check effect on logits
        with torch.no_grad():
            model.bigram_table.fill_(1.0)
        
        input_ids = torch.randint(0, small_config.vocab_size, (2, 8))
        logits1, _, _ = model(input_ids)
        
        with torch.no_grad():
            model.bigram_table.fill_(0.0)
        
        logits2, _, _ = model(input_ids)
        
        # Logits should be different
        assert not torch.allclose(logits1, logits2)
    
    def test_no_bigram_table(self, small_sparsity_config):
        """Test model works without bigram table."""
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_ctx=32,
            d_head=16,
            vocab_size=100,
            use_bigram_table=False,
        )
        model = create_model(config, small_sparsity_config)
        
        assert model.bigram_table is None
        
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        logits, _, _ = model(input_ids)
        
        assert logits.shape == (2, 8, config.vocab_size)
    
    def test_positional_embeddings(self, small_sparsity_config):
        """Test model with positional embeddings."""
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_ctx=32,
            d_head=16,
            vocab_size=100,
            use_positional_embeddings=True,
        )
        model = create_model(config, small_sparsity_config)
        
        assert model.wpe is not None
        
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        logits, _, _ = model(input_ids)
        
        assert logits.shape == (2, 16, config.vocab_size)
    
    def test_tied_embeddings(self, small_sparsity_config):
        """Test model with tied embeddings."""
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_ctx=32,
            d_head=16,
            vocab_size=100,
            tie_embeddings=True,
        )
        model = create_model(config, small_sparsity_config)
        
        # Check weights are the same object
        assert model.wte.weight is model.lm_head.weight
    
    def test_generate(self, small_config, small_sparsity_config):
        """Test text generation."""
        model = create_model(small_config, small_sparsity_config)
        model.eval()
        
        input_ids = torch.randint(0, small_config.vocab_size, (2, 4))
        
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=8)
        
        assert generated.shape == (2, 12)  # 4 + 8 new tokens
    
    def test_gradient_flow(self, small_config, small_sparsity_config):
        """Test that gradients flow through the model."""
        model = create_model(small_config, small_sparsity_config)
        
        input_ids = torch.randint(0, small_config.vocab_size, (2, 8))
        labels = torch.randint(0, small_config.vocab_size, (2, 8))
        
        logits, loss, _ = model(input_ids, labels=labels)
        loss.backward()
        
        # Check that all trainable parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_rms_norm(self, small_config, small_sparsity_config):
        """Test that RMSNorm is used when configured."""
        model = create_model(small_config, small_sparsity_config)
        
        assert isinstance(model.ln_f, nn.RMSNorm)
        for block in model.blocks:
            assert isinstance(block.ln_1, nn.RMSNorm)
            assert isinstance(block.ln_2, nn.RMSNorm)
    
    def test_layer_norm_option(self, small_sparsity_config):
        """Test that LayerNorm can be used instead."""
        config = ModelConfig(
            n_layer=2,
            d_model=64,
            n_ctx=32,
            d_head=16,
            vocab_size=100,
            use_rms_norm=False,
        )
        model = create_model(config, small_sparsity_config)
        
        assert isinstance(model.ln_f, nn.LayerNorm)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

