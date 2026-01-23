"""
Tests for gradient buffering in bridges training.

Tests that the GradientBuffer class and hybrid KL loss computation work correctly.
"""

import torch
import torch.nn as nn
import pytest

from sparse_pretrain.src.config import ModelConfig, SparsityConfig
from sparse_pretrain.src.model import SparseGPT
from sparse_pretrain.src.bridges import (
    BridgeSet,
    compute_hybrid_kl_losses,
    GradientBuffer,
)


def create_test_models(n_layers=2, d_dense=64, d_sparse=128, vocab_size=100, n_ctx=32):
    """Create test dense and sparse models with matching layer counts."""
    # Dense model (no sparsity)
    dense_config = ModelConfig(
        n_layer=n_layers,
        d_model=d_dense,
        n_ctx=n_ctx,
        d_head=16,
        vocab_size=vocab_size,
        use_bigram_table=False,
        use_attention_sinks=False,
    )
    dense_sparsity = SparsityConfig(
        enable_weight_sparsity=False,
        enable_activation_sparsity=False,
    )
    dense_model = SparseGPT(dense_config, dense_sparsity)
    dense_model.eval()
    for p in dense_model.parameters():
        p.requires_grad = False
    
    # Sparse model
    sparse_config = ModelConfig(
        n_layer=n_layers,
        d_model=d_sparse,
        n_ctx=n_ctx,
        d_head=16,
        vocab_size=vocab_size,
        use_bigram_table=False,
        use_attention_sinks=False,
    )
    sparse_sparsity = SparsityConfig(
        enable_weight_sparsity=False,
        enable_activation_sparsity=False,
    )
    sparse_model = SparseGPT(sparse_config, sparse_sparsity)
    sparse_model.train()
    
    return dense_model, sparse_model


class TestGradientBuffer:
    """Tests for the GradientBuffer class."""
    
    def test_accumulator_returns_same_shape(self):
        """Test that the accumulator returns the same shape as input."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        buffer = GradientBuffer(x)
        
        assert buffer.accumulator.shape == x.shape
    
    def test_gradient_accumulation_and_release(self):
        """Test that gradients are accumulated and released correctly."""
        W = torch.randn(10, 5, requires_grad=True)
        x = torch.randn(5)
        
        # Linear transform
        h = W @ x  # h has shape [10], requires_grad via W
        
        # Create buffer
        buffer = GradientBuffer(h)
        
        # Compute loss using accumulator
        loss = buffer.accumulator.sum()
        loss.backward(retain_graph=True)
        
        # W should not have gradients yet (they're buffered)
        assert W.grad is None, "Gradient should be buffered, not propagated"
        
        # Release gradients
        buffer.release_gradients()
        
        # Now W should have gradients
        assert W.grad is not None, "Gradient should be released"
        
        # Check the gradient value (d/dW[sum(W @ x)] = outer(1, x))
        expected_grad = torch.ones(10).unsqueeze(1) @ x.unsqueeze(0)
        assert torch.allclose(W.grad, expected_grad), "Gradient value mismatch"
    
    def test_multiple_backward_accumulation(self):
        """Test that multiple backward calls accumulate correctly."""
        W = torch.randn(10, 5, requires_grad=True)
        x = torch.randn(5)
        
        h = W @ x
        buffer = GradientBuffer(h)
        
        # Multiple losses
        loss1 = buffer.accumulator[:5].sum()
        loss2 = buffer.accumulator[5:].sum()
        
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        
        # Release accumulated gradients
        buffer.release_gradients()
        
        assert W.grad is not None
        
        # The gradient should be from both losses combined
        expected_grad = torch.ones(10).unsqueeze(1) @ x.unsqueeze(0)
        assert torch.allclose(W.grad, expected_grad), "Accumulated gradient mismatch"


class TestHybridKLLosses:
    """Tests for hybrid KL loss computation."""
    
    @pytest.fixture
    def setup_models(self):
        """Set up models and data for testing."""
        torch.manual_seed(42)
        
        n_layers = 2
        d_dense = 64
        d_sparse = 96
        vocab_size = 100
        batch_size = 2
        seq_len = 16
        
        dense_model, sparse_model = create_test_models(
            n_layers=n_layers,
            d_dense=d_dense,
            d_sparse=d_sparse,
            vocab_size=vocab_size,
        )
        
        bridge_set = BridgeSet(
            n_layers=n_layers,
            d_dense=d_dense,
            d_sparse=d_sparse,
            encoder_afrac=0.25,
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        return dense_model, sparse_model, bridge_set, input_ids
    
    def test_hybrid_kl_losses_compute(self, setup_models):
        """Test that hybrid KL losses can be computed."""
        dense_model, sparse_model, bridge_set, input_ids = setup_models
        
        # Get activations
        with torch.no_grad():
            y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
        
        y_sparse, h_sparse_pre_list, h_sparse_post_list = sparse_model.forward_with_bridge_sites(input_ids)
        
        # Compute hybrid KL losses
        result = compute_hybrid_kl_losses(
            dense_model=dense_model,
            sparse_model=sparse_model,
            bridge_set=bridge_set,
            h_dense_list=h_dense_list,
            h_sparse_pre_list=h_sparse_pre_list,
            y_dense=y_dense,
            input_ids=input_ids,
        )
        
        assert result.kl_d2s.item() > 0, "d2s KL loss should be positive"
        assert result.kl_s2d.item() > 0, "s2d KL loss should be positive"
        assert torch.allclose(result.total, result.kl_d2s + result.kl_s2d)
    
    def test_hybrid_kl_losses_gradients(self, setup_models):
        """Test that gradients flow correctly through hybrid KL losses."""
        dense_model, sparse_model, bridge_set, input_ids = setup_models
        
        sparse_model.zero_grad()
        bridge_set.zero_grad()
        
        # Get activations
        with torch.no_grad():
            y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
        
        y_sparse, h_sparse_pre_list, h_sparse_post_list = sparse_model.forward_with_bridge_sites(input_ids)
        
        # Compute hybrid KL losses
        result = compute_hybrid_kl_losses(
            dense_model=dense_model,
            sparse_model=sparse_model,
            bridge_set=bridge_set,
            h_dense_list=h_dense_list,
            h_sparse_pre_list=h_sparse_pre_list,
            y_dense=y_dense,
            input_ids=input_ids,
        )
        
        # Backward
        result.total.backward(retain_graph=True)
        result.release_gradients()
        
        # Check that gradients exist
        sparse_grad_count = sum(1 for p in sparse_model.parameters() if p.grad is not None)
        bridge_grad_count = sum(1 for p in bridge_set.parameters() if p.grad is not None)
        
        assert sparse_grad_count > 0, "Sparse model should have gradients"
        assert bridge_grad_count > 0, "Bridges should have gradients"
        
        # Dense model should NOT have gradients (frozen)
        dense_grad_count = sum(1 for p in dense_model.parameters() if p.grad is not None)
        assert dense_grad_count == 0, "Dense model should not have gradients (frozen)"


def test_end_to_end():
    """Quick end-to-end test."""
    print("Running end-to-end gradient buffer test...")
    
    torch.manual_seed(42)
    
    n_layers = 2
    d_dense = 64
    d_sparse = 96
    vocab_size = 100
    batch_size = 2
    seq_len = 16
    
    dense_model, sparse_model = create_test_models(
        n_layers=n_layers,
        d_dense=d_dense,
        d_sparse=d_sparse,
        vocab_size=vocab_size,
    )
    
    bridge_set = BridgeSet(
        n_layers=n_layers,
        d_dense=d_dense,
        d_sparse=d_sparse,
        encoder_afrac=0.25,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Get activations
    with torch.no_grad():
        y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
    
    sparse_model.zero_grad()
    bridge_set.zero_grad()
    
    y_sparse, h_sparse_pre_list, h_sparse_post_list = sparse_model.forward_with_bridge_sites(input_ids)
    
    # Compute hybrid KL losses
    result = compute_hybrid_kl_losses(
        dense_model, sparse_model, bridge_set,
        h_dense_list, h_sparse_pre_list, y_dense, input_ids,
    )
    
    print(f"KL d2s: {result.kl_d2s.item():.4f}")
    print(f"KL s2d: {result.kl_s2d.item():.4f}")
    print(f"Total: {result.total.item():.4f}")
    
    # Backward and release
    result.total.backward(retain_graph=True)
    result.release_gradients()
    
    # Check gradients
    sparse_grad_count = sum(1 for p in sparse_model.parameters() if p.grad is not None)
    bridge_grad_count = sum(1 for p in bridge_set.parameters() if p.grad is not None)
    
    print(f"Sparse model params with grad: {sparse_grad_count}")
    print(f"Bridge params with grad: {bridge_grad_count}")
    
    assert sparse_grad_count > 0, "Sparse model should have gradients"
    assert bridge_grad_count > 0, "Bridges should have gradients"
    
    print("✓ PASS: End-to-end test successful!")


if __name__ == "__main__":
    test_end_to_end()
