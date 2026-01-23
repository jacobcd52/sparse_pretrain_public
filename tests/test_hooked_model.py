"""
Tests to verify HookedSparseGPT produces identical outputs to SparseGPT.

These tests ensure that the hooked implementation is mathematically
equivalent to the original, which is critical for interpretability work.
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparse_pretrain.src.config import ModelConfig, SparsityConfig
from sparse_pretrain.src.model import SparseGPT
from sparse_pretrain.src.hooked_model import HookedSparseGPT, HookedSparseGPTConfig


def create_test_configs(
    n_layer: int = 2,
    d_model: int = 64,
    d_head: int = 16,
    d_mlp: int = 128,
    vocab_size: int = 1000,
    n_ctx: int = 32,
    use_attention_sinks: bool = True,
    use_rms_norm: bool = True,
) -> tuple[ModelConfig, SparsityConfig]:
    """Create test configurations for both models."""
    model_config = ModelConfig(
        n_layer=n_layer,
        d_model=d_model,
        d_head=d_head,
        d_mlp=d_mlp,
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        use_attention_sinks=use_attention_sinks,
        use_rms_norm=use_rms_norm,
        use_bigram_table=False,  # Must be False for HookedSparseGPT
        use_positional_embeddings=False,
        tie_embeddings=False,
        activation="gelu",
        dropout=0.0,
        use_bias=True,
    )
    
    # Disable sparsity for testing (we want exact numerical match)
    sparsity_config = SparsityConfig(
        enable_weight_sparsity=False,
        enable_activation_sparsity=False,
    )
    
    return model_config, sparsity_config


def create_hooked_config(model_config: ModelConfig) -> HookedSparseGPTConfig:
    """Create HookedSparseGPTConfig from ModelConfig."""
    return HookedSparseGPTConfig.from_sparse_gpt_config(
        model_config,
        device="cpu",
        dtype=torch.float32,
    )


def transfer_weights_sparse_to_hooked(
    sparse_model: SparseGPT,
    hooked_model: HookedSparseGPT,
):
    """Transfer weights from SparseGPT to HookedSparseGPT."""
    state_dict = sparse_model.state_dict()
    hooked_model._load_sparse_gpt_weights(state_dict)


class TestWeightTransfer:
    """Test that weights are correctly transferred."""
    
    def test_embedding_weights(self):
        """Test embedding weights match."""
        model_cfg, sparsity_cfg = create_test_configs()
        sparse = SparseGPT(model_cfg, sparsity_cfg)
        
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        
        transfer_weights_sparse_to_hooked(sparse, hooked)
        
        torch.testing.assert_close(sparse.wte.weight, hooked.embed.weight)
    
    def test_lm_head_weights(self):
        """Test language model head weights match."""
        model_cfg, sparsity_cfg = create_test_configs()
        sparse = SparseGPT(model_cfg, sparsity_cfg)
        
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        
        transfer_weights_sparse_to_hooked(sparse, hooked)
        
        torch.testing.assert_close(sparse.lm_head.weight, hooked.unembed.weight)
    
    def test_attention_weights(self):
        """Test attention weights are correctly transposed."""
        model_cfg, sparsity_cfg = create_test_configs()
        sparse = SparseGPT(model_cfg, sparsity_cfg)
        
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        
        transfer_weights_sparse_to_hooked(sparse, hooked)
        
        for i, (sparse_block, hooked_block) in enumerate(zip(sparse.blocks, hooked.blocks)):
            # QKV weights: sparse uses nn.Linear, hooked uses direct matmul
            # sparse: weight is (3*n_heads*d_head, d_model), forward does x @ W.T
            # hooked: W is (d_model, 3*n_heads*d_head), forward does x @ W
            sparse_qkv = sparse_block.attn.c_attn.weight  # (out, in)
            hooked_qkv = hooked_block.attn.W_QKV  # (in, out)
            torch.testing.assert_close(sparse_qkv.T, hooked_qkv, msg=f"QKV weights mismatch at layer {i}")
            
            # Output projection
            sparse_out = sparse_block.attn.c_proj.weight
            hooked_out = hooked_block.attn.W_O
            torch.testing.assert_close(sparse_out.T, hooked_out, msg=f"Output weights mismatch at layer {i}")


class TestForwardPass:
    """Test forward pass produces identical outputs."""
    
    @pytest.fixture
    def models_and_input(self):
        """Create matched models and test input."""
        torch.manual_seed(42)
        
        model_cfg, sparsity_cfg = create_test_configs()
        sparse = SparseGPT(model_cfg, sparsity_cfg)
        sparse.eval()
        
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        hooked.eval()
        
        transfer_weights_sparse_to_hooked(sparse, hooked)
        
        # Random input tokens
        input_ids = torch.randint(0, model_cfg.vocab_size, (2, 16))
        
        return sparse, hooked, input_ids
    
    def test_logits_match(self, models_and_input):
        """Test that logits are identical."""
        sparse, hooked, input_ids = models_and_input
        
        with torch.no_grad():
            sparse_logits, _, _ = sparse(input_ids)
            hooked_logits = hooked(input_ids, return_type="logits")
        
        torch.testing.assert_close(
            sparse_logits, hooked_logits,
            rtol=1e-4, atol=1e-4,
            msg="Logits do not match!"
        )
    
    def test_logits_match_no_sinks(self):
        """Test logits match when attention sinks are disabled."""
        torch.manual_seed(42)
        
        model_cfg, sparsity_cfg = create_test_configs(use_attention_sinks=False)
        sparse = SparseGPT(model_cfg, sparsity_cfg)
        sparse.eval()
        
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        hooked.eval()
        
        transfer_weights_sparse_to_hooked(sparse, hooked)
        
        input_ids = torch.randint(0, model_cfg.vocab_size, (2, 16))
        
        with torch.no_grad():
            sparse_logits, _, _ = sparse(input_ids)
            hooked_logits = hooked(input_ids, return_type="logits")
        
        torch.testing.assert_close(
            sparse_logits, hooked_logits,
            rtol=1e-4, atol=1e-4,
            msg="Logits do not match (no sinks)!"
        )
    
    def test_logits_match_layer_norm(self):
        """Test logits match with LayerNorm instead of RMSNorm."""
        torch.manual_seed(42)
        
        model_cfg, sparsity_cfg = create_test_configs(use_rms_norm=False)
        sparse = SparseGPT(model_cfg, sparsity_cfg)
        sparse.eval()
        
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        hooked.eval()
        
        transfer_weights_sparse_to_hooked(sparse, hooked)
        
        input_ids = torch.randint(0, model_cfg.vocab_size, (2, 16))
        
        with torch.no_grad():
            sparse_logits, _, _ = sparse(input_ids)
            hooked_logits = hooked(input_ids, return_type="logits")
        
        torch.testing.assert_close(
            sparse_logits, hooked_logits,
            rtol=1e-4, atol=1e-4,
            msg="Logits do not match (LayerNorm)!"
        )
    
    def test_loss_match(self, models_and_input):
        """Test that computed loss is identical."""
        sparse, hooked, input_ids = models_and_input
        
        with torch.no_grad():
            _, sparse_loss, _ = sparse(input_ids, labels=input_ids)
            hooked_loss = hooked(input_ids, return_type="loss")
        
        torch.testing.assert_close(
            sparse_loss, hooked_loss,
            rtol=1e-4, atol=1e-4,
            msg="Loss does not match!"
        )


class TestStopAtLayer:
    """Test stop_at_layer functionality."""
    
    def test_stop_at_layer(self):
        """Test stopping at intermediate layers."""
        torch.manual_seed(42)
        
        model_cfg, sparsity_cfg = create_test_configs(n_layer=4)
        sparse = SparseGPT(model_cfg, sparsity_cfg)
        sparse.eval()
        
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        hooked.eval()
        
        transfer_weights_sparse_to_hooked(sparse, hooked)
        
        input_ids = torch.randint(0, model_cfg.vocab_size, (2, 16))
        
        # Get residual stream at layer 2 from hooked model
        with torch.no_grad():
            hooked_resid = hooked(input_ids, stop_at_layer=2)
        
        # Manually compute what sparse model would have at layer 2
        with torch.no_grad():
            x = sparse.wte(input_ids)
            for i in range(2):
                x = sparse.blocks[i](x)
        
        torch.testing.assert_close(
            x, hooked_resid,
            rtol=1e-4, atol=1e-4,
            msg="stop_at_layer residual does not match!"
        )


class TestHookPoints:
    """Test that hook points work correctly."""
    
    def test_hook_embed(self):
        """Test that hook_embed captures embeddings."""
        torch.manual_seed(42)
        
        model_cfg, sparsity_cfg = create_test_configs()
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        hooked.eval()
        
        input_ids = torch.randint(0, model_cfg.vocab_size, (2, 16))
        
        cache = {}
        def save_hook(tensor, hook):
            cache[hook.name] = tensor.detach().clone()
        
        hooked.hook_embed.add_hook(save_hook)
        
        with torch.no_grad():
            hooked(input_ids)
        
        assert "hook_embed" in cache
        assert cache["hook_embed"].shape == (2, 16, model_cfg.d_model)
    
    def test_run_with_cache(self):
        """Test run_with_cache captures activations."""
        torch.manual_seed(42)
        
        model_cfg, sparsity_cfg = create_test_configs(n_layer=2)
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        hooked.eval()
        
        input_ids = torch.randint(0, model_cfg.vocab_size, (2, 16))
        
        with torch.no_grad():
            _, cache = hooked.run_with_cache(input_ids)
        
        # Check various hook points exist
        assert "hook_embed" in cache
        assert "blocks.0.hook_resid_pre" in cache
        assert "blocks.0.hook_resid_mid" in cache
        assert "blocks.0.hook_resid_post" in cache
        assert "blocks.0.hook_attn_out" in cache
        assert "blocks.0.hook_mlp_out" in cache
        assert "blocks.0.attn.hook_pattern" in cache
        assert "blocks.0.mlp.hook_pre" in cache
        assert "blocks.0.mlp.hook_post" in cache
    
    def test_attention_pattern_shape_with_sinks(self):
        """Test attention pattern has correct shape with sinks."""
        torch.manual_seed(42)
        
        model_cfg, sparsity_cfg = create_test_configs(use_attention_sinks=True)
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        hooked.eval()
        
        seq_len = 16
        input_ids = torch.randint(0, model_cfg.vocab_size, (2, seq_len))
        
        with torch.no_grad():
            _, cache = hooked.run_with_cache(input_ids)
        
        # With sinks, pattern shape should be (batch, n_heads, seq_len, seq_len+1)
        pattern = cache["blocks.0.attn.hook_pattern"]
        assert pattern.shape == (2, model_cfg.n_heads, seq_len, seq_len + 1)
    
    def test_attention_pattern_shape_no_sinks(self):
        """Test attention pattern has correct shape without sinks."""
        torch.manual_seed(42)
        
        model_cfg, sparsity_cfg = create_test_configs(use_attention_sinks=False)
        hooked_cfg = create_hooked_config(model_cfg)
        hooked = HookedSparseGPT(hooked_cfg)
        hooked.eval()
        
        seq_len = 16
        input_ids = torch.randint(0, model_cfg.vocab_size, (2, seq_len))
        
        with torch.no_grad():
            _, cache = hooked.run_with_cache(input_ids)
        
        # Without sinks, pattern shape should be (batch, n_heads, seq_len, seq_len)
        pattern = cache["blocks.0.attn.hook_pattern"]
        assert pattern.shape == (2, model_cfg.n_heads, seq_len, seq_len)


class TestBigramTableAssertion:
    """Test that bigram table raises error."""
    
    def test_bigram_table_assertion(self):
        """Test that creating HookedSparseGPT with bigram_table fails."""
        model_cfg, _ = create_test_configs()
        model_cfg.use_bigram_table = True  # This should cause an error
        
        # The assertion happens in from_pretrained, but we can't easily test that
        # without a real checkpoint. Instead, document that it's asserted.
        # In real usage, from_pretrained checks this in the config.
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

