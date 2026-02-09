"""
Tests for KL divergence top-k approximation consistency.

Verifies that:
1. KLTargetCache produces the same result as direct topk computation
2. topk is correctly forwarded through compute_hybrid_kl_losses
3. Token dimensions align between shifted (kl_sparse) and unshifted (hybrid KL) paths
"""

import torch
import torch.nn.functional as F
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import ModelConfig, SparsityConfig
from src.model import SparseGPT
from src.bridges import (
    BridgeSet,
    KLTargetCache,
    kl_divergence,
    compute_hybrid_kl_losses,
)


def create_test_models(n_layers=2, d_dense=64, d_sparse=96, vocab_size=100, n_ctx=32):
    """Create test dense and sparse models."""
    dense_config = ModelConfig(
        n_layer=n_layers, d_model=d_dense, n_ctx=n_ctx, d_head=16,
        vocab_size=vocab_size, use_bigram_table=False, use_attention_sinks=False,
    )
    dense_model = SparseGPT(dense_config, SparsityConfig(
        enable_weight_sparsity=False, enable_activation_sparsity=False,
    ))
    dense_model.eval()
    for p in dense_model.parameters():
        p.requires_grad = False

    sparse_config = ModelConfig(
        n_layer=n_layers, d_model=d_sparse, n_ctx=n_ctx, d_head=16,
        vocab_size=vocab_size, use_bigram_table=False, use_attention_sinks=False,
    )
    sparse_model = SparseGPT(sparse_config, SparsityConfig(
        enable_weight_sparsity=False, enable_activation_sparsity=False,
    ))
    sparse_model.train()

    return dense_model, sparse_model


class TestKLApproxConsistency:
    """Tests that KL top-k approximation is applied consistently."""

    def test_cache_matches_direct_topk(self):
        """KLTargetCache with topk must produce identical result to direct topk call."""
        torch.manual_seed(42)
        vocab_size = 100
        topk = 16

        logits_target = torch.randn(2, 8, vocab_size)
        logits_source = torch.randn(2, 8, vocab_size)

        # Direct computation
        kl_direct = kl_divergence(logits_target, logits_source, topk=topk)

        # Via cache
        cache = KLTargetCache(logits_target, temperature=1.0, topk=topk)
        kl_cached = kl_divergence(logits_target, logits_source, target_cache=cache)

        assert torch.allclose(kl_direct, kl_cached, atol=1e-5), (
            f"Cache KL ({kl_cached.item():.6f}) != direct KL ({kl_direct.item():.6f})"
        )

    def test_cache_topk_flag_is_set(self):
        """KLTargetCache should have use_topk=True when topk is specified."""
        logits = torch.randn(2, 8, 100)

        cache_with_topk = KLTargetCache(logits, topk=16)
        assert cache_with_topk.use_topk is True
        assert cache_with_topk.topk_indices is not None
        assert cache_with_topk.topk_indices.shape[-1] == 16

        cache_without_topk = KLTargetCache(logits, topk=None)
        assert cache_without_topk.use_topk is False
        assert cache_without_topk.topk_indices is None

    def test_topk_changes_kl_value(self):
        """KL with topk should differ from full-vocab KL (sanity check)."""
        torch.manual_seed(42)
        vocab_size = 100
        topk = 16

        logits_target = torch.randn(2, 8, vocab_size)
        logits_source = torch.randn(2, 8, vocab_size)

        kl_full = kl_divergence(logits_target, logits_source, topk=None)
        kl_topk = kl_divergence(logits_target, logits_source, topk=topk)

        assert not torch.allclose(kl_full, kl_topk, atol=1e-3), (
            "Top-k KL should differ from full-vocab KL for random logits"
        )

    def test_topk_ignored_when_cache_provided(self):
        """When target_cache is provided, topk parameter should be ignored."""
        torch.manual_seed(42)
        vocab_size = 100

        logits_target = torch.randn(2, 8, vocab_size)
        logits_source = torch.randn(2, 8, vocab_size)

        cache = KLTargetCache(logits_target, topk=16)

        # Passing a different topk should have no effect when cache is used
        kl_cache_only = kl_divergence(logits_target, logits_source, target_cache=cache)
        kl_cache_with_topk = kl_divergence(logits_target, logits_source, target_cache=cache, topk=32)

        assert torch.allclose(kl_cache_only, kl_cache_with_topk, atol=1e-6), (
            "topk param should be ignored when target_cache is provided"
        )

    def test_hybrid_kl_uses_topk(self):
        """compute_hybrid_kl_losses with topk should differ from without topk."""
        torch.manual_seed(42)
        n_layers = 2
        d_dense = 64
        d_sparse = 96
        vocab_size = 100
        topk = 16

        dense_model, sparse_model = create_test_models(
            n_layers=n_layers, d_dense=d_dense, d_sparse=d_sparse,
            vocab_size=vocab_size,
        )

        bridge_set = BridgeSet(
            n_layers=n_layers, d_dense=d_dense, d_sparse=d_sparse,
            encoder_afrac=0.25,
        )

        input_ids = torch.randint(0, vocab_size, (2, 16))

        with torch.no_grad():
            y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
        y_sparse, h_sparse_pre_list, _ = sparse_model.forward_with_bridge_sites(input_ids)

        # Without topk (full vocab)
        result_full = compute_hybrid_kl_losses(
            dense_model=dense_model, sparse_model=sparse_model,
            bridge_set=bridge_set, h_dense_list=h_dense_list,
            h_sparse_pre_list=h_sparse_pre_list, y_dense=y_dense,
            input_ids=input_ids, topk=None,
        )

        # With topk
        result_topk = compute_hybrid_kl_losses(
            dense_model=dense_model, sparse_model=sparse_model,
            bridge_set=bridge_set, h_dense_list=h_dense_list,
            h_sparse_pre_list=h_sparse_pre_list, y_dense=y_dense,
            input_ids=input_ids, topk=topk,
        )

        assert result_topk.kl_d2s.item() > 0, "d2s KL with topk should be positive"
        assert result_topk.kl_s2d.item() > 0, "s2d KL with topk should be positive"

        assert not torch.allclose(result_full.kl_d2s, result_topk.kl_d2s, atol=1e-3), (
            f"d2s: topk KL ({result_topk.kl_d2s.item():.6f}) should differ from "
            f"full-vocab KL ({result_full.kl_d2s.item():.6f})"
        )
        assert not torch.allclose(result_full.kl_s2d, result_topk.kl_s2d, atol=1e-3), (
            f"s2d: topk KL ({result_topk.kl_s2d.item():.6f}) should differ from "
            f"full-vocab KL ({result_full.kl_s2d.item():.6f})"
        )

    def test_token_alignment_shifted_vs_unshifted(self):
        """
        Verify token dimensions are correct in the training pattern:
        - kl_sparse uses shifted logits (seq-1)
        - hybrid KL uses unshifted logits (full seq)
        Both should be internally consistent (target and source same shape).
        """
        torch.manual_seed(42)
        vocab_size = 100
        batch_size = 2
        seq_len = 16
        topk = 16

        dense_model, sparse_model = create_test_models(vocab_size=vocab_size)

        bridge_set = BridgeSet(
            n_layers=2, d_dense=64, d_sparse=96, encoder_afrac=0.25,
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
        y_sparse, h_sparse_pre_list, _ = sparse_model.forward_with_bridge_sites(input_ids)

        # -- Shifted path (kl_sparse) --
        shift_logits_dense = y_dense[:, :-1, :].contiguous()
        shift_logits_sparse = y_sparse[:, :-1, :].contiguous()

        assert shift_logits_dense.shape == (batch_size, seq_len - 1, vocab_size)
        assert shift_logits_sparse.shape == (batch_size, seq_len - 1, vocab_size)

        # Cache is built from shifted logits
        cache = KLTargetCache(shift_logits_dense, topk=topk)
        assert cache.p_target.shape[0] == batch_size * (seq_len - 1), (
            f"Cache should have {batch_size * (seq_len - 1)} positions, "
            f"got {cache.p_target.shape[0]}"
        )
        assert cache.topk_indices.shape == (batch_size * (seq_len - 1), topk)

        # kl_sparse: shifted target vs shifted source — no shape error
        kl_sparse = kl_divergence(shift_logits_dense, shift_logits_sparse, target_cache=cache)
        assert kl_sparse.isfinite(), "kl_sparse should be finite"

        # -- Unshifted path (hybrid KL) --
        # hybrid outputs have full seq length
        result = compute_hybrid_kl_losses(
            dense_model=dense_model, sparse_model=sparse_model,
            bridge_set=bridge_set, h_dense_list=h_dense_list,
            h_sparse_pre_list=h_sparse_pre_list, y_dense=y_dense,
            input_ids=input_ids, topk=topk,
        )
        assert result.kl_d2s.isfinite(), "hybrid d2s KL should be finite"
        assert result.kl_s2d.isfinite(), "hybrid s2d KL should be finite"

        # Verify the cache CANNOT be used for hybrid (shape mismatch)
        # Cache has batch*(seq-1) positions, but hybrid has batch*seq positions
        # The cache indices have dim 0 = batch*(seq-1), but hybrid source
        # flattens to batch*seq — these don't match, confirming the cache
        # must not be reused for the unshifted hybrid path.
        hybrid_flattened_positions = batch_size * seq_len
        cache_positions = cache.topk_indices.shape[0]
        assert cache_positions != hybrid_flattened_positions, (
            f"Cache positions ({cache_positions}) should differ from hybrid "
            f"positions ({hybrid_flattened_positions}) — shape mismatch expected"
        )
        assert cache_positions == batch_size * (seq_len - 1)

    def test_hybrid_kl_matches_manual_topk(self):
        """
        Verify that compute_hybrid_kl_losses with topk produces the same result
        as manually calling kl_divergence with topk on each site's hybrid output.

        Both paths must run in eval mode to ensure deterministic behavior.
        """
        torch.manual_seed(42)
        n_layers = 2
        d_dense = 64
        d_sparse = 96
        vocab_size = 100
        topk = 16

        dense_model, sparse_model = create_test_models(
            n_layers=n_layers, d_dense=d_dense, d_sparse=d_sparse,
            vocab_size=vocab_size,
        )

        bridge_set = BridgeSet(
            n_layers=n_layers, d_dense=d_dense, d_sparse=d_sparse,
            encoder_afrac=0.25,
        )

        # Put everything in eval mode for deterministic comparison
        sparse_model.eval()
        bridge_set.eval()

        input_ids = torch.randint(0, vocab_size, (2, 16))

        with torch.no_grad():
            y_dense, h_dense_list, _ = dense_model.forward_with_bridge_sites(input_ids)
            y_sparse, h_sparse_pre_list, _ = sparse_model.forward_with_bridge_sites(input_ids)

            # Compute via compute_hybrid_kl_losses
            result = compute_hybrid_kl_losses(
                dense_model=dense_model, sparse_model=sparse_model,
                bridge_set=bridge_set, h_dense_list=h_dense_list,
                h_sparse_pre_list=h_sparse_pre_list, y_dense=y_dense,
                input_ids=input_ids, topk=topk,
            )

            # Compute manually (same as validation path)
            manual_d2s = 0.0
            manual_s2d = 0.0
            for i in range(bridge_set.n_sites):
                h_encoded = bridge_set.encode(i, h_dense_list[i], sharpness=None, hard=True)
                y_hybrid_d2s = sparse_model.forward_from_site(h_encoded, i, input_ids)
                manual_d2s += kl_divergence(y_dense, y_hybrid_d2s, topk=topk).item()

                h_decoded = bridge_set.decode(i, h_sparse_pre_list[i])
                y_hybrid_s2d = dense_model.forward_from_site(h_decoded, i, input_ids)
                manual_s2d += kl_divergence(y_dense, y_hybrid_s2d, topk=topk).item()

        assert abs(result.kl_d2s.item() - manual_d2s) < 1e-3, (
            f"d2s mismatch: compute_hybrid={result.kl_d2s.item():.6f}, manual={manual_d2s:.6f}"
        )
        assert abs(result.kl_s2d.item() - manual_s2d) < 1e-3, (
            f"s2d mismatch: compute_hybrid={result.kl_s2d.item():.6f}, manual={manual_s2d:.6f}"
        )
