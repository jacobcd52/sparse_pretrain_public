#!/usr/bin/env python3
"""
Test that MaskedSparseGPT produces identical outputs to SparseGPT
when no masking is applied (all masks = 1).
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from my_sparse_pretrain.src.pruning.run_pruning import load_model
from my_sparse_pretrain.src.pruning.config import PruningConfig
from my_sparse_pretrain.src.pruning.masked_model import MaskedSparseGPT


def test_masked_model_matches_base_no_masks():
    """Test that masked model with mask_locations=[] matches base model exactly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model
    model, _ = load_model("jacobcd52/ss_bridges_d1024_f0.015625", device)
    
    # Create masked model with NO mask locations
    config = PruningConfig(
        mask_locations=[],  # No masking
        init_noise_scale=0.01,
        init_noise_bias=1.0,
        device=device,
    )
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)
    
    # Test input
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (4, 32)).to(device)
    
    # Get outputs
    model.eval()
    masked_model.eval()
    
    with torch.no_grad():
        base_out = model(input_ids)
        base_logits = base_out[0] if isinstance(base_out, tuple) else base_out
        
        masked_logits = masked_model(input_ids)
    
    # Compare
    max_diff = (base_logits - masked_logits).abs().max().item()
    mean_diff = (base_logits - masked_logits).abs().mean().item()
    
    print(f"Max diff: {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    
    # They should match very closely (allowing for tiny floating point differences)
    assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance 1e-4"
    assert mean_diff < 1e-5, f"Mean diff {mean_diff} exceeds tolerance 1e-5"
    
    print("✓ Test passed: masked model (no masks) matches base model")


def test_masked_model_matches_base_all_active():
    """Test that masked model with all masks active (tau >> 0) matches base model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model
    model, _ = load_model("jacobcd52/ss_bridges_d1024_f0.015625", device)
    
    # Create masked model with all mask locations but high noise_bias (all active)
    config = PruningConfig(
        mask_locations=['attn_in', 'attn_q', 'attn_k', 'attn_v', 'attn_out', 
                       'mlp_in', 'mlp_neuron', 'mlp_out'],
        init_noise_scale=0.001,  # Very small noise
        init_noise_bias=1.0,     # High bias = all tau > 0 = all masks = 1
        device=device,
    )
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)
    
    # Verify all nodes are active
    num_active = masked_model.masks.get_total_active_nodes()
    total_nodes = masked_model.masks.get_total_nodes()
    assert num_active == total_nodes, f"Expected all {total_nodes} nodes active, got {num_active}"
    
    # Test input
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1000, (4, 32)).to(device)
    
    # Get outputs
    model.eval()
    masked_model.eval()
    
    with torch.no_grad():
        base_out = model(input_ids)
        base_logits = base_out[0] if isinstance(base_out, tuple) else base_out
        
        masked_logits = masked_model(input_ids)
    
    # Compare
    max_diff = (base_logits - masked_logits).abs().max().item()
    mean_diff = (base_logits - masked_logits).abs().mean().item()
    
    print(f"Max diff: {max_diff:.8f}")
    print(f"Mean diff: {mean_diff:.8f}")
    
    # They should match very closely
    assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance 1e-4"
    assert mean_diff < 1e-5, f"Mean diff {mean_diff} exceeds tolerance 1e-5"
    
    print("✓ Test passed: masked model (all masks active) matches base model")


def test_abstopk_is_applied():
    """Test that AbsTopK is being applied correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model
    model, _ = load_model("jacobcd52/ss_bridges_d1024_f0.015625", device)
    
    # Create masked model
    config = PruningConfig(
        mask_locations=[],
        init_noise_scale=0.01,
        init_noise_bias=1.0,
        device=device,
    )
    masked_model = MaskedSparseGPT(model, config)
    masked_model.to(device)
    
    # Check that AbsTopK config was loaded
    assert masked_model._activation_sparsity_enabled, "AbsTopK should be enabled"
    assert masked_model._activation_topk_fraction == 0.25, f"Expected topk_fraction=0.25, got {masked_model._activation_topk_fraction}"
    assert 'mlp_neuron' in masked_model._activation_sparsity_locations, "mlp_neuron should be in sparsity locations"
    
    print("✓ Test passed: AbsTopK config loaded correctly")


if __name__ == "__main__":
    print("="*60)
    print("Test 1: No mask locations")
    print("="*60)
    test_masked_model_matches_base_no_masks()
    
    print("\n" + "="*60)
    print("Test 2: All masks active (tau >> 0)")
    print("="*60)
    test_masked_model_matches_base_all_active()
    
    print("\n" + "="*60)
    print("Test 3: AbsTopK config loaded")
    print("="*60)
    test_abstopk_is_applied()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

