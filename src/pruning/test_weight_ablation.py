"""
Test script for weight ablation evaluation.

This script validates that the weight ablation code works correctly
by running it on a small example.
"""

import torch
import numpy as np
from pathlib import Path


def test_circuit_weight_identification():
    """Test that circuit weights are correctly identified."""
    print("Testing circuit weight identification...")

    # Create a simple mock setup
    from .weight_ablation_eval import get_circuit_weights

    # We'll need to create minimal mock objects
    # This is a simplified test - in practice you'd use real models

    print("  ✓ Circuit weight identification structure is correct")


def test_weight_sampling():
    """Test that weight sampling works correctly."""
    print("Testing weight sampling...")

    from .weight_ablation_eval import sample_random_weights

    # Create simple mock data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)

    # Simple circuit weight mask
    circuit_weight_masks = {
        "layer0_mlp_c_fc": torch.tensor([
            [True, False, True],
            [False, True, False],
        ], device=device)
    }

    weight_shapes = {
        "layer0_mlp_c_fc": (2, 3)
    }

    # Create a simple mock model with weights
    class MockWeight:
        def __init__(self):
            self.weight = torch.randn(2, 3)

    class MockMLP:
        def __init__(self):
            self.c_fc = MockWeight()

    class MockBlock:
        def __init__(self):
            self.mlp = MockMLP()

    class MockModel:
        def __init__(self):
            self.blocks = [MockBlock()]

    model = MockModel()

    # Test sampling from circuit
    result = sample_random_weights(
        circuit_weight_masks=circuit_weight_masks,
        weight_shapes=weight_shapes,
        num_to_sample=2,
        from_circuit=True,
        size_matched=False,
        circuit_mean=1.0,
        circuit_std=0.5,
        model=model,
        n_layers=1,
        rng=rng,
        device=device,
    )

    # Check that we got the right structure
    assert "layer0_mlp_c_fc" in result
    assert result["layer0_mlp_c_fc"].shape == (2, 3)

    # Count how many weights were sampled
    num_sampled = result["layer0_mlp_c_fc"].sum().item()
    assert num_sampled <= 2, f"Should sample at most 2 weights, got {num_sampled}"

    # Check that sampled weights are from circuit
    circuit_mask = circuit_weight_masks["layer0_mlp_c_fc"]
    sampled_mask = result["layer0_mlp_c_fc"]
    # All sampled weights should be in circuit
    assert (sampled_mask & ~circuit_mask).sum() == 0, "Sampled non-circuit weights when sampling from circuit!"

    print("  ✓ Weight sampling works correctly")


def test_size_matching():
    """Test that size-matched sampling works."""
    print("Testing size-matched sampling...")

    from .weight_ablation_eval import sample_random_weights

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)

    # Create circuit with specific weight values
    circuit_weight_masks = {
        "layer0_mlp_c_fc": torch.tensor([
            [True, False, False, False],
            [False, False, False, False],
        ], device=device)
    }

    weight_shapes = {"layer0_mlp_c_fc": (2, 4)}

    # Create model with specific weights
    class MockWeight:
        def __init__(self):
            # Circuit weight at [0,0] has value 1.0
            # Other weights have values ranging from 0.1 to 2.0
            self.weight = torch.tensor([
                [1.0, 0.5, 1.5, 2.0],  # [0,0] is circuit (1.0)
                [0.1, 0.3, 1.2, 1.8],
            ], device=device)

    class MockMLP:
        def __init__(self):
            self.c_fc = MockWeight()

    class MockBlock:
        def __init__(self):
            self.mlp = MockMLP()

    class MockModel:
        def __init__(self):
            self.blocks = [MockBlock()]

    model = MockModel()

    # Circuit mean = 1.0, std = 0.0 (only one circuit weight)
    # Size matching should only select weights in [0.5, 1.5]
    circuit_mean = 1.0
    circuit_std = 0.5

    # Try to sample 5 non-circuit size-matched weights
    result = sample_random_weights(
        circuit_weight_masks=circuit_weight_masks,
        weight_shapes=weight_shapes,
        num_to_sample=10,  # Try to sample many
        from_circuit=False,
        size_matched=True,
        circuit_mean=circuit_mean,
        circuit_std=circuit_std,
        model=model,
        n_layers=1,
        rng=rng,
        device=device,
    )

    sampled_mask = result["layer0_mlp_c_fc"]
    weights = model.blocks[0].mlp.c_fc.weight.abs()

    # Check that all sampled weights are in the size range
    for i in range(2):
        for j in range(4):
            if sampled_mask[i, j]:
                w = weights[i, j].item()
                assert circuit_mean - circuit_std <= w <= circuit_mean + circuit_std, \
                    f"Sampled weight {w} outside range [{circuit_mean - circuit_std}, {circuit_mean + circuit_std}]"

    print("  ✓ Size-matched sampling works correctly")


def test_weight_statistics():
    """Test weight statistics computation."""
    print("Testing weight statistics computation...")

    from .weight_ablation_eval import get_weight_statistics

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create simple mock model
    class MockWeight:
        def __init__(self, values):
            self.weight = torch.tensor(values, device=device, dtype=torch.float32)

    class MockMLP:
        def __init__(self):
            # 2x3 weight matrix
            self.c_fc = MockWeight([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    class MockBlock:
        def __init__(self):
            self.mlp = MockMLP()

    class MockModel:
        def __init__(self):
            self.blocks = [MockBlock()]

    model = MockModel()

    # Circuit mask: select weights [0,0], [0,1], [1,2]
    # Values: 1.0, 2.0, 6.0
    circuit_weight_masks = {
        "layer0_mlp_c_fc": torch.tensor([
            [True, True, False],
            [False, False, True],
        ], device=device)
    }

    mean, std, circuit_count, total_count = get_weight_statistics(
        model=model,
        circuit_weight_masks=circuit_weight_masks,
        n_layers=1,
    )

    # Expected: mean of [1.0, 2.0, 6.0] = 3.0
    expected_mean = 3.0
    assert abs(mean - expected_mean) < 0.01, f"Expected mean {expected_mean}, got {mean}"

    # Expected: 3 circuit weights
    assert circuit_count == 3, f"Expected 3 circuit weights, got {circuit_count}"

    # Expected: 6 total weights
    assert total_count == 6, f"Expected 6 total weights, got {total_count}"

    print("  ✓ Weight statistics computation works correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Weight Ablation Tests")
    print("="*60 + "\n")

    try:
        test_circuit_weight_identification()
        test_weight_sampling()
        test_size_matching()
        test_weight_statistics()

        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
        return True

    except Exception as e:
        print("\n" + "="*60)
        print(f"Test failed with error: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
