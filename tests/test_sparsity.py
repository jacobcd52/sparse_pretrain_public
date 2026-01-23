"""
Tests for weight sparsity utilities.
"""

import pytest
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from sparse_pretrain.src.sparsity import WeightSparsifier, SharkfinScheduler, clip_grad_rms_


class SimpleModel(nn.Module):
    """Simple model for testing sparsity."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 32)
        self.ln = nn.LayerNorm(32)  # Should be skipped by sparsifier
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        return x


class TestWeightSparsifier:
    """Tests for WeightSparsifier."""
    
    def test_initialization(self):
        """Test sparsifier initialization."""
        model = SimpleModel()
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.1,
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.5,
            min_weights_per_neuron=4,
            total_steps=1000,
        )
        
        assert sparsifier.target_l0_fraction == 0.1
        assert sparsifier.state.anneal_end_step == 500
        assert sparsifier.get_current_l0_fraction() == 1.0  # Start dense
    
    def test_l0_annealing(self):
        """Test L0 fraction annealing schedule."""
        model = SimpleModel()
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.1,
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.5,
            total_steps=1000,
        )
        
        # At step 0, should be fully dense
        assert sparsifier.get_current_l0_fraction() == 1.0
        
        # At step 250 (50% of annealing), should be at 0.55
        sparsifier.state.current_step = 250
        l0 = sparsifier.get_current_l0_fraction()
        assert abs(l0 - 0.55) < 0.01  # 1.0 - 0.5 * (1.0 - 0.1) = 0.55
        
        # At step 500 (end of annealing), should be at target
        sparsifier.state.current_step = 500
        assert abs(sparsifier.get_current_l0_fraction() - 0.1) < 0.01
        
        # After annealing, should stay at target
        sparsifier.state.current_step = 750
        assert abs(sparsifier.get_current_l0_fraction() - 0.1) < 0.01
    
    def test_apply_sparsity(self):
        """Test that sparsity is applied correctly."""
        model = SimpleModel()
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.5,  # Keep 50% of weights
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.0,  # Immediate sparsity (anneal completes at step 0)
            min_weights_per_neuron=1,
            total_steps=100,
        )
        
        # Count initial nonzero
        initial_nonzero = sum(
            (p != 0).sum().item() 
            for p in model.parameters() 
            if p.requires_grad and "ln" not in [n for n, _ in model.named_parameters() if _ is p][0]
        )
        
        sparsifier.apply_sparsity()
        
        # Count after sparsification
        final_nonzero = sum(
            (p != 0).sum().item() 
            for p in model.parameters() 
            if p.requires_grad and "ln" not in [n for n, _ in model.named_parameters() if _ is p][0]
        )
        
        # Should have roughly half the weights (within tolerance)
        # Note: LayerNorm is skipped, so we only check Linear layers
        ratio = final_nonzero / initial_nonzero
        assert 0.4 < ratio < 0.6
    
    def test_min_weights_per_neuron(self):
        """Test that minimum weights per neuron is respected."""
        model = SimpleModel()
        min_j = 4
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.01,  # Very sparse
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.0,  # Immediate
            min_weights_per_neuron=min_j,
            total_steps=100,
        )
        
        sparsifier.apply_sparsity()
        
        # Check that each row of weight matrices has at least min_j nonzero
        for name, param in model.named_parameters():
            if "weight" in name and len(param.shape) == 2 and "ln" not in name:
                nonzero_per_row = (param != 0).sum(dim=1)
                assert (nonzero_per_row >= min_j).all(), f"Row in {name} has fewer than {min_j} nonzero"
    
    def test_layernorm_skipped(self):
        """Test that LayerNorm parameters are not sparsified."""
        model = SimpleModel()
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.1,
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.0,  # Immediate
            total_steps=100,
        )
        
        # Record LayerNorm weights before
        ln_weight_before = model.ln.weight.clone()
        
        sparsifier.apply_sparsity()
        
        # LayerNorm weights should be unchanged
        assert torch.allclose(model.ln.weight, ln_weight_before)
    
    def test_sharkfin_multiplier(self):
        """Test sharkfin LR multiplier calculation."""
        model = SimpleModel()
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.01,  # 1/100 nonzero
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.5,
            total_steps=1000,
        )
        
        # At full density (L0=1), multiplier should be 1
        assert sparsifier.get_sharkfin_lr_multiplier() == 1.0
        
        # At L0=0.01, multiplier should be 10 (1/sqrt(0.01))
        sparsifier.state.current_step = 500
        multiplier = sparsifier.get_sharkfin_lr_multiplier()
        assert abs(multiplier - 10.0) < 0.1
    
    def test_get_sparsity_stats(self):
        """Test sparsity statistics."""
        model = SimpleModel()
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.5,
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.0,  # Immediate
            total_steps=100,
        )
        
        sparsifier.apply_sparsity()
        stats = sparsifier.get_sparsity_stats()
        
        assert "current_l0_fraction" in stats
        assert "actual_l0_fraction" in stats
        assert "actual_nonzero_params" in stats
        assert 0 < stats["actual_l0_fraction"] < 1


class TestSharkfinScheduler:
    """Tests for SharkfinScheduler."""
    
    def test_warmup(self):
        """Test warmup phase."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        scheduler = SharkfinScheduler(
            optimizer=optimizer,
            base_lr=0.001,
            total_steps=1000,
            warmup_fraction=0.1,  # 10% warmup = 100 steps
            sparsifier=None,
            use_sharkfin=False,
        )
        
        # At step 0, LR should be 0
        assert scheduler.get_lr() == 0.0
        
        # At step 50, LR should be half of base
        scheduler.current_step = 50
        assert abs(scheduler.get_lr() - 0.0005) < 0.0001
        
        # At step 100, LR should be at base
        scheduler.current_step = 100
        assert abs(scheduler.get_lr() - 0.001) < 0.0001
    
    def test_decay(self):
        """Test decay phase."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        scheduler = SharkfinScheduler(
            optimizer=optimizer,
            base_lr=0.001,
            total_steps=1000,
            warmup_fraction=0.1,
            sparsifier=None,
            use_sharkfin=False,
        )
        
        # After warmup, should decay linearly to 0
        scheduler.current_step = 100  # End of warmup
        assert abs(scheduler.get_lr() - 0.001) < 0.0001
        
        scheduler.current_step = 550  # Halfway through decay
        assert abs(scheduler.get_lr() - 0.0005) < 0.0001
        
        scheduler.current_step = 1000  # End
        assert scheduler.get_lr() == 0.0
    
    def test_sharkfin_integration(self):
        """Test sharkfin multiplier with sparsifier."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        sparsifier = WeightSparsifier(
            model=model,
            target_l0_fraction=0.01,
            anneal_start_fraction=0.0,
            anneal_end_fraction=0.5,
            total_steps=1000,
        )
        
        scheduler = SharkfinScheduler(
            optimizer=optimizer,
            base_lr=0.001,
            total_steps=1000,
            warmup_fraction=0.1,
            sparsifier=sparsifier,
            use_sharkfin=True,
        )
        
        # At step 0, both L0=1 and warmup=0, so LR = 0
        scheduler.current_step = 0
        sparsifier.state.current_step = 0
        assert scheduler.get_lr() == 0.0
        
        # At step 500, L0=0.01, so sharkfin multiplier = 10
        scheduler.current_step = 500
        sparsifier.state.current_step = 500
        lr = scheduler.get_lr()
        # decay_factor at step 500 = 1 - (500-100)/900 ≈ 0.556
        # sharkfin_multiplier = 10
        # expected_lr ≈ 0.001 * 0.556 * 10 ≈ 0.00556
        assert lr > 0.004  # Should be boosted by sharkfin
    
    def test_step_updates_optimizer(self):
        """Test that step() updates optimizer LR."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        scheduler = SharkfinScheduler(
            optimizer=optimizer,
            base_lr=0.001,
            total_steps=100,
            warmup_fraction=0.5,
            sparsifier=None,
            use_sharkfin=False,
        )
        
        # Take multiple steps through warmup
        for _ in range(25):
            scheduler.step()
        
        # Optimizer LR should match what scheduler computes
        # (Note: step() sets LR then increments counter, so get_lr uses next step's value)
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr > 0  # Should be partway through warmup
        assert current_lr < 0.001  # Should not have reached full LR yet


class TestGradientClipping:
    """Tests for gradient clipping."""
    
    def test_clip_grad_rms(self):
        """Test RMS gradient clipping."""
        model = SimpleModel()
        
        # Create large gradients
        x = torch.randn(8, 32) * 100
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Get initial RMS
        total_sq = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)
        total_count = sum(p.grad.numel() for p in model.parameters() if p.grad is not None)
        initial_rms = (total_sq / total_count) ** 0.5
        
        # Clip
        max_rms = 1.0
        returned_rms = clip_grad_rms_(model.parameters(), max_rms=max_rms)
        
        # Check returned value is initial RMS
        assert abs(returned_rms - initial_rms) < 0.01
        
        # Check gradients are clipped
        clipped_sq = sum(p.grad.pow(2).sum().item() for p in model.parameters() if p.grad is not None)
        clipped_rms = (clipped_sq / total_count) ** 0.5
        
        if initial_rms > max_rms:
            assert clipped_rms <= max_rms + 0.01
    
    def test_clip_grad_rms_no_clip_needed(self):
        """Test that small gradients are not clipped."""
        model = SimpleModel()
        
        # Create small gradients
        x = torch.randn(8, 32) * 0.01
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Store original gradients
        original_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        
        # Clip with high threshold
        clip_grad_rms_(model.parameters(), max_rms=100.0)
        
        # Gradients should be unchanged
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, original_grads[n])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

