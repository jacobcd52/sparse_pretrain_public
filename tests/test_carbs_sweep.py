"""
Tests for CARBS hyperparameter sweep functionality.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from carbs import CARBS, CARBSParams, Param, LogSpace, LogitSpace, LinearSpace, ObservationInParam


class TestCarbsIntegration:
    """Test CARBS integration."""
    
    def test_carbs_import(self):
        """Test that CARBS can be imported."""
        from carbs import CARBS, CARBSParams
        assert CARBS is not None
        assert CARBSParams is not None
    
    def test_param_spaces_defined(self):
        """Test that our parameter spaces are valid."""
        from my_sparse_pretrain.scripts.run_carbs_sweep import get_carbs_param_spaces
        
        param_spaces = get_carbs_param_spaces()
        assert len(param_spaces) == 8
        
        # Check names
        names = {p.name for p in param_spaces}
        expected_names = {
            "k_coef", "init_noise_scale", "init_noise_bias", "weight_decay",
            "lr", "beta2", "lr_warmup_frac", "heaviside_temp"
        }
        assert names == expected_names
    
    def test_carbs_initialization(self):
        """Test CARBS can be initialized with our param spaces."""
        from my_sparse_pretrain.scripts.run_carbs_sweep import get_carbs_param_spaces
        
        param_spaces = get_carbs_param_spaces()
        
        carbs_params = CARBSParams(
            better_direction_sign=-1,
            is_wandb_logging_enabled=False,
            resample_frequency=0,
            num_random_samples=2,
        )
        
        carbs = CARBS(carbs_params, param_spaces)
        assert carbs is not None
    
    def test_carbs_suggest_observe_cycle(self):
        """Test CARBS suggest/observe cycle."""
        from my_sparse_pretrain.scripts.run_carbs_sweep import get_carbs_param_spaces
        
        param_spaces = get_carbs_param_spaces()
        
        carbs_params = CARBSParams(
            better_direction_sign=-1,
            is_wandb_logging_enabled=False,
            resample_frequency=0,
            num_random_samples=2,
        )
        
        carbs = CARBS(carbs_params, param_spaces)
        
        # Get suggestion
        suggestion_out = carbs.suggest()
        suggestion = suggestion_out.suggestion
        
        # Check all params are present
        assert "k_coef" in suggestion
        assert "lr" in suggestion
        assert "beta2" in suggestion
        
        # Check values are reasonable
        assert 1e-8 < suggestion["k_coef"] < 1e-1
        assert 1e-5 < suggestion["lr"] < 1e-1
        assert 0 < suggestion["beta2"] < 1
        
        # Observe result
        obs_out = carbs.observe(ObservationInParam(
            input=suggestion,
            output=1000.0,  # Simulated circuit size
            cost=1.0,
        ))
        
        assert obs_out is not None
    
    def test_sweep_config(self):
        """Test SweepConfig dataclass."""
        from my_sparse_pretrain.scripts.run_carbs_sweep import SweepConfig
        
        config = SweepConfig()
        assert config.task_name == "dummy_pronoun"
        assert config.carbs_iterations == 32
        assert config.parallel_suggestions == 8
        assert config.num_steps == 2000
        assert config.target_loss == 0.15
        
        # Test to_dict
        d = config.to_dict()
        assert d["task_name"] == "dummy_pronoun"
        assert d["carbs_iterations"] == 32


class TestPaperDefaults:
    """Test that default values match paper's Table 2."""
    
    def test_table2_centers(self):
        """Test search centers match Table 2 from Appendix A.5."""
        from my_sparse_pretrain.scripts.run_carbs_sweep import get_carbs_param_spaces
        
        param_spaces = get_carbs_param_spaces()
        centers = {p.name: p.search_center for p in param_spaces}
        
        # Table 2 values
        assert centers["k_coef"] == pytest.approx(1e-4, rel=1e-6)
        assert centers["init_noise_scale"] == pytest.approx(1e-2, rel=1e-6)
        assert centers["init_noise_bias"] == pytest.approx(1e-1, rel=1e-6)
        assert centers["weight_decay"] == pytest.approx(1e-3, rel=1e-6)
        assert centers["lr"] == pytest.approx(3e-3, rel=1e-6)
        assert centers["beta2"] == pytest.approx(0.95, rel=1e-6)  # inv_beta2 = 0.05
        assert centers["lr_warmup_frac"] == pytest.approx(0.05, rel=1e-6)
        assert centers["heaviside_temp"] == pytest.approx(1.0, rel=1e-6)
    
    def test_pruning_config_defaults(self):
        """Test PruningConfig has paper defaults."""
        from my_sparse_pretrain.src.pruning.config import PruningConfig
        
        config = PruningConfig()
        
        # Table 2 defaults
        assert config.k_coef == pytest.approx(1e-4, rel=1e-6)
        assert config.init_noise_scale == pytest.approx(1e-2, rel=1e-6)
        assert config.init_noise_bias == pytest.approx(1e-1, rel=1e-6)
        assert config.weight_decay == pytest.approx(1e-3, rel=1e-6)
        assert config.lr == pytest.approx(3e-3, rel=1e-6)
        assert config.beta2 == pytest.approx(0.95, rel=1e-6)
        assert config.heaviside_temp == pytest.approx(1.0, rel=1e-6)
        
        # Paper defaults
        assert config.target_loss == 0.15
        assert config.batch_size == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

