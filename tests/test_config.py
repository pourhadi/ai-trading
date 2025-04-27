import pytest
from trading_system.config import Config

def test_config_defaults():
    cfg = Config()
    assert isinstance(cfg.prediction_horizon, (int, float))
    assert cfg.prediction_horizon == 20
    assert cfg.alpha_threshold_up == 0.6
    assert cfg.alpha_threshold_down == 0.4
    assert cfg.profit_target == 0.5
    assert cfg.stop_loss == 0.25
