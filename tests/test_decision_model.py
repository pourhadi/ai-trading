import pytest
from trading_system.decision_model import DecisionModel

class DummyConfig:
    alpha_threshold_up = 0.6
    alpha_threshold_down = 0.4

def test_decide_buy():
    dm = DecisionModel(DummyConfig())
    result = dm.decide(0.7, {})
    assert result['action'] == 'BUY'
    assert result['confidence'] == 0.7

def test_decide_sell():
    dm = DecisionModel(DummyConfig())
    result = dm.decide(0.3, {})
    assert result['action'] == 'SELL'
    assert result['confidence'] == pytest.approx(1 - 0.3)

def test_decide_hold():
    dm = DecisionModel(DummyConfig())
    result = dm.decide(0.5, {})
    assert result['action'] == 'HOLD'
    assert result['confidence'] == 0.5
