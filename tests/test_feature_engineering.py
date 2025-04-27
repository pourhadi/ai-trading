import pytest
from trading_system.feature_engineering import FeatureEngineer

class DummyConfig:
    prediction_horizon = 10
    data_feed_interval = 1.0

def make_tick(best_bid=100, best_ask=102, bid_size=5, ask_size=3, last_price=101):
    return {
        'best_bid': best_bid,
        'best_ask': best_ask,
        'bid_size': bid_size,
        'ask_size': ask_size,
        'last_price': last_price
    }

def test_compute_features_empty():
    fe = FeatureEngineer(DummyConfig())
    assert fe.compute_features() == {}

def test_compute_features_single_tick():
    fe = FeatureEngineer(DummyConfig())
    tick = make_tick()
    fe.update(tick)
    f = fe.compute_features()
    assert f['mid_price'] == pytest.approx((100 + 102) / 2)
    assert f['spread'] == pytest.approx(2)
    assert f['bid_ask_ratio'] == pytest.approx(5 / 3)
    assert f['recent_return'] == 0.0

def test_compute_features_multiple_ticks():
    fe = FeatureEngineer(DummyConfig())
    tick1 = make_tick(last_price=100)
    tick2 = make_tick(last_price=110)
    fe.update(tick1)
    fe.update(tick2)
    f = fe.compute_features()
    assert f['recent_return'] == pytest.approx((110 - 100) / 100)
