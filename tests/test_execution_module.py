import pytest
import asyncio
from trading_system.execution_module import ExecutionModule

class DummyConfig:
    prediction_horizon = 10.0
    profit_target = 1.0
    stop_loss = 0.5
    max_positions = 2
    use_position_rl = False

def make_tick(price, timestamp):
    return {'last_price': price, 'timestamp': timestamp}

@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

def test_long_entry_and_profit_exit(loop, capsys):
    cfg = DummyConfig()
    exec_mod = ExecutionModule(cfg)
    tick1 = make_tick(price=100.0, timestamp=0.0)
    loop.run_until_complete(exec_mod.execute({'action': 'BUY'}, tick1))
    assert exec_mod.position == 1
    assert exec_mod.entry_price == 100.0
    assert exec_mod.entry_time == 0.0
    captured = capsys.readouterr()
    assert 'Entered LONG unit at 100.00' in captured.out

    tick2 = make_tick(price=101.0, timestamp=1.0)
    loop.run_until_complete(exec_mod.execute({'action': 'HOLD'}, tick2))
    assert exec_mod.position == 0
    captured = capsys.readouterr()
    assert 'Exited LONG position of 1 units for profit at 101.00' in captured.out

def test_long_addition_and_max_positions(loop):
    cfg = DummyConfig()
    exec_mod = ExecutionModule(cfg)
    loop.run_until_complete(exec_mod.execute({'action': 'BUY'}, make_tick(100.0, 0.0)))
    loop.run_until_complete(exec_mod.execute({'action': 'BUY'}, make_tick(102.0, 1.0)))
    assert exec_mod.position == 2
    assert exec_mod.entry_price == pytest.approx((100.0 + 102.0) / 2)
    loop.run_until_complete(exec_mod.execute({'action': 'BUY'}, make_tick(104.0, 2.0)))
    assert exec_mod.position == 2

def test_short_entry_and_time_exit(loop, capsys):
    cfg = DummyConfig()
    exec_mod = ExecutionModule(cfg)
    loop.run_until_complete(exec_mod.execute({'action': 'SELL'}, make_tick(200.0, 0.0)))
    assert exec_mod.position == -1

    tick2 = make_tick(price=199.0, timestamp=cfg.prediction_horizon + 1)
    loop.run_until_complete(exec_mod.execute({'action': 'HOLD'}, tick2))
    assert exec_mod.position == 0
    captured = capsys.readouterr()
    assert 'Exited SHORT position of 1 units on time at 199.00' in captured.out
