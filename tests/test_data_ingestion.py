import pytest
import asyncio
from trading_system.data_ingestion import DataIngestion

class DummyConfig:
    data_feed_interval = 0.0

def test_data_ingestion_ticks():
    di = DataIngestion(DummyConfig())
    ticks = []
    class StopException(Exception):
        pass

    async def cb(tick):
        ticks.append(tick)
        if len(ticks) >= 3:
            raise StopException()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with pytest.raises(StopException):
        loop.run_until_complete(di.start(cb))
    assert len(ticks) == 3
    for tick in ticks:
        assert 'timestamp' in tick and isinstance(tick['timestamp'], float)
        assert 'best_bid' in tick
        assert 'best_ask' in tick
        assert 'bid_size' in tick and isinstance(tick['bid_size'], int)
        assert 'ask_size' in tick and isinstance(tick['ask_size'], int)
        assert 'last_price' in tick
        assert 'last_size' in tick
