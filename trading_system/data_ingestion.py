"""
Data ingestion module: simulates or connects to a real-time market data feed.
"""
import asyncio
import time
import random


class DataIngestion:
    """
    Simulated data feed for E-mini S&P 500 futures.
    Calls a provided callback with each new tick data.
    """
    def __init__(self, config):
        self.interval = config.data_feed_interval

    async def start(self, callback):
        while True:
            tick = {
                'timestamp': time.time(),
                'best_bid': random.uniform(4400, 4500),
                'best_ask': random.uniform(4400, 4500),
                'bid_size': random.randint(1, 20),
                'ask_size': random.randint(1, 20),
                'last_price': random.uniform(4400, 4500),
                'last_size': random.randint(1, 20),
            }
            await callback(tick)
            await asyncio.sleep(self.interval)