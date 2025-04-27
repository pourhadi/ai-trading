"""
Entry point for the automated trading system.
"""
import asyncio
from trading_system.run import run

if __name__ == '__main__':
    # Launch the trading system event loop
    asyncio.run(run())