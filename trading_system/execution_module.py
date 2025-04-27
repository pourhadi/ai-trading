"""
Execution module: handles order placement and exit strategy.
"""
import asyncio


class ExecutionModule:
    """
    Executes trades and manages open positions.
    """
    def __init__(self, config):
        self.config = config
        # Position units: positive for long, negative for short, 0 for no position
        self.position = 0
        self.entry_price = None
        self.entry_time = None
        # Flag for RL-based position management after first entry
        self.management_mode = False

    async def execute(self, decision, tick):
        action = decision.get('action')
        curr_time = tick['timestamp']
        price = tick.get('last_price')
        # RL-based position management: override entry/exit/add logic
        if self.config.use_position_rl and self.management_mode:
            # Explicit exit decision
            if action == 'EXIT' and self.position != 0:
                units = abs(self.position)
                if self.position > 0:
                    print(f"Exited LONG position of {units} units via RL decision at {price:.2f}")
                else:
                    print(f"Exited SHORT position of {units} units via RL decision at {price:.2f}")
                self.position = 0
                self.entry_price = None
                self.entry_time = None
                self.management_mode = False
                return
            # Explicit add decision
            elif action == 'ADD':
                if self.position > 0:
                    prev_units = self.position
                    if prev_units < self.config.max_positions and curr_time < self.entry_time + self.config.prediction_horizon:
                        new_units = prev_units + 1
                        self.entry_price = (self.entry_price * prev_units + price) / new_units
                        self.position = new_units
                        print(f"Added LONG unit at {price:.2f}, new avg entry price {self.entry_price:.2f} via RL decision")
                elif self.position < 0:
                    prev_units = abs(self.position)
                    if prev_units < self.config.max_positions and curr_time < self.entry_time + self.config.prediction_horizon:
                        new_units = prev_units + 1
                        self.entry_price = (self.entry_price * prev_units + price) / new_units
                        self.position = -new_units
                        print(f"Added SHORT unit at {price:.2f}, new avg entry price {self.entry_price:.2f} via RL decision")
                return
            # Explicit hold: do nothing
            return

        # Entry logic: allow up to config.max_positions units to average entry price
        if action == 'BUY' and self.position >= 0:
            prev_units = self.position
            # First entry
            if prev_units == 0:
                self.position = 1
                self.entry_price = price
                self.entry_time = curr_time
                print(f"Entered LONG unit at {price:.2f} (t={curr_time:.2f})")
                # Activate RL-based position management if enabled
                if self.config.use_position_rl:
                    self.management_mode = True
            # Add additional unit within prediction horizon
            elif prev_units < self.config.max_positions and curr_time < self.entry_time + self.config.prediction_horizon:
                new_units = prev_units + 1
                self.entry_price = (self.entry_price * prev_units + price) / new_units
                self.position = new_units
                print(f"Added LONG unit at {price:.2f}, new avg entry price {self.entry_price:.2f}")
        elif action == 'SELL' and self.position <= 0:
            prev_units = abs(self.position)
            # First entry
            if prev_units == 0:
                self.position = -1
                self.entry_price = price
                self.entry_time = curr_time
                print(f"Entered SHORT unit at {price:.2f} (t={curr_time:.2f})")
                # Activate RL-based position management if enabled
                if self.config.use_position_rl:
                    self.management_mode = True
            # Add additional unit within prediction horizon
            elif prev_units < self.config.max_positions and curr_time < self.entry_time + self.config.prediction_horizon:
                new_units = prev_units + 1
                self.entry_price = (self.entry_price * prev_units + price) / new_units
                self.position = -new_units
                print(f"Added SHORT unit at {price:.2f}, new avg entry price {self.entry_price:.2f}")

        # Exit logic: only evaluate exits on HOLD actions
        if action == 'HOLD':
            # Long position exit: check time expiration first, then profit target
            if self.position > 0:
                units = self.position
                # Time expiration
                if curr_time >= self.entry_time + self.config.prediction_horizon:
                    print(f"Exited LONG position of {units} units on time at {price:.2f}")
                    self.position = 0
                    self.entry_price = None
                    self.entry_time = None
                    self.management_mode = False
                # Profit target
                elif price >= self.entry_price + self.config.profit_target:
                    print(f"Exited LONG position of {units} units for profit at {price:.2f}")
                    self.position = 0
                    self.entry_price = None
                    self.entry_time = None
                    self.management_mode = False
            # Short position exit: check time expiration first, then profit target
            elif self.position < 0:
                units = abs(self.position)
                # Time expiration
                if curr_time >= self.entry_time + self.config.prediction_horizon:
                    print(f"Exited SHORT position of {units} units on time at {price:.2f}")
                    self.position = 0
                    self.entry_price = None
                    self.entry_time = None
                    self.management_mode = False
                # Profit target for short
                elif price <= self.entry_price - self.config.profit_target:
                    print(f"Exited SHORT position of {units} units for profit at {price:.2f}")
                    self.position = 0
                    self.entry_price = None
                    self.entry_time = None
                    self.management_mode = False