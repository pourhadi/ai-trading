"""
Gym-compatible trading environment for discrete-action RL.

Observation (state) consists of a sliding window of market features:
 - Normalized mid-price returns
 - Normalized spread
 - Log bid-ask ratio
 Plus current position, cash balance, and unrealized PnL normalized.

Actions: Discrete {0: SELL (-1), 1: HOLD (0), 2: BUY (+1)}.

Reward: Δ Portfolio Value - fee * |ΔPosition| - risk_lambda * position^2.
"""
"""Gym-compatible trading environment stub for discrete-action RL."""
import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=50, fee=0.0001, risk_lambda=0.0):
        # Basic initialization
        self.window_size = window_size
        self.length = len(df)
        self.fee = fee
        self.risk_lambda = risk_lambda
        self.current_step = window_size
        self.position = 0
        self.cash = 0.0
        self.prev_price = 0.0
        self.done = False
        # Define action and observation spaces
        obs_dim = self.window_size * 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self):
        # Reset to initial state
        self.current_step = self.window_size
        self.position = 0
        self.cash = 0.0
        self.prev_price = 0.0
        self.done = False
        # Return zeroed observation
        import numpy as np
        obs_shape = self.window_size * 3 + 3
        return np.zeros(obs_shape, dtype=np.float32)

    def step(self, action):
        # Ensure valid sequence
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        # Map action to position change
        if action == 0:
            delta_pos = -1
        elif action == 2:
            delta_pos = 1
        else:
            delta_pos = 0
        # Update position
        self.position += delta_pos
        # Stub reward always zero
        reward = 0.0
        # Advance time step
        self.current_step += 1
        if self.current_step >= self.length:
            self.done = True
        # Return zeroed observation
        import numpy as np
        obs_shape = self.window_size * 3 + 3
        obs = np.zeros(obs_shape, dtype=np.float32)
        return obs, reward, self.done, {}

    def render(self, mode='human'):
        # Simple render stub
        print(f"Step: {self.current_step}, Position: {self.position}, Cash: {self.cash:.2f}")

    def close(self):
        pass