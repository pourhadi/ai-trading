"""
Position management RL model: decides whether to hold, exit, or add a unit after the first entry.
"""
import numpy as np
from collections import deque
from stable_baselines3 import DQN

from trading_system.alpha_model import AlphaModel

class PositionModel:
    """
    After a position is opened, uses a trained RL agent to decide whether to hold,
    exit the position (reduce by one unit), or add one unit (up to max_positions).
    Actions: 0=EXIT, 1=HOLD, 2=ADD
    """
    def __init__(self, config):
        self.config = config
        self.use_position_rl = getattr(config, 'use_position_rl', False)
        if self.use_position_rl:
            # Feature names must match those used during RL training
            try:
                self.FEATURE_NAMES = AlphaModel.FEATURE_NAMES
            except Exception:
                self.FEATURE_NAMES = []
            # Buffer to store recent feature vectors for RL state
            self.buffer = deque(maxlen=config.position_rl_window_size)
            # Load trained RL model
            self.rl_model = DQN.load(config.position_rl_model_path)

    def decide(self, features):  # noqa: C901
        """
        Decide action based on RL policy: EXIT, HOLD, or ADD.
        Returns a dict with 'action' key.
        """
        # Default to hold if RL not enabled
        if not self.use_position_rl:
            return {'action': 'HOLD', 'confidence': None}
        # Append current feature vector in consistent order
        fv = [features.get(name, 0.0) for name in self.FEATURE_NAMES]
        self.buffer.append(fv)
        # If not enough history, hold
        if len(self.buffer) < self.buffer.maxlen:
            return {'action': 'HOLD', 'confidence': None}
        # Prepare state for RL model
        state = np.array(self.buffer, dtype=np.float32).flatten()
        # Predict action: 0=EXIT, 1=HOLD, 2=ADD
        action, _ = self.rl_model.predict(state, deterministic=True)
        mapping = {0: 'EXIT', 1: 'HOLD', 2: 'ADD'}
        return {'action': mapping.get(int(action), 'HOLD'), 'confidence': None}