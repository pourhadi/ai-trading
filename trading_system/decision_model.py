"""
Decision and risk management model logic.
"""


class DecisionModel:
    """
    Converts prediction probabilities into trade decisions (BUY/SELL/HOLD).
    """
    def __init__(self, config):
        self.config = config
        # Determine whether to use RL-based decision model
        self.use_rl = getattr(config, 'use_rl', False)
        if self.use_rl:
            # RL feature names (must match those used during RL training)
            try:
                from trading_system.alpha_model import AlphaModel
                self.RL_FEATURE_NAMES = AlphaModel.FEATURE_NAMES
            except ImportError:
                self.RL_FEATURE_NAMES = []
            # Buffer to store recent feature vectors for RL state
            from collections import deque
            self.buffer = deque(maxlen=config.rl_window_size)
            # Load trained RL model
            from stable_baselines3 import DQN
            self.rl_model = DQN.load(config.rl_model_path)

    def decide(self, prediction, features):
        """
        Decide whether to buy, sell, or hold based on prediction.
        """
        # Use RL-based decisions if enabled
        if self.use_rl:
            # Append current feature vector in consistent order
            fv = [features.get(name, 0.0) for name in self.RL_FEATURE_NAMES]
            self.buffer.append(fv)
            # If not enough history, hold
            if len(self.buffer) < self.buffer.maxlen:
                return {'action': 'HOLD', 'confidence': None}
            # Prepare state for RL model
            import numpy as np
            state = np.array(self.buffer, dtype=np.float32).flatten()
            # Predict action: 0=SELL, 1=HOLD, 2=BUY
            action, _ = self.rl_model.predict(state, deterministic=True)
            mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            return {'action': mapping.get(int(action), 'HOLD'), 'confidence': None}
        # Default threshold-based decision
        if prediction > self.config.alpha_threshold_up:
            return {'action': 'BUY', 'confidence': prediction}
        elif prediction < self.config.alpha_threshold_down:
            return {'action': 'SELL', 'confidence': 1 - prediction}
        else:
            return {'action': 'HOLD', 'confidence': prediction}