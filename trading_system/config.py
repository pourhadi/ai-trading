"""
Configuration parameters for the trading system.
"""

class Config:
    def __init__(self):
        # Prediction horizon (seconds)
        self.prediction_horizon = 20
        # Thresholds for trade decision
        self.alpha_threshold_up = 0.6
        self.alpha_threshold_down = 0.4
        # Profit target and stop-loss levels (price points)
        self.profit_target = 0.5
        self.stop_loss = 0.25
        # Data feed polling interval (seconds)
        self.data_feed_interval = 1.0
        # Maximum concurrent positions (allow averaging up to two units)
        self.max_positions = 2
        # Path to the trained alpha model file (PyTorch .pth format)
        self.alpha_model_path = 'models/alpha_model.pth'
        # LSTM model hyperparameters
        # Sequence length for time-series input
        self.sequence_length = 50
        # Hidden size of the LSTM
        self.hidden_size = 64
        # Number of LSTM layers
        self.num_layers = 2
        # Training parameters
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.num_epochs = 10
        # Reinforcement Learning parameters
        # Enable RL-based decision model instead of threshold-based model
        self.use_rl = False
        # Window size for RL state (number of timesteps)
        self.rl_window_size = self.sequence_length
        # Transaction cost penalty used in RL reward shaping (per unit executed)
        self.rl_fee = 0.0001
        # Risk penalty coefficient used in RL reward shaping
        self.rl_risk_lambda = 0.0
        # Path to the trained RL model file (Stable-Baselines3 .zip format)
        self.rl_model_path = 'models/rl_model.zip'
        # Position management RL parameters
        # Enable RL-based position management after first entry (hold/add/exit)
        self.use_position_rl = False
        # Window size (number of feature vectors) for position RL state
        self.position_rl_window_size = self.sequence_length
        # Path to the trained position management RL model file (.zip format)
        self.position_rl_model_path = 'models/position_rl_model.zip'