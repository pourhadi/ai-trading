"""
Alpha (price prediction) model interface.
"""
import os
from collections import deque
import torch
import torch.nn as nn

class LSTMAlphaNet(nn.Module):
    """
    LSTM-based neural network for price movement prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAlphaNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (hn, _) = self.lstm(x)
        # Use the last layer hidden state
        out = hn[-1]
        # out: (batch, hidden_size)
        return self.fc(out).squeeze(-1)

class AlphaModel:
    """
    Price prediction model that loads a trained LSTM PyTorch model.
    """
    FEATURE_NAMES = ['mid_price', 'spread', 'bid_ask_ratio', 'recent_return']

    def __init__(self, config):
        self.config = config
        model_path = config.alpha_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Alpha model not found at {model_path}. "
                "Please train the model using scripts/train_alpha_model.py"
            )
        # Initialize the network
        input_size = len(self.FEATURE_NAMES)
        self.model = LSTMAlphaNet(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers
        )
        # Load trained parameters
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # Buffer to store recent feature vectors
        self.buffer = deque(maxlen=config.sequence_length)

    def predict(self, features):
        """
        Returns a probability of upward price movement.
        Maintains a rolling window of feature vectors.
        """
        # Append current features
        x = [features.get(name, 0.0) for name in self.FEATURE_NAMES]
        self.buffer.append(x)
        # If not enough data, return neutral probability
        if len(self.buffer) < self.config.sequence_length:
            return 0.5
        # Prepare input tensor
        seq = torch.tensor([list(self.buffer)], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(seq)
            proba = torch.sigmoid(logits).item()
        return proba