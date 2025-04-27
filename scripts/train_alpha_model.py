#!/usr/bin/env python3
"""
Train alpha (price prediction) model using LSTM-based neural network.
"""

import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from trading_system.config import Config
from trading_system.feature_engineering import FeatureEngineer
from trading_system.alpha_model import LSTMAlphaNet, AlphaModel

def main():
    parser = argparse.ArgumentParser(description="Train alpha LSTM model.")
    parser.add_argument('--data-path', type=str, required=True,
                        help="Path to historical tick data CSV file.")
    parser.add_argument('--model-output-path', type=str, default=None,
                        help="Output path for the trained model file.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="Enable verbose logging")
    args = parser.parse_args()

    config = Config()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.debug(f"Data path: {args.data_path}, Model output path: {args.model_output_path or config.alpha_model_path}")

    model_path = args.model_output_path or config.alpha_model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load and sort historical data
    df = pd.read_csv(args.data_path)
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Compute features for each tick
    fe = FeatureEngineer(config)
    features_list = []
    for _, row in df.iterrows():
        tick = row.to_dict()
        fe.update(tick)
        features_list.append(fe.compute_features())

    # Determine shift based on prediction horizon
    n_shift = int(config.prediction_horizon / config.data_feed_interval)
    seq_len = config.sequence_length

    # Build sequences and labels
    X_seq, y_seq = [], []
    for t in range(seq_len - 1, len(df) - n_shift):
        seq = [
            [features_list[j].get(name, 0.0) for name in AlphaModel.FEATURE_NAMES]
            for j in range(t - seq_len + 1, t + 1)
        ]
        current_price = df.loc[t, 'last_price']
        future_price = df.loc[t + n_shift, 'last_price']
        label = 1.0 if future_price > current_price else 0.0
        X_seq.append(seq)
        y_seq.append(label)

    # Convert to tensors and create DataLoader
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    input_size = len(AlphaModel.FEATURE_NAMES)
    model = LSTMAlphaNet(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    logging.info("Training LSTM model...")
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(loader.dataset)
        logging.info(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()