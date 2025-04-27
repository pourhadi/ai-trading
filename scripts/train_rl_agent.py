#!/usr/bin/env python3
"""
Train a DQN agent on historical tick data for trading.
"""
import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_system.config import Config
from trading_system.rl_env import TradingEnv


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agent using DQN.")
    parser.add_argument('--data-path', type=str, required=True,
                        help="Path to historical tick data CSV file.")
    parser.add_argument('--model-output-path', type=str, default=None,
                        help="Output path for the trained RL model (zip format).")
    parser.add_argument('--total-timesteps', type=int, default=100000,
                        help="Total timesteps for training the RL agent.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="Enable verbose logging")
    args = parser.parse_args()

    # Load config
    config = Config()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.debug(f"Data path: {args.data_path}, Model output path: {args.model_output_path or config.rl_model_path}, Total timesteps: {args.total_timesteps}")
    # Override RL params if needed
    model_path = args.model_output_path or config.rl_model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Load historical data
    df = pd.read_csv(args.data_path)

    # Initialize environment
    env = TradingEnv(
        df,
        window_size=config.rl_window_size,
        fee=config.rl_fee,
        risk_lambda=config.rl_risk_lambda
    )
    # Use vectorized env for SB3
    vec_env = DummyVecEnv([lambda: env])

    # Initialize DQN agent
    model = DQN(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=32,
        target_update_interval=1000,
        train_freq=4,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1
    )

    # Train the agent
    logging.info(f"Training DQN agent for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)

    # Save the trained model
    model.save(model_path)
    logging.info(f"RL model saved to {model_path}")


if __name__ == '__main__':
    main()