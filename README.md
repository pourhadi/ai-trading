# Automated Trading System for E-mini S&P 500 Futures

This repository provides a skeleton implementation of an automated trading system for E-mini S&P 500 futures, driven by machine learning components and following best-practice design recommendations.

## Project Structure
- `.gitignore`: Ignored files and directories
- `requirements.txt`: Python dependencies
- `run.py`: Entry point to start the trading system
- `trading_system/`
  - `config.py`: Configuration parameters (horizon, thresholds, targets, intervals)
  - `data_ingestion.py`: Simulated or real-time data feed ingestion module
  - `feature_engineering.py`: Feature computation for model inputs
  - `alpha_model.py`: Price prediction (alpha) model interface
  - `decision_model.py`: Trade decision and risk management logic
  - `execution_module.py`: Order execution and exit handling
  - `run.py`: Orchestrates the asynchronous trading loop

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the trading system:
   ```bash
   python run.py
   ```

## Next Steps
- Integrate a real market data feed (e.g., ZeroMQ, Kafka)
- Load and deploy trained ML models (PyTorch, XGBoost)
- Connect to a brokerage API for live or paper trading
- Implement logging, monitoring, backtesting, and error handling

For detailed design rationale and component descriptions, see `instructions.txt`.
## Training the Alpha Model

Before starting the trading system, train the alpha (price prediction) model using historical data:

```bash
python scripts/train_alpha_model.py --data-path path/to/historical_data.csv
```

This will save the trained PyTorch LSTM model to `models/alpha_model.pth`. The trading system will automatically load this model on startup.
## Training the RL Agent

To replace the neural-network alpha model with a reinforcement learning (RL) agent using DQN:

1. Train the RL agent on historical tick data:
   ```bash
   python scripts/train_rl_agent.py --data-path path/to/historical_data.csv \
       --total-timesteps 200000
   ```
   This will save the RL model to `models/rl_model.zip` by default.

2. Enable RL-based decision making in the configuration:
   - Open `trading_system/config.py` and set `use_rl = True`.
   - (Optional) Adjust `rl_window_size`, `rl_fee`, `rl_risk_lambda`, and `rl_model_path` as needed.

3. Run the trading system with the RL agent:
   ```bash
   python run.py
   ```

## Training the Position Management RL Agent

If you want to enable RL-based position management after the first entry:

1. Train the position RL agent:
   ```bash
   python scripts/train_position_rl_agent.py --data-path path/to/historical_data.csv \\
       --total-timesteps 100000
   ```
   This will save the RL model to `models/position_rl_model.zip` by default.

2. Enable RL-based position management in the configuration:
   - Open `trading_system/config.py` and set `use_position_rl = True`.
   - (Optional) Adjust `position_rl_window_size`, `rl_fee`, `rl_risk_lambda`, and `position_rl_model_path` as needed.

3. Run the trading system with the position RL agent:
   ```bash
   python run.py
   ```