"""
Main run loop for the trading system.
"""
import asyncio

from trading_system.config import Config
from trading_system.data_ingestion import DataIngestion
from trading_system.feature_engineering import FeatureEngineer
from trading_system.alpha_model import AlphaModel
from trading_system.decision_model import DecisionModel
from trading_system.execution_module import ExecutionModule
from trading_system.position_model import PositionModel


async def run():
    # Load configuration
    config = Config()

    # Initialize modules
    ingestion = DataIngestion(config)
    features = FeatureEngineer(config)
    alpha_model = AlphaModel(config)
    decision_model = DecisionModel(config)
    execution_module = ExecutionModule(config)
    # Position management RL model (after first entry)
    pos_model = PositionModel(config)

    async def on_tick(tick):
        # Update features and compute
        features.update(tick)
        feat_vec = features.compute_features()
        # Choose decision logic: initial entry vs. position management
        if execution_module.position != 0 and config.use_position_rl:
            # After first entry, use RL-based position management
            decision = pos_model.decide(feat_vec)
        else:
            # Initial trade decision using alpha model and threshold/RL
            prediction = alpha_model.predict(feat_vec)
            decision = decision_model.decide(prediction, feat_vec)
        # Execute orders based on decision
        await execution_module.execute(decision, tick)

    print("Starting trading system...")
    await ingestion.start(on_tick)

if __name__ == '__main__':
    asyncio.run(run())