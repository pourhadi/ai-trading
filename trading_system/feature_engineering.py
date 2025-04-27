"""
Feature engineering: computes input features for prediction and decision models.
"""
from collections import deque


class FeatureEngineer:
    """
    Maintains a history of market data and computes features.
    """
    def __init__(self, config):
        # Size of history window (based on horizon and feed interval)
        maxlen = int(config.prediction_horizon / config.data_feed_interval) * 2
        self.order_book_history = deque(maxlen=maxlen)
        self.trade_history = deque(maxlen=maxlen)

    def update(self, tick):
        # Append the latest market tick
        self.order_book_history.append(tick)
        self.trade_history.append(tick)

    def compute_features(self):
        """
        Returns a feature dictionary for the current state.
        """
        if not self.order_book_history:
            return {}
        latest = self.order_book_history[-1]
        bid = latest['best_bid']
        ask = latest['best_ask']
        mid_price = (bid + ask) / 2
        spread = ask - bid
        # Compute a simple return over the window
        if len(self.trade_history) > 1:
            start = self.trade_history[0]['last_price']
            end = self.trade_history[-1]['last_price']
            recent_return = (end - start) / start if start else 0.0
        else:
            recent_return = 0.0
        return {
            'mid_price': mid_price,
            'spread': spread,
            'bid_ask_ratio': latest['bid_size'] / (latest['ask_size'] + 1e-6),
            'recent_return': recent_return,
        }