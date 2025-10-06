import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class TradingBot:
    def __init__(self, balance: float, model_path: str = "bot/trading_model.pkl"):
        self.balance = balance
        self.drawdown_cap = 0.06 * balance
        self.risk_per_trade = 0.02 * balance
        self.total_loss = 0

        # Load model safely
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            try:
                self.model = joblib.load(model_path)
            except (EOFError, FileNotFoundError):
                print(f"Warning: Could not load model from {model_path}. Initializing a dummy model.")
                self.model = RandomForestClassifier()
        else:
            print(f"Warning: Model file missing or empty at {model_path}. Initializing a dummy model.")
            self.model = RandomForestClassifier()

    def should_trade(self) -> bool:
        """Check if drawdown cap has been reached."""
        return self.total_loss < self.drawdown_cap

    def execute_trade(self, signal: int) -> float:
        """
        Execute a simulated trade.
        signal: 1 for long, 0 for short
        Returns PnL for this trade
        """
        if not self.should_trade():
            print("Drawdown cap reached. Stop trading.")
            return 0.0

        trade_amount = self.risk_per_trade
        pnl = trade_amount if signal == 1 else -trade_amount
        self.total_loss += max(-pnl, 0)
        self.balance += pnl
        return pnl

    def predict_signal(self, features: np.ndarray) -> int:
        """Predict trading signal based on features."""
        if hasattr(self.model, "predict"):
            return self.model.predict([features])[0]
        else:
            print("Model not trained yet. Defaulting to no trade signal.")
            return 0
