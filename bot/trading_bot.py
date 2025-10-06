import joblib
import numpy as np

class TradingBot:
    def __init__(self, balance: float):
        self.balance = balance
        self.drawdown_cap = 0.06 * balance
        self.risk_per_trade = 0.02 * balance
        self.total_loss = 0
        self.model = joblib.load("bot/trading_model.pkl")
    
    def should_trade(self):
        return self.total_loss < self.drawdown_cap
    
    def execute_trade(self, signal: int):
        if not self.should_trade():
            print("Drawdown cap reached. Stop trading.")
            return 0
        
        # Trade size calculation
        trade_amount = self.risk_per_trade
        # Simulated profit/loss (1 = up, 0 = down)
        pnl = trade_amount if signal == 1 else -trade_amount
        self.total_loss += max(-pnl, 0)
        self.balance += pnl
        return pnl
    
    def predict_signal(self, features: np.ndarray):
        return self.model.predict([features])[0]
