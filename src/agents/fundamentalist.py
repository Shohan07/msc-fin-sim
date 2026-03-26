import numpy as np

from src.agents.base_agent import BaseTrader


class FundamentalistAgent(BaseTrader):
    def __init__(self, model, cash=10000.0, shares=50):
        super().__init__(model, cash=cash, shares=shares)
        self.fundamental = np.random.normal(100, 5)

    def decide(self, price, news=None):
        deviation = (self.fundamental - price) / price
        if deviation > 0.02 and self.cash >= price:
            self.action = "BUY"
        elif deviation < -0.02 and self.shares > 0:
            self.action = "SELL"
        else:
            self.action = "HOLD"
        self.quantity = 1

    def step(self):
        price = self.model.order_book.current_price
        self.decide(price)
        self.execute_trade(price)
