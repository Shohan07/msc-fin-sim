import random

from src.agents.base_agent import BaseTrader


class NoiseTrader(BaseTrader):
    def __init__(self, model, cash=10000.0, shares=50):
        super().__init__(model, cash=cash, shares=shares)

    def decide(self, price, news=None):
        self.action = random.choice(["BUY", "SELL", "HOLD"])
        self.quantity = 1

    def step(self):
        price = self.model.order_book.current_price
        noise = random.uniform(-0.005, 0.005)
        price_with_noise = price * (1 + noise)
        self.decide(price_with_noise)
        self.execute_trade(price_with_noise)
