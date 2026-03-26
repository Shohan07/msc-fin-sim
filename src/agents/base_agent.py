import mesa


class BaseTrader(mesa.Agent):
    def __init__(self, model, cash=10000.0, shares=50):
        super().__init__(model)
        self.cash = cash
        self.shares = shares
        self.action = "HOLD"
        self.quantity = 1

    def decide(self, price, news=None):
        """Override in subclasses to set self.action and self.quantity."""
        self.action = "HOLD"
        self.quantity = 1

    def execute_trade(self, price):
        """Apply self.action / self.quantity against current cash and shares."""
        if self.action == "BUY":
            cost = price * self.quantity
            if self.cash >= cost:
                self.cash -= cost
                self.shares += self.quantity
        elif self.action == "SELL":
            if self.shares >= self.quantity:
                self.cash += price * self.quantity
                self.shares -= self.quantity

    def step(self):
        """Mesa step: subclasses should call decide() then execute_trade()."""
        pass
