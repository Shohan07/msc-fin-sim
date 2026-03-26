import random


class OrderBook:
    def __init__(self, initial_price=100.0, price_sensitivity=0.05, noise_std=0.002):
        self.current_price = initial_price
        self.price_sensitivity = price_sensitivity
        self.noise_std = noise_std
        self.price_history = [initial_price]

    def process_orders(self, agent_actions):
        """
        agent_actions: list of dicts with keys 'action' (BUY/SELL/HOLD) and 'quantity' (int).
        Returns the new price after applying the price-impact model.
        """
        total_agents = len(agent_actions)
        if total_agents == 0:
            self.price_history.append(self.current_price)
            return self.current_price

        buy_volume = sum(a["quantity"] for a in agent_actions if a["action"] == "BUY")
        sell_volume = sum(a["quantity"] for a in agent_actions if a["action"] == "SELL")
        net_demand = buy_volume - sell_volume

        price_impact = (net_demand / total_agents) * self.price_sensitivity
        noise = random.gauss(0, self.noise_std)
        self.current_price = max(0.01, self.current_price * (1 + price_impact + noise))
        self.price_history.append(self.current_price)
        return self.current_price
