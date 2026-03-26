import os

import mesa
import numpy as np
from dotenv import load_dotenv

from src.agents.fundamentalist import FundamentalistAgent
from src.agents.noise_trader import NoiseTrader
from src.agents.rag_agent import RAGAgent
from src.agents.sentiment_momentum import SentimentMomentumAgent
from src.market.order_book import OrderBook


class MarketSimulation(mesa.Model):
    def __init__(
        self,
        n_fundamentalist=10,
        n_sentiment=10,
        n_noise=10,
        n_rag=0,
        use_random_news=False,
        seed=42,
    ):
        super().__init__(seed=seed)

        self.order_book = OrderBook(initial_price=100.0)

        # Build shared RAG components only when needed
        retriever = None
        llm_agent = None
        if n_rag > 0:
            load_dotenv(".env")
            if use_random_news:
                from src.rag.random_store import RandomNewsStore
                retriever = RandomNewsStore()
            else:
                from src.rag.retriever import NewsRetriever
                retriever = NewsRetriever()
            from src.rag.llm_agent import GeminiTrader
            llm_agent = GeminiTrader()

        for _ in range(n_fundamentalist):
            FundamentalistAgent(self)

        for _ in range(n_sentiment):
            SentimentMomentumAgent(self)

        for _ in range(n_noise):
            NoiseTrader(self)

        for _ in range(n_rag):
            RAGAgent(self, retriever=retriever, llm_agent=llm_agent)

    def step(self):
        """
        Two-phase market step:
          1. All agents decide (shuffled) — sets agent.action / agent.quantity
          2. Collect actions → update price via order book
          3. All agents execute trades at the new price
        """
        price = self.order_book.current_price
        price_history = self.order_book.price_history

        # Phase 1: decisions (shuffled for fairness)
        agent_list = list(self.agents)
        self.random.shuffle(agent_list)
        for agent in agent_list:
            if isinstance(agent, RAGAgent):
                agent.decide(price, price_history)
            elif isinstance(agent, SentimentMomentumAgent):
                agent.decide(price, news=None)
            else:
                agent.decide(price)

        # Phase 2: price formation from aggregate demand
        agent_actions = [
            {"action": a.action, "quantity": a.quantity} for a in self.agents
        ]
        new_price = self.order_book.process_orders(agent_actions)

        # Phase 3: execute trades at the cleared price
        for agent in self.agents:
            agent.execute_trade(new_price)

        # Increment Mesa step counter
        super().step()

    def get_results(self):
        prices = self.order_book.price_history
        returns = list(np.diff(np.log(prices))) if len(prices) > 1 else []

        decision_logs = []
        for agent in self.agents:
            if isinstance(agent, RAGAgent) and agent.decision_log:
                decision_logs.extend(agent.decision_log)

        return {
            "price_history": prices,
            "returns": returns,
            "decision_logs": decision_logs,
        }
