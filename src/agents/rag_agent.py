from src.agents.base_agent import BaseTrader


class RAGAgent(BaseTrader):
    def __init__(self, model, retriever, llm_agent, cash=10000.0, shares=50):
        super().__init__(model, cash=cash, shares=shares)
        self.retriever = retriever
        self.llm_agent = llm_agent
        self.decision_log = []

    def decide(self, current_price, price_history, news=None):
        try:
            trend = "rising" if len(price_history) >= 2 and price_history[-1] > price_history[-2] else "falling"
            query = f"stock market price {current_price:.2f} trend {trend}"

            news_items = self.retriever.retrieve(query, k=3)

            result = self.llm_agent.decide(news_items, current_price, price_history)

            action = result.get("action", "HOLD")
            quantity = int(result.get("quantity", 1))
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")

            if confidence < 0.5:
                action = "HOLD"

            self.action = action
            self.quantity = quantity

            self.decision_log.append({
                "step": self.model.steps,
                "action": action,
                "quantity": quantity,
                "confidence": confidence,
                "reasoning": reasoning,
                "news_used": news_items,
            })

        except Exception as e:
            print(f"RAGAgent error (agent {self.unique_id}): {e}")
            self.action = "HOLD"
            self.quantity = 1

    def step(self):
        price = self.model.order_book.current_price
        price_history = self.model.order_book.price_history
        self.decide(price, price_history)
        self.execute_trade(price)
