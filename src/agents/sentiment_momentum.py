from src.agents.base_agent import BaseTrader

POS_WORDS = {"rise", "gain", "surge", "beat", "profit", "strong", "growth", "bull", "up"}
NEG_WORDS = {"fall", "drop", "loss", "miss", "weak", "decline", "bear", "down", "cut"}


def _news_tone(news):
    """Return 1 (positive), -1 (negative), or 0 (neutral) based on keyword match."""
    if not news:
        return 0
    words = news.lower().split()
    pos = any(w in POS_WORDS for w in words)
    neg = any(w in NEG_WORDS for w in words)
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return 0


class SentimentMomentumAgent(BaseTrader):
    def __init__(self, model, cash=10000.0, shares=50):
        super().__init__(model, cash=cash, shares=shares)
        self.price_window = []  # last 5 prices

    def decide(self, price, news=None):
        self.price_window.append(price)
        if len(self.price_window) > 5:
            self.price_window.pop(0)

        # Trend: rising if latest price > oldest in window
        if len(self.price_window) >= 2:
            trend = 1 if self.price_window[-1] > self.price_window[0] else -1
        else:
            trend = 0

        tone = _news_tone(news)

        if trend == 1 and tone == 1:
            self.action = "BUY"
            self.quantity = 3
        elif trend == -1 and tone == -1:
            self.action = "SELL"
            self.quantity = 3
        elif trend == 1 and tone != 1:
            self.action = "BUY"
            self.quantity = 1
        elif trend == -1 and tone != -1:
            self.action = "SELL"
            self.quantity = 1
        else:
            self.action = "HOLD"
            self.quantity = 1

    def step(self):
        price = self.model.order_book.current_price
        news = getattr(self.model, "current_news", None)
        self.decide(price, news=news)
        self.execute_trade(price)
