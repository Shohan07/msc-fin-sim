import hashlib
import json
import os
import pickle
import time

from dotenv import load_dotenv
from google import genai

load_dotenv(".env")

CACHE_FILE = "data/embeddings/llm_cache.pkl"
MODEL = "gemini-2.5-flash-lite"
PROMPT_TEMPLATE_PATH = "src/rag/prompt_template.txt"
FALLBACK = {"action": "HOLD", "quantity": 1, "confidence": 0.0, "reasoning": "fallback"}

with open(PROMPT_TEMPLATE_PATH, "r") as f:
    _PROMPT_TEMPLATE = f.read()


def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def _cache_key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


class GeminiTrader:
    def __init__(self, sleep_time=0.5):
        self.sleep_time = sleep_time
        self.client = genai.Client()
        self.cache = _load_cache()

    def decide(self, news_items, current_price, price_history):
        """
        Call Gemini to produce a trading decision.
        Returns dict with action, quantity, confidence, reasoning.
        Checks cache first; saves cache after every new API call.
        Overrides action to HOLD if confidence < 0.5.
        """
        news_block = "\n".join(f"- {n}" for n in news_items)
        trend = [round(p, 2) for p in price_history[-5:]]
        prompt = (
            _PROMPT_TEMPLATE
            .replace("{current_price}", str(round(current_price, 2)))
            .replace("{price_history}", str(trend))
            .replace("{news_items}", news_block)
        )

        key = _cache_key(prompt)
        if key in self.cache:
            return self.cache[key]

        result = self._call_gemini(prompt)

        if result["confidence"] < 0.5:
            result["action"] = "HOLD"

        self.cache[key] = result
        _save_cache(self.cache)
        return result

    def _call_gemini(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.7,
                        "max_output_tokens": 150,
                    },
                )
                result = json.loads(response.text)
                time.sleep(self.sleep_time)
                return result
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    wait = 60 * (attempt + 1)
                    print(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"Gemini error: {e}")
                    return {**FALLBACK, "reasoning": f"error: {e}"}
        return {**FALLBACK, "reasoning": "fallback after retries"}
