"""
Run all 9 experiments: 3 conditions × 3 seeds × 200 steps.
Results saved to experiments/results/{condition}_{seed}.json
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(".env")

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import MarketSimulation  # noqa: E402

STEPS = 200
SEEDS = [42, 123, 999]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

EXPERIMENTS = [
    {
        "condition": "baseline",
        "kwargs": {
            "n_fundamentalist": 10,
            "n_sentiment": 10,
            "n_noise": 10,
            "n_rag": 0,
        },
    },
    {
        "condition": "random_news",
        "kwargs": {
            "n_fundamentalist": 10,
            "n_sentiment": 10,
            "n_noise": 0,
            "n_rag": 10,
            "use_random_news": True,
        },
    },
    {
        "condition": "gemini_rag",
        "kwargs": {
            "n_fundamentalist": 10,
            "n_sentiment": 10,
            "n_noise": 0,
            "n_rag": 10,
            "use_random_news": False,
        },
    },
]


def run_single(condition, kwargs, seed):
    print(f"\n--- {condition} | seed={seed} ---")
    model = MarketSimulation(seed=seed, **kwargs)

    start = time.time()
    for _ in tqdm(range(STEPS), desc=f"{condition}/{seed}", unit="step"):
        model.step()
    elapsed = time.time() - start

    results = model.get_results()
    results["condition"] = condition
    results["seed"] = seed
    results["steps"] = STEPS
    results["elapsed_seconds"] = round(elapsed, 2)

    print(f"  Completed in {elapsed:.1f}s — {len(results['price_history'])} prices, "
          f"{len(results['returns'])} returns")

    # Save LLM cache after every RAG run
    if kwargs.get("n_rag", 0) > 0:
        _save_llm_cache(model)

    return results


def _save_llm_cache(model):
    """Persist the shared GeminiTrader cache if it exists on any RAGAgent."""
    from src.agents.rag_agent import RAGAgent
    from src.rag.llm_agent import _save_cache

    for agent in model.agents:
        if isinstance(agent, RAGAgent) and hasattr(agent, "llm_agent"):
            _save_cache(agent.llm_agent.cache)
            break


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total = len(EXPERIMENTS) * len(SEEDS)
    completed = 0

    for exp in EXPERIMENTS:
        condition = exp["condition"]
        kwargs = exp["kwargs"]
        for seed in SEEDS:
            out_path = os.path.join(RESULTS_DIR, f"{condition}_{seed}.json")
            if os.path.exists(out_path):
                print(f"Skipping {condition}_{seed} (already exists)")
                completed += 1
                continue

            results = run_single(condition, kwargs, seed)

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved → {out_path}")

            completed += 1
            print(f"  Progress: {completed}/{total} runs done")

    print(f"\nAll {total} experiments complete. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
