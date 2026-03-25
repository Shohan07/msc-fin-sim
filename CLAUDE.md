# CLAUDE.md — MSc Financial Market Simulation
# Auto-read by Claude Code on every session open
# Last updated: 25 March 2026

## Project Overview
Stock market simulation where trading agents retrieve relevant financial news (RAG) and use Gemini to reason about it before deciding to BUY, SELL, or HOLD. Three controlled experiments compare: (1) baseline with no RAG, (2) random news control, (3) full Gemini RAG. Success = more realistic stylised facts: kurtosis, volatility clustering, ADF test.

## CRITICAL — Model Changes (March 2026)
> **gemini-1.5-flash is RETIRED — returns 404. Do NOT use it.**
> **text-embedding-004 is SHUT DOWN — returns 404. Do NOT use it.**

| Component | Model to use | Notes |
|-----------|-------------|-------|
| LLM reasoning | `gemini-2.5-flash-lite` | Free tier: 15 RPM, 1,000 RPD |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, LOCAL) | No API needed, 384-dim vectors |
| Vector store | FAISS-cpu (latest) | Index dim = 384 |

## Machine & Environment
- OS: Windows
- Shell: Git Bash (NOT PowerShell)
- venv activation: `source venv/Scripts/activate` (NOT `venv\Scripts\activate`)
- Project path: `G:/UNIVERSITY/Ulster/MSC Research/msc-fin-sim/`
- Python: 3.10+

## Tech Stack — All Locked
| Component | Tool |
|-----------|------|
| LLM reasoning | Gemini 2.5 Flash-Lite via google-genai SDK |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (LOCAL, no API) |
| Vector store | FAISS-cpu |
| RAG framework | LangChain + langchain-google-genai |
| Multi-agent | Mesa |
| Stats | scipy + statsmodels |
| Figures | matplotlib + seaborn |
| Python env | venv inside project folder |

### NOT using
Ollama, phi3:mini, any local LLM, HuggingFace local LLMs, OpenAI API, NewsAPI, Alpha Vantage, Twitter API, any paid data APIs, gemini-1.5-flash, text-embedding-004.

## Packages to Install
```bash
pip install google-genai langchain-google-genai langchain-community faiss-cpu mesa pandas numpy scipy matplotlib seaborn tqdm python-dotenv datasets jupyter statsmodels sentence-transformers torch
```

## Dataset Strategy
Three datasets merged into `data/raw/combined_news_corpus.csv` (~26,000 items):

| Dataset | Source | Size |
|---------|--------|------|
| Financial PhraseBank (Malo et al., 2014) | HuggingFace / Kaggle | 4,840 sentences |
| FinGPT fingpt-sentiment-train | HuggingFace: FinGPT/fingpt-sentiment-train | 10,000 sampled from 76k |
| Twitter Financial News Sentiment | HuggingFace: zeroshot/twitter-financial-news-sentiment | 11,932 tweets |

All labels normalised to: positive / negative / neutral

## API Tier — Check Before Running Experiments
> **Before your first experiment run, check your tier at aistudio.google.com → Settings**
> If free tier: keep `time.sleep(4.5)` — you have 15 RPM / 1,000 RPD
> If Tier 1 (billing linked, even $0 cap): use `time.sleep(0.5)` — you have 150+ RPM / 1,500+ RPD
> **Recommendation:** Link Google Cloud billing with $0 spend cap to get Tier 1 instantly.
> You get $300 free credit. The whole project costs <$2 in API calls. This turns 5-day experiments into 3-hour runs.

## Mesa 3.x Syntax (CRITICAL — current version is 3.5.1)
> **Mesa 2.x code WILL NOT WORK. Use Mesa 3.x syntax only.**

### Agent __init__:
```python
# CORRECT — Mesa 3.x
class MyAgent(mesa.Agent):
    def __init__(self, model, some_param):
        super().__init__(model)  # NO unique_id — auto-assigned
        self.some_param = some_param

# WRONG — Mesa 2.x (will crash)
class MyAgent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)  # unique_id is GONE
```

### Model __init__:
```python
class MarketModel(mesa.Model):
    def __init__(self, n_agents=30, seed=None):
        super().__init__(seed=seed)  # seed still works but rng preferred
        # Create agents — NO scheduler needed
        for i in range(n_agents):
            MyAgent(self, some_param=value)  # agents auto-register with model
```

### Stepping agents (NO scheduler):
```python
# CORRECT — Mesa 3.x
def step(self):
    self.agents.shuffle_do("step")  # replaces RandomActivation
    # OR for specific agent types:
    self.agents_by_type[RAGAgent].shuffle_do("step")

# WRONG — Mesa 2.x
self.schedule.step()  # schedulers are DEPRECATED
```

### Step counter:
```python
# CORRECT: self.steps (auto-incremented by model.step())
# WRONG: self.model._steps or self.schedule.steps
```

### Running the model:
```python
model = MarketModel(n_agents=30, seed=42)
for _ in range(200):
    model.step()  # self.steps auto-increments
```

### Cite Mesa as:
ter Hoeven, E., Kwakkel, J., Hess, V., Pike, T., Wang, B., rht, & Kazil, J. (2025). Mesa 3: Agent-based modeling with Python in 2025. Journal of Open Source Software, 10(107), 7668.

## Four Agent Types

### FundamentalistAgent — src/agents/fundamentalist.py
- Compares price to personal fundamental value from `np.random.normal(100, 5)`
- BUY if: `(fundamental - price) / price > 0.02` AND cash available
- SELL if: `(fundamental - price) / price < -0.02` AND shares held
- HOLD otherwise. Does NOT use news.

### SentimentMomentumAgent — src/agents/sentiment_momentum.py
- Combines 5-step price trend with news keyword tone matching — no LLM call
- POS_WORDS: rise, gain, surge, beat, profit, strong, growth, bull, up
- NEG_WORDS: fall, drop, loss, miss, weak, decline, bear, down, cut
- Strong BUY (qty=3): trend rising AND news positive
- Strong SELL (qty=3): trend falling AND news negative
- Weak BUY/SELL (qty=1): trend and news disagree
- HOLD: conflicting or neutral signals

### NoiseTrader — src/agents/noise_trader.py
- `random.choice(['BUY', 'SELL', 'HOLD'])` with price noise `random.uniform(-0.005, 0.005)`
- Must be at least 30% of all agents

### RAGAgent — src/agents/rag_agent.py
- Retrieve 3 news items from FAISS → pass to Gemini 2.5 Flash-Lite → parse JSON → execute
- Gemini must return: `{"action": "BUY|SELL|HOLD", "quantity": int, "confidence": float, "reasoning": str}`
- **Confidence filter:** If confidence < 0.5, override action to HOLD (agent uncertain = don't trade)
- Logs every decision to `self.decision_log` with step, action, confidence, reasoning, news used
- Safety: BUY blocked if `cash < price * qty`. SELL blocked if `shares < qty`
- Fallback: any exception → return HOLD, never crash the simulation

### Agent Counts — always 30 total
- Experiment 1: 10 Fundamentalist + 10 SentimentMomentum + 10 NoiseTrader
- Experiments 2 & 3: 10 Fundamentalist + 10 SentimentMomentum + 10 RAGAgent

## Initial Conditions
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Starting price | 100.0 | Round number, standard in ABM literature |
| Cash per agent | 10,000.0 | Allows ~100 shares at starting price |
| Shares per agent | 50 | Each agent starts with a position |
| Fundamental value | `np.random.normal(100, 5)` per agent | Heterogeneous beliefs |
| Price noise factor | 0.005 | Prevents price stagnation |
| Simulation steps | 200 | Constrained by API rate limits |

## Price Formation / Order Book
Use a simple price-impact model (standard in ABM finance):
```python
buy_volume = sum(qty for agent in agents if agent.action == 'BUY')
sell_volume = sum(qty for agent in agents if agent.action == 'SELL')
net_demand = buy_volume - sell_volume
price_impact = (net_demand / total_agents) * price_sensitivity
noise = random.gauss(0, noise_std)
new_price = max(0.01, current_price * (1 + price_impact + noise))
```
Where `price_sensitivity = 0.05` and `noise_std = 0.002`.

## Three Experiments

| # | Condition | Agents | Seeds | Output |
|---|-----------|--------|-------|--------|
| 1 | Baseline (no RAG) | 10 Fund + 10 SentMom + 10 Noise | 42, 123, 999 | baseline.json |
| 2 | Random news control | 10 Fund + 10 SentMom + 10 RAGAgent w/ RandomNewsStore | 42, 123, 999 | random_news.json |
| 3 | Full Gemini RAG | 10 Fund + 10 SentMom + 10 RAGAgent w/ real FAISS | 42, 123, 999 | gemini_rag.json |

200 steps per run. 9 total runs (3 conditions × 3 seeds). Results averaged per condition.

## Evaluation Metrics
| Metric | Target | Code |
|--------|--------|------|
| Kurtosis | > 3.0 | `scipy.stats.kurtosis(returns)` |
| ACF squared returns lag-1 | > 0.10 | `statsmodels.tsa.stattools.acf(returns**2, nlags=5)[1]` |
| ADF p-value | > 0.05 | `statsmodels.tsa.stattools.adfuller(prices)[1]` |
| Jarque-Bera p-value | < 0.05 | `scipy.stats.jarque_bera(returns)` |
| ACF absolute returns lag-1 | > 0.10 | `statsmodels.tsa.stattools.acf(np.abs(returns), nlags=5)[1]` |
| Run time | < 3 minutes (baseline), < 60 min (RAG) | `time.time()` wrapper |

## Gemini API Rules

### Every file that calls Gemini MUST start with:
```python
import os
from dotenv import load_dotenv
load_dotenv()  # always before any genai calls
```

### Gemini call pattern — ALWAYS wrap in try/except with retry:
```python
import time
import json
from google import genai

client = genai.Client()

def call_gemini(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.7,
                    "max_output_tokens": 150
                }
            )
            result = json.loads(response.text)
            time.sleep(4.5)  # FREE TIER = 15 req/min — NEVER go below 4.0
            return result
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait = 60 * (attempt + 1)  # 60s, 120s, 180s
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"Gemini error: {e}")
                return {"action": "HOLD", "quantity": 1, "confidence": 0.0, "reasoning": "fallback"}
    return {"action": "HOLD", "quantity": 1, "confidence": 0.0, "reasoning": "fallback after retries"}
```

### LLM Response Cache — MANDATORY
```python
import hashlib
import pickle
import os

CACHE_FILE = "data/embeddings/llm_cache.pkl"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

def get_cached_response(prompt, cache):
    key = hashlib.sha256(prompt.encode()).hexdigest()
    return cache.get(key)

def set_cached_response(prompt, response, cache):
    key = hashlib.sha256(prompt.encode()).hexdigest()
    cache[key] = response
```
Every RAG agent call MUST check cache first. Save cache after every run.

### Embedding — LOCAL, no API:
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, runs on CPU
embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
# Build FAISS index
dimension = 384
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalisation)
faiss.normalize_L2(embeddings)
index.add(embeddings.astype('float32'))
# Save
faiss.write_index(index, "data/embeddings/faiss_index/index.faiss")
```

### Critical Rules
| Rule | Detail |
|------|--------|
| Only model allowed | `gemini-2.5-flash-lite` — never gemini-1.5-flash, never gemini-pro |
| JSON mode | `response_mime_type='application/json'` must always be set |
| Rate limit | `time.sleep(4.5)` after every Gemini call + exponential backoff on 429 |
| API key | Always from `os.getenv('GOOGLE_API_KEY')` — never hardcoded |
| FAISS | Build once with local embeddings — never rebuild |
| Embeddings | LOCAL `all-MiniLM-L6-v2` — no API calls for embeddings |
| Cache | LLM response cache is MANDATORY — check before every Gemini call |
| Smoke test | Must pass on seeds 42, 123, 999 before running experiments |
| Git commits | Format: `feat: filename — one sentence description` |

## Project File Structure
```
msc-fin-sim/
├── .vscode/settings.json
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── fundamentalist.py
│   │   ├── sentiment_momentum.py
│   │   ├── noise_trader.py
│   │   └── rag_agent.py
│   ├── market/
│   │   ├── __init__.py
│   │   └── order_book.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_store.py
│   │   ├── retriever.py
│   │   ├── llm_agent.py
│   │   ├── random_store.py
│   │   └── prompt_template.txt
│   └── simulation.py
├── data/raw/combined_news_corpus.csv
├── data/embeddings/faiss_index/
├── data/embeddings/llm_cache.pkl
├── experiments/
│   ├── run_experiments.py
│   ├── analysis.py
│   ├── results/
│   └── plots/
├── scripts/build_corpus.py
├── dissertation/
├── venv/                    # NEVER commit
├── .env                     # NEVER commit
├── .gitignore
├── CLAUDE.md                # THIS FILE — auto-read by Claude Code
├── STATUS.md                # Progress tracker
├── requirements.txt
└── README.md                # Created near submission
```

## Build Order — Strict Sequence
1. `src/market/order_book.py` — price formation logic
2. `src/agents/base_agent.py` — abstract base with cash, shares, action interface
3. `src/agents/fundamentalist.py`
4. `src/agents/sentiment_momentum.py`
5. `src/agents/noise_trader.py`
6. `src/rag/document_store.py` — FAISS wrapper with local MiniLM embeddings
7. `src/rag/retriever.py` — query FAISS, return top-k news
8. `src/rag/llm_agent.py` — Gemini call with cache
9. `src/rag/random_store.py` — returns random non-financial text for Experiment 2
10. `src/rag/prompt_template.txt` — structured prompt for Gemini
11. `src/agents/rag_agent.py` — combines retriever + llm_agent
12. `src/simulation.py` — Mesa model orchestrating all agents + order book
13. `experiments/run_experiments.py` — runs all 9 experiments
14. `experiments/analysis.py` — computes metrics, generates plots
15. `scripts/build_corpus.py` — downloads and merges 3 datasets

## Common Errors and Fixes
| Error | Cause | Fix |
|-------|-------|-----|
| venv won't activate | Wrong shell | Use Git Bash + `source venv/Scripts/activate` |
| 404 model not found | Using retired model | Use `gemini-2.5-flash-lite` not `gemini-1.5-flash` |
| ResourceExhausted / 429 | Hit rate limit | Exponential backoff: wait 60s, 120s, 180s |
| KeyError: GOOGLE_API_KEY | .env not loaded | `load_dotenv()` before any genai calls |
| allow_dangerous_deserialization | LangChain security | Pass `allow_dangerous_deserialization=True` |
| json.JSONDecodeError | Gemini non-JSON | Check `response_mime_type='application/json'` |
| Prices stuck flat | Not enough crossing orders | NoiseTrader must be 30%+ of agents |
| (venv) not showing | Wrong terminal | VS Code must use Git Bash not PowerShell |
| ModuleNotFoundError | Package not in venv | Confirm (venv) active then pip install |
| ImportError on src modules | Missing __init__.py | Add `__init__.py` to src/, src/agents/, src/market/, src/rag/ |

## Dissertation Structure (~4,500 words)
| Section | Words | Key content |
|---------|-------|-------------|
| 1. Introduction | ~400 | Problem, solution, structure — written last |
| 2. Background | ~900 | 15+ papers: ABMs, RAG, LLMs in agents, stylised facts, research gap |
| 3. Methodology | ~1000 | Architecture, 4 agents, 3-dataset corpus, price mechanism, metrics |
| 4. Implementation | ~600 | Tools, design decisions, Gemini + local embeddings justification |
| 5. Results | ~800 | fig1 + fig2 embedded, stats table, interpret each condition |
| 6. Discussion | ~600 | Results meaning, limitations, compare to Cont (2001) |
| 7. Conclusion | ~350 | 3 contributions, future work |

## Key References (target 15-20)
- LeBaron et al. (1999) — Santa Fe Artificial Stock Market
- Cont (2001) — Stylised facts of financial time series
- Gould et al. (2013) — Limit order books
- Malo et al. (2014) — Financial PhraseBank dataset
- Lewis et al. (2020) — RAG paper (ArXiv 2005.11401)
- Kazil et al. (2020) — Mesa: ABM framework in Python
- Park et al. (2023) — Generative Agents (ArXiv 2304.03442)
- Vaswani et al. (2017) — Attention is all you need (Transformer foundation)
- Reimers & Gurevych (2019) — Sentence-BERT / sentence-transformers
- Huang et al. (2023) — FinGPT: Open-source financial LLMs
- Iaroshev et al. (2024) — Financial RAG
- Chen et al. — Hybrid Multi-agent Financial Markets
- Axtell & Farmer (2025) — ABM in Economics and Finance
- Cont (2007) — Volatility clustering in financial markets
- Farmer et al. (2005) — Is economics the next physical science?
