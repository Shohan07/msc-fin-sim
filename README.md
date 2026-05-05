# MSc Financial Market Simulation

An agent-based stock market simulation where trading agents retrieve relevant financial news via RAG (Retrieval-Augmented Generation) and use Gemini to reason about it before deciding to BUY, SELL, or HOLD.

Three controlled experiments compare the effect of news-informed AI reasoning on market realism, measured through stylised facts: excess kurtosis, volatility clustering, and ADF stationarity tests.

---

## Overview

| Experiment | Condition | Agents |
|---|---|---|
| 1 | Baseline — no RAG | 10 Fundamentalist + 10 SentimentMomentum + 10 NoiseTrader |
| 2 | Random news control | 10 Fundamentalist + 10 SentimentMomentum + 10 RAGAgent (random text) |
| 3 | Full Gemini RAG | 10 Fundamentalist + 10 SentimentMomentum + 10 RAGAgent (FAISS + Gemini) |

Each condition runs 3 seeds (42, 123, 999) × 200 steps = 9 total runs.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM reasoning | Gemini 2.5 Flash-Lite (`gemini-2.5-flash-lite`) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, no API) |
| Vector store | FAISS-cpu |
| RAG framework | LangChain + langchain-google-genai |
| Multi-agent framework | Mesa 3.5.x |
| Stats | scipy + statsmodels |
| Figures | matplotlib + seaborn |

---

## Project Structure

```
msc-fin-sim/
├── src/
│   ├── agents/
│   │   ├── base_agent.py           # Abstract base with cash/shares interface
│   │   ├── fundamentalist.py       # Trades on personal fundamental value
│   │   ├── sentiment_momentum.py   # Combines price trend + keyword tone
│   │   ├── noise_trader.py         # Random BUY/SELL/HOLD
│   │   └── rag_agent.py            # FAISS retrieval + Gemini reasoning
│   ├── market/
│   │   └── order_book.py           # Price-impact order book
│   ├── rag/
│   │   ├── document_store.py       # FAISS wrapper (local MiniLM embeddings)
│   │   ├── retriever.py            # Query FAISS, return top-k news
│   │   ├── llm_agent.py            # Gemini call with LLM response cache
│   │   ├── random_store.py         # Random non-financial text (Experiment 2)
│   │   └── prompt_template.txt     # Structured prompt for Gemini
│   └── simulation.py               # Mesa model orchestrating agents + order book
├── experiments/
│   ├── run_experiments.py          # Runs all 9 experiment runs
│   ├── analysis.py                 # Computes metrics, generates plots
│   ├── results/                    # baseline.json, random_news.json, gemini_rag.json
│   └── plots/                      # fig1_price_series.png, fig2_metrics.png, etc.
├── data/
│   ├── raw/combined_news_corpus.csv   # ~26,000 financial news items
│   └── embeddings/
│       ├── faiss_index/               # Prebuilt FAISS index (384-dim)
│       └── llm_cache.pkl              # Cached Gemini responses
├── scripts/
│   └── build_corpus.py             # Downloads and merges 3 HuggingFace datasets
├── dissertation/                   # LaTeX / Word source
├── .env                            # GOOGLE_API_KEY (never committed)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd msc-fin-sim
```

### 2. Create and activate the virtual environment

```bash
python -m venv venv
source venv/Scripts/activate   # Git Bash on Windows
# macOS/Linux: source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install google-genai langchain-google-genai langchain-community faiss-cpu mesa \
    pandas numpy scipy matplotlib seaborn tqdm python-dotenv datasets jupyter \
    statsmodels sentence-transformers torch
```

### 4. Add your Gemini API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com). Free tier: 15 RPM / 1,000 RPD.

### 5. Build the news corpus and FAISS index

```bash
python scripts/build_corpus.py
```

Downloads three datasets from HuggingFace and builds the local FAISS index using `all-MiniLM-L6-v2`. No API calls are made during indexing.

---

## Running the Experiments

```bash
python experiments/run_experiments.py
```

Runs all 9 simulations (3 conditions × 3 seeds, 200 steps each). Results are saved to `experiments/results/`.

**Expected runtimes:**
- Baseline (no API calls): ~3 minutes
- Random news + Full RAG: up to 60 minutes on free tier (rate-limited to 15 RPM)

### Analyse results and generate plots

```bash
python experiments/analysis.py
```

Outputs metric tables and figures to `experiments/plots/`.

---

## Evaluation Metrics

| Metric | Target | Interpretation |
|---|---|---|
| Excess kurtosis | > 3.0 | Fat tails — realistic return distribution |
| ACF squared returns (lag-1) | > 0.10 | Volatility clustering |
| ADF p-value | > 0.05 | Non-stationarity of price series |
| Jarque-Bera p-value | < 0.05 | Non-normal return distribution |
| ACF absolute returns (lag-1) | > 0.10 | Long-range dependence |

---

## Dataset Sources

| Dataset | Source | Size |
|---|---|---|
| Financial PhraseBank (Malo et al., 2014) | HuggingFace / Kaggle | 4,840 sentences |
| FinGPT fingpt-sentiment-train | HuggingFace: FinGPT/fingpt-sentiment-train | 10,000 sampled |
| Twitter Financial News Sentiment | HuggingFace: zeroshot/twitter-financial-news-sentiment | 11,932 tweets |

All labels normalised to: `positive` / `negative` / `neutral`.

---

## Key References

- LeBaron et al. (1999) — Santa Fe Artificial Stock Market
- Cont (2001) — Stylised facts of financial time series
- Lewis et al. (2020) — RAG (ArXiv 2005.11401)
- Park et al. (2023) — Generative Agents (ArXiv 2304.03442)
- Reimers & Gurevych (2019) — Sentence-BERT / sentence-transformers
- Huang et al. (2023) — FinGPT
- ter Hoeven et al. (2025) — Mesa 3 (JOSS 10.21105/joss.07668)
