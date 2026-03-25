# STATUS.md — Progress Tracker
# Updated: 25 March 2026 | Deadline: 12 April 2026

## Phase 1: Environment Setup (Day 1)
- [x] VS Code installed and configured
- [x] Claude Code CLI installed via npm
- [x] Claude Code authenticated with Pro account
- [x] Claude Code VS Code extension installed
- [x] Project folder msc-fin-sim created
- [x] Git initialised
- [x] Folder structure created (src/, data/, experiments/, etc.)
- [x] All __init__.py files created
- [x] .gitignore created
- [x] .vscode/settings.json created (cross-platform)
- [x] Virtual environment created
- [x] venv activated
- [x] pip upgraded to latest
- [ ] All packages installed
- [ ] requirements.txt frozen (VENV LOCKED)
- [ ] .env created with GOOGLE_API_KEY
- [ ] CLAUDE.md dropped into project root
- [ ] STATUS.md dropped into project root
- [ ] Gemini 2.5 Flash-Lite API test passes
- [ ] Local MiniLM embedding test passes
- [ ] Git commit: environment locked

## Phase 1.5: Tier 1 Upgrade (Day 1)
- [ ] Google Cloud billing linked at aistudio.google.com
- [ ] Budget alert set at $5
- [ ] Tier 1 speed test passes (5 fast requests)

## Phase 2: Data Pipeline (Day 1-2)
- [ ] scripts/build_corpus.py built by Claude Code
- [ ] build_corpus.py run — combined_news_corpus.csv created (~26,000 rows)
- [ ] src/rag/document_store.py built by Claude Code
- [ ] FAISS index built with local MiniLM embeddings
- [ ] Git commit: data pipeline done

## Phase 3: Source Code (Day 2-4)
- [ ] src/market/order_book.py — test passes
- [ ] src/agents/base_agent.py — test passes
- [ ] src/agents/fundamentalist.py — test passes
- [ ] src/agents/sentiment_momentum.py — test passes
- [ ] src/agents/noise_trader.py — test passes
- [ ] src/rag/retriever.py — test passes
- [ ] src/rag/prompt_template.txt — created
- [ ] src/rag/llm_agent.py — test passes (1 API call)
- [ ] src/rag/random_store.py — test passes
- [ ] src/agents/rag_agent.py — test passes
- [ ] src/simulation.py — baseline smoke test passes
- [ ] Git commit: all source modules complete

## Phase 4: Experiment Runner + Analysis (Day 4-5)
- [ ] experiments/run_experiments.py built by Claude Code
- [ ] experiments/analysis.py built by Claude Code
- [ ] Baseline smoke test (10 steps, seed 42) passes

## Phase 5: Run Experiments (Day 5-8)
- [ ] Experiment 1: Baseline — seeds 42, 123, 999 complete
- [ ] Experiment 2: Random news — seeds 42, 123, 999 complete
- [ ] Experiment 3: Full RAG — seeds 42, 123, 999 complete
- [ ] All 9 result JSONs saved in experiments/results/
- [ ] Git commit: experiments complete

## Phase 6: Analysis + Dissertation (Day 6-16)
- [ ] analysis.py run — all metrics computed
- [ ] fig1_price_series.png generated
- [ ] fig2_return_distributions.png generated
- [ ] fig3_metrics_comparison.png generated
- [ ] stats_table.csv generated
- [ ] Section 2: Background (~900 words)
- [ ] Section 3: Methodology (~1000 words)
- [ ] Section 4: Implementation (~600 words)
- [ ] Section 5: Results (~800 words)
- [ ] Section 6: Discussion (~600 words)
- [ ] Section 1: Introduction (~400 words) — write last
- [ ] Section 7: Conclusion (~350 words) — write last
- [ ] References: 15-20 listed

## Phase 7: Final (Day 17-18)
- [ ] All sections proofread
- [ ] Formatting checked
- [ ] README.md created
- [ ] Final git commit
- [ ] SUBMITTED
