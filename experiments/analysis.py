"""
Load all 9 experiment results, compute stylised-fact metrics,
generate plots, and save stats_table.csv.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

CONDITIONS = ["baseline", "random_news", "gemini_rag"]
SEEDS = [42, 123, 999]

CONDITION_LABELS = {
    "baseline": "Baseline",
    "random_news": "Random News",
    "gemini_rag": "Gemini RAG",
}

METRIC_TARGETS = {
    "kurtosis":        ("higher", "> 3.0"),
    "acf_sq_lag1":     ("higher", "> 0.10"),
    "acf_abs_lag1":    ("higher", "> 0.10"),
    "adf_pvalue":      ("higher", "> 0.05"),
    "jarque_bera_p":   ("lower",  "< 0.05"),
}

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(returns, prices):
    r = np.array(returns)
    if len(r) < 10:
        return {k: np.nan for k in METRIC_TARGETS}

    kurt = float(stats.kurtosis(r, fisher=True))  # excess kurtosis
    acf_sq = float(acf(r ** 2, nlags=5, fft=True)[1])
    acf_abs = float(acf(np.abs(r), nlags=5, fft=True)[1])
    adf_p = float(adfuller(prices, autolag="AIC")[1])
    jb_p = float(stats.jarque_bera(r).pvalue)

    return {
        "kurtosis":      kurt,
        "acf_sq_lag1":   acf_sq,
        "acf_abs_lag1":  acf_abs,
        "adf_pvalue":    adf_p,
        "jarque_bera_p": jb_p,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results():
    records = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = os.path.join(RESULTS_DIR, f"{condition}_{seed}.json")
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            prices = data["price_history"]
            returns = data["returns"]
            metrics = compute_metrics(returns, prices)
            records.append({
                "condition": condition,
                "seed": seed,
                "prices": prices,
                "returns": returns,
                **metrics,
            })
    return records


def average_metrics(records):
    rows = []
    for condition in CONDITIONS:
        cond_records = [r for r in records if r["condition"] == condition]
        if not cond_records:
            continue
        row = {"condition": CONDITION_LABELS[condition]}
        for metric in METRIC_TARGETS:
            vals = [r[metric] for r in cond_records if not np.isnan(r[metric])]
            row[metric] = round(np.mean(vals), 4) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_price_series(records):
    fig, ax = plt.subplots(figsize=(10, 5))
    for condition in CONDITIONS:
        run = next((r for r in records if r["condition"] == condition and r["seed"] == 42), None)
        if run is None:
            continue
        ax.plot(run["prices"], label=CONDITION_LABELS[condition], linewidth=1.2)
    ax.set_title("Price Series — Seed 42", fontsize=13)
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig1_price_series.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_return_distributions(records):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, condition in zip(axes, CONDITIONS):
        # Pool returns from all 3 seeds for richer distribution
        all_returns = []
        for r in records:
            if r["condition"] == condition:
                all_returns.extend(r["returns"])
        if not all_returns:
            continue
        ax.hist(all_returns, bins=50, density=True, color=sns.color_palette("muted")[CONDITIONS.index(condition)],
                edgecolor="white", linewidth=0.3)
        # Overlay normal fit
        mu, sigma = np.mean(all_returns), np.std(all_returns)
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), "k--", linewidth=1, label="Normal fit")
        ax.set_title(CONDITION_LABELS[condition], fontsize=11)
        ax.set_xlabel("Log Return")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density")
    fig.suptitle("Return Distributions (all seeds pooled)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig2_return_distributions.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_metrics_comparison(df_avg):
    metrics_to_plot = ["kurtosis", "acf_sq_lag1", "acf_abs_lag1"]
    metric_labels = {
        "kurtosis":    "Excess Kurtosis",
        "acf_sq_lag1": "ACF(r²) lag-1",
        "acf_abs_lag1": "ACF(|r|) lag-1",
    }

    x = np.arange(len(metrics_to_plot))
    width = 0.25
    palette = sns.color_palette("muted", n_colors=3)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (_, row) in enumerate(df_avg.iterrows()):
        vals = [row[m] for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width, label=row["condition"], color=palette[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels([metric_labels[m] for m in metrics_to_plot])
    ax.set_ylabel("Value")
    ax.set_title("Stylised Fact Metrics by Condition (averaged over seeds)", fontsize=12)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "fig3_metrics_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_stats_table(df_avg):
    col_order = ["condition", "kurtosis", "acf_sq_lag1", "acf_abs_lag1", "adf_pvalue", "jarque_bera_p"]
    headers = {
        "condition":     "Condition",
        "kurtosis":      "Kurtosis",
        "acf_sq_lag1":   "ACF(r²) lag1",
        "acf_abs_lag1":  "ACF(|r|) lag1",
        "adf_pvalue":    "ADF p-val",
        "jarque_bera_p": "JB p-val",
    }
    targets = {
        "kurtosis":      "> 3.0",
        "acf_sq_lag1":   "> 0.10",
        "acf_abs_lag1":  "> 0.10",
        "adf_pvalue":    "> 0.05",
        "jarque_bera_p": "< 0.05",
    }
    print("\n" + "=" * 72)
    print(f"{'Condition':<18} {'Kurtosis':>10} {'ACF(r²)':>10} {'ACF(|r|)':>10} {'ADF p':>10} {'JB p':>10}")
    print(f"{'':18} {'(>3.0)':>10} {'(>0.10)':>10} {'(>0.10)':>10} {'(>0.05)':>10} {'(<0.05)':>10}")
    print("-" * 72)
    for _, row in df_avg.iterrows():
        print(f"{row['condition']:<18} "
              f"{row['kurtosis']:>10.4f} "
              f"{row['acf_sq_lag1']:>10.4f} "
              f"{row['acf_abs_lag1']:>10.4f} "
              f"{row['adf_pvalue']:>10.4f} "
              f"{row['jarque_bera_p']:>10.4f}")
    print("=" * 72)
    print(f"Targets:           {'':>10} {'(>0.10)':>10} {'(>0.10)':>10} {'(>0.05)':>10} {'(<0.05)':>10}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading results...")
    records = load_all_results()
    print(f"  Loaded {len(records)} runs")

    print("Computing averages...")
    df_avg = average_metrics(records)

    csv_path = os.path.join(RESULTS_DIR, "stats_table.csv")
    df_avg.to_csv(csv_path, index=False)
    print(f"  Stats saved → {csv_path}")

    print("Generating plots...")
    plot_price_series(records)
    plot_return_distributions(records)
    plot_metrics_comparison(df_avg)

    print_stats_table(df_avg)


if __name__ == "__main__":
    main()
