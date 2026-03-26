"""
Build combined news corpus from three HuggingFace datasets.
Outputs: data/raw/combined_news_corpus.csv
Columns: text, label, source
Labels normalised to: positive / negative / neutral
"""

import os
import pandas as pd
from datasets import load_dataset

OUTPUT_PATH = "data/raw/combined_news_corpus.csv"
FINGPT_SAMPLE = 10_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Label normalisers
# ---------------------------------------------------------------------------


def normalise_fingpt(label_str):
    """
    FinGPT labels are free-text strings like 'positive', 'negative', 'neutral',
    but sometimes 'mildly positive', 'strong negative', etc.
    """
    s = str(label_str).lower().strip()
    if "positive" in s:
        return "positive"
    if "negative" in s:
        return "negative"
    return "neutral"


def normalise_twitter(label):
    """
    zeroshot/twitter-financial-news-sentiment uses integer labels:
    0 = bearish (negative), 1 = bullish (positive), 2 = neutral
    """
    mapping = {0: "negative", 1: "positive", 2: "neutral"}
    return mapping.get(int(label), "neutral")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

PHRASEBANK_INT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
PHRASEBANK_PARQUET_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main"
    "/sentences_allagree/financial_phrasebank-train.parquet"
)


def load_phrasebank():
    print("Downloading Financial PhraseBank...")

    # Strategy 1: alternative mirror dataset
    try:
        print("  Trying gtfintechlab mirror...")
        ds = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        text_col = "sentence" if "sentence" in df.columns else "text"
        df = df.rename(columns={text_col: "text"})
        if df["label"].dtype != object:
            df["label"] = df["label"].map(PHRASEBANK_INT_LABELS)
        df["label"] = df["label"].str.lower()
        df["source"] = "financial_phrasebank"
        print(f"  PhraseBank loaded via mirror ({len(df):,} rows)")
        return df[["text", "label", "source"]]
    except Exception as e:
        print(f"  Mirror failed: {e}")

    # Strategy 2: direct parquet download
    try:
        print("  Trying direct parquet download...")
        df = pd.read_parquet(PHRASEBANK_PARQUET_URL)
        df = df.rename(columns={"sentence": "text"})
        df["label"] = df["label"].map(PHRASEBANK_INT_LABELS)
        df["source"] = "financial_phrasebank"
        print(f"  PhraseBank loaded via parquet ({len(df):,} rows)")
        return df[["text", "label", "source"]]
    except Exception as e:
        print(f"  Parquet failed: {e}")

    raise RuntimeError("All PhraseBank loading strategies failed.")


def load_fingpt():
    print("Downloading FinGPT/fingpt-sentiment-train (sampling 10,000 rows)...")
    ds = load_dataset("FinGPT/fingpt-sentiment-train")
    df = ds["train"].to_pandas()
    df = df.sample(n=FINGPT_SAMPLE, random_state=RANDOM_SEED).reset_index(drop=True)
    # FinGPT columns: input (news text), output (label)
    df = df.rename(columns={"input": "text", "output": "label"})
    df["label"] = df["label"].apply(normalise_fingpt)
    df["source"] = "fingpt_sentiment"
    return df[["text", "label", "source"]]


def load_twitter():
    print("Downloading zeroshot/twitter-financial-news-sentiment...")
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    frames = []
    for split in ds.keys():
        frames.append(ds[split].to_pandas())
    df = pd.concat(frames, ignore_index=True)
    # Columns: text, label
    df["label"] = df["label"].apply(normalise_twitter)
    df["source"] = "twitter_financial"
    return df[["text", "label", "source"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    phrasebank = load_phrasebank()
    fingpt = load_fingpt()
    twitter = load_twitter()

    combined = pd.concat([phrasebank, fingpt, twitter], ignore_index=True)
    combined = combined.dropna(subset=["text"]).reset_index(drop=True)
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"] != ""].reset_index(drop=True)

    combined.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print("\n--- Row counts per source ---")
    for source, count in combined.groupby("source").size().items():
        print(f"  {source}: {count:,}")
    print(f"  TOTAL: {len(combined):,}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
