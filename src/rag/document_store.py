import os
import pickle

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/raw/combined_news_corpus.csv"
INDEX_DIR = "data/embeddings/faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.pkl")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384


def build_index():
    print(f"Loading corpus from {CORPUS_PATH}...")
    df = pd.read_csv(CORPUS_PATH)
    texts = df["text"].astype(str).tolist()
    print(f"  {len(texts):,} documents loaded")

    print(f"Embedding with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(embeddings)
    print(f"  Index contains {index.ntotal:,} vectors")

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"  Index saved to {INDEX_PATH}")

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    print(f"  Texts saved to {TEXTS_PATH}")


def load_index():
    """Return (faiss_index, texts_list)."""
    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
    return index, texts


if __name__ == "__main__":
    build_index()
