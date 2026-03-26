import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.document_store import load_index, EMBEDDING_MODEL


class NewsRetriever:
    def __init__(self):
        self.index, self.texts = load_index()
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def retrieve(self, query, k=3):
        """Return top-k news text strings most similar to query."""
        embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(embedding)
        _, indices = self.index.search(embedding, k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]
