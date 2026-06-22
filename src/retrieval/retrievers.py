"""
Query-time BM25 and dense retrievers over the prebuilt FiQA indexes.

These are the single source of truth for retrieval: the offline build steps
(indexing/bm25.py, indexing/embed.py) persist the indexes, and everything
downstream (fusion, reranking, evaluation) loads its search logic from here.
"""

import json
import os
from pathlib import Path

import bm25s
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).parents[2]
BM25_DIR = ROOT / "data" / "indexes" / "bm25"
DENSE_DIR = ROOT / "data" / "indexes" / "dense"

EMBEDDING_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --------------------------------------------------------------
# BM25
# --------------------------------------------------------------


class BM25Retriever:
    def __init__(self) -> None:
        self._retriever = bm25s.BM25.load(str(BM25_DIR))
        self._doc_ids = (BM25_DIR / "doc_ids.txt").read_text().splitlines()

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = bm25s.tokenize([query], stopwords="en")
        indices, scores = self._retriever.retrieve(tokens, k=k)
        return [(self._doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())]


# --------------------------------------------------------------
# Dense
# --------------------------------------------------------------


class DenseRetriever:
    def __init__(self) -> None:
        self._doc_ids = (DENSE_DIR / "doc_ids.txt").read_text().splitlines()
        raw = np.load(DENSE_DIR / "embeddings.npy")
        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        # Query embedding cache: a query is embedded against the API only the
        # first time it is seen; every later call for it is a local lookup.
        queries_dir = DENSE_DIR / "queries"
        queries_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = queries_dir / "query_cache.json"
        self._cache_embeddings_path = queries_dir / "query_embeddings.npy"

        if self._cache_path.exists():
            self._query_cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
        else:
            self._query_cache = {}

        if self._cache_embeddings_path.exists():
            self._query_embeddings = np.load(self._cache_embeddings_path)
        else:
            dim = self._embeddings.shape[1]
            self._query_embeddings = np.empty((0, dim), dtype=np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        if query in self._query_cache:
            vec = self._query_embeddings[self._query_cache[query]]
        else:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            self._query_cache[query] = len(self._query_cache)
            self._query_embeddings = np.vstack([self._query_embeddings, vec[np.newaxis, :]])
            self._cache_path.write_text(json.dumps(self._query_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            np.save(self._cache_embeddings_path, self._query_embeddings)
        return vec / np.linalg.norm(vec)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores = self._embeddings @ self._embed_query(query)
        top_k = np.argsort(-scores)[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_k]
