"""
Reranking: reorder the hybrid candidates with a cross-encoder and return the
top k.

A bi-encoder (the dense retriever) embeds query and document separately and
compares them with cosine similarity. A cross-encoder scores query and document
jointly in a single pass, capturing interactions that two independent
embeddings miss, at higher cost. Reranking is therefore applied only to the top
candidates from fusion.
"""

import json
import os
import time
from pathlib import Path

import cohere
import pandas as pd
from dotenv import load_dotenv

from src.retrieval.fusion import hybrid_candidates
from src.retrieval.retrievers import BM25Retriever, DenseRetriever

load_dotenv()

co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
RERANK_MODEL = "rerank-v4.0-fast"

# Cohere's trial key allows only 10 requests/minute, so each (query, candidates)
# pair is reranked against the API once and cached on disk. A cache key pins the
# query together with the exact candidate list, so a different set of candidates
# is reranked afresh.
CACHE_PATH = Path(__file__).parents[2] / "data" / "cache" / "rerank" / "rerank_cache.json"

# The trial limit is per 60s window, so on a 429 we wait for the window to
# refill and retry instead of crashing the whole evaluation run.
RATE_LIMIT_WAIT_SECONDS = 60


def _load_cache() -> dict[str, list[list]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _rerank_call(query: str, documents: list[str], k: int):
    """Call Cohere's reranker, waiting out the trial rate limit on a 429."""
    while True:
        try:
            return co.rerank(model=RERANK_MODEL, query=query, documents=documents, top_n=k)
        except cohere.TooManyRequestsError:
            print(f"  rate limited, waiting {RATE_LIMIT_WAIT_SECONDS}s for the trial quota to refill...")
            time.sleep(RATE_LIMIT_WAIT_SECONDS)


def rerank_with_cohere(
    query: str,
    candidate_ids: list[str],
    corpus_by_id: pd.DataFrame,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Rescore candidate docs against the query with the cross-encoder.

    Results are cached on disk keyed by the query and its candidate list, so a
    repeated call is a local lookup rather than another API request.
    """
    cache = _load_cache()
    cache_key = f"{query}::{'||'.join(candidate_ids)}"
    if cache_key in cache:
        return [(doc_id, score) for doc_id, score in cache[cache_key][:k]]

    documents = [corpus_by_id.loc[d, "text"] for d in candidate_ids]
    response = _rerank_call(query, documents, k)
    ranked = [(candidate_ids[r.index], r.relevance_score) for r in response.results]

    cache[cache_key] = [[doc_id, score] for doc_id, score in ranked]
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return ranked


def search_reranked(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    corpus_by_id: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    """Full pipeline: hybrid retrieve candidate_k docs, then rerank to top k."""
    candidates = hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    return rerank_with_cohere(query, candidate_ids, corpus_by_id, k=k)


# Smoke test: hybrid alone vs hybrid + cross-encoder rerank on a single query.
if __name__ == "__main__":
    from pathlib import Path

    corpus_path = Path(__file__).parents[2] / "data" / "0.raw_fiqa" / "corpus.parquet"
    corpus_by_id = pd.read_parquet(corpus_path).set_index("_id")
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    def show(label: str, results: list[tuple[str, float]]) -> None:
        print(f"\n{label}")
        for i, (doc_id, score) in enumerate(results[:5], 1):
            text = corpus_by_id.loc[doc_id, "text"]
            print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")

    hybrid = hybrid_candidates(query, bm25, dense, candidate_k=50)
    show("Hybrid (RRF) only", hybrid[:5])
    show("Hybrid + Cohere rerank-v4.0-fast", search_reranked(query, bm25, dense, corpus_by_id, k=5))
