"""
Reciprocal Rank Fusion (RRF): combine the BM25 and dense rankings into one.

BM25 scores are unbounded while cosine similarities lie in [0, 1], so the two
cannot be averaged directly. RRF fuses the rankings rather than the scores.
"""

from collections import defaultdict
from pathlib import Path

from src.retrieval.retrievers import BM25Retriever, DenseRetriever

K_RRF = 60


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = K_RRF) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of doc_ids into one ranked list."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_candidates(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int = 50,
) -> list[tuple[str, float]]:
    """Retrieve candidate_k docs from each retriever and fuse them with RRF."""
    bm25_ids = [doc_id for doc_id, _ in bm25.search(query, k=candidate_k)]
    dense_ids = [doc_id for doc_id, _ in dense.search(query, k=candidate_k)]
    return reciprocal_rank_fusion([bm25_ids, dense_ids])[:candidate_k]


# Smoke test: BM25 vs dense vs their fusion on a single query.
if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd

    corpus = pd.read_parquet(Path(__file__).parents[2] / "data" / "0.raw_fiqa" / "corpus.parquet")
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    def show(label: str, results: list[tuple[str, float]]) -> None:
        print(f"\n{label}")
        for i, (doc_id, score) in enumerate(results[:5], 1):
            text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
            print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")

    show("BM25 only", bm25.search(query, k=5))
    show("Dense only", dense.search(query, k=5))
    show("Hybrid (RRF)", hybrid_candidates(query, bm25, dense, candidate_k=50)[:5])
