"""
Evaluate retrieval quality with NDCG@10 on the FiQA-2018 test set, comparing
BM25, dense, hybrid (RRF), and hybrid + rerank. Also reports Recall@K of the
hybrid candidate pool, to show how often the relevant doc reaches the reranker.

By default only 50 sampled test queries are scored, to stay within Cohere's
free reranking tier. Set RERANK_SAMPLE_SIZE to None to score all test queries.
"""

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retrieval.fusion import hybrid_candidates
from src.retrieval.reranker import rerank_with_cohere
from src.retrieval.retrievers import BM25Retriever, DenseRetriever

ROOT = Path(__file__).parents[0]
DATA_DIR = ROOT / "data" / "0.raw_fiqa"

RERANK_SAMPLE_SIZE = 50
SEED = 42


def load_corpus() -> pd.DataFrame:
    """Load the FiQA corpus: the documents the pipeline searches over."""
    return pd.read_parquet(DATA_DIR / "corpus.parquet")


def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Normalized discounted cumulative gain for a single query."""
    dcg = sum(relevant.get(doc_id, 0) / math.log2(rank + 2) for rank, doc_id in enumerate(predicted_ids[:k]))
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of relevant docs that appear in the top k predictions.

    Unlike NDCG this ignores ordering; it answers whether the relevant docs are
    retrieved at all. Measured on the fused candidate pool, it shows whether the
    reranker is even given a chance to surface the right document.
    """
    relevant_ids = {doc_id for doc_id, rel in relevant.items() if rel > 0}
    if not relevant_ids:
        return 0.0
    retrieved = relevant_ids.intersection(predicted_ids[:k])
    return len(retrieved) / len(relevant_ids)


def load_qrels() -> dict[str, dict[str, int]]:
    """Ground truth as {query_id: {doc_id: relevance}}."""
    qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    return qrels


def main() -> None:
    qrels = load_qrels()

    # Only queries with at least one judgment can be scored.
    queries = pd.read_parquet(DATA_DIR / "queries.parquet")
    queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()
    n = RERANK_SAMPLE_SIZE or len(queries_with_qrels)
    sample = queries_with_qrels.sample(n=n, random_state=SEED)
    print(f"Evaluating on {len(sample)} queries (sampled from {len(queries_with_qrels)})")

    # Wire up the four retrieval methods.
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    corpus_by_id = load_corpus().set_index("_id")

    CANDIDATE_K = 50

    # Score every method on the sample.
    results: dict[str, list[float]] = defaultdict(list)
    candidate_recall: list[float] = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating"):
        query_text = row["text"]
        relevant = qrels[str(row["_id"])]

        # The fused candidate pool feeds both the recall measurement and the
        # reranker, so compute it once and reuse it.
        candidates = [d for d, _ in hybrid_candidates(query_text, bm25, dense, candidate_k=CANDIDATE_K)]
        reranked = [d for d, _ in rerank_with_cohere(query_text, candidates, corpus_by_id, k=10)]

        results["BM25"].append(ndcg_at_k([d for d, _ in bm25.search(query_text, k=10)], relevant))
        results["Dense"].append(ndcg_at_k([d for d, _ in dense.search(query_text, k=10)], relevant))
        results["Hybrid (RRF)"].append(ndcg_at_k(candidates[:10], relevant))
        results["Hybrid + Rerank"].append(ndcg_at_k(reranked, relevant))

        candidate_recall.append(recall_at_k(candidates, relevant, k=CANDIDATE_K))

    # Print the table.
    print(f"\nNDCG@10 x100 on FiQA ({len(sample)} sampled test queries)")
    print("-" * 42)
    for method, scores in results.items():
        print(f"  {method:<22} {np.mean(scores) * 100:.2f}")

    # How often the relevant doc even reaches the reranker.
    print(f"\nRecall@{CANDIDATE_K} of the hybrid candidate pool: {np.mean(candidate_recall) * 100:.2f}")


if __name__ == "__main__":
    main()
