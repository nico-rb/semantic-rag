"""
Build and persist the BM25 index over the FiQA corpus.

Every document is tokenized (lowercased, punctuation stripped, English
stopwords removed) and indexed by bm25s. The index plus the aligned list
of document ids is written to disk so the corpus is tokenized only once.

This is the offline build step. The resulting index is used
by the BM25 retriever at query time.
"""

from pathlib import Path

import bm25s
import pandas as pd

ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data" / "0.raw_fiqa"
BM25_DIR = ROOT / "data" / "indexes" / "bm25"


def build_bm25_index(documents: list[str]) -> bm25s.BM25:
    """
    Build and return a BM25 index from a list of documents.

    The tokenizer lowercases text, strips punctuation,
    and removes English stopwords.
    """
    tokens = bm25s.tokenize(documents, stopwords="en")

    retriever = bm25s.BM25()
    retriever.index(tokens)

    return retriever


corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")

doc_ids = corpus["_id"].tolist()
doc_texts = corpus["text"].tolist()

print(f"Indexing {len(doc_texts)} documents with BM25...")

retriever = build_bm25_index(doc_texts)


# Persist the index to disk
BM25_DIR.mkdir(parents=True, exist_ok=True)
retriever.save(str(BM25_DIR))
(BM25_DIR / "doc_ids.txt").write_text("\n".join(doc_ids))
