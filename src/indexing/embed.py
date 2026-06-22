"""
Build and persist the dense embedding matrix over the FiQA corpus.

Every document is embedded once and the resulting (N, 1536) matrix is 
cached as a single .npy file, loaded from disk whenever it already 
exists so nothing is ever embedded twice.

This is the offline build step. The resulting matrix is used
by the dense retriever at query time.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data" / "0.raw_fiqa"
DENSE_DIR = ROOT / "data" / "indexes" / "dense"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 256

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts and return a (len(texts), EMBEDDING_DIM) array."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in response.data], dtype=np.float32)


def build_dense_index(documents: list[str]) -> np.ndarray:
    """
    Build and return the dense embedding matrix from a list of documents.

    Documents are embedded in batches and stacked into a single
    (N, EMBEDDING_DIM) float32 matrix.
    """
    chunks = []
    for i in tqdm(range(0, len(documents), EMBED_BATCH_SIZE), desc="Embedding"):
        chunks.append(embed_batch(documents[i : i + EMBED_BATCH_SIZE]))
    return np.vstack(chunks)


corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")

doc_ids = corpus["_id"].tolist()
# OpenAI rejects empty strings in the embeddings endpoint. ~38 FiQA docs have
# blank text; we swap in a placeholder so the row order stays aligned with the
# BM25 index (which tolerates empty text just fine).
doc_texts = [t.strip() or "[empty document]" for t in corpus["text"].tolist()]


# Build the document embedding matrix, or load it if already cached
DENSE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_PATH = DENSE_DIR / "embeddings.npy"

if EMBEDDINGS_PATH.exists():
    print(f"Loading cached embeddings from {EMBEDDINGS_PATH}.")
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    print(f"Embedding {len(doc_texts)} docs.")
    embeddings = build_dense_index(doc_texts)
    np.save(EMBEDDINGS_PATH, embeddings)

# doc_ids must stay aligned with the matrix rows, so always write it next to
# the embeddings — including when the matrix was loaded from cache.
(DENSE_DIR / "doc_ids.txt").write_text("\n".join(doc_ids))
