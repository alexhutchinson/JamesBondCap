from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path(__file__).parent / "index.npz"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_index(index_path: Path = INDEX_PATH):
    data = np.load(index_path, allow_pickle=True)
    embeddings = data["embeddings"]
    documents = data["documents"].tolist()
    metadatas = data["metadatas"].tolist()
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    return embeddings, documents, metadatas, bm25


def query_index(question: str, model, embeddings, documents, metadatas, bm25, n_results: int = 15):
    # --- semantic search ---
    q_embedding = model.encode([question])[0]
    norms = np.linalg.norm(embeddings, axis=1)
    q_norm = np.linalg.norm(q_embedding)
    sem_scores = (embeddings @ q_embedding) / (norms * q_norm + 1e-9)
    sem_ranking = np.argsort(sem_scores)[::-1]

    # --- BM25 keyword search ---
    bm25_scores = bm25.get_scores(question.lower().split())
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    # --- Reciprocal Rank Fusion (k=60 is standard) ---
    k = 60
    rrf = {}
    for rank, idx in enumerate(sem_ranking[:100]):
        rrf[idx] = rrf.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranking[:100]):
        rrf[idx] = rrf.get(idx, 0) + 1 / (k + rank + 1)

    top_indices = sorted(rrf, key=rrf.get, reverse=True)[:n_results]
    chunks = [documents[i] for i in top_indices]
    metas = [metadatas[i] for i in top_indices]

    context = "\n\n---\n\n".join(
        f"[{m['chapter']}]\n{c}" for c, m in zip(chunks, metas)
    )
    return context, metas
