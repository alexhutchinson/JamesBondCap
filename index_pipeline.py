from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path(__file__).parent / "index.npz"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_index(index_path: Path = INDEX_PATH):
    data = np.load(index_path, allow_pickle=True)
    return data["embeddings"], data["documents"].tolist(), data["metadatas"].tolist()


def query_index(question: str, model, embeddings, documents, metadatas, n_results: int = 20):
    q_embedding = model.encode([question])[0]

    norms = np.linalg.norm(embeddings, axis=1)
    q_norm = np.linalg.norm(q_embedding)
    scores = (embeddings @ q_embedding) / (norms * q_norm + 1e-9)

    top_indices = np.argsort(scores)[::-1][:n_results]
    chunks = [documents[i] for i in top_indices]
    metas = [metadatas[i] for i in top_indices]

    context = "\n\n---\n\n".join(
        f"[{m['chapter']}]\n{c}" for c, m in zip(chunks, metas)
    )
    return context, metas
