from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path(__file__).parent / "index.npz"
MODEL_NAME = "all-MiniLM-L6-v2"


def query_index(
    question: str,
    index_path: Path = INDEX_PATH,
    n_results: int = 8,
) -> tuple[str, list[dict]]:
    data = np.load(index_path, allow_pickle=True)
    embeddings = data["embeddings"]
    documents = data["documents"].tolist()
    metadatas = data["metadatas"].tolist()

    model = SentenceTransformer(MODEL_NAME)
    q_embedding = model.encode([question])[0]

    # cosine similarity
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
