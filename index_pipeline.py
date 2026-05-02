from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).parent / "chroma_db"


def query_index(
    question: str,
    chroma_path: Path = CHROMA_PATH,
    n_results: int = 8,
) -> tuple[str, list[dict]]:
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection("novogradac")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding = model.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n_results)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    context = "\n\n---\n\n".join(
        f"[{m['chapter']}]\n{c}" for c, m in zip(chunks, metadatas)
    )
    return context, metadatas
