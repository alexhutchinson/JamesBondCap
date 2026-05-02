import os
import streamlit as st
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

from index_pipeline import load_index, query_index, MODEL_NAME


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_data():
    return load_index()  # returns (embeddings, documents, metadatas, bm25)


def expand_query(question: str, client: Anthropic) -> list[str]:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                "Generate 3 alternative phrasings of this question about affordable housing finance, "
                "using different terminology that might appear in policy documents, tax law, or handbooks. "
                "Return only the 3 phrasings as a numbered list, no explanation.\n\n"
                f"Question: {question}"
            )
        }]
    )
    lines = response.content[0].text.strip().split("\n")
    variants = [l.lstrip("123456789. )").strip() for l in lines if l.strip()]
    return [question] + variants[:3]


def get_answer(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    model = load_model()
    embeddings, documents, metadatas, bm25 = load_data()

    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    # expand query into multiple phrasings, merge retrieved chunks
    queries = expand_query(question, client)
    seen = set()
    merged_chunks = []
    merged_metas = []
    for q in queries:
        context, sources = query_index(q, model, embeddings, documents, metadatas, bm25, n_results=10)
        for chunk, meta in zip(context.split("\n\n---\n\n"), sources):
            key = meta["filename"] + meta["chapter"]
            if key not in seen:
                seen.add(key)
                merged_chunks.append(chunk)
                merged_metas.append(meta)

    # cap at 20 unique chunks to keep context manageable
    merged_chunks = merged_chunks[:20]
    merged_metas = merged_metas[:20]
    context = "\n\n---\n\n".join(merged_chunks)

    system = (
        "You are a senior tax credit accountant specializing in affordable housing finance in Washington State. "
        "You have deep expertise in Low Income Housing Tax Credits (LIHTC), tax-exempt bond financing, "
        "HUD programs, and Washington-specific agencies including the Washington State Housing Finance Commission (WSHFC). "
        "You speak precisely and practically — your audience is housing finance professionals who need accurate, "
        "actionable answers, not general overviews. Where relevant, note Washington State-specific rules, "
        "WSHFC requirements, or how federal rules interact with state practice.\n\n"
        "You have access to the following indexed resources (921 pages/chunks total):\n"
        "1. Novogradac 2022 Tax Exempt Bond Handbook — Chapters 1–6, Appendices A–J, Glossary, Index\n"
        "2. WSHFC Bond Policies 2026 — Washington State Housing Finance Commission 4% bond program policies\n"
        "3. IRC Section 42 — Federal Low-Income Housing Tax Credit statute\n\n"
        "For each question, the most relevant excerpts from these documents are provided below. "
        "Answer naturally and directly — never mention retrieval, semantic search, embeddings, chunks, or indexing. "
        "Never tell the user a section wasn't 'retrieved' or 'surfaced.' "
        "When citing content, always include the specific section number or subsection (e.g., 'WSHFC Bond Policies Section 3.31', "
        "'IRC §42(g)(1)', 'Novogradac Chapter 2, Section 2.3'). These section numbers appear in the excerpts — use them. "
        "If the excerpts don't cover the question, just answer from your own expertise and note it's not from the provided documents.\n\n"
        "Relevant excerpts:\n\n"
        f"{context}"
    )

    messages = history + [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    return response.content[0].text, merged_metas, merged_chunks


def main():
    st.title("Novogradac RAG Chatbot")
    st.caption("Ask questions about affordable housing finance.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about affordable housing finance..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources, chunks = get_answer(prompt, history)
            st.markdown(answer)
            with st.expander("Sources"):
                for s, chunk in zip(sources, chunks):
                    first_line = chunk.split("\n")[1].strip() if "\n" in chunk else chunk[:80]
                    st.write(f"**{s['chapter']}** — {first_line[:100]}")

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
