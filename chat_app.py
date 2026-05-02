import os
import streamlit as st
from anthropic import Anthropic

from index_pipeline import query_index


def get_answer(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    context, sources = query_index(question)
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    system = (
        "You are a senior tax credit accountant specializing in affordable housing finance in Washington State. "
        "You have deep expertise in Low Income Housing Tax Credits (LIHTC), tax-exempt bond financing, "
        "HUD programs, and Washington-specific agencies including the Washington State Housing Finance Commission (WSHFC). "
        "You speak precisely and practically — your audience is housing finance professionals who need accurate, "
        "actionable answers, not general overviews. Where relevant, note Washington State-specific rules, "
        "WSHFC requirements, or how federal rules interact with state practice.\n\n"
        "Use the following excerpts from the Novogradac 2022 Tax Exempt Bond Handbook to ground your answer. "
        "If the excerpts don't contain enough information, draw on your expertise but clearly distinguish "
        "textbook content from your own knowledge.\n\n"
        f"{context}"
    )

    messages = history + [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=messages,
    )
    return response.content[0].text, sources


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
                answer, sources = get_answer(prompt, history)
            st.markdown(answer)
            with st.expander("Sources"):
                for s in sources:
                    st.write(f"**{s['chapter']}** — {s['filename']}")

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
