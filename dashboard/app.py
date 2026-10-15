"""
dashboard/app.py
------------------
Streamlit frontend for docrag-engine. Talks to the FastAPI service over
HTTP (not by importing the pipeline directly) - the two are meant to run
as separate services, same as the rest of this project's Docker setup.

Run:
    streamlit run dashboard/app.py

Configuration (env var):
    DOCRAG_API_URL   Base URL of the FastAPI service (default: http://localhost:8000)
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("DOCRAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="docrag-engine", layout="wide")


def _get(path: str, **kwargs):
    resp = requests.get(f"{API_URL}{path}", timeout=30, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, json: dict, timeout: int = 60):
    resp = requests.post(f"{API_URL}{path}", json=json, timeout=timeout)
    if resp.status_code >= 400:
        detail = resp.json().get("detail", resp.text)
        raise RuntimeError(f"{resp.status_code}: {detail}")
    return resp.json()


# ── Sidebar: index status + document list + ingest form ────────────────────

with st.sidebar:
    st.markdown(f"**API:** `{API_URL}`")

    try:
        health = _get("/health")
        st.success(f"Index online - {health['indexed_chunks']} chunks")
    except Exception as e:
        st.error(f"Can't reach API: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("Indexed documents")
    try:
        docs = _get("/v1/documents")
        for d in docs["documents"]:
            st.caption(f"{d['source_file']} — {d['chunk_count']} chunks")
    except Exception as e:
        st.warning(f"Couldn't load document list: {e}")

    st.markdown("---")
    st.subheader("Ingest a document")
    with st.form("ingest_form", clear_on_submit=True):
        filename = st.text_input("Filename", placeholder="my-doc.md")
        content = st.text_area("Content (markdown or plain text)", height=120)
        submitted = st.form_submit_button("Ingest")
        if submitted and filename and content:
            try:
                result = _post("/v1/ingest", {"filename": filename, "content": content})
                st.success(f"Added {result['chunks_added']} chunks from {result['filename']}")
            except Exception as e:
                st.error(f"Ingest failed: {e}")


# ── Main: ask a question ────────────────────────────────────────────────────

st.title("docrag-engine")
st.caption("Hybrid-retrieval RAG over internal documentation")

question = st.text_input("Ask a question about the indexed docs", placeholder="What is the DEF concentration spec?")
compare_mode = st.checkbox("Compare hybrid vs. dense-only retrieval (no generation, retrieval only)")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question:
    if compare_mode:
        with st.spinner("Retrieving..."):
            try:
                result = _post("/v1/compare_retrieval", {"question": question, "top_k": 5})
            except Exception as e:
                st.error(f"Request failed: {e}")
                result = None

        if result:
            col1, col2 = st.columns(2)
            for col, key, label in [(col1, "hybrid", "Hybrid (dense + BM25 + RRF)"), (col2, "dense_only", "Dense only")]:
                with col:
                    st.subheader(label)
                    for i, chunk in enumerate(result[key], start=1):
                        with st.expander(f"{i}. {chunk['source_file']} (score: {chunk['fused_score']:.3f})"):
                            st.write(chunk["text"])

    else:
        with st.spinner("Thinking..."):
            try:
                result = _post("/v1/ask", {"question": question})
            except Exception as e:
                st.error(f"Request failed: {e}")
                result = None

        if result:
            if result["insufficient_context"]:
                st.info(result["answer"])
            else:
                st.markdown("### Answer")
                st.write(result["answer"])

                if result["confidence"]:
                    st.markdown("### Confidence")
                    c = result["confidence"]
                    cols = st.columns(4)
                    cols[0].metric("Retrieval", f"{c['retrieval_confidence']:.2f}")
                    cols[1].metric("Citation coverage", f"{c['citation_coverage']:.2f}")
                    cols[2].metric("Completeness", f"{c['completeness']:.2f}")
                    cols[3].metric("Composite", f"{c['composite']:.2f}")

                if result["citations"]:
                    st.markdown("### Citations")
                    for c in result["citations"]:
                        verdict = c.get("verdict") or "unverified"
                        icon = {"SUPPORTED": "✅", "PARTIAL": "⚠️", "UNSUPPORTED": "❌"}.get(verdict, "❔")
                        with st.expander(f"{icon} [{c['marker']}] {c['source_file']} — {verdict}"):
                            st.write(f"**Claim:** {c['claim_text']}")

                st.markdown("### Retrieved passages")
                for i, chunk in enumerate(result["used_chunks"], start=1):
                    score_label = (
                        f"rerank: {chunk['rerank_score']:.1f}"
                        if chunk.get("rerank_score") is not None
                        else f"fused: {chunk['fused_score']:.3f}"
                    )
                    with st.expander(f"[{i}] {chunk['source_file']} ({score_label})"):
                        st.write(chunk["text"])
elif ask_clicked:
    st.warning("Enter a question first.")
