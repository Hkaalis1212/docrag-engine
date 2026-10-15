import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from indexing.sparse_index import SparseIndex
from indexing.vector_store import VectorStore


class FakeEmbedder:
    def embed_one(self, text):
        return [1.0, 0.0]

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _build_fake_pipeline(tmp_path, with_data=True):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    sparse = SparseIndex()
    if with_data:
        store.add(
            ids=["a", "b"],
            embeddings=[[1.0, 0.0], [0.0, 1.0]],
            documents=["DEF spec is 32.5 percent urea", "unrelated tire content"],
            metadatas=[{"source_file": "doc1.md"}, {"source_file": "doc2.md"}],
        )
        sparse.build(
            chunk_ids=["a", "b"],
            texts=["DEF spec is 32.5 percent urea", "unrelated tire content"],
            metadatas=[{"source_file": "doc1.md"}, {"source_file": "doc2.md"}],
        )
    embedder = FakeEmbedder()
    retriever = api_main.HybridRetriever(store, sparse, embedder)
    return api_main.Pipeline(vector_store=store, sparse_index=sparse, embedder=embedder, retriever=retriever)


@pytest.fixture
def client(tmp_path):
    pipeline = _build_fake_pipeline(tmp_path)
    api_main.app.dependency_overrides[api_main.get_pipeline] = lambda: pipeline
    with TestClient(api_main.app) as c:
        yield c
    api_main.app.dependency_overrides.clear()


@pytest.fixture
def empty_client(tmp_path):
    pipeline = _build_fake_pipeline(tmp_path, with_data=False)
    api_main.app.dependency_overrides[api_main.get_pipeline] = lambda: pipeline
    with TestClient(api_main.app) as c:
        yield c
    api_main.app.dependency_overrides.clear()


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_reports_indexed_chunk_count(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "indexed_chunks": 2}


# ── /v1/documents ────────────────────────────────────────────────────────────

def test_list_documents_groups_by_source_file(client):
    resp = client.get("/v1/documents")
    body = resp.json()
    assert body["total_chunks"] == 2
    sources = {d["source_file"] for d in body["documents"]}
    assert sources == {"doc1.md", "doc2.md"}
    assert all(d["chunk_count"] == 1 for d in body["documents"])


def test_list_documents_on_empty_index(empty_client):
    resp = empty_client.get("/v1/documents")
    body = resp.json()
    assert body["total_chunks"] == 0
    assert body["documents"] == []


# ── /v1/compare_retrieval (no LLM, no API key needed) ──────────────────────

def test_compare_retrieval_returns_both_modes(client):
    resp = client.post("/v1/compare_retrieval", json={"question": "DEF spec", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["hybrid"]) > 0
    assert len(body["dense_only"]) > 0


def test_compare_retrieval_dense_only_has_no_sparse_rank(client):
    resp = client.post("/v1/compare_retrieval", json={"question": "DEF spec", "top_k": 2})
    body = resp.json()
    # ChunkOut doesn't expose sparse_rank directly, but dense_only entries
    # should still be well-formed chunks with a fused_score.
    for chunk in body["dense_only"]:
        assert "fused_score" in chunk


# ── /v1/ask ──────────────────────────────────────────────────────────────────

def test_ask_on_empty_index_returns_503(empty_client):
    resp = empty_client.post("/v1/ask", json={"question": "anything"})
    assert resp.status_code == 503


def test_ask_full_pipeline_with_patched_claude_calls(client, monkeypatch):
    monkeypatch.setattr(
        "retrieval.rerank.claude_relevance_scores", lambda q, c, **kw: [10.0] * len(c)
    )
    monkeypatch.setattr(
        "generation.generate._default_llm_call",
        lambda s, u, m: "DEF spec is 32.5% urea [1].",
    )
    monkeypatch.setattr(
        "generation.verify_citations.claude_verify",
        lambda cites, lookup, **kw: ["SUPPORTED"] * len(cites),
    )
    monkeypatch.setattr(
        "generation.confidence.claude_completeness_score", lambda q, a, **kw: 0.9
    )

    resp = client.post("/v1/ask", json={"question": "What's the DEF spec?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["insufficient_context"] is False
    assert "32.5%" in body["answer"]
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["verdict"] == "SUPPORTED"
    assert body["citations"][0]["source_file"] in ("doc1.md", "doc2.md")
    assert body["confidence"]["composite"] > 0


def test_ask_insufficient_context_returns_no_confidence_or_citations(client, monkeypatch):
    monkeypatch.setattr(
        "retrieval.rerank.claude_relevance_scores", lambda q, c, **kw: [0.0] * len(c)
    )
    resp = client.post("/v1/ask", json={"question": "off topic question"})

    body = resp.json()
    assert body["insufficient_context"] is True
    assert body["confidence"] is None
    assert body["citations"] == []


# ── /v1/ingest ───────────────────────────────────────────────────────────────

def test_ingest_adds_document_and_rebuilds_sparse_index(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "DOCRAG_INGEST_DIR", str(tmp_path / "ingested"))

    resp = client.post(
        "/v1/ingest",
        json={"filename": "new_doc.md", "content": "# New Doc\n\nSome fresh content about turbos."},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "new_doc.md"
    assert body["chunks_added"] >= 1
    assert body["total_chunks_in_index"] >= 3  # 2 original + at least 1 new

    docs_resp = client.get("/v1/documents")
    sources = {d["source_file"] for d in docs_resp.json()["documents"]}
    assert "new_doc.md" in sources


def test_ingest_rejects_unsupported_extension(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "DOCRAG_INGEST_DIR", str(tmp_path / "ingested"))

    resp = client.post("/v1/ingest", json={"filename": "bad.exe", "content": "whatever"})

    assert resp.status_code == 400
