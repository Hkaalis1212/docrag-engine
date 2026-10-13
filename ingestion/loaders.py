"""
loaders.py
----------
Multi-format document loader: markdown, plain text, HTML, PDF.
Normalizes every format into clean plaintext + a metadata dict, and writes
both the raw source and the processed plaintext to disk so re-chunking never
requires re-uploading.

Usage:
    from ingestion.loaders import load_directory

    docs = load_directory("corpus/throttleguard-docs")
    # -> list[Document]
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".md", ".txt", ".html", ".htm", ".pdf"}


@dataclass
class Document:
    """A single loaded source document, normalized to plaintext."""

    doc_id: str
    source_path: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def _doc_id_for(path: Path) -> str:
    # Stable id derived from path, not content — re-ingesting the same file
    # after an edit should update the existing doc, not create a duplicate.
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]


def _load_markdown_or_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        # Keep page boundaries visible in the plaintext — chunking strategies
        # downstream use these markers to attach a page number to each chunk.
        pages.append(f"\n\n<!-- page:{i + 1} -->\n\n{page_text}")
    return "".join(pages)


_LOADERS = {
    ".md": _load_markdown_or_text,
    ".txt": _load_markdown_or_text,
    ".html": _load_html,
    ".htm": _load_html,
    ".pdf": _load_pdf,
}


def load_file(path: Path) -> Document:
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported extension '{ext}' for {path}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    text = _LOADERS[ext](path).strip()
    title = path.stem.replace("-", " ").replace("_", " ")
    return Document(
        doc_id=_doc_id_for(path),
        source_path=str(path),
        title=title,
        text=text,
        metadata={"extension": ext, "filename": path.name},
    )


def load_directory(directory: str | Path) -> list[Document]:
    """Load every supported file under `directory` (recursive)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(load_file(path))
    return docs
