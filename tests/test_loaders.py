import pytest

from ingestion.loaders import load_directory, load_file


def test_load_directory_finds_all_markdown_docs(tmp_path):
    (tmp_path / "a.md").write_text("# Title\n\nSome content.")
    (tmp_path / "b.txt").write_text("Plain text content.")
    (tmp_path / "ignored.json").write_text("{}")

    docs = load_directory(tmp_path)

    assert len(docs) == 2
    titles = {d.title for d in docs}
    assert titles == {"a", "b"}


def test_load_file_assigns_stable_doc_id(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("content")

    doc1 = load_file(path)
    doc2 = load_file(path)

    assert doc1.doc_id == doc2.doc_id


def test_load_file_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "doc.exe"
    path.write_text("binary-ish")

    with pytest.raises(ValueError):
        load_file(path)


def test_load_directory_on_missing_dir_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_directory(tmp_path / "does-not-exist")
