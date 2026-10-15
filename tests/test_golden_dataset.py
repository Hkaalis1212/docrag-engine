import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_dataset.jsonl"

VALID_CATEGORIES = {"direct", "multi_hop", "unanswerable", "ambiguous"}
REQUIRED_KEYS = {"id", "question", "golden_answer", "category", "source_docs", "expect_insufficient"}
KNOWN_SOURCE_DOCS = {
    "claude-md-overview.md",
    "scoring-engine.md",
    "passive-regen.md",
    "telematics-ingestion.md",
    "subscription-billing.md",
}


def _load_rows():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_golden_dataset_has_at_least_50_cases():
    rows = _load_rows()
    assert len(rows) >= 50


def test_every_row_has_required_keys():
    for row in _load_rows():
        missing = REQUIRED_KEYS - set(row.keys())
        assert not missing, f"{row.get('id')} missing keys: {missing}"


def test_every_row_has_a_valid_category():
    for row in _load_rows():
        assert row["category"] in VALID_CATEGORIES, f"{row['id']}: bad category {row['category']!r}"


def test_ids_are_unique():
    ids = [row["id"] for row in _load_rows()]
    assert len(ids) == len(set(ids))


def test_source_docs_reference_known_corpus_files():
    for row in _load_rows():
        for doc in row["source_docs"]:
            assert doc in KNOWN_SOURCE_DOCS, f"{row['id']}: unknown source doc {doc!r}"


def test_unanswerable_cases_expect_insufficient_and_have_no_source_docs():
    for row in _load_rows():
        if row["category"] == "unanswerable":
            assert row["expect_insufficient"] is True, row["id"]
            assert row["source_docs"] == [], row["id"]


def test_direct_and_multi_hop_cases_do_not_expect_insufficient():
    for row in _load_rows():
        if row["category"] in ("direct", "multi_hop"):
            assert row["expect_insufficient"] is False, row["id"]
            assert len(row["source_docs"]) >= 1, row["id"]


def test_multi_hop_cases_reference_at_least_two_source_docs():
    for row in _load_rows():
        if row["category"] == "multi_hop":
            assert len(row["source_docs"]) >= 2, row["id"]


def test_direct_cases_reference_exactly_one_source_doc():
    for row in _load_rows():
        if row["category"] == "direct":
            assert len(row["source_docs"]) == 1, row["id"]


def test_all_four_categories_are_represented():
    categories = {row["category"] for row in _load_rows()}
    assert categories == VALID_CATEGORIES


def test_all_five_corpus_docs_are_referenced_at_least_once():
    referenced = set()
    for row in _load_rows():
        referenced.update(row["source_docs"])
    assert referenced == KNOWN_SOURCE_DOCS


def test_no_empty_questions_or_answers():
    for row in _load_rows():
        assert row["question"].strip(), row["id"]
        assert row["golden_answer"].strip(), row["id"]
