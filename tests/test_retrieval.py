"""Retrieval unit tests.

Two groups:
- pure-python tests (tokenize, RRF, parent/child split, fallback pipeline) —
  always run, no index or ML dependency needed;
- index-mode tests — skipped automatically when index_data/ has not been
  built yet (fresh clone) or ML dependencies are missing.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import retrieval
from retrieval import (
    Document,
    SearchResult,
    bm25_rank,
    dense_rank,
    hybrid_search,
    rrf_fusion,
    split_parent_child,
    tokenize,
)


# ── Pure-python tests (always run) ────────────────────────────────────────────

def test_tokenize_casefolds_and_keeps_accents():
    assert tokenize("Résumé de l'Article 5") == ["résumé", "de", "l", "article", "5"]


def test_rrf_fusion_rewards_docs_present_in_both_rankings():
    shared = Document("shared", "t", "x", "s")
    only_a = Document("only-a", "t", "x", "s")
    only_b = Document("only-b", "t", "x", "s")
    ranking_a = [SearchResult(only_a, 5.0, "bm25"), SearchResult(shared, 1.0, "bm25")]
    ranking_b = [SearchResult(only_b, 5.0, "dense"), SearchResult(shared, 1.0, "dense")]
    fused = rrf_fusion([ranking_a, ranking_b])
    # 'shared' is rank 2 in both lists: 2/(60+2) > 1/(60+1) for single-list docs
    assert fused[0].document.doc_id == "shared"
    assert "bm25" in fused[0].method and "dense" in fused[0].method


def test_split_parent_child_links_children_to_parent(tmp_path):
    path = tmp_path / "reglement_test.md"
    text = "# Titre du règlement\n" + " ".join(f"mot{i}" for i in range(300))
    path.write_text(text, encoding="utf-8")
    chunks = split_parent_child(path, path.read_text(encoding="utf-8"))
    assert len(chunks) >= 2
    assert all(c.parent_id == "reglement_test" for c in chunks)
    assert all(c.title == "Titre du règlement" for c in chunks)


def _force_fallback(monkeypatch):
    """Disable the index singleton so hybrid_search takes the fallback path,
    even on machines where index_data/ has been built."""
    monkeypatch.setattr(retrieval._IndexRetriever, "get", classmethod(lambda cls: None))


def test_fallback_hybrid_search_returns_results(tmp_path, monkeypatch):
    _force_fallback(monkeypatch)
    # Point data_dir at an empty folder -> built-in demo corpus, no index needed
    results = hybrid_search("systèmes IA à haut risque emploi crédit",
                            top_k=2, data_dir=tmp_path)
    assert len(results) == 2
    assert results[0].document.doc_id == "ai-act-high-risk"


def test_fallback_baseline_mode_skips_bm25_and_rerank(tmp_path, monkeypatch):
    _force_fallback(monkeypatch)
    results = hybrid_search("transparence contenu synthétique",
                            top_k=2, data_dir=tmp_path, mode="baseline")
    assert results
    assert all("baseline" in r.method for r in results)
    assert all("rerank" not in r.method for r in results)


def test_bm25_rank_prefers_matching_document():
    docs = [
        Document("a", "t", "le chat mange la souris", "s"),
        Document("b", "t", "les obligations de transparence des systèmes IA", "s"),
    ]
    top = bm25_rank("obligations de transparence", docs)[0]
    assert top.document.doc_id == "b"


def test_dense_rank_returns_all_documents_scored():
    docs = [Document(str(i), "t", f"texte {i}", "s") for i in range(3)]
    ranked = dense_rank("texte", docs)
    assert len(ranked) == 3


# ── Index-mode tests (skipped on fresh clones) ────────────────────────────────

def _index_available() -> bool:
    return retrieval._IndexRetriever.get() is not None


needs_index = pytest.mark.skipif(
    not _index_available(),
    reason="index_data/ absent ou dépendances ML manquantes — lancer python src/ingest.py",
)


@needs_index
def test_index_search_returns_metadata_and_parent_context():
    results = hybrid_search("high-risk AI systems classification", top_k=3)
    assert len(results) == 3
    for r in results:
        assert r.document.jurisdiction in {"EU", "US", "UK"}
        assert r.document.status
        assert r.document.context  # parent legal block attached
        assert "rrf" in r.method


@needs_index
def test_index_jurisdiction_filter_is_strict():
    for jur in ("EU", "US", "UK"):
        results = hybrid_search("risk management requirements", top_k=5, jurisdiction=jur)
        assert results, f"aucun résultat pour {jur}"
        assert all(r.document.jurisdiction == jur for r in results)


@needs_index
def test_index_baseline_mode_is_dense_only():
    results = hybrid_search("human oversight requirements", top_k=3, mode="baseline")
    assert results
    assert all(r.method == "dense-only (baseline)" for r in results)
