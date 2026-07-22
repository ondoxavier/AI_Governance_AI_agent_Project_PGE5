"""Hybrid retrieval: BM25, local dense vectors, RRF and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import log, sqrt
from pathlib import Path
import re


TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9_]+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    parent_id: str | None = None


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    method: str


DEFAULT_CORPUS = [
    Document(
        doc_id="ai-act-high-risk",
        title="AI Act - Haut risque",
        source="corpus intégré",
        text=(
            "Les systèmes IA utilisés dans l'emploi, l'éducation, le crédit, "
            "les services essentiels, la migration, la justice ou les infrastructures "
            "critiques peuvent relever du haut risque. Les obligations incluent "
            "gestion des risques, gouvernance des données, documentation, traçabilité, "
            "supervision humaine, robustesse et surveillance post-déploiement."
        ),
    ),
    Document(
        doc_id="ai-act-limited-risk",
        title="AI Act - Risque limité",
        source="corpus intégré",
        text=(
            "Les systèmes IA qui interagissent avec des personnes ou produisent "
            "du contenu synthétique imposent des obligations de transparence. "
            "L'utilisateur doit être informé lorsqu'il interagit avec une IA."
        ),
    ),
    Document(
        doc_id="ai-act-prohibited",
        title="AI Act - Usages interdits",
        source="corpus intégré",
        text=(
            "Les usages interdits comprennent la manipulation subliminale, "
            "l'exploitation de vulnérabilités, certaines notations sociales et "
            "certaines identifications biométriques à distance en temps réel."
        ),
    ),
]


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text or "")]


def load_corpus(data_dir: str | Path = "data") -> list[Document]:
    """Load Markdown and text files from data; fallback to the built-in corpus."""
    root = Path(data_dir)
    documents: list[Document] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.suffix.casefold() not in {".md", ".txt"}:
                continue
            if path.name.casefold() == "readme.md":
                continue
            text = path.read_text(encoding="utf-8")
            documents.extend(split_parent_child(path, text))
    return documents or DEFAULT_CORPUS


def split_parent_child(path: Path, text: str, chunk_words: int = 120) -> list[Document]:
    """Create child chunks while preserving a parent document identifier."""
    words = text.split()
    parent_id = path.stem
    heading = next(
        (line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")),
        "",
    )
    title = heading or path.stem
    chunks: list[Document] = []
    for index in range(0, len(words), chunk_words):
        chunk = " ".join(words[index : index + chunk_words])
        if chunk.strip():
            chunks.append(
                Document(
                    doc_id=f"{parent_id}-{index // chunk_words}",
                    parent_id=parent_id,
                    title=title,
                    text=chunk,
                    source=str(path),
                )
            )
    return chunks


def bm25_rank(query: str, documents: list[Document]) -> list[SearchResult]:
    query_terms = tokenize(query)
    doc_terms = [tokenize(doc.text) for doc in documents]
    avg_len = sum(len(terms) for terms in doc_terms) / max(1, len(doc_terms))
    k1 = 1.5
    b = 0.75
    results: list[SearchResult] = []
    for doc, terms in zip(documents, doc_terms):
        score = 0.0
        term_counts = {term: terms.count(term) for term in set(terms)}
        for term in query_terms:
            containing = sum(1 for candidate in doc_terms if term in candidate)
            if containing == 0:
                continue
            idf = log(1 + (len(documents) - containing + 0.5) / (containing + 0.5))
            tf = term_counts.get(term, 0)
            denom = tf + k1 * (1 - b + b * len(terms) / max(1, avg_len))
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
        results.append(SearchResult(doc, score, "bm25"))
    return sorted(results, key=lambda item: item.score, reverse=True)


def dense_vector(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
        bucket = int.from_bytes(digest, "big") % dimensions
        vector[bucket] += 1.0
    norm = sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def dense_rank(query: str, documents: list[Document]) -> list[SearchResult]:
    query_vector = dense_vector(query)
    return sorted(
        [
            SearchResult(doc, cosine(query_vector, dense_vector(doc.text)), "dense")
            for doc in documents
        ],
        key=lambda item: item.score,
        reverse=True,
    )


def rrf_fusion(rankings: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    methods: dict[str, list[str]] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            docs[result.document.doc_id] = result.document
            methods.setdefault(result.document.doc_id, []).append(result.method)
            scores[result.document.doc_id] = scores.get(result.document.doc_id, 0.0) + 1 / (k + rank)
    return sorted(
        [
            SearchResult(docs[doc_id], score, "+".join(sorted(set(methods[doc_id]))))
            for doc_id, score in scores.items()
        ],
        key=lambda item: item.score,
        reverse=True,
    )


def cross_encoder_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Deterministic reranker approximating cross-encoder relevance locally."""
    query_terms = set(tokenize(query))
    reranked: list[SearchResult] = []
    for result in results:
        doc_terms = set(tokenize(result.document.text))
        overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
        phrase_bonus = 0.1 if query.casefold()[:20] in result.document.text.casefold() else 0.0
        reranked.append(
            SearchResult(result.document, result.score + overlap + phrase_bonus, result.method + "+rerank")
        )
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def hybrid_search(query: str, top_k: int = 4, data_dir: str | Path = "data") -> list[SearchResult]:
    documents = load_corpus(data_dir)
    fused = rrf_fusion([bm25_rank(query, documents), dense_rank(query, documents)])
    return cross_encoder_rerank(query, fused)[:top_k]
