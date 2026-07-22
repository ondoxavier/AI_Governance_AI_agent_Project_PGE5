"""Ingestion pipeline: PDF corpus -> text -> legal-aware parent/child chunks -> embeddings.

Usage:
    python src/ingest.py            # incremental (skips already-extracted PDFs)
    python src/ingest.py --force    # re-extract and re-index everything

Reads   data/<corpus_dir>/*.pdf  (+ .md/.txt files)
Writes  processed_txt/<corpus_dir>/*.txt   (page-marked extractions, cached)
        index_data/chunks.json             (child chunks + metadata + parent text)
        index_data/embeddings.npy          (float32, L2-normalised, one row per chunk)

The index is loaded by src/retrieval.py. Both files are regenerable and
git-ignored; run this script once after cloning to enable the full RAG.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from constants import UNKNOWN_DATE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TXT_DIR = ROOT / "processed_txt"
INDEX_DIR = ROOT / "index_data"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Map corpus folders to a jurisdiction label used for filtered retrieval.
JURISDICTION_BY_DIR = {
    "ai_act_corpus": "EU",
    "gdpr_corpus": "EU",
    "us_ai_regulation_corpus": "US",
    "uk_ai_regulation_corpus": "UK",
}

# Regulatory status is attached to every chunk so the reasoning layer can
# refuse to present voluntary guidance as binding law (see data/*/README.md).
STATUS_BY_DIR = {
    "ai_act_corpus": "obligatoire",
    "gdpr_corpus": "obligatoire",
    "us_ai_regulation_corpus": "variable",   # refined per document below
    "uk_ai_regulation_corpus": "recommandation",
}
STATUS_BY_DOC_HINT = {
    # US: executive orders are binding on federal agencies, NIST/Blueprint are voluntary,
    # Colorado SB24-205 is a binding state law.
    "eo_": "obligatoire (federal, revocable)",
    "colorado": "obligatoire (etat)",
    "nist": "volontaire",
    "blueprint": "recommandation",
}
DATE_BY_DOC_ID = {
    "EO_14110_Biden_AI_Framework": "2023-10-30",
    "EO_14179_Trump_Deregulation": "2025-01-23",
    "EO_14365_National_Policy_Framework": "2025-12-11",
    "NIST_AI_RMF_1.0": "2023-01-26",
    "NIST_AI_RMF_Playbook": "mise à jour continue",
    "NIST_Generative_AI_Profile": "2024-07",
    "Blueprint_AI_Bill_of_Rights": "2022-10-04",
    "Colorado_SB24-205_Signed_Act": "2024-05-17",
    "UK_White_Paper_Mar2023": "2023-03",
    "UK_Government_Response_Feb2024": "2024-02-06",
}

# ── Legal structure detection (from the CIE 3.1 project, regexes kept as-is) ──
RE_CHAPTER = re.compile(r"^(?:chapter|chapitre|title|titre)\s+([IVXLCDMivxlcdm]+|\d+)\b", re.IGNORECASE)
RE_ARTICLE = re.compile(r"^(?:article|art\.?|section|sec\.?|§)\s+(\d+[a-z]?)\b", re.IGNORECASE)
RE_RECITAL = re.compile(r"^(?:\((\d+)\)|recital\s+(\d+)|considérant\s+(\d+))", re.IGNORECASE)
_HEADER_TITLE_MAX_LEN = 80

CHILD_CHUNK_WORDS = 180   # target size of an indexed (child) chunk
CHILD_OVERLAP_WORDS = 30
PARENT_MAX_CHARS = 4000   # cap on the parent context stored alongside each child


# ── Step 1: PDF -> text ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text with page markers (PyMuPDF)."""
    import fitz  # imported lazily so --help works without the dependency

    doc = fitz.open(pdf_path)
    pages = [f"--- Page {i + 1} ---\n{page.get_text()}" for i, page in enumerate(doc)]
    doc.close()
    return "\n".join(pages)


def extract_all_pdfs(force: bool = False) -> int:
    converted = 0
    for corpus_dir in sorted(DATA_DIR.iterdir()):
        if not corpus_dir.is_dir():
            continue
        out_dir = TXT_DIR / corpus_dir.name
        pdfs = sorted(corpus_dir.glob("*.pdf"))
        if not pdfs:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[extract] {corpus_dir.name} ({len(pdfs)} PDFs)")
        for pdf in pdfs:
            txt_path = out_dir / (pdf.stem + ".txt")
            if txt_path.exists() and not force:
                continue
            try:
                text = extract_pdf_text(pdf)
            except Exception as exc:
                print(f"  !! {pdf.name}: extraction failed ({exc})", file=sys.stderr)
                continue
            txt_path.write_text(text, encoding="utf-8")
            converted += 1
            print(f"  ok {pdf.name} -> {txt_path.name}")
    return converted


# ── Step 2: legal-aware parent/child chunking ────────────────────────────────

def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[ \t]+", " ", text)


def split_into_legal_blocks(text: str, detect_recitals: bool = True) -> list[dict]:
    """Split a page-marked document into parent blocks along legal headers.

    Returns [{chapter, article, text}] — one entry per Article/Recital/Section
    block; documents without detectable structure yield one block per ~15 lines
    of running text (handled by the caller's fallback).

    detect_recitals: recitals "(45)" only exist in EU legal texts — disabled for
    US/UK corpora where the pattern would falsely match years like "(2022)".
    """
    blocks: list[dict] = []
    current = {"chapter": "", "article": "", "lines": []}

    def flush():
        body = "\n".join(current["lines"]).strip()
        if len(body) > 50:
            blocks.append({"chapter": current["chapter"], "article": current["article"], "text": body})
        current["lines"] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if line.startswith("--- Page"):
            continue
        chap = RE_CHAPTER.match(line)
        art = RE_ARTICLE.match(line)
        rec = RE_RECITAL.match(line) if detect_recitals else None
        if chap:
            flush()
            title = line[chap.end():].strip(" :-—–.\t")
            current["chapter"] = f"Chapter {chap.group(1)}"
            if title and len(title) <= _HEADER_TITLE_MAX_LEN and not title.endswith((".", ",", ";")):
                current["chapter"] += f" — {title}"
            current["article"] = ""
        elif art:
            flush()
            current["article"] = f"Article {art.group(1)}"
        elif rec:
            flush()
            rid = next((g for g in rec.groups() if g), "?")
            current["article"] = f"Recital {rid}"
        else:
            current["lines"].append(raw_line)
    flush()
    return blocks


def split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


def build_chunks() -> list[dict]:
    """Chunk every extracted .txt (and native .md) into child chunks with parent context."""
    chunks: list[dict] = []

    for corpus_dir in sorted(TXT_DIR.iterdir()) if TXT_DIR.exists() else []:
        if not corpus_dir.is_dir():
            continue
        jurisdiction = JURISDICTION_BY_DIR.get(corpus_dir.name, "OTHER")
        base_status = STATUS_BY_DIR.get(corpus_dir.name, "inconnu")

        for txt in sorted(corpus_dir.glob("*.txt")):
            doc_id = txt.stem
            status = base_status
            date = DATE_BY_DOC_ID.get(doc_id, UNKNOWN_DATE)
            if base_status == "variable":
                low = doc_id.lower()
                status = next((s for hint, s in STATUS_BY_DOC_HINT.items() if hint in low), "variable")

            text = _clean(txt.read_text(encoding="utf-8"))
            blocks = split_into_legal_blocks(text, detect_recitals=(jurisdiction == "EU"))
            if not blocks:  # unstructured document -> whole text as parent blocks
                blocks = [{"chapter": "", "article": "", "text": t}
                          for t in split_words(text, CHILD_CHUNK_WORDS * 4, 0)]

            for b_idx, block in enumerate(blocks):
                parent_text = block["text"][:PARENT_MAX_CHARS]
                for c_idx, child in enumerate(split_words(block["text"], CHILD_CHUNK_WORDS, CHILD_OVERLAP_WORDS)):
                    chunks.append({
                        "chunk_id": f"{doc_id}#{b_idx}.{c_idx}",
                        "doc_id": doc_id,
                        "corpus": corpus_dir.name,
                        "jurisdiction": jurisdiction,
                        "status": status,
                        "date": date,
                        "chapter": block["chapter"],
                        "article": block["article"],
                        "text": child,
                        "parent_text": parent_text,
                    })

    # Also index loose .md knowledge files at the top of data/ (e.g. ai_act_reference.md)
    for md in sorted(DATA_DIR.glob("*.md")):
        if md.name.casefold() == "readme.md":
            continue
        text = _clean(md.read_text(encoding="utf-8"))
        for i, child in enumerate(split_words(text, CHILD_CHUNK_WORDS, CHILD_OVERLAP_WORDS)):
            chunks.append({
                "chunk_id": f"{md.stem}#0.{i}", "doc_id": md.stem, "corpus": "data",
                "jurisdiction": "EU", "status": "resume non officiel",
                "date": UNKNOWN_DATE,
                "chapter": "", "article": "", "text": child, "parent_text": text[:PARENT_MAX_CHARS],
            })
    return chunks


# ── Step 3: embeddings ────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-extract PDFs and rebuild the index")
    args = parser.parse_args()

    print("=== Step 1/3 — PDF extraction ===")
    n = extract_all_pdfs(force=args.force)
    print(f"{n} new file(s) extracted (cached files skipped)\n")

    print("=== Step 2/3 — Legal-aware chunking ===")
    chunks = build_chunks()
    if not chunks:
        print("No chunks produced — is data/ populated and step 1 done?", file=sys.stderr)
        return 1
    by_jur: dict[str, int] = {}
    for c in chunks:
        by_jur[c["jurisdiction"]] = by_jur.get(c["jurisdiction"], 0) + 1
    print(f"{len(chunks)} child chunks ({', '.join(f'{k}: {v}' for k, v in sorted(by_jur.items()))})\n")

    print("=== Step 3/3 — Embeddings ===")
    embeddings = embed_chunks(chunks)

    INDEX_DIR.mkdir(exist_ok=True)
    import numpy as np
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"\nIndex written: {INDEX_DIR}/embeddings.npy "
          f"({embeddings.shape[0]} x {embeddings.shape[1]}) + chunks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
