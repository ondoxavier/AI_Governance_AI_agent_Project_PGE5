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
from hashlib import blake2b
import json
from math import sqrt
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
    # ── US ──
    "EO_14110_Biden_AI_Framework": "2023-10-30",
    "EO_14179_Trump_Deregulation": "2025-01-23",
    "EO_14365_National_Policy_Framework": "2025-12-11",
    "NIST_AI_RMF_1.0": "2023-01-26",
    "NIST_AI_RMF_Playbook": "mise à jour continue",
    "NIST_Generative_AI_Profile": "2024-07",
    "Blueprint_AI_Bill_of_Rights": "2022-10-04",
    "Colorado_SB24-205_Signed_Act": "2024-05-17",
    # ── UK ──
    "UK_White_Paper_Mar2023": "2023-03",
    "UK_Government_Response_Feb2024": "2024-02-06",
    # ── EU — AI Act (Regulation (EU) 2024/1689, entry into force 2024-08-01;
    #    per-article "date of entry into force" 2026/2027 is preserved in the
    #    chunk text itself, this date is the instrument's own adoption date) ──
    "Article6_AIAct": "2024-08-01",
    "Article9_AIAct": "2024-08-01",
    "Article10_AIAct": "2024-08-01",
    "Article13_AIAct": "2024-08-01",
    "Article14_AIAct": "2024-08-01",
    "AI_act_articles": "2024-08-01",
    "20230601STO93804_en": "2023-06-01",
    "240827_FINAL_AI_ACT_Enforcement": "2024-08-27",
    "67_Artificial_Intelligence_Act_AI_Act_3d29a6adb6": "2024-10",
    "AL_Goodbody_-_Guide_to_the_AI_Act": "2024",
    "ELI_Response_on_the_definition_of_an_AI_System": "2024",
    "EU-AI-Act-Navigating-a-Brave-New-World": "2024-07",
    "European Union Artificial Intelligence Act Guide": "2025-04-07",
    "decoding-eu-ai-act": "2024",
    "en-pdf-file-ai-act-guide": "2024-02",
    "ey-gl-eu-ai-act-07-2024": "2024-07-12",
    # ── EU — GDPR (Regulation (EU) 2016/679, Official Journal L 119, 4.5.2016) ──
    "Article4_GDPR": "2016-05-04",
    "Article5_GDPR": "2016-05-04",
    "Article6_GDPR": "2016-05-04",
    "Article9_GDPR": "2016-05-04",
    "Article12_GDPR": "2016-05-04",
    "Article22_GDPR": "2016-05-04",
    "Article25_GDPR": "2016-05-04",
    "Article30_GDPR": "2016-05-04",
    "Article32_GDPR": "2016-05-04",
    "Article35_GDPR": "2016-05-04",
    "Article37_GDPR": "2016-05-04",
    "GDPR_recitals": "2016-05-04",
    "GDPR_recitals_official": "2016-05-04",
    "21-04-27_aepd-edps_anonymisation_en_5": "2021-04-27",
    "artificial-intelligence-systems-and-the-gdpr---a-data-protection-perspective-december": "2024-12",
    "cnil_guide_securite_personnelle_ven_0": "2024",
    "edpb_opinion_202428_ai-models_en": "2024-12-17",
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

    return chunks


# ── Step 3: embeddings ────────────────────────────────────────────────────────

def _hash_vector(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    for token in re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", text.casefold()):
        digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
        vector[int.from_bytes(digest, "big") % dimensions] += 1.0
    norm = sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_chunks(chunks: list[dict], backend: str):
    import numpy as np
    texts = [c["text"] for c in chunks]
    if backend == "hash":
        emb = np.asarray([_hash_vector(text) for text in texts], dtype="float32")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype("float32")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-extract PDFs and rebuild the index")
    parser.add_argument(
        "--embedding-backend",
        choices=("hash", "transformer"),
        default="hash",
        help="hash is autonomous; transformer downloads the configured SentenceTransformer model",
    )
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
    embeddings = embed_chunks(chunks, args.embedding_backend)

    INDEX_DIR.mkdir(exist_ok=True)
    import numpy as np
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "index_meta.json").write_text(
        json.dumps({"embedding_backend": args.embedding_backend, "model": EMBED_MODEL if args.embedding_backend == "transformer" else None}),
        encoding="utf-8",
    )
    print(f"\nIndex written: {INDEX_DIR}/embeddings.npy "
          f"({embeddings.shape[0]} x {embeddings.shape[1]}) + chunks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
