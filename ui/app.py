"""Flask server for the retrieval test UI.

Wires the "Test du retrieval" mockup to the real hybrid_search() pipeline —
no more static example cards, every query hits the actual 3-jurisdiction index
(or the labelled fallback corpus if index_data/ has not been built yet).

Run:
    python ui/app.py
Then open http://127.0.0.1:5050
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from retrieval import hybrid_search  # noqa: E402
from eval_retrieval import GOLD, evaluate  # noqa: E402

app = Flask(__name__)

VALID_MODES = {"baseline", "hybrid", "full"}
VALID_JURISDICTIONS = {"all", "eu", "us", "uk"}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/search")
def api_search():
    payload = request.get_json(force=True, silent=True) or {}
    query = (payload.get("query") or "").strip()
    jurisdiction = (payload.get("jurisdiction") or "all").lower()
    mode = (payload.get("mode") or "full").lower()
    try:
        top_k = max(1, min(20, int(payload.get("top_k") or 10)))
    except (TypeError, ValueError):
        top_k = 10

    if not query:
        return jsonify({"error": "query is required"}), 400
    if mode not in VALID_MODES:
        return jsonify({"error": f"mode must be one of {sorted(VALID_MODES)}"}), 400
    if jurisdiction not in VALID_JURISDICTIONS:
        return jsonify({"error": f"jurisdiction must be one of {sorted(VALID_JURISDICTIONS)}"}), 400

    t0 = time.perf_counter()
    results = hybrid_search(query, top_k=top_k, jurisdiction=jurisdiction, mode=mode)
    latency_ms = (time.perf_counter() - t0) * 1000

    items = []
    for rank, r in enumerate(results, start=1):
        title, _, tag = r.document.title.partition(" — ")
        items.append({
            "rank": rank,
            "score": round(r.score, 3),
            "title": title,
            "tag": tag,
            "jurisdiction": r.document.jurisdiction or "?",
            "status": r.document.status or "inconnu",
            "text": r.document.text,
            "context": r.document.context,
            "method": r.method,
        })

    return jsonify({"results": items, "latency_ms": round(latency_ms)})


@app.get("/api/eval")
def api_eval():
    """Re-run the 12-question gold-set evaluation (tolerant scoring) for one mode."""
    mode = request.args.get("mode", "full").lower()
    if mode not in VALID_MODES:
        return jsonify({"error": f"mode must be one of {sorted(VALID_MODES)}"}), 400

    metrics = evaluate(mode, strict=False)
    return jsonify({
        "hit_at_3": round(metrics["hit@3"], 2),
        "mrr": round(metrics["MRR"], 2),
        "avg_latency_ms": round(metrics["avg_latency_s"] * 1000),
        "n_questions": len(GOLD),
        "mode": mode,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
