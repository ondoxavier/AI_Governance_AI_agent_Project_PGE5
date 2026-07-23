"""Service layer: filesystem summaries, safe adapters and bounded analysis jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from inspect import signature
import json
from pathlib import Path
from queue import Empty, Queue
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import uuid4

try:
    from src.agent import AgentResponse, run_agent
    from src.constants import DISCLAIMER
    from src.mcp_server import MCP_TOOL_REGISTRY
    from src.observability import create_tracer
    from src.web_models import AnalysisRequest
    from src.web_tracing import WebTracer
except ModuleNotFoundError:
    from agent import AgentResponse, run_agent
    from constants import DISCLAIMER
    from mcp_server import MCP_TOOL_REGISTRY
    from observability import create_tracer
    from web_models import AnalysisRequest
    from web_tracing import WebTracer


ROOT = Path(__file__).resolve().parents[1]
RETENTION = timedelta(hours=1)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_sections(answer: str) -> dict[str, Any]:
    """Extract known headings server-side and tolerate missing/malformed blocks."""
    headings = ["PREUVES", "ANALYSE", "CONCLUSION", "CONFIANCE", "CRITIC_VERDICT", "AVERTISSEMENTS", "DISCLAIMER", "METADATA"]
    positions = [(answer.find(f"\n{heading}\n"), heading) for heading in headings]
    positions = sorted((index + 1, heading) for index, heading in positions if index >= 0)
    blocks: dict[str, str] = {}
    for index, (start, heading) in enumerate(positions):
        body_start = start + len(heading) + 1
        body_end = positions[index + 1][0] if index + 1 < len(positions) else len(answer)
        blocks[heading] = answer[body_start:body_end].strip()
    evidence = [line[2:].strip() for line in blocks.get("PREUVES", "").splitlines() if line.startswith("- ")]
    return {
        "evidence": evidence,
        "analysis": blocks.get("ANALYSE", ""),
        "conclusion": blocks.get("CONCLUSION", ""),
        "confidence": blocks.get("CONFIANCE", ""),
    }


def adapt_agent_response(response: AgentResponse) -> dict[str, Any]:
    return {
        "answer": response.answer,
        "sections": parse_sections(response.answer),
        "conclusion": response.conclusion,
        "confidence": response.confidence,
        "critic_verdict": response.critic_verdict,
        "sources": response.sources,
        "missing_information": response.missing_information,
        "warnings": response.warnings,
        "trace_id": response.trace_id,
        "latency_ms": round(response.latency_ms, 3),
        "metadata": response.metadata,
        "disclaimer": DISCLAIMER,
    }


def corpus_summary() -> dict[str, Any]:
    mapping = {
        "EU": ["ai_act_corpus", "gdpr_corpus"],
        "US": ["us_ai_regulation_corpus"],
        "UK": ["uk_ai_regulation_corpus"],
    }
    jurisdictions: dict[str, Any] = {}
    for jurisdiction, folders in mapping.items():
        corpora = []
        for folder_name in folders:
            folder = ROOT / "data" / folder_name
            pdfs = list(folder.glob("*.pdf")) if folder.is_dir() else []
            newest = max((p.stat().st_mtime for p in pdfs), default=None)
            corpora.append({
                "name": folder_name,
                "present": folder.is_dir(),
                "pdf_count": len(pdfs),
                "last_modified": datetime.fromtimestamp(newest, timezone.utc).isoformat() if newest else None,
            })
        jurisdictions[jurisdiction] = {"available": any(c["present"] for c in corpora), "corpora": corpora}
    index = ROOT / "index_data"
    return {
        "jurisdictions": jurisdictions,
        "index_available": index.is_dir() and any(index.iterdir()),
        "fallback_mode": not (index.is_dir() and any(index.iterdir())),
    }


def read_evaluation() -> dict[str, Any]:
    path = ROOT / "evaluation" / "latest_results.json"
    if not path.is_file():
        return {"available": False, "notes": ["Aucun résultat d’évaluation disponible."]}
    data = json.loads(path.read_text(encoding="utf-8"))
    usage = data.get("token_usage_by_model") or {}
    data["available"] = True
    data["cost_measurable"] = bool(usage)
    if not usage:
        data["cost_display"] = "Coût non mesurable avec les données actuelles"
    limitations = []
    if any("fallback-demo" in str(row.get("top_method", "")) for row in data.get("rows", [])):
        limitations.append("Retrieval exécuté en mode fallback-demo.")
    if not usage:
        limitations.append("L’API fournisseur n’a pas exposé de compteur de tokens.")
    limitations.append("Les métriques sont des métriques locales, pas un juge RAGAS cloud.")
    data["limitations"] = limitations
    return data


def read_traces() -> dict[str, Any]:
    path = ROOT / "observability" / "latest_trace.jsonl"
    if not path.is_file():
        return {"available": False, "spans": [], "warnings": ["Aucune trace locale disponible."]}
    spans, warnings = [], []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            record.pop("prompt", None)
            record["metadata"] = {
                k: v for k, v in (record.get("metadata") or {}).items()
                if not any(marker in k.casefold() for marker in ("key", "secret", "token", "password", "authorization", "prompt"))
            }
            spans.append(record)
        except json.JSONDecodeError:
            warnings.append(f"Ligne {line_number} ignorée car invalide.")
    return {"available": bool(spans), "spans": spans, "warnings": warnings}


def tool_catalog() -> list[dict[str, Any]]:
    catalog = []
    for name, function in MCP_TOOL_REGISTRY.items():
        parameters = []
        for parameter in signature(function).parameters.values():
            parameters.append({
                "name": parameter.name,
                "required": parameter.default is parameter.empty,
                "default": None if parameter.default is parameter.empty else parameter.default,
                "annotation": str(parameter.annotation),
            })
        catalog.append({"name": name, "description": (function.__doc__ or "").strip(), "arguments": parameters, "risk": "monitor"})
    return catalog


def invoke_tool(name: str, arguments: dict[str, Any]) -> Any:
    function = MCP_TOOL_REGISTRY.get(name)
    if function is None:
        raise KeyError(name)
    signature(function).bind(**arguments)
    return function(**arguments)


class AnalysisJobs:
    """In-memory demo store; production should use durable Redis/Celery workers."""

    def __init__(self, max_workers: int = 2):
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="regula-analysis")

    def _cleanup(self) -> None:
        threshold = utc_now() - RETENTION
        for job_id in [key for key, value in self._jobs.items() if value["created_at"] < threshold]:
            self._jobs.pop(job_id, None)

    def create(self, request: AnalysisRequest) -> dict[str, str]:
        job_id = str(uuid4())
        events: Queue[dict[str, Any]] = Queue()
        with self._lock:
            self._cleanup()
            self._jobs[job_id] = {
                "analysis_id": job_id,
                "status": "queued",
                "created_at": utc_now(),
                "started_at": None,
                "completed_at": None,
                "events": events,
                "result": None,
                "error": None,
            }
        events.put({"type": "analysis.queued", "timestamp": utc_now().isoformat(), "analysis_id": job_id})
        self._executor.submit(self._run, job_id, request)
        return {"analysis_id": job_id, "status": "queued"}

    def create_sync(self, request: AnalysisRequest) -> dict[str, Any]:
        """Run one analysis within its HTTP request for serverless safety."""
        job_id = str(uuid4())
        events: Queue[dict[str, Any]] = Queue()
        job = {
            "analysis_id": job_id,
            "status": "running",
            "created_at": utc_now(),
            "started_at": utc_now(),
            "completed_at": None,
            "events": events,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._cleanup()
            self._jobs[job_id] = job
        tracer = WebTracer(create_tracer("agent.run", {"analysis_id": job_id}), events)
        try:
            response = run_agent(
                request.agent_question(),
                jurisdiction=request.jurisdiction,
                top_k=request.top_k,
                self_consistency_k=request.self_consistency_k,
                tracer=tracer,
            )
            job["result"] = adapt_agent_response(response)
            job["status"] = "completed"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = f"{exc.__class__.__name__}: analyse interrompue"
        finally:
            job["completed_at"] = utc_now()
            tracer.flush()
        return self.get(job_id) or {
            "analysis_id": job_id,
            "status": "failed",
            "result": None,
            "error": "Analyse interrompue.",
        }

    def _run(self, job_id: str, request: AnalysisRequest) -> None:
        job = self._jobs[job_id]
        job["status"], job["started_at"] = "running", utc_now()
        job["events"].put({"type": "analysis.started", "timestamp": utc_now().isoformat(), "analysis_id": job_id})
        tracer = WebTracer(create_tracer("agent.run", {"analysis_id": job_id}), job["events"])
        try:
            response = run_agent(
                request.agent_question(),
                jurisdiction=request.jurisdiction,
                top_k=request.top_k,
                self_consistency_k=request.self_consistency_k,
                tracer=tracer,
            )
            job["result"] = adapt_agent_response(response)
            job["status"] = "completed"
            event_type = "analysis.completed"
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = f"{exc.__class__.__name__}: analyse interrompue"
            event_type = "analysis.failed"
        finally:
            job["completed_at"] = utc_now()
            job["events"].put({"type": event_type, "timestamp": utc_now().isoformat(), "analysis_id": job_id})
            tracer.flush()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return {
                key: (value.isoformat() if isinstance(value, datetime) else value)
                for key, value in job.items() if key != "events"
            }

    def next_event(self, job_id: str, timeout: float = 15.0) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        try:
            return job["events"].get(timeout=timeout)
        except Empty:
            return None


analysis_jobs = AnalysisJobs()
