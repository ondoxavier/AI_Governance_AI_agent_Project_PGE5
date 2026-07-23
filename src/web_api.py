"""FastAPI adapter over the existing synchronous agent and MCP functions."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.constants import DISCLAIMER
    from src.llm_client import is_available
    from src.mcp_server import compare_jurisdiction, hybrid_search, security_screen
    from src.observability import create_tracer
    from src.web_models import AnalysisRequest, CompareRequest, SearchRequest, SecurityRequest, ToolInvokeRequest
    from src.web_services import analysis_jobs, corpus_summary, invoke_tool, read_evaluation, read_traces, tool_catalog
except ModuleNotFoundError:
    from constants import DISCLAIMER
    from llm_client import is_available
    from mcp_server import compare_jurisdiction, hybrid_search, security_screen
    from observability import create_tracer
    from web_models import AnalysisRequest, CompareRequest, SearchRequest, SecurityRequest, ToolInvokeRequest
    from web_services import analysis_jobs, corpus_summary, invoke_tool, read_evaluation, read_traces, tool_catalog


ROOT = Path(__file__).resolve().parents[1]
API_PREFIX = "/api/v1"
app = FastAPI(title="RegulaAI API", version="1.0.0", docs_url="/api/docs", redoc_url=None)
origins = [item.strip() for item in os.getenv("REGULAAI_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if item.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, Any]:
    corpus = corpus_summary()
    tracer = create_tracer("web.health")
    result = {
        "status": "ok",
        "agent_version": tracer.agent_version,
        "llm": {"configured": is_available(), "provider": "DeepInfra"},
        "observability": {"provider": tracer.provider, "configured": tracer.provider == "langfuse"},
        "retrieval": {"index_available": corpus["index_available"], "fallback_mode": corpus["fallback_mode"]},
        "mcp": {"available": len(tool_catalog()) == 4, "tools_count": len(tool_catalog())},
    }
    tracer.flush()
    return result


@app.get(f"{API_PREFIX}/corpus")
def corpus() -> dict[str, Any]:
    return corpus_summary()


@app.post(f"{API_PREFIX}/analyses")
def create_analysis(request: AnalysisRequest) -> dict[str, Any]:
    # Background threads can be frozen once a serverless response is returned.
    return analysis_jobs.create_sync(request)


@app.get(f"{API_PREFIX}/analyses/{{analysis_id}}")
def get_analysis(analysis_id: str) -> dict[str, Any]:
    job = analysis_jobs.get(analysis_id)
    if not job:
        raise HTTPException(404, "Analyse introuvable ou expirée.")
    return job


@app.get(f"{API_PREFIX}/analyses/{{analysis_id}}/events")
async def analysis_events(analysis_id: str) -> StreamingResponse:
    if not analysis_jobs.get(analysis_id):
        raise HTTPException(404, "Analyse introuvable ou expirée.")

    async def stream():
        while True:
            event = await asyncio.to_thread(analysis_jobs.next_event, analysis_id, 15.0)
            if event is None:
                yield ": keep-alive\n\n"
                continue
            yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in {"analysis.completed", "analysis.failed"}:
                break

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post(f"{API_PREFIX}/search")
def search(request: SearchRequest) -> dict[str, Any]:
    return hybrid_search(request.query, request.top_k, request.jurisdiction)


@app.post(f"{API_PREFIX}/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    return compare_jurisdiction(request.topic, request.top_k)


@app.post(f"{API_PREFIX}/security/screen")
def screen(request: SecurityRequest) -> dict[str, Any]:
    return security_screen(request.text)


@app.get(f"{API_PREFIX}/tools")
def tools() -> dict[str, Any]:
    return {"tools": tool_catalog()}


@app.post(f"{API_PREFIX}/tools/{{tool_name}}/invoke")
def tool_invoke(tool_name: str, request: ToolInvokeRequest) -> dict[str, Any]:
    try:
        return {"tool": tool_name, "result": invoke_tool(tool_name, request.arguments)}
    except KeyError:
        raise HTTPException(404, "Outil MCP inconnu.") from None
    except TypeError as exc:
        raise HTTPException(422, f"Arguments invalides: {exc}") from None


@app.get(f"{API_PREFIX}/evaluation/latest")
def evaluation_latest() -> dict[str, Any]:
    return read_evaluation()


@app.get(f"{API_PREFIX}/traces/latest")
def traces_latest() -> dict[str, Any]:
    return read_traces()
