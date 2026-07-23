"""HTTP adapter tests. Agent work is mocked; no LLM key is required."""
from pathlib import Path
import json, sys
import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.agent import AgentResponse
from src.web_api import app
from src import web_services
client = TestClient(app)

def test_health_and_no_secrets():
    payload = client.get("/api/v1/health").json()
    assert payload["mcp"]["tools_count"] == 4
    assert "secret_key" not in json.dumps(payload).casefold()

def test_corpus():
    assert client.get("/api/v1/corpus").json()["jurisdictions"]["EU"]["available"]

def test_analysis_job(monkeypatch):
    monkeypatch.setattr(web_services, "run_agent", lambda *a, **k: AgentResponse(
        answer="ANALYSE\nPreuves.\n\nCONCLUSION\nValidation.", conclusion="Validation.",
        confidence=.75, critic_verdict="APPROVE"))
    created = client.post("/api/v1/analyses", json={"mode": "free", "question": "Analyser ce système de crédit IA."})
    assert created.status_code == 200
    result = created.json()
    assert result["status"] == "completed"

@pytest.mark.parametrize("payload", [{}, {"mode": "free", "question": ""}, {"mode": "guided", "system": None}])
def test_invalid_analysis(payload):
    assert client.post("/api/v1/analyses", json=payload).status_code == 422

def test_tools_and_refusal():
    tools = client.get("/api/v1/tools").json()["tools"]
    assert {t["name"] for t in tools} == {"hybrid_search", "classify_ai_act_risk", "security_screen", "compare_jurisdiction"}
    assert client.post("/api/v1/tools/arbitrary/invoke", json={"arguments": {}}).status_code == 404

def test_evaluation_and_traces():
    assert client.get("/api/v1/evaluation/latest").status_code == 200
    assert client.get("/api/v1/traces/latest").status_code == 200
