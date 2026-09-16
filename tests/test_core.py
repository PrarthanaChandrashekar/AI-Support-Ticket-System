from pathlib import Path

from fastapi.testclient import TestClient

from app.anomaly_detector import detect_anomalies
from app.data_loader import load_tickets
from app.main import app, df
from app.llm_client import OllamaClient

DATA = Path(__file__).resolve().parents[1] / "data" / "support_tickets.csv"
client = TestClient(app)


def test_dataset_loads():
    loaded = load_tickets(DATA)
    assert len(loaded) == 500
    assert loaded["created_at"].notna().all()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["rows_loaded"] == 500


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["open_tickets"] == int((df["status"] == "Open").sum())


def test_anomalies_have_summary():
    result = detect_anomalies(df)
    assert "summary" in result
    assert "anomalies" in result
    assert result["summary"]["total_anomalies"] == len(result["anomalies"])


def test_common_llm_misinterpretation_is_corrected():
    plan = {
        "operation": "count",
        "metric": None,
        "group_by": None,
        "filters": {
            "category": "Critical",
            "status": "Open",
            "priority": "High",
        },
    }
    corrected = OllamaClient._validate_plan(
        plan,
        "How many critical tickets are unresolved?",
    )
    assert corrected["filters"]["priority"] == "Critical"
    assert corrected["filters"]["status"] == ["Open", "Escalated"]
    assert "category" not in corrected["filters"]


def test_resolved_agent_ranking_plan_is_normalized():
    plan = {
        "operation": "count",
        "metric": None,
        "group_by": None,
        "filters": {},
    }
    corrected = OllamaClient._validate_plan(
        plan,
        "Which agent resolved the most tickets?",
    )
    assert corrected["operation"] == "group_by"
    assert corrected["group_by"] == "agent_id"
    assert corrected["metric"] == "resolved_ticket_count"
    assert corrected["filters"]["status"] == "Resolved"
