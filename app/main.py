from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .anomaly_detector import detect_anomalies
from .data_loader import load_tickets
from .llm_client import OllamaClient, OllamaError
from .query_executor import answer_text, execute_query

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "data" / "support_tickets.csv")))
if not DATA_PATH.is_absolute():
    DATA_PATH = BASE_DIR / DATA_PATH

df = load_tickets(DATA_PATH)
llm = OllamaClient()
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))
app = FastAPI(title="AI Support Ticket Analytics", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> Any:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "rows_loaded": int(len(df)), "llm_model": llm.model}


@app.get("/stats")
def stats() -> dict[str, Any]:
    return {
        "total_tickets": int(len(df)),
        "open_tickets": int((df["status"] == "Open").sum()),
        "resolved_tickets": int((df["status"] == "Resolved").sum()),
        "escalated_tickets": int((df["status"] == "Escalated").sum()),
        "average_response_time_hrs": round(float(df["response_time_hrs"].mean()), 2),
    }


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        plan = llm.parse_question(request.question)
        if plan["operation"] == "anomaly_check":
            result = detect_anomalies(df)
            answer = f"I found {result['summary']['total_anomalies']} anomalies."
        else:
            result = execute_query(df, plan)
            answer = answer_text(request.question, plan, result)
        return {
            "question": request.question,
            "interpreted_query": plan,
            "answer": answer,
            "data": result,
        }
    except (OllamaError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not complete the query") from exc


@app.get("/anomalies")
def anomalies() -> dict[str, Any]:
    return detect_anomalies(df)
