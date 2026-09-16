from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You convert customer-support questions into JSON query plans.
Return JSON only. Never include Markdown or explanations.

Allowed operations:
- count: count matching tickets
- average: calculate an average
- list: return matching ticket rows
- group_by: group and rank results
- anomaly_check: the application handles anomalies separately

Allowed fields and values:
- category: Billing, Technical, General
- priority: Low, Medium, High, Critical
- status: Open, Resolved, Escalated
- agent_id: strings such as AGT-01
- metric: response_time_hrs, resolution_time_hrs, customer_rating,
  resolved_ticket_count, count, average_customer_rating
- group_by: agent_id, category, priority, status

Use status Open and Escalated when the user says unresolved.
For numeric filters use objects such as {\"greater_than\": 12}.
If the question is about anomalies, use operation anomaly_check.

Return exactly this shape:
{
  \"operation\": \"count\",
  \"metric\": null,
  \"group_by\": null,
  \"filters\": {}
}
"""


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    def parse_question(self, question: str) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question.strip()},
            ],
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=90.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(
                "Could not connect to Ollama. Start Ollama and make sure the model is downloaded."
            ) from exc

        try:
            content = response.json()["message"]["content"]
            plan = json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise OllamaError("Ollama returned an invalid JSON query plan") from exc

        return self._validate_plan(plan, question)

    @staticmethod
    def _validate_plan(plan: dict[str, Any], question: str) -> dict[str, Any]:
        if not isinstance(plan, dict):
            raise OllamaError("Query plan must be a JSON object")

        allowed_operations = {"count", "average", "list", "group_by", "anomaly_check"}
        operation = plan.get("operation")
        if operation not in allowed_operations:
            raise OllamaError(f"Unsupported operation from LLM: {operation}")

        plan.setdefault("filters", {})
        plan.setdefault("metric", None)
        plan.setdefault("group_by", None)
        if not isinstance(plan["filters"], dict):
            raise OllamaError("Query filters must be an object")

        # Correct two common small-model mistakes using the user's wording:
        # "Critical" is a priority, never a category, and unresolved includes
        # both Open and Escalated tickets in this dataset.
        filters = plan["filters"]
        priority_values = {"Low", "Medium", "High", "Critical"}
        category_value = filters.get("category")
        if category_value in priority_values:
            filters.pop("category")
            filters["priority"] = category_value

        question_lower = question.lower()
        if "unresolved" in question_lower:
            filters["status"] = ["Open", "Escalated"]

        if "critical" in question_lower:
            filters["priority"] = "Critical"

        if (
            "which agent" in question_lower
            and "resolved" in question_lower
            and ("most" in question_lower or "highest" in question_lower)
        ):
            plan["operation"] = "group_by"
            plan["group_by"] = "agent_id"
            plan["metric"] = "resolved_ticket_count"
            filters["status"] = "Resolved"

        return plan
