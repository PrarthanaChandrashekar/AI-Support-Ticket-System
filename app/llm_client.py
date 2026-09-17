from __future__ import annotations

import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)


# ============================================================
# CUSTOM ERROR
# ============================================================

class OllamaError(Exception):
    """Raised when Ollama cannot process a query."""


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI query planner for a support-ticket analytics system.

Your job is to convert a user's natural-language question into
ONE valid JSON query plan.

The dataset contains these columns:

- ticket_id
- created_at
- category
- priority
- status
- response_time_hrs
- resolution_time_hrs
- agent_id
- customer_rating
- issue_summary

Allowed category values:
- Billing
- Technical
- General

Allowed priority values:
- Low
- Medium
- High
- Critical

Allowed status values:
- Open
- Resolved
- Escalated

Allowed operations:
- count
- average
- list
- group_by
- anomaly_check

Allowed group_by values:
- agent_id
- category
- priority
- status

Allowed metrics:
- ticket_id
- response_time_hrs
- resolution_time_hrs
- customer_rating

The JSON structure MUST be:

{
  "operation": "...",
  "metric": "...",
  "group_by": "...",
  "filters": {}
}

Rules:

1. "How many tickets..." means:
   operation = "count"
   metric = "ticket_id"

2. "How many open tickets..." means:
   operation = "count"
   metric = "ticket_id"
   status = "Open"

3. "How many resolved tickets..." means:
   operation = "count"
   metric = "ticket_id"
   status = "Resolved"

4. "How many escalated tickets..." means:
   operation = "count"
   metric = "ticket_id"
   status = "Escalated"

5. "Which agent resolved the most tickets..." means:
   operation = "group_by"
   metric = "ticket_id"
   group_by = "agent_id"
   status = "Resolved"

6. "Which category has the most tickets..." means:
   operation = "group_by"
   metric = "ticket_id"
   group_by = "category"

7. "Average customer rating..." means:
   operation = "average"
   metric = "customer_rating"

8. "Average response time..." means:
   operation = "average"
   metric = "response_time_hrs"

9. "Average resolution time..." means:
   operation = "average"
   metric = "resolution_time_hrs"

10. "Show me all Critical tickets not resolved within 12 hours"
    means:
    operation = "list"
    metric = null
    group_by = null
    filters must contain:
    "critical_not_resolved_within_hours": 12
    "priority": "Critical"

    This includes:
    - Critical tickets that are Open
    - Critical tickets that are Escalated
    - Critical tickets that are Resolved but took more than 12 hours

11. "Are there any anomalies in resolution times..."
    means:
    operation = "anomaly_check"
    metric = "resolution_time_hrs"

12. Date filters:

    "this month" -> date_range period "this_month"
    "this week" -> date_range period "this_week"
    "last 7 days" -> date_range period "last_7_days"
    "today" -> date_range period "today"
    "yesterday" -> date_range period "yesterday"
    "this year" -> date_range period "this_year"
    "last month" -> date_range period "last_month"

13. Use created_at for date filtering.

14. Do NOT invent columns.

15. Do NOT invent group_by values.

16. Do NOT invent metric names.

17. Return ONLY valid JSON.
"""


# ============================================================
# JSON EXTRACTION
# ============================================================

def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    # Remove markdown code fences if the model adds them.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        result = json.loads(text)

        if not isinstance(result, dict):
            raise ValueError(
                "LLM response must be a JSON object"
            )

        return result

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL
        )

        if not match:
            raise ValueError(
                "LLM returned invalid JSON"
            )

        try:
            result = json.loads(
                match.group(0)
            )

            if not isinstance(result, dict):
                raise ValueError(
                    "LLM response must be a JSON object"
                )

            return result

        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM returned invalid JSON"
            ) from exc


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _normalize_status(value: Any) -> Any:

    if not isinstance(value, str):
        return value

    mapping = {
        "open": "Open",
        "opened": "Open",
        "unresolved": "Open",

        "resolved": "Resolved",
        "closed": "Resolved",

        "escalated": "Escalated",
        "escalated tickets": "Escalated",
    }

    return mapping.get(
        value.strip().lower(),
        value
    )


def _normalize_priority(value: Any) -> Any:

    if not isinstance(value, str):
        return value

    mapping = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "critical": "Critical",
    }

    return mapping.get(
        value.strip().lower(),
        value
    )


def _normalize_category(value: Any) -> Any:

    if not isinstance(value, str):
        return value

    mapping = {
        "billing": "Billing",
        "technical": "Technical",
        "general": "General",
    }

    return mapping.get(
        value.strip().lower(),
        value
    )


# ============================================================
# PLAN VALIDATION + NORMALIZATION
# ============================================================

def _validate_plan(
    plan: dict[str, Any],
    question: str
) -> dict[str, Any]:

    question_lower = question.lower()

    if not isinstance(plan, dict):
        raise ValueError(
            "LLM query plan must be a JSON object"
        )

    # --------------------------------------------------------
    # Ensure filters exists
    # --------------------------------------------------------

    filters = plan.get(
        "filters",
        {}
    )

    if not isinstance(filters, dict):
        filters = {}

    plan["filters"] = filters

    # --------------------------------------------------------
    # Metric aliases
    # --------------------------------------------------------

    metric = plan.get("metric")

    metric_aliases = {

        "resolved_ticket_count":
            "ticket_id",

        "resolved_count":
            "ticket_id",

        "total_resolved":
            "ticket_id",

        "number_of_tickets":
            "ticket_id",

        "ticket_count":
            "ticket_id",

        "tickets_count":
            "ticket_id",

        "count":
            "ticket_id",

        "number_of_resolved_tickets":
            "ticket_id",
    }

    if isinstance(metric, str):

        metric_key = (
            metric.strip().lower()
        )

        if metric_key in metric_aliases:

            plan["metric"] = metric_aliases[
                metric_key
            ]

    # --------------------------------------------------------
    # Critical unresolved > X hours aliases
    # --------------------------------------------------------

    if (
        filters.get(
            "custom_unresolved_critical_gt_12"
        )
        is True
    ):

        limit = filters.get(
            "critical_resolution_limit_hours",
            12
        )

        filters[
            "critical_not_resolved_within_hours"
        ] = limit

        filters.pop(
            "custom_unresolved_critical_gt_12",
            None
        )

        filters.pop(
            "critical_resolution_limit_hours",
            None
        )

    critical_aliases = [

        "critical_unresolved_over_hours",

        "critical_not_resolved_over_hours",

        "critical_tickets_over_hours",

        "unresolved_critical_over_hours",
    ]

    for alias in critical_aliases:

        if alias in filters:

            filters[
                "critical_not_resolved_within_hours"
            ] = filters.pop(alias)

    # --------------------------------------------------------
    # Normalize standard filters
    # --------------------------------------------------------

    if "status" in filters:

        filters["status"] = _normalize_status(
            filters["status"]
        )

    if "priority" in filters:

        filters["priority"] = _normalize_priority(
            filters["priority"]
        )

    if "category" in filters:

        filters["category"] = _normalize_category(
            filters["category"]
        )

    # --------------------------------------------------------
    # Explicit question overrides
    # --------------------------------------------------------

    # Most resolved tickets by agent.
    if (
        "which agent" in question_lower
        and (
            "resolved the most"
            in question_lower
            or "most tickets"
            in question_lower
        )
    ):

        plan["operation"] = "group_by"
        plan["metric"] = "ticket_id"
        plan["group_by"] = "agent_id"

        filters["status"] = "Resolved"

    # Most tickets by category.
    elif (
        "which category" in question_lower
        and (
            "most tickets"
            in question_lower
            or "most" in question_lower
        )
    ):

        plan["operation"] = "group_by"
        plan["metric"] = "ticket_id"
        plan["group_by"] = "category"

    # Critical tickets not resolved within X hours.
    critical_match = re.search(
        r"critical.*?"
        r"(?:not resolved|unresolved).*?"
        r"(?:within|under|in)\s*"
        r"(\d+(?:\.\d+)?)\s*hours?",
        question_lower
    )

    if critical_match:

        limit = float(
            critical_match.group(1)
        )

        if limit.is_integer():
            limit = int(limit)

        plan["operation"] = "list"
        plan["metric"] = None
        plan["group_by"] = None

        filters[
            "critical_not_resolved_within_hours"
        ] = limit

        filters["priority"] = "Critical"

        # The special filter controls status.
        filters.pop(
            "status",
            None
        )

    # Resolution-time anomaly question.
    if (
        "anomal" in question_lower
        and "resolution" in question_lower
    ):

        plan["operation"] = "anomaly_check"
        plan["metric"] = "resolution_time_hrs"
        plan["group_by"] = None

    # Average customer rating.
    if (
        "average customer rating"
        in question_lower
    ):

        plan["operation"] = "average"
        plan["metric"] = "customer_rating"
        plan["group_by"] = None

    # Average response time.
    elif (
        "average response time"
        in question_lower
    ):

        plan["operation"] = "average"
        plan["metric"] = "response_time_hrs"
        plan["group_by"] = None

    # Average resolution time.
    elif (
        "average resolution time"
        in question_lower
    ):

        plan["operation"] = "average"
        plan["metric"] = "resolution_time_hrs"
        plan["group_by"] = None

    # How many resolved tickets?
    if (
        "how many" in question_lower
        and "resolved" in question_lower
        and "which agent" not in question_lower
    ):

        plan["operation"] = "count"
        plan["metric"] = "ticket_id"
        plan["group_by"] = None

        filters["status"] = "Resolved"

    # How many open tickets?
    if (
        "how many" in question_lower
        and "open" in question_lower
    ):

        plan["operation"] = "count"
        plan["metric"] = "ticket_id"
        plan["group_by"] = None

        filters["status"] = "Open"

    # How many escalated tickets?
    if (
        "how many" in question_lower
        and "escalated" in question_lower
    ):

        plan["operation"] = "count"
        plan["metric"] = "ticket_id"
        plan["group_by"] = None

        filters["status"] = "Escalated"

    # --------------------------------------------------------
    # Explicit date detection
    # --------------------------------------------------------

    if (
        "this month"
        in question_lower
    ):

        filters["date_range"] = {
            "period": "this_month"
        }

    elif (
        "this week"
        in question_lower
    ):

        filters["date_range"] = {
            "period": "this_week"
        }

    elif (
        "last 7 days"
        in question_lower
    ):

        filters["date_range"] = {
            "period": "last_7_days"
        }

    elif (
        "today"
        in question_lower
    ):

        filters["date_range"] = {
            "period": "today"
        }

    elif (
        "yesterday"
        in question_lower
    ):

        filters["date_range"] = {
            "period": "yesterday"
        }

    elif (
        "this year"
        in question_lower
    ):

        filters["date_range"] = {
            "period": "this_year"
        }

    elif (
        "last month"
        in question_lower
    ):

        filters["date_range"] = {
            "period": "last_month"
        }

    else:

        # Do not allow the LLM to invent an
        # unsupported date range.
        filters.pop(
            "date_range",
            None
        )

    # --------------------------------------------------------
    # Canonical operation
    # --------------------------------------------------------

    allowed_operations = {
        "count",
        "average",
        "list",
        "group_by",
        "anomaly_check",
    }

    operation = plan.get(
        "operation"
    )

    if operation not in allowed_operations:

        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    # --------------------------------------------------------
    # Canonical metric
    # --------------------------------------------------------

    allowed_metrics = {
        "ticket_id",
        "response_time_hrs",
        "resolution_time_hrs",
        "customer_rating",
    }

    metric = plan.get(
        "metric"
    )

    if metric is not None and metric not in allowed_metrics:

        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    # --------------------------------------------------------
    # Canonical group_by
    # --------------------------------------------------------

    allowed_group_by = {
        "agent_id",
        "category",
        "priority",
        "status",
        None,
    }

    group_by = plan.get(
        "group_by"
    )

    if group_by not in allowed_group_by:

        raise ValueError(
            f"Unsupported group_by: {group_by}"
        )

    # --------------------------------------------------------
    # Critical special filter wins.
    # --------------------------------------------------------

    if (
        "critical_not_resolved_within_hours"
        in filters
    ):

        plan["operation"] = "list"
        plan["metric"] = None
        plan["group_by"] = None

        filters["priority"] = "Critical"

        filters.pop(
            "status",
            None
        )

    # --------------------------------------------------------
    # Final canonical defaults
    # --------------------------------------------------------

    plan.setdefault(
        "metric",
        None
    )

    plan.setdefault(
        "group_by",
        None
    )

    plan["filters"] = filters

    return plan


# ============================================================
# OLLAMA CLIENT
# ============================================================

class OllamaClient:

    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = 120
    ) -> None:

        self.base_url = base_url.rstrip(
            "/"
        )

        self.model = model

        self.timeout = timeout

    # --------------------------------------------------------
    # Parse natural-language question
    # --------------------------------------------------------

    def parse_question(
        self,
        question: str
    ) -> dict[str, Any]:

        if not question.strip():

            raise ValueError(
                "Question cannot be empty"
            )

        payload = {

            "model":
                self.model,

            "messages": [

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": question
                }
            ],

            "stream":
                False,

            "format":
                "json",

            "options": {

                "temperature":
                    0
            }
        }

        try:

            response = requests.post(

                f"{self.base_url}/api/chat",

                json=payload,

                timeout=self.timeout
            )

        except requests.RequestException as exc:

            raise OllamaError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ) from exc

        if response.status_code != 200:

            raise OllamaError(
                f"Ollama returned HTTP "
                f"{response.status_code}: "
                f"{response.text[:300]}"
            )

        try:

            response_json = response.json()

        except ValueError as exc:

            raise OllamaError(
                "Ollama returned invalid JSON"
            ) from exc

        message = response_json.get(
            "message",
            {}
        )

        content = message.get(
            "content",
            ""
        )

        if not content:

            raise OllamaError(
                "Ollama returned an empty response"
            )

        plan = _extract_json(
            content
        )

        return _validate_plan(
            plan,
            question
        )