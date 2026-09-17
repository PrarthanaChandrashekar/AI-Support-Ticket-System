from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """
You convert customer-support questions into JSON query plans.

Return JSON only. Never include Markdown or explanations.

Allowed operations:
- count
- average
- list
- group_by
- anomaly_check

Allowed fields:
- category: Billing, Technical, General
- priority: Low, Medium, High, Critical
- status: Open, Resolved, Escalated
- agent_id: strings such as AGT-01

Allowed metrics:
- response_time_hrs
- resolution_time_hrs
- customer_rating
- resolved_ticket_count
- count
- average_customer_rating

Allowed group_by fields:
- agent_id
- category
- priority
- status

DATE HANDLING
-------------
For "this month":
"date_range": {"period": "this_month"}

For "this week":
"date_range": {"period": "this_week"}

For "last 7 days":
"date_range": {"period": "last_7_days"}

The application resolves these periods using the latest date available
in the historical dataset.

UNRESOLVED
----------
Unresolved means BOTH:
- Open
- Escalated

Therefore use:
"status": ["Open", "Escalated"]

ESCALATED
---------
If the user explicitly asks for Escalated tickets,
use ONLY:
"status": "Escalated"

Do not include Open unless the user explicitly asks for unresolved.

CRITICAL NOT RESOLVED WITHIN HOURS
----------------------------------
For:
"Show me all Critical tickets not resolved within 12 hours."

use:
operation = "list"
priority = "Critical"
custom_unresolved_critical_gt_12 = true
critical_resolution_limit_hours = 12

MOST RESOLVED AGENT
-------------------
For:
"Which agent resolved the most tickets this month?"

use:
operation = "group_by"
group_by = "agent_id"
metric = "resolved_ticket_count"
status = "Resolved"
date_range period = "this_month"

SIMPLE RESOLVED COUNT
---------------------
For:
"How many tickets are resolved?"

use:
operation = "count"
metric = null
group_by = null
status = "Resolved"

CATEGORY WITH MOST TICKETS
--------------------------
For:
"Which category has the most tickets?"

use:
operation = "group_by"
metric = "count"
group_by = "category"

Do not add a date range unless the user explicitly mentions
a date period.

AVERAGE CUSTOMER RATING
-----------------------
For:
"What is the average customer rating for Billing tickets?"

use:
operation = "average"
metric = "customer_rating"
group_by = null

filters:
{
    "category": "Billing"
}

Do not group by category when a specific category is already selected.

ANOMALIES
---------
If the user asks about anomalies, use:
operation = "anomaly_check"

Return this structure:

{
  "operation": "count",
  "metric": null,
  "group_by": null,
  "filters": {}
}
"""


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self):
        self.base_url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434"
        ).rstrip("/")

        self.model = os.getenv(
            "OLLAMA_MODEL",
            "llama3.2:3b"
        )

    def parse_question(self, question: str) -> dict[str, Any]:

        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0
            },
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )

            response.raise_for_status()

        except httpx.HTTPError as exc:
            raise OllamaError(
                f"Could not connect to Ollama: {exc}"
            ) from exc

        try:
            data = response.json()

            content = data["message"]["content"]

            plan = json.loads(content)

        except (KeyError, json.JSONDecodeError) as exc:
            raise OllamaError(
                "Ollama returned an invalid JSON query plan."
            ) from exc

        return self._validate_plan(
            plan,
            question
        )

    @staticmethod
    def _validate_plan(
        plan: dict[str, Any],
        question: str
    ) -> dict[str, Any]:

        # ---------------------------------------------------------
        # ALLOWED VALUES
        # ---------------------------------------------------------

        allowed_operations = {
            "count",
            "average",
            "list",
            "group_by",
            "anomaly_check"
        }

        allowed_metrics = {
            None,
            "response_time_hrs",
            "resolution_time_hrs",
            "customer_rating",
            "resolved_ticket_count",
            "count",
            "average_customer_rating"
        }

        allowed_group_by = {
            None,
            "agent_id",
            "category",
            "priority",
            "status"
        }

        # ---------------------------------------------------------
        # BASIC VALIDATION
        # ---------------------------------------------------------

        operation = plan.get(
            "operation",
            "count"
        )

        if operation not in allowed_operations:
            operation = "count"

        metric = plan.get("metric")

        if metric not in allowed_metrics:
            metric = None

        group_by = plan.get("group_by")

        if group_by not in allowed_group_by:
            group_by = None

        filters = plan.get(
            "filters",
            {}
        )

        if not isinstance(filters, dict):
            filters = {}

        question_lower = question.lower()

        # ---------------------------------------------------------
        # HANDLE NESTED FILTERS
        # ---------------------------------------------------------

        nested_filters = filters.pop(
            "filters",
            None
        )

        if isinstance(nested_filters, dict):
            filters.update(
                nested_filters
            )

        # ---------------------------------------------------------
        # FIX CATEGORY / PRIORITY CONFUSION
        # ---------------------------------------------------------

        priority_values = {
            "Low",
            "Medium",
            "High",
            "Critical"
        }

        category_value = filters.get(
            "category"
        )

        if category_value in priority_values:

            filters.pop(
                "category",
                None
            )

            filters["priority"] = category_value

        # ---------------------------------------------------------
        # UNRESOLVED
        #
        # Unresolved = Open + Escalated
        # ---------------------------------------------------------

        if "unresolved" in question_lower:

            filters["status"] = [
                "Open",
                "Escalated"
            ]

        # ---------------------------------------------------------
        # NOT RESOLVED
        # ---------------------------------------------------------

        if "not resolved" in question_lower:

            filters["status"] = [
                "Open",
                "Escalated"
            ]

        # ---------------------------------------------------------
        # CRITICAL
        # ---------------------------------------------------------

        if "critical" in question_lower:

            filters["priority"] = "Critical"

        # ---------------------------------------------------------
        # EXPLICIT ESCALATED
        #
        # "Escalated" alone = only Escalated
        #
        # Do not include Open unless the question says
        # unresolved / not resolved.
        # ---------------------------------------------------------

        if (
            "escalated" in question_lower
            and "unresolved" not in question_lower
            and "not resolved" not in question_lower
        ):

            filters["status"] = "Escalated"

        # ---------------------------------------------------------
        # CRITICAL NOT RESOLVED WITHIN X HOURS
        # ---------------------------------------------------------

        limit_match = re.search(
            r"(?:not\s+resolved|unresolved)"
            r".{0,50}?"
            r"(?:within|in|after)?\s*"
            r"(\d+(?:\.\d+)?)\s*hours?",
            question_lower
        )

        if (
            "critical" in question_lower
            and limit_match
        ):

            limit_hours = float(
                limit_match.group(1)
            )

            operation = "list"
            metric = None
            group_by = None

            filters["priority"] = "Critical"

            filters[
                "custom_unresolved_critical_gt_12"
            ] = True

            filters[
                "critical_resolution_limit_hours"
            ] = limit_hours

        # ---------------------------------------------------------
        # RESOLUTION TIME GREATER THAN X
        # ---------------------------------------------------------

        resolution_match = re.search(
            r"resolution\s*time"
            r"[^\d]*"
            r"(?:greater\s+than|over|above|more\s+than)"
            r"\s*(\d+(?:\.\d+)?)",
            question_lower
        )

        if resolution_match:

            filters["resolution_time_hrs"] = {
                "greater_than": float(
                    resolution_match.group(1)
                )
            }

        # ---------------------------------------------------------
        # DATE RANGE
        # ---------------------------------------------------------

        if "this month" in question_lower:

            filters["date_range"] = {
                "period": "this_month"
            }

        elif (
            "this week" in question_lower
            or "last 7 days" in question_lower
            or "last seven days" in question_lower
        ):

            filters["date_range"] = {
                "period": "this_week"
            }

        # ---------------------------------------------------------
        # WHICH AGENT RESOLVED THE MOST
        # ---------------------------------------------------------

        if (
            "which agent" in question_lower
            and "resolved" in question_lower
            and (
                "most" in question_lower
                or "highest" in question_lower
            )
        ):

            operation = "group_by"
            metric = "resolved_ticket_count"
            group_by = "agent_id"

            filters["status"] = "Resolved"

        # ---------------------------------------------------------
        # SIMPLE RESOLVED COUNT
        # ---------------------------------------------------------

        is_simple_resolved_count = (
            "resolved" in question_lower
            and (
                "how many" in question_lower
                or "number of" in question_lower
                or "count of" in question_lower
                or "count" in question_lower
            )
            and "which agent" not in question_lower
            and "most" not in question_lower
            and "highest" not in question_lower
            and "not resolved" not in question_lower
            and "unresolved" not in question_lower
        )

        if is_simple_resolved_count:

            operation = "count"
            metric = None
            group_by = None

            filters["status"] = "Resolved"

        # ---------------------------------------------------------
        # OPEN COUNT
        # ---------------------------------------------------------

        if (
            "how many" in question_lower
            and "open" in question_lower
            and "ticket" in question_lower
        ):

            operation = "count"
            metric = None
            group_by = None

            filters["status"] = "Open"

        # ---------------------------------------------------------
        # ESCALATED COUNT
        # ---------------------------------------------------------

        if (
            "how many" in question_lower
            and "escalated" in question_lower
            and "ticket" in question_lower
        ):

            operation = "count"
            metric = None
            group_by = None

            filters["status"] = "Escalated"

        # ---------------------------------------------------------
        # CATEGORY WITH MOST TICKETS
        #
        # Example:
        # "Which category has the most tickets?"
        #
        # IMPORTANT:
        # Do not invent "this_month".
        # ---------------------------------------------------------

        if (
            "which category" in question_lower
            and "most tickets" in question_lower
            and "this month" not in question_lower
            and "this week" not in question_lower
            and "last 7 days" not in question_lower
            and "last seven days" not in question_lower
        ):

            operation = "group_by"
            metric = "count"
            group_by = "category"

            filters.pop(
                "date_range",
                None
            )

        # ---------------------------------------------------------
        # AVERAGE CUSTOMER RATING FOR SPECIFIC CATEGORY
        #
        # Example:
        # "What is the average customer rating for Billing tickets?"
        #
        # Do not group again because the category is already
        # selected as a filter.
        # ---------------------------------------------------------

        if (
            "average" in question_lower
            and "customer rating" in question_lower
            and filters.get("category") in {
                "Billing",
                "Technical",
                "General"
            }
        ):

            operation = "average"
            metric = "customer_rating"
            group_by = None

        # ---------------------------------------------------------
        # RESPONSE TIME AVERAGE
        # ---------------------------------------------------------

        if (
            "average response time" in question_lower
            or "average response_time_hrs" in question_lower
        ):

            operation = "average"
            metric = "response_time_hrs"
            group_by = None

        # ---------------------------------------------------------
        # RESOLUTION TIME AVERAGE
        # ---------------------------------------------------------

        if (
            "average resolution time" in question_lower
            or "average resolution_time_hrs" in question_lower
        ):

            operation = "average"
            metric = "resolution_time_hrs"
            group_by = None

        # ---------------------------------------------------------
        # ANOMALY CHECK
        # ---------------------------------------------------------

        if "anomal" in question_lower:

            operation = "anomaly_check"
            metric = None
            group_by = None

        # ---------------------------------------------------------
        # EXPLICIT CATEGORY EXTRACTION
        #
        # This makes sure questions containing:
        # Billing
        # Technical
        # General
        #
        # always get the correct category filter.
        # ---------------------------------------------------------

        category_names = {
            "billing": "Billing",
            "technical": "Technical",
            "general": "General"
        }

        for category_text, category_value in category_names.items():

            if re.search(
                rf"\b{category_text}\b",
                question_lower
            ):

                filters["category"] = category_value

                break

        # ---------------------------------------------------------
        # RETURN FINAL CLEAN PLAN
        # ---------------------------------------------------------

        return {
            "operation": operation,
            "metric": metric,
            "group_by": group_by,
            "filters": filters
        }