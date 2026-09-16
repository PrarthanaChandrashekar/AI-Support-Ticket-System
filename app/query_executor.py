from __future__ import annotations

from typing import Any

import pandas as pd


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply only allow-listed filters produced by the LLM."""
    result = df.copy()
    allowed_columns = {
        "category",
        "priority",
        "status",
        "agent_id",
        "ticket_id",
    }

    for column, value in filters.items():
        if column in allowed_columns:
            if isinstance(value, list):
                result = result[result[column].isin(value)]
            else:
                result = result[result[column] == value]
        elif column in {"response_time_hrs", "resolution_time_hrs", "customer_rating"}:
            if not isinstance(value, dict):
                continue
            for operator, number in value.items():
                number = float(number)
                if operator == "greater_than":
                    result = result[result[column] > number]
                elif operator == "greater_than_or_equal":
                    result = result[result[column] >= number]
                elif operator == "less_than":
                    result = result[result[column] < number]
                elif operator == "less_than_or_equal":
                    result = result[result[column] <= number]

    return result


def execute_query(df: pd.DataFrame, plan: dict[str, Any]) -> dict[str, Any]:
    """Execute a validated query plan and return JSON-serializable data."""
    operation = plan.get("operation")
    filtered = _apply_filters(df, plan.get("filters") or {})

    if operation == "count":
        return {"count": int(len(filtered))}

    if operation == "average":
        metric = plan.get("metric")
        if metric not in {"response_time_hrs", "resolution_time_hrs", "customer_rating"}:
            raise ValueError("Average metric is not supported")
        values = filtered[metric].dropna()
        return {
            "metric": metric,
            "count": int(len(values)),
            "average": round(float(values.mean()), 2) if len(values) else None,
        }

    if operation == "list":
        rows = filtered.copy()
        rows["created_at"] = rows["created_at"].dt.strftime("%Y-%m-%d %H:%M")
        rows = rows.where(pd.notna(rows), None)
        return {"count": int(len(rows)), "tickets": rows.to_dict(orient="records")[:100]}

    if operation == "group_by":
        group_by = plan.get("group_by")
        metric = plan.get("metric", "count")
        if group_by not in {"agent_id", "category", "priority", "status"}:
            raise ValueError("Group-by field is not supported")

        if metric == "resolved_ticket_count":
            filtered = filtered[filtered["status"] == "Resolved"]
            grouped = filtered.groupby(group_by).size().reset_index(name="count")
            grouped = grouped.sort_values("count", ascending=False)
        elif metric == "count":
            grouped = filtered.groupby(group_by).size().reset_index(name="count")
            grouped = grouped.sort_values("count", ascending=False)
        elif metric == "average_customer_rating":
            grouped = (
                filtered.groupby(group_by)["customer_rating"]
                .mean()
                .round(2)
                .reset_index(name="average_customer_rating")
                .sort_values("average_customer_rating", ascending=False)
            )
        else:
            raise ValueError("Group-by metric is not supported")

        return {"groups": grouped.to_dict(orient="records")}

    raise ValueError(f"Unsupported query operation: {operation}")


def answer_text(question: str, plan: dict[str, Any], result: dict[str, Any]) -> str:
    """Create a concise factual answer without asking the LLM to calculate."""
    operation = plan.get("operation")
    if operation == "count":
        return f"There are {result['count']} matching tickets."
    if operation == "average":
        if result["average"] is None:
            return "There is not enough data to calculate that average."
        return f"The average {result['metric']} is {result['average']} across {result['count']} tickets."
    if operation == "list":
        return f"I found {result['count']} matching tickets."
    if operation == "group_by":
        groups = result.get("groups", [])
        if groups and "count" in groups[0]:
            top_count = groups[0]["count"]
            group_plan_key = plan.get("group_by", "group")
            leaders = [row[group_plan_key] for row in groups if row.get("count") == top_count]
            if len(leaders) == 1:
                return f"{leaders[0]} resolved the most tickets, with {top_count} resolved tickets."
            return f"The top agents are {', '.join(leaders)}, tied at {top_count} resolved tickets."
        return "Here are the grouped results, sorted from highest to lowest."
    return "The query was completed."
