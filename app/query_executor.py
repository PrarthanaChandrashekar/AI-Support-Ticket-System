from __future__ import annotations

import pandas as pd

from app.anomaly_detector import detect_anomalies


# ============================================================
# DATE FILTER
# ============================================================

def _apply_date_filter(
    df: pd.DataFrame,
    date_range: dict | None
) -> pd.DataFrame:

    if not date_range:
        return df

    period = date_range.get("period")

    if not period:
        return df

    reference_time = df["created_at"].max()

    if period == "this_month":

        start = reference_time.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        return df[
            df["created_at"] >= start
        ]

    if period in {
        "this_week",
        "last_7_days"
    }:

        cutoff = (
            reference_time
            - pd.Timedelta(days=7)
        )

        return df[
            df["created_at"] >= cutoff
        ]

    if period == "today":

        start = reference_time.normalize()

        return df[
            df["created_at"] >= start
        ]

    if period == "yesterday":

        end = reference_time.normalize()

        start = (
            end
            - pd.Timedelta(days=1)
        )

        return df[
            (df["created_at"] >= start)
            &
            (df["created_at"] < end)
        ]

    if period == "this_year":

        start = reference_time.replace(
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        return df[
            df["created_at"] >= start
        ]

    if period == "last_month":

        first_this_month = reference_time.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        last_month_start = (
            first_this_month
            - pd.DateOffset(months=1)
        )

        return df[
            (df["created_at"] >= last_month_start)
            &
            (df["created_at"] < first_this_month)
        ]

    return df


# ============================================================
# GENERAL FILTERS
# ============================================================

def _apply_filters(
    df: pd.DataFrame,
    filters: dict
) -> pd.DataFrame:

    if not filters:
        return df

    result = df.copy()

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    category = filters.get("category")

    if category:

        if isinstance(category, str):

            result = result[
                result["category"]
                .astype(str)
                .str.lower()
                ==
                category.lower()
            ]

        elif isinstance(category, list):

            categories = {
                str(value).lower()
                for value in category
            }

            result = result[
                result["category"]
                .astype(str)
                .str.lower()
                .isin(categories)
            ]

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    priority = filters.get("priority")

    if priority:

        if isinstance(priority, str):

            result = result[
                result["priority"]
                .astype(str)
                .str.lower()
                ==
                priority.lower()
            ]

        elif isinstance(priority, list):

            priorities = {
                str(value).lower()
                for value in priority
            }

            result = result[
                result["priority"]
                .astype(str)
                .str.lower()
                .isin(priorities)
            ]

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status = filters.get("status")

    # --------------------------------------------------------
    # SPECIAL CASE:
    #
    # "Critical tickets not resolved within 12 hours"
    #
    # Critical AND
    # (
    #     Open
    #     OR Escalated
    #     OR Resolved with resolution_time > 12
    # )
    # --------------------------------------------------------

    critical_not_resolved = filters.get(
        "critical_not_resolved_within_hours"
    )

    if critical_not_resolved is not None:

        limit = float(
            critical_not_resolved
        )

        result = result[
            result["priority"]
            .astype(str)
            .str.lower()
            ==
            "critical"
        ]

        result = result[
            (
                result["status"]
                .astype(str)
                .str.lower()
                .isin(
                    {
                        "open",
                        "escalated"
                    }
                )
            )
            |
            (
                result["status"]
                .astype(str)
                .str.lower()
                ==
                "resolved"
            )
            &
            (
                result["resolution_time_hrs"]
                > limit
            )
        ]

        # Do NOT apply the normal status filter here.
        status = None

    # --------------------------------------------------------
    # NORMAL STATUS FILTER
    # --------------------------------------------------------

    if status:

        if isinstance(status, str):

            result = result[
                result["status"]
                .astype(str)
                .str.lower()
                ==
                status.lower()
            ]

        elif isinstance(status, list):

            statuses = {
                str(value).lower()
                for value in status
            }

            result = result[
                result["status"]
                .astype(str)
                .str.lower()
                .isin(statuses)
            ]

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent_id = filters.get("agent_id")

    if agent_id:

        if isinstance(agent_id, str):

            result = result[
                result["agent_id"]
                .astype(str)
                .str.lower()
                ==
                agent_id.lower()
            ]

        elif isinstance(agent_id, list):

            agents = {
                str(value).lower()
                for value in agent_id
            }

            result = result[
                result["agent_id"]
                .astype(str)
                .str.lower()
                .isin(agents)
            ]

    # --------------------------------------------------------
    # TICKET ID
    # --------------------------------------------------------

    ticket_id = filters.get("ticket_id")

    if ticket_id:

        if isinstance(ticket_id, str):

            result = result[
                result["ticket_id"]
                .astype(str)
                .str.lower()
                ==
                ticket_id.lower()
            ]

        elif isinstance(ticket_id, list):

            ticket_ids = {
                str(value).lower()
                for value in ticket_id
            }

            result = result[
                result["ticket_id"]
                .astype(str)
                .str.lower()
                .isin(ticket_ids)
            ]

    # --------------------------------------------------------
    # NUMERIC FILTERS
    # --------------------------------------------------------

    numeric_columns = {
        "response_time_hrs",
        "resolution_time_hrs",
        "customer_rating"
    }

    for column in numeric_columns:

        condition = filters.get(column)

        if not condition:
            continue

        if not isinstance(
            condition,
            dict
        ):
            continue

        if column not in result.columns:
            continue

        if "gt" in condition:

            result = result[
                result[column]
                > float(condition["gt"])
            ]

        if "gte" in condition:

            result = result[
                result[column]
                >= float(condition["gte"])
            ]

        if "lt" in condition:

            result = result[
                result[column]
                < float(condition["lt"])
            ]

        if "lte" in condition:

            result = result[
                result[column]
                <= float(condition["lte"])
            ]

        if "eq" in condition:

            result = result[
                result[column]
                == float(condition["eq"])
            ]

    return result


# ============================================================
# EXECUTE QUERY
# ============================================================

def execute_query(
    df: pd.DataFrame,
    plan: dict
):

    operation = plan.get(
        "operation",
        "count"
    )

    metric = plan.get(
        "metric"
    )

    group_by = plan.get(
        "group_by"
    )

    filters = plan.get(
        "filters",
        {}
    )

    if not isinstance(
        filters,
        dict
    ):
        filters = {}

    # --------------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------------

    if operation == "anomaly":

        anomaly_result = detect_anomalies(
            df
        )

        date_range = filters.get(
            "date_range"
        )

        # ----------------------------------------------------
        # ONLY RESOLUTION-TIME ANOMALIES
        #
        # The question:
        # "Are there any anomalies in resolution times?"
        #
        # refers specifically to:
        # long_resolution_time
        #
        # It should NOT include:
        # old_unresolved_priority
        # ----------------------------------------------------

        resolution_anomalies = [
            anomaly
            for anomaly
            in anomaly_result.get(
                "anomalies",
                []
            )
            if anomaly.get("type")
            ==
            "long_resolution_time"
        ]

        # ----------------------------------------------------
        # DATE FILTER
        # ----------------------------------------------------

        if isinstance(
            date_range,
            dict
        ):

            period = date_range.get(
                "period"
            )

            if period in {
                "this_week",
                "last_7_days"
            }:

                reference_time = (
                    df["created_at"].max()
                )

                cutoff = (
                    reference_time
                    -
                    pd.Timedelta(days=7)
                )

                resolution_anomalies = [
                    anomaly
                    for anomaly
                    in resolution_anomalies
                    if pd.to_datetime(
                        anomaly["created_at"]
                    )
                    >= cutoff
                ]

        # ----------------------------------------------------
        # RETURN ONLY RESOLUTION-TIME ANOMALIES
        # ----------------------------------------------------

        anomaly_result["anomalies"] = (
            resolution_anomalies
        )

        anomaly_result["summary"] = {

            "total_anomalies":
                len(
                    resolution_anomalies
                ),

            "long_resolution_time":
                len(
                    resolution_anomalies
                ),

            "old_unresolved_priority":
                0
        }

        return anomaly_result

    # --------------------------------------------------------
    # DATE FILTER
    # --------------------------------------------------------

    date_range = filters.get(
        "date_range"
    )

    filtered_df = _apply_date_filter(
        df,
        date_range
    )

    # --------------------------------------------------------
    # OTHER FILTERS
    # --------------------------------------------------------

    filtered_df = _apply_filters(
        filtered_df,
        filters
    )

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    if operation == "count":

        return {
            "count": int(
                len(filtered_df)
            )
        }

    # --------------------------------------------------------
    # AVERAGE
    # --------------------------------------------------------

    if operation == "average":

        if metric not in filtered_df.columns:

            return {
                "average": None,
                "count": 0
            }

        values = pd.to_numeric(
            filtered_df[metric],
            errors="coerce"
        ).dropna()

        if len(values) == 0:

            return {
                "average": None,
                "count": 0
            }

        return {
            "average": round(
                float(values.mean()),
                2
            ),
            "count": int(
                len(values)
            )
        }

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    if operation == "list":

        columns = [
            "ticket_id",
            "created_at",
            "category",
            "priority",
            "status",
            "response_time_hrs",
            "resolution_time_hrs",
            "agent_id",
            "customer_rating",
            "issue_summary"
        ]

        available_columns = [
            column
            for column in columns
            if column in filtered_df.columns
        ]

        records = filtered_df[
            available_columns
        ].copy()

        if "created_at" in records.columns:

            records["created_at"] = (
                records["created_at"]
                .dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
            )

        return records.to_dict(
            orient="records"
        )

    # --------------------------------------------------------
    # GROUP BY
    # --------------------------------------------------------

    if operation == "group_by":

        if group_by not in filtered_df.columns:

            return []

        if metric == "ticket_id":
            grouped = (
                filtered_df
                .groupby(group_by)["ticket_id"]
                .count()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )

            return grouped.to_dict(
                orient="records"
            )


        if metric:
            if metric not in filtered_df.columns:
                return []

            grouped = (
                filtered_df
                .groupby(group_by)[metric]
                .agg(["count", "mean"])
                .reset_index()
            )

            grouped["mean"] = grouped["mean"].round(2)

            return grouped.to_dict(
                orient="records"
            )

        grouped = (
            filtered_df
            .groupby(group_by)
            .size()
            .reset_index(
                name="count"
            )
        )

        return grouped.to_dict(
            orient="records"
        )

    # --------------------------------------------------------
    # UNKNOWN OPERATION
    # --------------------------------------------------------

    raise ValueError(
        f"Unsupported operation: {operation}"
    )


# ============================================================
# ANSWER TEXT
# ============================================================

def answer_text(
    question: str,
    plan: dict,
    result
) -> str:

    operation = plan.get(
        "operation"
    )

    metric = plan.get(
        "metric"
    )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    if operation == "anomaly":

        summary = result.get(
            "summary",
            {}
        )

        total = summary.get(
            "total_anomalies",
            0
        )

        if total == 0:

            return (
                "No resolution-time anomalies "
                "were detected for the requested period."
            )

        return (
            f"I found {total} resolution-time "
            f"anomalies for the requested period."
        )

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    if operation == "count":

        count = result.get(
            "count",
            0
        )

        return (
            f"There are {count} tickets "
            f"matching the requested criteria."
        )

    # --------------------------------------------------------
    # AVERAGE
    # --------------------------------------------------------

    if operation == "average":

        average = result.get(
            "average"
        )

        count = result.get(
            "count",
            0
        )

        if average is None:

            return (
                f"No valid {metric} values "
                f"were found for the requested criteria."
            )

        return (
            f"The average {metric} is "
            f"{average} across {count} tickets."
        )

    # --------------------------------------------------------
    # LIST
    # --------------------------------------------------------

    if operation == "list":

        count = len(result)

        if count == 0:

            return (
                "No tickets matched "
                "the requested criteria."
            )

        return (
            f"I found {count} tickets "
            f"matching the requested criteria."
        )

    # --------------------------------------------------------
    # GROUP BY
    # --------------------------------------------------------

    if operation == "group_by":

        if not result:

            return (
                "No tickets matched "
                "the requested criteria."
            )

        group_by = plan.get(
            "group_by"
        )

        # ----------------------------------------------------
        # Find largest group
        # ----------------------------------------------------

        if all(
            "count" in row
            for row in result
        ):

            top = max(
                result,
                key=lambda row: row["count"]
            )

            return (
                f"{top[group_by]} has the most "
                f"tickets, with {top['count']} tickets."
            )

        return str(result)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return str(result)