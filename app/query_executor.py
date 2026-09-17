from __future__ import annotations

from typing import Any

import pandas as pd


# ============================================================
# DATE FILTER
# ============================================================

def _apply_date_filter(
    df: pd.DataFrame,
    date_range: dict[str, Any],
    use_resolution_date: bool = False
) -> pd.DataFrame:

    if not date_range:
        return df

    if df.empty:
        return df

    period = date_range.get(
        "period"
    )

    # --------------------------------------------------------
    # USE ACTUAL RESOLUTION DATE
    #
    # resolution date =
    # created_at + resolution_time_hrs
    # --------------------------------------------------------

    if use_resolution_date:

        result = df[
            df[
                "resolution_time_hrs"
            ].notna()
        ].copy()

        result["resolved_at"] = (

            result["created_at"]

            +

            pd.to_timedelta(

                result[
                    "resolution_time_hrs"
                ],

                unit="h"
            )
        )

        date_column = (
            result["resolved_at"]
        )

    else:

        result = df.copy()

        date_column = (
            result["created_at"]
        )

    if result.empty:
        return result

    latest_date = (
        date_column.max()
    )

    # --------------------------------------------------------
    # THIS MONTH
    # --------------------------------------------------------

    if period == "this_month":

        return result[
            date_column.dt.to_period("M")
            ==
            latest_date.to_period("M")
        ]

    # --------------------------------------------------------
    # THIS WEEK / LAST 7 DAYS
    # --------------------------------------------------------

    if period in {
        "this_week",
        "last_7_days"
    }:

        cutoff = (

            latest_date

            -

            pd.Timedelta(
                days=7
            )
        )

        return result[
            date_column >= cutoff
        ]

    return result


# ============================================================
# APPLY FILTERS
# ============================================================

def _apply_filters(
    df: pd.DataFrame,
    filters: dict[str, Any],
    plan_metric: str | None = None
) -> pd.DataFrame:

    result = df.copy()

    filters = dict(
        filters
    )

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    date_range = filters.pop(
        "date_range",
        None
    )

    if isinstance(
        date_range,
        dict
    ):

        result = _apply_date_filter(
            result,
            date_range,
            False
        )

    # --------------------------------------------------------
    # CRITICAL NOT RESOLVED WITHIN X HOURS
    #
    # Critical AND
    #
    # (
    #     Open
    #     OR Escalated
    #     OR resolution_time > X
    # )
    # --------------------------------------------------------

    custom_flag = filters.pop(

        "custom_unresolved_critical_gt_12",

        False
    )

    limit_hours = filters.pop(

        "critical_resolution_limit_hours",

        12
    )

    if custom_flag:

        result = result[

            (
                result[
                    "priority"
                ]
                ==
                "Critical"
            )

            &

            (

                result[
                    "status"
                ].isin(
                    [
                        "Open",
                        "Escalated"
                    ]
                )

                |

                (

                    result[
                        "resolution_time_hrs"
                    ]

                    >

                    float(
                        limit_hours
                    )
                )
            )
        ]

    # --------------------------------------------------------
    # ALLOWED TEXT COLUMNS
    # --------------------------------------------------------

    allowed_columns = {

        "category",
        "priority",
        "status",
        "agent_id",
        "ticket_id"
    }

    for column, value in filters.items():

        if column in allowed_columns:

            if isinstance(
                value,
                list
            ):

                result = result[
                    result[
                        column
                    ].isin(value)
                ]

            else:

                result = result[
                    result[
                        column
                    ]
                    ==
                    value
                ]

        # ----------------------------------------------------
        # NUMERIC FILTERS
        # ----------------------------------------------------

        elif column in {

            "response_time_hrs",
            "resolution_time_hrs",
            "customer_rating"
        }:

            if not isinstance(
                value,
                dict
            ):

                continue

            for operator, number in value.items():

                number = float(
                    number
                )

                if operator == "greater_than":

                    result = result[
                        result[column]
                        >
                        number
                    ]

                elif operator == "greater_than_or_equal":

                    result = result[
                        result[column]
                        >=
                        number
                    ]

                elif operator == "less_than":

                    result = result[
                        result[column]
                        <
                        number
                    ]

                elif operator == "less_than_or_equal":

                    result = result[
                        result[column]
                        <=
                        number
                    ]

    return result


# ============================================================
# EXECUTE QUERY
# ============================================================

def execute_query(
    df: pd.DataFrame,
    plan: dict[str, Any]
) -> dict[str, Any]:

    operation = plan.get(
        "operation"
    )

    filters = plan.get(
        "filters"
    ) or {}

    filtered = _apply_filters(

        df,

        filters,

        plan.get("metric")
    )

    # ========================================================
    # COUNT
    # ========================================================

    if operation == "count":

        return {

            "count":
                int(
                    len(filtered)
                )
        }

    # ========================================================
    # AVERAGE
    # ========================================================

    if operation == "average":

        metric = plan.get(
            "metric"
        )

        if metric not in {

            "response_time_hrs",
            "resolution_time_hrs",
            "customer_rating"
        }:

            raise ValueError(
                "Average metric is not supported"
            )

        values = filtered[
            metric
        ].dropna()

        return {

            "metric":
                metric,

            "count":
                int(
                    len(values)
                ),

            "average":

                (
                    round(
                        float(
                            values.mean()
                        ),
                        2
                    )

                    if len(values)

                    else None
                )
        }

    # ========================================================
    # LIST
    # ========================================================

    if operation == "list":

        rows = filtered.copy()

        rows["created_at"] = (

            rows[
                "created_at"
            ]

            .dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

        # Remove helper column before returning
        if "resolved_at" in rows.columns:

            rows = rows.drop(
                columns=[
                    "resolved_at"
                ]
            )

        rows = rows.where(
            pd.notna(rows),
            None
        )

        return {

            "count":
                int(
                    len(rows)
                ),

            "tickets":
                rows.to_dict(
                    orient="records"
                )[:100]
        }

    # ========================================================
    # GROUP BY
    # ========================================================

    if operation == "group_by":

        group_by = plan.get(
            "group_by"
        )

        metric = plan.get(
            "metric",
            "count"
        )

        if group_by not in {

            "agent_id",
            "category",
            "priority",
            "status"
        }:

            raise ValueError(
                "Group-by field is not supported"
            )

        # ----------------------------------------------------
        # RESOLVED TICKET COUNT
        # ----------------------------------------------------

        if metric == "resolved_ticket_count":

            filtered = filtered[
                filtered[
                    "status"
                ]
                ==
                "Resolved"
            ]

            grouped = (

                filtered

                .groupby(
                    group_by
                )

                .size()

                .reset_index(
                    name="count"
                )
            )

            grouped = (
                grouped
                .sort_values(
                    "count",
                    ascending=False
                )
            )

        # ----------------------------------------------------
        # NORMAL COUNT
        # ----------------------------------------------------

        elif metric == "count":

            grouped = (

                filtered

                .groupby(
                    group_by
                )

                .size()

                .reset_index(
                    name="count"
                )
            )

            grouped = (
                grouped
                .sort_values(
                    "count",
                    ascending=False
                )
            )

        # ----------------------------------------------------
        # AVERAGE CUSTOMER RATING
        # ----------------------------------------------------

        elif metric == (
            "average_customer_rating"
        ):

            grouped = (

                filtered

                .groupby(
                    group_by
                )[
                    "customer_rating"
                ]

                .mean()

                .round(2)

                .reset_index(
                    name=
                    "average_customer_rating"
                )

                .sort_values(
                    "average_customer_rating",
                    ascending=False
                )
            )

        else:

            raise ValueError(
                "Group-by metric is not supported"
            )

        return {

            "groups":
                grouped.to_dict(
                    orient="records"
                )
        }

    raise ValueError(
        f"Unsupported query operation: "
        f"{operation}"
    )


# ============================================================
# CREATE HUMAN-READABLE ANSWER
# ============================================================

def answer_text(
    question: str,
    plan: dict[str, Any],
    result: dict[str, Any]
) -> str:

    operation = plan.get(
        "operation"
    )

    # ========================================================
    # COUNT
    # ========================================================

    if operation == "count":

        return (

            f"There are "
            f"{result['count']} "
            f"matching tickets."
        )

    # ========================================================
    # AVERAGE
    # ========================================================

    if operation == "average":

        if result[
            "average"
        ] is None:

            return (
                "There is not enough data "
                "to calculate that average."
            )

        return (

            f"The average "
            f"{result['metric']} is "
            f"{result['average']} across "
            f"{result['count']} tickets."
        )

    # ========================================================
    # LIST
    # ========================================================

    if operation == "list":

        return (

            f"I found "
            f"{result['count']} "
            f"matching tickets."
        )

    # ========================================================
    # GROUP BY
    # ========================================================

    if operation == "group_by":

        groups = result.get(
            "groups",
            []
        )

        if not groups:

            return (
                "No matching grouped results "
                "were found."
            )

        # ----------------------------------------------------
        # RESOLVED TICKET RANKING
        # ----------------------------------------------------

        if plan.get(
            "metric"
        ) == "resolved_ticket_count":

            top_count = groups[
                0
            ][
                "count"
            ]

            group_column = plan.get(
                "group_by"
            )

            leaders = [

                row[
                    group_column
                ]

                for row in groups

                if row.get(
                    "count"
                )
                ==
                top_count
            ]

            if len(leaders) == 1:

                return (

                    f"{leaders[0]} "
                    f"resolved the most tickets, "
                    f"with {top_count} "
                    f"resolved tickets."
                )

            return (

                f"The top agents are "
                f"{', '.join(leaders)}, "
                f"tied at {top_count} "
                f"resolved tickets."
            )

        # ----------------------------------------------------
        # OTHER GROUPED QUERIES
        # ----------------------------------------------------

        return (
            "Here are the grouped results, "
            "sorted from highest to lowest."
        )

    return (
        "The query was completed."
    )