from __future__ import annotations

import pandas as pd


def detect_anomalies(df: pd.DataFrame) -> dict:
    """Find transparent, rule-based support-ticket anomalies."""
    reference_time = df["created_at"].max()

    unresolved = df["status"].isin(["Open", "Escalated"])
    high_priority = df["priority"].isin(["High", "Critical"])
    age_hours = (reference_time - df["created_at"]).dt.total_seconds() / 3600
    old_unresolved = df[unresolved & high_priority & (age_hours > 24)].copy()

    resolved_times = df.loc[df["resolution_time_hrs"].notna(), "resolution_time_hrs"]
    if len(resolved_times):
        q1 = float(resolved_times.quantile(0.25))
        q3 = float(resolved_times.quantile(0.75))
        iqr = q3 - q1
        long_threshold = q3 + 1.5 * iqr
    else:
        q1 = q3 = long_threshold = 0.0

    long_resolution = df[
        df["resolution_time_hrs"].notna()
        & (df["resolution_time_hrs"] > long_threshold)
    ].copy()

    anomalies = []
    for _, row in old_unresolved.iterrows():
        anomalies.append(
            {
                "ticket_id": row["ticket_id"],
                "type": "old_unresolved_priority",
                "reason": "High or Critical ticket is unresolved and older than 24 hours",
                "priority": row["priority"],
                "status": row["status"],
                "age_hours": round(float((reference_time - row["created_at"]).total_seconds() / 3600), 2),
            }
        )

    for _, row in long_resolution.iterrows():
        anomalies.append(
            {
                "ticket_id": row["ticket_id"],
                "type": "long_resolution_time",
                "reason": f"Resolution time exceeds the IQR threshold of {long_threshold:.2f} hours",
                "priority": row["priority"],
                "status": row["status"],
                "resolution_time_hrs": round(float(row["resolution_time_hrs"]), 2),
            }
        )

    return {
        "reference_time": reference_time.strftime("%Y-%m-%d %H:%M"),
        "long_resolution_threshold_hrs": round(long_threshold, 2),
        "summary": {
            "total_anomalies": len(anomalies),
            "old_unresolved_priority": len(old_unresolved),
            "long_resolution_time": len(long_resolution),
        },
        "anomalies": anomalies,
    }
