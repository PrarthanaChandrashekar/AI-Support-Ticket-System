from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]


def load_tickets(path: str | Path) -> pd.DataFrame:
    """Load and normalize the support-ticket CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if df["created_at"].isna().any():
        raise ValueError("Some created_at values could not be parsed")

    for column in ["response_time_hrs", "resolution_time_hrs", "customer_rating"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df
