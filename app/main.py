from __future__ import annotations

import os

import pandas as pd

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.responses import HTMLResponse

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

from pydantic import (
    BaseModel,
    Field
)

from .anomaly_detector import (
    detect_anomalies
)

from .data_loader import (
    load_tickets
)

from .llm_client import (
    OllamaClient,
    OllamaError
)

from .query_executor import (
    answer_text,
    execute_query
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent


DATA_PATH = Path(

    os.getenv(

        "DATA_PATH",

        str(

            BASE_DIR
            / "data"
            / "support_tickets.csv"
        )
    )
)


if not DATA_PATH.is_absolute():

    DATA_PATH = (
        BASE_DIR
        / DATA_PATH
    )


# ============================================================
# LOAD DATASET
# ============================================================

df = load_tickets(
    DATA_PATH
)


# ============================================================
# OLLAMA LLM
# ============================================================

llm = OllamaClient()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

templates = Jinja2Templates(

    directory=str(

        BASE_DIR
        / "app"
        / "templates"
    )
)


app = FastAPI(

    title="AI Support Ticket Analytics",

    version="1.0.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(

    "/static",

    StaticFiles(

        directory=str(

            BASE_DIR
            / "app"
            / "static"
        )
    ),

    name="static"
)


# ============================================================
# QUERY REQUEST
# ============================================================

class QueryRequest(BaseModel):

    question: str = Field(

        min_length=1,

        max_length=500
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get(

    "/",

    response_class=HTMLResponse
)
def home(
    request: Request
) -> Any:

    return templates.TemplateResponse(

        request=request,

        name="index.html",

        context={}
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:

    return {

        "status":
            "ok",

        "rows_loaded":
            int(
                len(df)
            ),

        "llm_model":
            llm.model
    }


# ============================================================
# BASIC STATISTICS
# ============================================================

@app.get("/stats")
def stats() -> dict[str, Any]:

    return {

        "total_tickets":
            int(
                len(df)
            ),

        "open_tickets":
            int(

                (
                    df["status"]
                    ==
                    "Open"
                ).sum()
            ),

        "resolved_tickets":
            int(

                (
                    df["status"]
                    ==
                    "Resolved"
                ).sum()
            ),

        "escalated_tickets":
            int(

                (
                    df["status"]
                    ==
                    "Escalated"
                ).sum()
            ),

        "average_response_time_hrs":
            round(

                float(

                    df[
                        "response_time_hrs"
                    ].mean()
                ),

                2
            )
    }


# ============================================================
# NATURAL LANGUAGE QUERY
# ============================================================

@app.post("/query")
def query(
    request: QueryRequest
) -> dict[str, Any]:

    try:

        # ----------------------------------------------------
        # STEP 1
        # LLM converts question into JSON plan
        # ----------------------------------------------------

        plan = llm.parse_question(

            request.question
        )

        # ----------------------------------------------------
        # STEP 2
        # ANOMALY QUERY
        # ----------------------------------------------------

        if (

            plan["operation"]
            ==
            "anomaly_check"
        ):

            result = detect_anomalies(
                df
            )

            question_lower = (
                request.question.lower()
            )

            # ------------------------------------------------
            # Resolution-time anomaly question
            # ------------------------------------------------

            if (
                "resolution"
                in question_lower
            ):

                resolution_anomalies = [

                    item

                    for item
                    in result[
                        "anomalies"
                    ]

                    if item[
                        "type"
                    ]
                    ==
                    "long_resolution_time"
                ]

                # --------------------------------------------
                # THIS WEEK / LAST 7 DAYS
                # --------------------------------------------

                date_range = (
                    plan
                    .get("filters", {})
                    .get("date_range")
                )

                if (

                    isinstance(
                        date_range,
                        dict
                    )

                    and

                    date_range.get(
                        "period"
                    )
                    in {
                        "this_week",
                        "last_7_days"
                    }
                ):

                    cutoff = (

                        df[
                            "created_at"
                        ].max()

                        -

                        pd.Timedelta(
                            days=7
                        )
                    )

                    resolution_anomalies = [

                        item

                        for item
                        in resolution_anomalies

                        if pd.to_datetime(

                            item[
                                "created_at"
                            ]

                        )
                        >= cutoff
                    ]

                # --------------------------------------------
                # Return filtered anomaly result
                # --------------------------------------------

                result = {

                    **result,

                    "summary": {

                        "total_anomalies":
                            len(
                                resolution_anomalies
                            ),

                        "long_resolution_time":
                            len(
                                resolution_anomalies
                            )
                    },

                    "anomalies":
                        resolution_anomalies
                }

            answer = (

                f"I found "
                f"{result['summary']['total_anomalies']} "
                f"anomalies."
            )

        # ----------------------------------------------------
        # STEP 3
        # NORMAL QUERY
        # ----------------------------------------------------

        else:

            result = execute_query(

                df,

                plan
            )

            answer = answer_text(

                request.question,

                plan,

                result
            )

        # ----------------------------------------------------
        # STEP 4
        # RETURN RESULT
        # ----------------------------------------------------

        return {

            "question":
                request.question,

            "interpreted_query":
                plan,

            "answer":
                answer,

            "data":
                result
        }

    # --------------------------------------------------------
    # EXPECTED USER / LLM ERROR
    # --------------------------------------------------------

    except (

        OllamaError,
        ValueError

    ) as exc:

        raise HTTPException(

            status_code=422,

            detail=str(exc)

        ) from exc

    # --------------------------------------------------------
    # UNEXPECTED ERROR
    # --------------------------------------------------------

    except Exception as exc:

        print(
            "Query error:",
            repr(exc)
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Could not complete "
                "the query"
            )

        ) from exc


# ============================================================
# ANOMALIES ENDPOINT
# ============================================================

@app.get("/anomalies")
def anomalies() -> dict[str, Any]:

    return detect_anomalies(
        df
    )