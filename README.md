# AI Support Ticket Analytics

A Python/FastAPI prototype for querying a customer-support ticket CSV in natural language and detecting operational anomalies.

## Features

The application loads `data/support_tickets.csv`, uses a local Ollama model to translate natural-language questions into validated JSON query plans, executes those plans deterministically with Pandas, detects anomalies, exposes a REST API, and provides a minimal browser UI.

The LLM understands the question. Python calculates the factual result. This separation reduces hallucinated numbers.

## Requirements

- Python 3.10+
- Ollama
- The `llama3.2:3b` model

Install and download the model:

```bash
ollama pull llama3.2:3b
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to change the defaults. The default configuration is:

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
DATA_PATH=data/support_tickets.csv
```

Start Ollama in another terminal if needed:

```bash
ollama serve
```

Start the application:

```bash
python run.py
```

Open the UI at <http://127.0.0.1:8000/>. Open the interactive REST API documentation at <http://127.0.0.1:8000/docs>.

## API endpoints

### Health

```http
GET /health
```

### Statistics

```http
GET /stats
```

### Natural-language query

```http
POST /query
Content-Type: application/json

{"question":"How many critical tickets are unresolved?"}
```

### Anomalies

```http
GET /anomalies
```

The anomaly detector uses two transparent rules:

1. High or Critical tickets that are Open or Escalated and more than 24 hours old relative to the latest timestamp in the dataset.
2. Resolved tickets whose resolution time exceeds `Q3 + 1.5 * IQR`.

## Example questions

- How many tickets are currently open?
- How many critical tickets are unresolved?
- What is the average customer rating for Technical tickets?
- Which agent resolved the most tickets?
- Show Critical tickets with resolution time greater than 12 hours.
- Are there any anomalies in resolution times?

## Architecture

```text
Browser UI / REST client
          |
       FastAPI
          |
   Ollama query planner
          |
   Validated JSON plan
          |
   Pandas query executor ---- anomaly detector
          |
      Exact JSON result
```

The query planner is allow-listed and validated. It cannot execute arbitrary Python or SQL. The executor supports counts, averages, lists, and grouped results over known fields.

## Tests

Run:

```bash
pytest -q
```

## Limitations

The natural-language parser supports the operations documented above. It is not a general SQL engine. The local LLM must be running for `/query`; health, statistics, and anomaly endpoints do not require the LLM. For a production deployment, the dataset should move to a database and the service should add authentication, logging, caching, and stronger schema validation.

## Example anomaly output

The anomaly detector identified 101 anomalies in the supplied dataset:

- 80 unresolved High or Critical tickets older than 24 hours.
- 21 resolved tickets with unusually long resolution times.
- The long-resolution threshold was 48.15 hours.

Ticket age is calculated relative to the latest `created_at` timestamp in the dataset so that results are reproducible.
