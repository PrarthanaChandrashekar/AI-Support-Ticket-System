# Beginner Guide: Run This Project in VS Code

## 1. Open the project

Open the folder `support-ticket-ai` in VS Code.

## 2. Open the VS Code terminal

Use **Terminal > New Terminal**.

## 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Make sure Ollama is ready

Open another terminal and run:

```bash
ollama pull llama3.2:3b
ollama serve
```

If Ollama is already running as a desktop application, `ollama serve` may report that the port is already in use. That is okay.

Test the model:

```bash
ollama run llama3.2:3b
```

Type a short question, then type `/bye` to exit.

## 6. Start the application

In the VS Code terminal, from the project root:

```bash
python run.py
```

## 7. Open the application

Open these addresses in your browser:

- UI: http://127.0.0.1:8000/
- REST API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Statistics: http://127.0.0.1:8000/stats
- Anomalies: http://127.0.0.1:8000/anomalies

## 8. Try these questions in the UI

- How many tickets are currently open?
- How many critical tickets are unresolved?
- What is the average customer rating for Technical tickets?
- Which agent resolved the most tickets?
- Show Critical tickets with resolution time greater than 12 hours.
- Are there any anomalies in resolution times?

## 9. Run tests

Stop the server with `Ctrl+C` if necessary, then run:

```bash
pytest -q
```

The expected result is four passing tests.

## 10. Understand the important files

- `app/data_loader.py`: reads and validates the CSV.
- `app/llm_client.py`: asks Ollama to convert English into a JSON query plan.
- `app/query_executor.py`: calculates exact answers with Pandas.
- `app/anomaly_detector.py`: detects long-resolution and old unresolved priority tickets.
- `app/main.py`: defines the FastAPI routes.
- `app/templates/index.html`: browser page.
- `app/static/app.js`: browser-to-API calls.
- `tests/test_core.py`: automated tests.

The LLM understands the question, but Python calculates the final numbers. This is the key design decision to explain in the interview.

## 11. Upload to GitHub

Create a new GitHub repository, then run these commands in the project terminal:

```bash
git init
git add .
git commit -m "Build AI support ticket analytics system"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Do not upload `.env` or API keys. This project uses local Ollama and does not need a secret key.
