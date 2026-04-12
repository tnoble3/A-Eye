# A-Eye

A-Eye is a browser-assisted image authenticity checker built around a thin
extension client and a FastAPI backend. The current prototype is organized to
support a hybrid detection pipeline that combines a CNN baseline with
feature-level forensic analysis.

## Project Layout

```text
backend/    FastAPI service, API routes, analysis pipeline, and tests
docs/       Architecture, privacy, Docker, training, and evaluation notes
extension/  Browser extension service worker and popup UI
ml/         Baseline model training workspace
```

## Current Prototype Focus

- Structured repository layout for backend, ML, docs, extension, and tests
- FastAPI and Uvicorn backend setup
- Request and response schemas for the `/analyze` endpoint
- Stub hybrid pipeline with feature analysis and a CNN deployment placeholder
- Dockerized backend workflow for local container testing
- Early UI direction for the browser popup

## Running The Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Running With Docker

```bash
docker compose -f docs/docker-compose.yml up --build
```

## Documentation

- `docs/architecture.md`
- `docs/privacy.md`
- `docs/model-evaluation.md`
- `docs/model-training.md`
- `docs/ui-direction.md`
