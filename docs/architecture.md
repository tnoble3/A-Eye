# A-Eye Architecture

## System Direction

A-Eye uses a hybrid detection pipeline that combines:

- A retrained CNN baseline for learned visual patterns.
- Feature-level analysis for interpretable forensic signals.
- A lightweight aggregation layer that merges those signals into a single score.

## Client-Server Split

The browser extension is intentionally thin:

- It captures the selected image URL.
- It forwards the request to the FastAPI backend.
- It renders the backend score and supporting signals in the popup UI.

The backend owns the invasive analysis:

- Image fetching and validation.
- Feature extraction.
- CNN inference once deployment weights are ready.
- Score aggregation and response formatting.

Keeping analysis on the backend protects the extension from heavy compute,
simplifies future model updates, and centralizes privacy and scaling controls.

## Current Implementation Status

- FastAPI and Uvicorn are installed for the backend service.
- Request and response schemas are defined in `backend/app/api/schemas.py`.
- A stub hybrid pipeline is wired through the `/analyze` route.
- The browser extension can send an image URL to the backend and display the result.
- Docker artifacts are included for containerized backend deployment.
