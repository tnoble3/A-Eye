from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="A-Eye API",
    version="0.1.0",
    description=(
        "Backend analysis service for the A-Eye browser extension. "
        "The extension stays thin while the API hosts the heavier detection logic."
    ),
)
app.add_middleware(
    CORSMiddleware,
    # Early extension work benefits from permissive CORS so the client can evolve
    # without being blocked by local origin changes.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
