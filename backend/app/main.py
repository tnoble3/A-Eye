from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.api.routes import router

settings = get_settings()

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
    allow_origins=[],
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def apply_privacy_headers(request, call_next):
    response = await call_next(request)
    #these headers focus on privacy and secutity, in addition to opting out of caching
    response.headers["Cache-Control"] = settings.cache_control_header
    response.headers["Pragma"] = "no-cache"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Permissions-Policy"] = "browsing-topics=(), interest-cohort=()"
    response.headers["X-A-Eye-Processing-Mode"] = (
        "stateless" if settings.stateless_processing else "stateful"
    )
    return response

app.include_router(router)
