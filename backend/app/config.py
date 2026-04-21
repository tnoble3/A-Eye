from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


@dataclass(frozen=True, slots=True)
class AppSettings:
    # These settings keep privacy-sensitive behavior explicit instead of
    # depending on incidental implementation details.
    stateless_processing: bool
    image_retention: str
    persist_request_logs: bool
    allow_private_image_hosts: bool
    fetch_timeout_s: float
    max_image_bytes: int
    cors_allow_origin_regex: str
    cache_control_header: str


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings(
        stateless_processing=_get_bool("A_EYE_STATELESS_PROCESSING", True),
        image_retention=os.getenv("A_EYE_IMAGE_RETENTION", "memory_only"),
        persist_request_logs=_get_bool("A_EYE_PERSIST_REQUEST_LOGS", False),
        allow_private_image_hosts=_get_bool("A_EYE_ALLOW_PRIVATE_IMAGE_HOSTS", False),
        fetch_timeout_s=_get_float("A_EYE_FETCH_TIMEOUT_S", 6.0),
        max_image_bytes=_get_int("A_EYE_MAX_IMAGE_BYTES", 5_000_000),
        cors_allow_origin_regex=os.getenv(
            "A_EYE_CORS_ALLOW_ORIGIN_REGEX",
            r"chrome-extension://.*|moz-extension://.*|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?",
        ),
        cache_control_header=os.getenv(
            "A_EYE_CACHE_CONTROL",
            "no-store, no-cache, must-revalidate, max-age=0",
        ),
    )
