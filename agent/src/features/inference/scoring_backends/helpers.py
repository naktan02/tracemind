"""Scoring backend helper functions."""

from __future__ import annotations

from .base import ScoringBackend


def resolve_scoring_backend_name(backend: ScoringBackend) -> str:
    """backend instance에서 canonical backend name을 읽는다."""

    backend_name = getattr(backend, "backend_name", None)
    return backend_name if isinstance(backend_name, str) else "unknown"
