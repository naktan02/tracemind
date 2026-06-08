"""Scoring backend helper functions."""

from __future__ import annotations

from .base import ScoringBackend


def resolve_scoring_backend_name(backend: ScoringBackend) -> str:
    """backend instance에서 canonical backend name을 읽는다."""

    backend_name = getattr(backend, "backend_name", None)
    return backend_name if isinstance(backend_name, str) else "unknown"


def resolve_scoring_confidence_kind(backend: ScoringBackend) -> str:
    """backend instance에서 score confidence kind를 읽는다."""

    confidence_kind = getattr(backend, "confidence_kind", None)
    if isinstance(confidence_kind, str) and confidence_kind.strip():
        return confidence_kind
    backend_name = resolve_scoring_backend_name(backend)
    if backend_name == "unknown":
        return "unknown"
    return f"{backend_name}_top1"
