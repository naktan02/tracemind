"""Explicit builtin imports for pseudo-label evidence backends."""

from __future__ import annotations

_BUILTINS_LOADED = False


def load_builtin_pseudo_label_evidence_backends() -> None:
    """Import builtin backend modules so local decorators register factories."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from agent.src.services.training.backends.evidence import (  # noqa: F401
        prototype_similarity as _prototype_similarity,
    )

    _BUILTINS_LOADED = True
