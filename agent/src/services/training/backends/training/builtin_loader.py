"""Explicit builtin imports for shared adapter training backends."""

from __future__ import annotations

_BUILTINS_LOADED = False


def load_builtin_shared_adapter_training_backends() -> None:
    """Import builtin backend modules so local decorators register factories."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from agent.src.services.training.backends.training import (  # noqa: F401
        diagonal_scale_heuristic as _diagonal_scale_heuristic,
    )
    from agent.src.services.training.backends.training.lora_classifier import (  # noqa: F401
        backend as _lora_classifier_backend,
    )

    _BUILTINS_LOADED = True
