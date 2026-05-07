"""Explicit builtin imports for training example backends."""

from __future__ import annotations

_BUILTINS_LOADED = False


def load_builtin_training_example_backends() -> None:
    """Import builtin backend modules so local decorators register factories."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from agent.src.services.training.backends.inputs import (  # noqa: F401
        prototype_rescore as _prototype_rescore,
    )
    from agent.src.services.training.backends.inputs import (  # noqa: F401
        weak_strong_pair as _weak_strong_pair,
    )

    _BUILTINS_LOADED = True
