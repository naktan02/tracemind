"""Explicit builtin imports for scoring backends."""

from __future__ import annotations

_BUILTINS_LOADED = False


def load_builtin_scoring_backends() -> None:
    """Import builtin backend modules so local decorators register factories."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from agent.src.services.inference.scoring_backends import (  # noqa: F401
        classifier_head_logits as _classifier_head_logits,
    )
    from agent.src.services.inference.scoring_backends import (  # noqa: F401
        prototype_similarity as _prototype_similarity,
    )

    _BUILTINS_LOADED = True
