"""Explicit builtin imports for pseudo-label acceptance policy metadata."""

from __future__ import annotations

_BUILTINS_LOADED = False


def load_builtin_pseudo_label_acceptance_policies() -> None:
    """Import builtin policy modules so local decorators register metadata."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from agent.src.services.training.acceptance_policies import (  # noqa: F401
        top1 as _top1,
    )

    _BUILTINS_LOADED = True
