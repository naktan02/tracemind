"""Explicit builtin imports for SSL hooks."""

from __future__ import annotations

_BUILTINS_LOADED = False


def load_builtin_ssl_hooks() -> None:
    """Import builtin hook modules so local decorators register factories."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from methods.ssl.hooks import selection as _selection_hooks  # noqa: F401

    _BUILTINS_LOADED = True
