"""Explicit builtin imports for shared adapter aggregation backends."""

from __future__ import annotations

_BUILTINS_LOADED = False


def load_builtin_shared_adapter_aggregation_backends() -> None:
    """Import builtin backend modules so local decorators register factories."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from main_server.src.services.federation.rounds.aggregation import (  # noqa: F401
        classifier_head as _classifier_head,
    )
    from main_server.src.services.federation.rounds.aggregation import (  # noqa: F401
        diagonal_scale as _diagonal_scale,
    )
    from main_server.src.services.federation.rounds.aggregation import (  # noqa: F401
        lora_classifier as _lora_classifier,
    )

    _BUILTINS_LOADED = True
