"""Explicit builtin imports for federated aggregation methods."""

from __future__ import annotations

_BUILTINS_LOADED = False


def load_builtin_federated_aggregation_methods() -> None:
    """Import builtin aggregation modules so local decorators register metadata."""

    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from methods.federated.aggregation.fedavg import (  # noqa: F401
        classifier_head_fedavg as _classifier_head_fedavg,
    )
    from methods.federated.aggregation.fedavg import (  # noqa: F401
        diagonal_scale_fedavg as _diagonal_scale_fedavg,
    )
    from methods.federated.aggregation.fedavg import (  # noqa: F401
        lora_classifier_fedavg as _lora_classifier_fedavg,
    )

    _BUILTINS_LOADED = True
