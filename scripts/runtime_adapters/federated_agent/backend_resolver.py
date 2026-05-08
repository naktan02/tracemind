"""FL simulation agent runtime backend resolver."""

from __future__ import annotations

from typing import Any


def resolve_example_generation_backend_name(
    *,
    objective_config: Any | None,
) -> str:
    """objective config에서 training example backend 이름을 정한다."""

    from methods.federated_ssl.runtime_fallbacks import (
        RUNTIME_FALLBACK_TRAINING_PROFILE,
    )

    if objective_config is None:
        return RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    return (
        objective_config.example_generation_backend_name
        or RUNTIME_FALLBACK_TRAINING_PROFILE.example_generation_backend_name
    )


def resolve_federated_training_backend_adapter_kind(
    *,
    objective_config: Any,
) -> str:
    """simulation config 검증용으로 local training backend의 adapter kind를 읽는다."""

    from methods.adaptation.local_update_registry import (
        build_shared_adapter_training_backend,
    )

    backend = build_shared_adapter_training_backend(
        objective_config.training_backend_name,
        objective_config=objective_config,
    )
    return str(backend.adapter_kind)
