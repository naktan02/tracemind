"""Training example backend compatibility checks."""

from __future__ import annotations

from .base import ANY_ADAPTER_KIND, TrainingExampleBackend


def require_training_example_backend_adapter_kind_support(
    *,
    backend: TrainingExampleBackend,
    adapter_kind: str,
) -> None:
    """example backend가 training backend adapter_kind를 지원하는지 검증한다."""

    normalized_supported = tuple(
        value.strip().lower() for value in backend.supported_adapter_kinds
    )
    normalized_adapter_kind = adapter_kind.strip().lower()
    if (
        ANY_ADAPTER_KIND in normalized_supported
        or normalized_adapter_kind in normalized_supported
    ):
        return
    raise ValueError(
        f"Incompatible training example backend: {backend.backend_name} does not "
        f"support adapter_kind={adapter_kind}."
    )
