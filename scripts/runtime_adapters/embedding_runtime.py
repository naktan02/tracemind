"""scripts용 embedding runtime bridge."""

from __future__ import annotations

from shared.src.domain.services.embedding_adapter import EmbeddingAdapter
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def create_embedding_adapter(spec: EmbeddingAdapterSpec) -> EmbeddingAdapter:
    """agent embedding factory를 scripts에서 직접 import하지 않게 감싼다."""

    from agent.src.infrastructure.model_adapters.embedding.factory import (
        EmbeddingAdapterFactory,
    )

    return EmbeddingAdapterFactory.create(spec)


def resolve_runtime_device_name(device: str) -> str:
    """agent runtime device resolver를 scripts 경계에서 호출한다."""

    from agent.src.infrastructure.model_adapters.runtime_device import (
        resolve_runtime_device,
    )

    return resolve_runtime_device(device)
