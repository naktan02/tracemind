"""Agent global/local adapter state composition service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from shared.src.contracts.model_contracts import ModelManifestPayload
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState


class SharedAdapterRuntimeProvider(Protocol):
    """Agent가 캐시한 global shared adapter state 제공자."""

    def get_active_state(self) -> SharedAdapterState:
        """현재 active shared adapter state를 반환한다."""
        ...

    def get_active_manifest(self) -> ModelManifestPayload:
        """현재 active shared adapter manifest를 반환한다."""
        ...


class LocalAdapterRuntimeProvider(Protocol):
    """Agent-private local adapter state 제공자."""

    def get_active_state(self) -> SharedAdapterState:
        """현재 active local/private adapter state를 반환한다."""
        ...


@dataclass(frozen=True, slots=True)
class AdapterRuntimeContext:
    """Agent inference/training이 사용할 adapter state 조합 컨텍스트."""

    shared_manifest: ModelManifestPayload | None = None
    shared_state: SharedAdapterState | None = None
    local_state: SharedAdapterState | None = None

    @property
    def shared_model_revision(self) -> str | None:
        if self.shared_state is None:
            return None
        return self.shared_state.model_revision

    @property
    def local_adapter_revision(self) -> str | None:
        if self.local_state is None:
            return None
        return self.local_state.model_revision

    def require_shared_manifest(self) -> ModelManifestPayload:
        if self.shared_manifest is None:
            raise FileNotFoundError("No active shared adapter manifest is cached.")
        return self.shared_manifest

    def require_shared_state(self) -> SharedAdapterState:
        if self.shared_state is None:
            raise FileNotFoundError("No active shared adapter state is cached.")
        return self.shared_state

    def apply_shared(self, embedding: Sequence[float]) -> list[float]:
        values = [float(value) for value in embedding]
        if self.shared_state is None:
            return values
        return self.shared_state.apply(values)

    def apply_for_inference(self, embedding: Sequence[float]) -> list[float]:
        values = self.apply_shared(embedding)
        if self.local_state is None:
            return values
        return self.local_state.apply(values)

    def model_revision_for_record(self, fallback_revision: str) -> str:
        return self.shared_model_revision or fallback_revision

    def query_buffer_metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if self.shared_state is not None:
            metadata.update(
                {
                    "adapter_kind": self.shared_state.adapter_kind,
                    "shared_adapter_kind": self.shared_state.adapter_kind,
                    "shared_model_revision": self.shared_state.model_revision,
                }
            )
        if self.local_state is not None:
            metadata.update(
                {
                    "local_adapter_kind": self.local_state.adapter_kind,
                    "local_adapter_revision": self.local_state.model_revision,
                }
            )
        return metadata


@dataclass(slots=True)
class AdapterCompositionService:
    """Global shared state와 agent-private local state의 조합 지점을 제공한다."""

    shared_adapter_provider: SharedAdapterRuntimeProvider | None = None
    local_adapter_provider: LocalAdapterRuntimeProvider | None = None

    def get_context(self, *, require_shared: bool = False) -> AdapterRuntimeContext:
        shared_manifest = None
        shared_state = None
        if self.shared_adapter_provider is not None:
            shared_manifest = self.shared_adapter_provider.get_active_manifest()
            shared_state = self.shared_adapter_provider.get_active_state()
        elif require_shared:
            raise FileNotFoundError("No shared adapter provider is configured.")

        local_state = None
        if self.local_adapter_provider is not None:
            try:
                local_state = self.local_adapter_provider.get_active_state()
            except FileNotFoundError:
                local_state = None

        return AdapterRuntimeContext(
            shared_manifest=shared_manifest,
            shared_state=shared_state,
            local_state=local_state,
        )
