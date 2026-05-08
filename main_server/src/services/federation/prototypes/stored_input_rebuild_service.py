"""저장된 canonical input 기반 prototype rebuild orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from main_server.src.services.federation.prototypes.models import (
    PrototypeRebuildInputRecord,
    PrototypeRebuildResult,
    ReferencePrototypeRebuildRequest,
    StoredReferencePrototypeRebuildRequest,
)
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


class PrototypeRebuildInputRepositoryProtocol(Protocol):
    """canonical prototype rebuild input 저장소 protocol."""

    def load_input(self, input_id: str) -> PrototypeRebuildInputRecord:
        """지정된 canonical rebuild input을 읽는다."""

    def load_active_input(self) -> PrototypeRebuildInputRecord:
        """현재 활성 canonical rebuild input을 읽는다."""


class ReferenceRowPrototypeRebuildServiceProtocol(Protocol):
    """reference row rebuild를 수행할 수 있는 runtime service protocol."""

    def rebuild_from_reference_rows(
        self,
        request: ReferencePrototypeRebuildRequest,
    ) -> PrototypeRebuildResult:
        """reference row로부터 rebuild를 실행한다."""


class EmbeddingAdapterFactoryProtocol(Protocol):
    """EmbeddingAdapterSpec으로 runtime adapter를 생성하는 protocol."""

    def create(self, spec: EmbeddingAdapterSpec) -> EmbeddingAdapter:
        """spec으로부터 임베딩 adapter를 생성한다."""


def _default_prototype_rebuild_service() -> ReferenceRowPrototypeRebuildServiceProtocol:
    from .prototype_rebuild_service import (
        PrototypeRebuildService,
    )

    return PrototypeRebuildService()


@dataclass(slots=True)
class StoredReferencePrototypeRebuildService:
    """저장된 canonical input을 읽어 runtime rebuild를 실행한다."""

    input_repository: PrototypeRebuildInputRepositoryProtocol
    adapter_factory: EmbeddingAdapterFactoryProtocol
    prototype_rebuild_service: ReferenceRowPrototypeRebuildServiceProtocol = field(
        default_factory=_default_prototype_rebuild_service
    )

    def rebuild(
        self,
        request: StoredReferencePrototypeRebuildRequest,
    ) -> PrototypeRebuildResult:
        input_record = (
            self.input_repository.load_active_input()
            if request.input_id is None
            else self.input_repository.load_input(request.input_id)
        )
        adapter = self.adapter_factory.create(input_record.embedding_spec)
        rebuild_result = self.prototype_rebuild_service.rebuild_from_reference_rows(
            ReferencePrototypeRebuildRequest(
                rows=input_record.rows,
                adapter=adapter,
                adapter_state=request.adapter_state,
                prototype_version=request.prototype_version,
                embedding_model_id=request.embedding_model_id,
                embedding_model_revision=request.embedding_model_revision,
                embedding_backend=input_record.embedding_spec.backend,
                mapping_version=input_record.mapping_version,
                built_at=request.built_at,
                normalize_embeddings=input_record.normalize_embeddings,
                task_prefix=input_record.embedding_spec.task_prefix,
                translation_model_id=input_record.translation_model_id,
                translation_model_revision=input_record.translation_model_revision,
                translation_direction=input_record.translation_direction,
                required_categories=input_record.required_categories,
            )
        )
        return replace(rebuild_result, source_input_id=input_record.input_id)
