"""Prototype rebuild와 publication 조정 서비스."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from main_server.src.services.prototypes.prototype_build_state_service import (
    PrototypeBuildStateService,
)
from main_server.src.services.prototypes.prototype_pack_service import (
    PrototypePackService,
)
from shared.src.contracts.prototype_build_state_contracts import (
    PrototypeBuildStatePayload,
    dump_prototype_build_state_payload,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    dump_prototype_pack_payload,
)
from shared.src.domain.entities.training.vector_adapter_state import VectorAdapterState
from shared.src.domain.services.clock import Clock, SystemUtcClock
from shared.src.services.prototypes.build_strategies import (
    PrototypeBuildArtifacts,
    PrototypeBuildRequest,
    PrototypeBuildStrategy,
    SinglePrototypeBuildStrategy,
)


@dataclass(slots=True)
class PrototypeRebuildRequest:
    """runtime rebuild에 필요한 canonical build 입력."""

    build_request: PrototypeBuildRequest


@dataclass(slots=True, frozen=True)
class PrototypeRebuildInputRecord:
    """server-owned canonical prototype rebuild 입력."""

    input_id: str
    embedding_spec: EmbeddingAdapterSpec
    rows: tuple["ReferencePrototypeSourceRow", ...]
    mapping_version: str
    normalize_embeddings: bool = True
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    required_categories: tuple[str, ...] | None = None


@dataclass(slots=True)
class StoredReferencePrototypeRebuildRequest:
    """저장된 canonical input 기반 rebuild 요청."""

    adapter_state: VectorAdapterState
    prototype_version: str
    embedding_model_id: str
    embedding_model_revision: str
    input_id: str | None = None
    built_at: datetime | None = None


@dataclass(slots=True)
class PrototypeRebuildResult:
    """rebuild와 publication 이후의 결과 요약."""

    pack_payload: PrototypePackPayload
    build_state_payload: PrototypeBuildStatePayload | None
    source_input_id: str | None = None
    published_pack_path: Path | None = None
    published_build_state_path: Path | None = None
    reference_pack_path: Path | None = None
    reference_build_state_path: Path | None = None


class PrototypePublicationStrategy(Protocol):
    """rebuild 결과물을 어디에 어떻게 발행할지 결정하는 전략 인터페이스."""

    def publish(
        self,
        build_artifacts: PrototypeBuildArtifacts,
    ) -> PrototypeRebuildResult:
        """빌드 결과물을 발행하고 경로를 반환한다."""


@dataclass(slots=True)
class InMemoryPrototypePublicationStrategy:
    """빌드 결과를 메모리에서만 반환하는 publication 전략."""

    def publish(
        self,
        build_artifacts: PrototypeBuildArtifacts,
    ) -> PrototypeRebuildResult:
        return PrototypeRebuildResult(
            pack_payload=build_artifacts.pack_payload,
            build_state_payload=build_artifacts.build_state_payload,
        )


@dataclass(slots=True)
class ReferencePrototypeSourceRow:
    """reference rebuild용 canonical row 표현."""

    text: str
    category: str


@dataclass(slots=True)
class ReferencePrototypeRebuildRequest:
    """reference row 기반 prototype rebuild 요청."""

    rows: tuple[ReferencePrototypeSourceRow, ...] | list[ReferencePrototypeSourceRow]
    adapter: Any
    adapter_state: VectorAdapterState
    prototype_version: str
    embedding_model_id: str
    embedding_model_revision: str
    embedding_backend: str
    mapping_version: str
    built_at: datetime | None = None
    normalize_embeddings: bool = True
    task_prefix: str = ""
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    required_categories: tuple[str, ...] | list[str] | None = None


class PrototypeRebuildInputRepositoryProtocol(Protocol):
    """canonical prototype rebuild input 저장소 protocol."""

    def load_input(self, input_id: str) -> PrototypeRebuildInputRecord:
        """지정된 canonical rebuild input을 읽는다."""

    def load_active_input(self) -> PrototypeRebuildInputRecord:
        """현재 활성 canonical rebuild input을 읽는다."""


@dataclass(slots=True)
class ReferenceRebuildPrototypePublicationStrategy:
    """reference rebuild 산출물을 local copy와 main_server state에 함께 발행한다."""

    reference_pack_output_dir: Path | None = None
    reference_build_state_output_dir: Path | None = None
    prototype_pack_service: PrototypePackService = field(
        default_factory=PrototypePackService
    )
    prototype_build_state_service: PrototypeBuildStateService = field(
        default_factory=PrototypeBuildStateService
    )

    def publish(
        self,
        build_artifacts: PrototypeBuildArtifacts,
    ) -> PrototypeRebuildResult:
        pack_payload = build_artifacts.pack_payload
        build_state_payload = build_artifacts.build_state_payload
        prototype_version = pack_payload.prototype_version

        reference_pack_path: Path | None = None
        if self.reference_pack_output_dir is not None:
            reference_pack_path = (
                self.reference_pack_output_dir / f"{prototype_version}.json"
            )
            dump_prototype_pack_payload(reference_pack_path, pack_payload)

        reference_build_state_path: Path | None = None
        if (
            self.reference_build_state_output_dir is not None
            and build_state_payload is not None
        ):
            reference_build_state_path = (
                self.reference_build_state_output_dir / f"{prototype_version}.json"
            )
            dump_prototype_build_state_payload(
                reference_build_state_path,
                build_state_payload,
            )

        published_build_state_path: Path | None = None
        if build_state_payload is not None:
            published_build_state_path = (
                self.prototype_build_state_service.publish_state(build_state_payload)
            )

        published_pack_path = self.prototype_pack_service.publish_pack(pack_payload)
        return PrototypeRebuildResult(
            pack_payload=pack_payload,
            build_state_payload=build_state_payload,
            published_pack_path=published_pack_path,
            published_build_state_path=published_build_state_path,
            reference_pack_path=reference_pack_path,
            reference_build_state_path=reference_build_state_path,
        )


@dataclass(slots=True)
class StoredReferencePrototypeRebuildService:
    """저장된 canonical input을 읽어 runtime rebuild를 실행한다."""

    input_repository: PrototypeRebuildInputRepositoryProtocol
    prototype_rebuild_service: "PrototypeRebuildService" = field(
        default_factory=lambda: PrototypeRebuildService()
    )
    adapter_factory: type[EmbeddingAdapterFactory] = EmbeddingAdapterFactory

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


@dataclass(slots=True)
class PrototypeRebuildService:
    """runtime builder 선택과 publication 전략을 한곳에서 소유한다."""

    build_strategy: PrototypeBuildStrategy = field(
        default_factory=SinglePrototypeBuildStrategy
    )
    publication_strategy: PrototypePublicationStrategy = field(
        default_factory=ReferenceRebuildPrototypePublicationStrategy
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def rebuild(self, request: PrototypeRebuildRequest) -> PrototypeRebuildResult:
        build_request = request.build_request
        if build_request.built_at is None:
            build_request = replace(build_request, built_at=self.clock.now())

        build_artifacts = self.build_strategy.build(build_request)
        return self.publication_strategy.publish(build_artifacts)

    def rebuild_from_reference_rows(
        self,
        request: ReferencePrototypeRebuildRequest,
    ) -> PrototypeRebuildResult:
        if not request.rows:
            raise ValueError("Reference rebuild rows must not be empty.")

        texts = [row.text for row in request.rows]
        base_embeddings = request.adapter.embed_texts(texts)
        embeddings_by_category: dict[str, list[list[float]]] = defaultdict(list)
        for row, base_embedding in zip(request.rows, base_embeddings, strict=True):
            embeddings_by_category[row.category].append(
                request.adapter_state.apply(base_embedding)
            )

        required_categories = (
            tuple(request.required_categories)
            if request.required_categories is not None
            else tuple(sorted(embeddings_by_category))
        )
        return self.rebuild(
            PrototypeRebuildRequest(
                build_request=PrototypeBuildRequest(
                    embeddings_by_category=embeddings_by_category,
                    prototype_version=request.prototype_version,
                    embedding_backend=request.embedding_backend,
                    embedding_model_id=request.embedding_model_id,
                    embedding_model_revision=request.embedding_model_revision,
                    normalize_embeddings=request.normalize_embeddings,
                    task_prefix=request.task_prefix,
                    translation_model_id=request.translation_model_id,
                    translation_model_revision=request.translation_model_revision,
                    translation_direction=request.translation_direction,
                    mapping_version=request.mapping_version,
                    built_at=request.built_at,
                    required_categories=required_categories,
                )
            )
        )
