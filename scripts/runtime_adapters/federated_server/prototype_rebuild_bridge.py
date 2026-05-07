"""Prototype rebuild bridge for federated simulation server runtime."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from main_server.src.infrastructure.repositories import (
    prototype_rebuild_input_repository as rebuild_input_repo,
)
from main_server.src.services.federation.assets.prototypes import (
    models as prototype_models,
)
from main_server.src.services.federation.assets.prototypes import (
    prototype_build_state_service as build_state_service_module,
)
from main_server.src.services.federation.assets.prototypes import (
    prototype_pack_service as pack_service_module,
)
from main_server.src.services.federation.assets.prototypes import (
    prototype_rebuild_service as rebuild_service_module,
)
from main_server.src.services.federation.assets.prototypes import (
    publication_strategies as publication_strategy_module,
)
from main_server.src.services.federation.assets.prototypes import (
    stored_input_rebuild_service as stored_rebuild_service_module,
)
from methods.prototype.building.base import PrototypeBuildStrategy
from scripts.io.labeled_query_rows import LabeledQueryRow
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    load_prototype_pack_payload,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec

from .repositories import (
    build_runtime_prototype_build_state_repository,
    build_runtime_prototype_pack_repository,
)


class SimulationEmbeddingAdapterFactory:
    """Simulation 중 실제 adapter 인스턴스를 재사용하는 test/runtime seam."""

    adapter: EmbeddingAdapter | None = None

    @classmethod
    def create(cls, spec: EmbeddingAdapterSpec) -> EmbeddingAdapter:
        del spec
        if cls.adapter is None:
            raise ValueError("Simulation embedding adapter is not configured.")
        return cls.adapter


def store_prototype_rebuild_input(
    *,
    rows: list[LabeledQueryRow],
    embedding_spec: EmbeddingAdapterSpec,
    repository: rebuild_input_repo.PrototypeRebuildInputRepository,
    rebuild_config: Any,
) -> str:
    """bootstrap row를 canonical prototype rebuild input으로 저장한다."""
    input_id = "bootstrap_v1"
    repository.save_input(
        prototype_models.PrototypeRebuildInputRecord(
            input_id=input_id,
            embedding_spec=embedding_spec,
            rows=tuple(
                prototype_models.ServerReferencePrototypeSourceRow(
                    text=str(row["text"]),
                    category=str(row["mapped_label_4"]),
                )
                for row in rows
            ),
            mapping_version=rebuild_config.mapping_version,
            translation_model_id=rebuild_config.translation_model_id,
            translation_model_revision=rebuild_config.translation_model_revision,
            translation_direction=rebuild_config.translation_direction,
        )
    )
    repository.set_active(input_id)
    return input_id


def build_prototype_rebuild_runtime_service(
    *,
    output_dir: Path,
    build_strategy: PrototypeBuildStrategy,
    input_repository: rebuild_input_repo.PrototypeRebuildInputRepository,
) -> stored_rebuild_service_module.StoredReferencePrototypeRebuildService:
    """simulation output 경로 기준 prototype rebuild runtime을 조립한다."""
    pack_repository = build_runtime_prototype_pack_repository(output_dir)
    build_state_repository = build_runtime_prototype_build_state_repository(output_dir)
    return stored_rebuild_service_module.StoredReferencePrototypeRebuildService(
        input_repository=input_repository,
        prototype_rebuild_service=rebuild_service_module.PrototypeRebuildService(
            build_strategy=build_strategy,
            publication_strategy=(
                publication_strategy_module.ReferenceRebuildPrototypePublicationStrategy(
                    reference_pack_output_dir=(
                        output_dir / "main_server" / "prototype_packs"
                    ),
                    reference_build_state_output_dir=(
                        output_dir / "main_server" / "prototype_build_states"
                    ),
                    prototype_pack_service=pack_service_module.PrototypePackService(
                        repository=pack_repository
                    ),
                    prototype_build_state_service=(
                        build_state_service_module.PrototypeBuildStateService(
                            repository=build_state_repository
                        )
                    ),
                )
            ),
        ),
        adapter_factory=SimulationEmbeddingAdapterFactory,
    )


def rebuild_reference_prototype_pack(
    *,
    stored_rebuild_service: (
        stored_rebuild_service_module.StoredReferencePrototypeRebuildService
    ),
    adapter_state: SharedAdapterState,
    prototype_version: str,
    embedding_model_id: str,
    embedding_model_revision: str,
    built_at: datetime,
) -> PrototypePackPayload:
    """reference row 기반 rebuild를 stored runtime service로 실행한다."""
    result = stored_rebuild_service.rebuild(
        prototype_models.StoredReferencePrototypeRebuildRequest(
            adapter_state=adapter_state,
            prototype_version=prototype_version,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            built_at=built_at,
        )
    )
    if result.published_pack_path is None:
        raise ValueError("Stored rebuild runtime must publish a prototype pack.")
    return load_prototype_pack_payload(result.published_pack_path)
