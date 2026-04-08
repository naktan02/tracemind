"""Federated simulation용 runtime wiring helper."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from main_server.src.infrastructure.repositories import (
    prototype_build_state_repository,
    prototype_pack_repository,
    prototype_rebuild_input_repository,
    shared_adapter_state_repository,
)
from main_server.src.services.prototypes import (
    PrototypeBuildStateService,
    PrototypePackService,
    PrototypeRebuildInputRecord,
    PrototypeRebuildService,
    ReferencePrototypeSourceRow,
    ReferenceRebuildPrototypePublicationStrategy,
    StoredReferencePrototypeRebuildRequest,
    StoredReferencePrototypeRebuildService,
)
from main_server.src.services.rounds.round_manager_service import RoundManagerService
from scripts.labeled_query_rows import LabeledQueryRow
from shared.src.contracts.adapter_contracts import VectorAdapterState
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    load_prototype_pack_payload,
)
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter
from shared.src.domain.value_objects import EmbeddingAdapterSpec
from shared.src.services.prototypes.build_strategies import PrototypeBuildStrategy

from .models import FederatedPrototypeRebuildConfig


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
    repository: (
        prototype_rebuild_input_repository.PrototypeRebuildInputRepository
    ),
    rebuild_config: FederatedPrototypeRebuildConfig,
) -> str:
    """bootstrap row를 canonical prototype rebuild input으로 저장한다."""
    input_id = "bootstrap_v1"
    repository.save_input(
        PrototypeRebuildInputRecord(
            input_id=input_id,
            embedding_spec=embedding_spec,
            rows=tuple(
                ReferencePrototypeSourceRow(
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
    input_repository: (
        prototype_rebuild_input_repository.PrototypeRebuildInputRepository
    ),
) -> StoredReferencePrototypeRebuildService:
    """simulation output 경로 기준 prototype rebuild runtime을 조립한다."""
    pack_repository = prototype_pack_repository.PrototypePackRepository(
        state_root=output_dir / "main_server" / "runtime_prototype_packs"
    )
    build_state_repository = (
        prototype_build_state_repository.PrototypeBuildStateRepository(
            state_root=output_dir / "main_server" / "runtime_prototype_build_states"
        )
    )
    return StoredReferencePrototypeRebuildService(
        input_repository=input_repository,
        prototype_rebuild_service=PrototypeRebuildService(
            build_strategy=build_strategy,
            publication_strategy=ReferenceRebuildPrototypePublicationStrategy(
                reference_pack_output_dir=(
                    output_dir / "main_server" / "prototype_packs"
                ),
                reference_build_state_output_dir=(
                    output_dir / "main_server" / "prototype_build_states"
                ),
                prototype_pack_service=PrototypePackService(repository=pack_repository),
                prototype_build_state_service=PrototypeBuildStateService(
                    repository=build_state_repository
                ),
            ),
        ),
        adapter_factory=SimulationEmbeddingAdapterFactory,
    )


def rebuild_reference_prototype_pack(
    *,
    stored_rebuild_service: StoredReferencePrototypeRebuildService,
    adapter_state: VectorAdapterState,
    prototype_version: str,
    embedding_model_id: str,
    embedding_model_revision: str,
    built_at: datetime,
) -> PrototypePackPayload:
    """reference row 기반 rebuild를 stored runtime service로 실행한다."""
    result = stored_rebuild_service.rebuild(
        StoredReferencePrototypeRebuildRequest(
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


def load_active_state(
    *,
    manifest: ModelManifest,
    state_repository: shared_adapter_state_repository.SharedAdapterStateRepository,
    round_manager: RoundManagerService,
) -> VectorAdapterState:
    """현재 active manifest가 가리키는 shared adapter state를 domain으로 읽는다."""
    payload = state_repository.load_state_from_ref(manifest.artifact_ref)
    state = round_manager.adapter_family.state_from_payload(payload)
    if not isinstance(state, VectorAdapterState):
        raise TypeError(
            f"Expected VectorAdapterState from simulation runtime, got {type(state)!r}."
        )
    return state
