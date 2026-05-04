"""Federated simulation용 runtime wiring helper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from main_server.src.infrastructure.repositories import (
    prototype_build_state_repository as build_state_repo,
)
from main_server.src.infrastructure.repositories import (
    prototype_pack_repository as pack_repo,
)
from main_server.src.infrastructure.repositories import (
    prototype_rebuild_input_repository as rebuild_input_repo,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as adapter_state_repo,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.federation.assets.prototypes import (
    PrototypeBuildStateService,
    PrototypePackService,
    PrototypeRebuildInputRecord,
    PrototypeRebuildService,
    ReferencePrototypeSourceRow,
    ReferenceRebuildPrototypePublicationStrategy,
    StoredReferencePrototypeRebuildRequest,
    StoredReferencePrototypeRebuildService,
)
from main_server.src.services.federation.rounds import build_shared_adapter_round_family
from main_server.src.services.federation.rounds.boundary.models import (
    RoundFinalizeRequest,
    RoundOpenRequest,
    RoundRecord,
    RoundTaskConfig,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    RoundManagerService,
)
from methods.prototype.building.base import PrototypeBuildStrategy
from scripts.labeled_query_rows import LabeledQueryRow
from shared.src.contracts.adapter_contracts import (
    ClassifierHeadState,
    VectorAdapterState,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    extract_category_centroids,
    load_prototype_pack_payload,
)
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter
from shared.src.domain.value_objects import EmbeddingAdapterSpec


class SimulationEmbeddingAdapterFactory:
    """Simulation 중 실제 adapter 인스턴스를 재사용하는 test/runtime seam."""

    adapter: EmbeddingAdapter | None = None

    @classmethod
    def create(cls, spec: EmbeddingAdapterSpec) -> EmbeddingAdapter:
        del spec
        if cls.adapter is None:
            raise ValueError("Simulation embedding adapter is not configured.")
        return cls.adapter


@dataclass(frozen=True, slots=True)
class SimulationServerRuntime:
    """main_server federation runtime을 simulation용 저장소로 조립한 adapter."""

    output_dir: Path
    state_repository: adapter_state_repo.SharedAdapterStateRepository
    input_repository: rebuild_input_repo.PrototypeRebuildInputRepository
    round_manager: RoundManagerService
    lifecycle_service: RoundLifecycleService
    stored_rebuild_service: StoredReferencePrototypeRebuildService

    @classmethod
    def build(
        cls,
        *,
        output_dir: Path,
        round_runtime_config: Any,
        prototype_build_strategy: PrototypeBuildStrategy,
    ) -> "SimulationServerRuntime":
        """simulation output root 기준 main_server runtime adapter를 만든다."""

        state_repository = adapter_state_repo.SharedAdapterStateRepository(
            state_root=output_dir / "main_server" / "shared_adapter_states"
        )
        round_manager = RoundManagerService(
            adapter_family=build_simulation_round_family(
                adapter_family_name=round_runtime_config.adapter_family_name,
                aggregation_backend_name=(
                    round_runtime_config.aggregation_backend_name
                ),
            ),
            artifact_repository=state_repository,
        )
        input_repository = rebuild_input_repo.PrototypeRebuildInputRepository(
            state_root=output_dir / "main_server" / "prototype_rebuild_inputs"
        )
        stored_rebuild_service = build_prototype_rebuild_runtime_service(
            output_dir=output_dir,
            build_strategy=prototype_build_strategy,
            input_repository=input_repository,
        )
        lifecycle_service = RoundLifecycleService(
            round_repository=RoundRepository(
                state_root=output_dir / "main_server" / "rounds"
            ),
            round_manager_service=round_manager,
            prototype_rebuild_runtime_service=stored_rebuild_service,
        )
        return cls(
            output_dir=output_dir,
            state_repository=state_repository,
            input_repository=input_repository,
            round_manager=round_manager,
            lifecycle_service=lifecycle_service,
            stored_rebuild_service=stored_rebuild_service,
        )

    def set_embedding_adapter(self, adapter: EmbeddingAdapter) -> None:
        """prototype rebuild runtime이 simulation adapter instance를 재사용하게 한다."""

        SimulationEmbeddingAdapterFactory.adapter = adapter

    def store_prototype_rebuild_input(
        self,
        *,
        rows: list[LabeledQueryRow],
        embedding_spec: EmbeddingAdapterSpec,
        rebuild_config: Any,
    ) -> str:
        """bootstrap row를 main_server prototype rebuild input으로 저장한다."""

        return store_prototype_rebuild_input(
            rows=rows,
            embedding_spec=embedding_spec,
            repository=self.input_repository,
            rebuild_config=rebuild_config,
        )

    def rebuild_reference_prototype_pack(
        self,
        *,
        adapter_state: SharedAdapterState,
        prototype_version: str,
        embedding_model_id: str,
        embedding_model_revision: str,
        built_at: datetime,
    ) -> PrototypePackPayload:
        """현재 저장된 reference input으로 prototype pack을 rebuild한다."""

        return rebuild_reference_prototype_pack(
            stored_rebuild_service=self.stored_rebuild_service,
            adapter_state=adapter_state,
            prototype_version=prototype_version,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            built_at=built_at,
        )

    def save_shared_adapter_state(self, state: SharedAdapterState) -> Path:
        """simulation용 main_server state repository에 active state를 저장한다."""

        return self.state_repository.save_shared_adapter_state(state)

    def open_round(self, request: RoundOpenRequest) -> RoundRecord:
        """main_server round lifecycle을 통해 round를 연다."""

        return self.lifecycle_service.open_round(request)

    def accept_update(
        self,
        round_id: str,
        update_envelope: TrainingUpdateEnvelope,
    ) -> None:
        """main_server round lifecycle에 client update를 제출한다."""

        self.lifecycle_service.accept_update(round_id, update_envelope)

    def finalize_round(
        self,
        *,
        round_id: str,
        next_model_revision: str,
        next_prototype_version: str,
    ) -> RoundRecord:
        """main_server round lifecycle을 통해 aggregate/finalize를 실행한다."""

        return self.lifecycle_service.finalize_round(
            round_id,
            RoundFinalizeRequest(
                next_model_revision=next_model_revision,
                next_prototype_version=next_prototype_version,
            ),
        )

    def load_active_state(self, manifest: ModelManifest) -> SharedAdapterState:
        """active manifest가 가리키는 shared adapter state를 domain으로 읽는다."""

        return load_active_state(
            manifest=manifest,
            state_repository=self.state_repository,
            round_manager=self.round_manager,
        )


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
    input_repository: rebuild_input_repo.PrototypeRebuildInputRepository,
) -> StoredReferencePrototypeRebuildService:
    """simulation output 경로 기준 prototype rebuild runtime을 조립한다."""
    pack_repository = pack_repo.PrototypePackRepository(
        state_root=output_dir / "main_server" / "runtime_prototype_packs"
    )
    build_state_repository = build_state_repo.PrototypeBuildStateRepository(
        state_root=output_dir / "main_server" / "runtime_prototype_build_states"
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
    adapter_state: SharedAdapterState,
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
    state_repository: adapter_state_repo.SharedAdapterStateRepository,
    round_manager: RoundManagerService,
) -> SharedAdapterState:
    """현재 active manifest가 가리키는 shared adapter state를 domain으로 읽는다."""
    payload = state_repository.load_state_from_ref(manifest.artifact_ref)
    return round_manager.adapter_family.state_from_payload(payload)


def build_simulation_round_family(
    *,
    adapter_family_name: str,
    aggregation_backend_name: str,
):
    """simulation이 사용할 round family 조합을 만든다."""
    return build_shared_adapter_round_family(
        adapter_family_name,
        aggregation_backend_name=aggregation_backend_name,
    )


def build_initial_shared_state(
    *,
    adapter_family_name: str,
    model_id: str,
    model_revision: str,
    training_scope: str,
    embedding_dim: int,
    labels: tuple[str, ...] | list[str],
    updated_at: datetime,
) -> SharedAdapterState:
    """simulation bootstrap용 초기 shared state를 family별로 만든다."""
    if adapter_family_name.strip().lower() == "classifier_head":
        return ClassifierHeadState.zero_initialized(
            model_id=model_id,
            model_revision=model_revision,
            labels=labels,
            embedding_dim=embedding_dim,
            training_scope=training_scope,
            updated_at=updated_at,
        )
    return VectorAdapterState.identity(
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        embedding_dim=embedding_dim,
        updated_at=updated_at,
    )


def build_classifier_head_state_from_prototype_pack(
    *,
    prototype_pack: PrototypePackPayload,
    model_id: str,
    model_revision: str,
    training_scope: str,
    updated_at: datetime,
    logit_scale: float = 8.0,
) -> SharedAdapterState:
    """bootstrap prototype centroid로 classifier-head 초기 상태를 만든다."""
    centroids = extract_category_centroids(prototype_pack)
    if not centroids:
        raise ValueError(
            "Classifier-head initialization requires at least one centroid."
        )
    return ClassifierHeadState(
        schema_version="classifier_head_state.v1",
        adapter_kind="classifier_head",
        model_id=model_id,
        model_revision=model_revision,
        training_scope=training_scope,
        updated_at=updated_at,
        label_weights={
            label: [float(value) * logit_scale for value in centroid]
            for label, centroid in centroids.items()
        },
        label_biases={label: 0.0 for label in centroids},
    )


def build_federated_training_task_config(
    *,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
    min_required_examples: int,
    gradient_clip_norm: float | None,
    objective_config: Any,
    selection_policy: Any,
) -> Any:
    """FL simulation용 RoundTaskConfig 생성을 main_server bridge로 격리한다."""

    return RoundTaskConfig(
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        min_required_examples=min_required_examples,
        gradient_clip_norm=gradient_clip_norm,
        objective_config=objective_config,
        selection_policy=selection_policy,
    )


def build_round_open_request(
    *,
    active_manifest: ModelManifest,
    round_id: str,
    training_task_config: Any,
) -> Any:
    """simulation task template을 canonical round open request로 변환한다."""

    return training_task_config.to_round_open_request(
        active_manifest=active_manifest,
        round_id=round_id,
    )
