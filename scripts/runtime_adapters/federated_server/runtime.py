"""Federated simulation server runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from main_server.src.infrastructure.repositories import (
    prototype_rebuild_input_repository as rebuild_input_repo,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as adapter_state_repo,
)
from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregationArtifactStore,
)
from main_server.src.services.federation.rounds.boundary.models import (
    RoundFinalizeRequest,
    RoundOpenDraftRequest,
    RoundRecord,
)
from main_server.src.services.federation.rounds.families.registry import (
    build_shared_adapter_round_family,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    RoundManagerService,
)
from methods.adaptation.federated_ssl_server_update import (
    resolve_federated_ssl_server_update_backend_name,
)
from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from methods.prototype.building.base import PrototypeBuildStrategy
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingUpdateEnvelope,
    make_training_update_submission,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec

from .prototype_rebuild_bridge import (
    SimulationEmbeddingAdapterFactory,
    build_prototype_rebuild_runtime_service,
    rebuild_reference_prototype_pack,
    store_prototype_rebuild_input,
)
from .repositories import (
    build_model_manifest_repository,
    build_prototype_rebuild_input_repository,
    build_round_repository,
    build_shared_adapter_state_repository,
    build_shared_adapter_update_repository,
)


@dataclass(frozen=True, slots=True)
class SimulationServerRuntime:
    """main_server federation runtime을 simulation용 저장소로 조립한 adapter."""

    output_dir: Path
    state_repository: adapter_state_repo.SharedAdapterStateRepository
    input_repository: rebuild_input_repo.PrototypeRebuildInputRepository
    round_manager: RoundManagerService
    lifecycle_service: RoundLifecycleService
    stored_rebuild_service: Any

    @classmethod
    def build(
        cls,
        *,
        output_dir: Path,
        round_runtime_config: Any,
        prototype_build_strategy: PrototypeBuildStrategy,
        method_descriptor: FederatedSslMethodDescriptor | None = None,
        capability_plan: FederatedSslCapabilityPlan | None = None,
    ) -> "SimulationServerRuntime":
        """simulation output root 기준 main_server runtime adapter를 만든다."""

        state_repository = build_shared_adapter_state_repository(output_dir)
        round_manager = RoundManagerService(
            adapter_family=build_simulation_round_family(
                adapter_family_name=round_runtime_config.adapter_family_name,
                aggregation_backend_name=resolve_simulation_aggregation_backend_name(
                    adapter_family_name=round_runtime_config.adapter_family_name,
                    aggregation_backend_name=(
                        round_runtime_config.aggregation_backend_name
                    ),
                    capability_plan=capability_plan,
                ),
                aggregation_backend_overrides=(
                    None
                    if capability_plan is None
                    else {
                        "weight_policy": capability_plan.aggregation_weight_policy.name
                    }
                ),
                output_dir=output_dir,
            ),
            artifact_repository=state_repository,
        )
        input_repository = build_prototype_rebuild_input_repository(output_dir)
        stored_rebuild_service = build_prototype_rebuild_runtime_service(
            output_dir=output_dir,
            build_strategy=prototype_build_strategy,
            input_repository=input_repository,
        )
        lifecycle_service = RoundLifecycleService(
            round_repository=build_round_repository(output_dir),
            update_payload_repository=build_shared_adapter_update_repository(
                output_dir
            ),
            active_manifest_service=ActiveModelManifestService(
                manifest_repository=build_model_manifest_repository(output_dir)
            ),
            round_manager_service=round_manager,
            prototype_rebuild_runtime_service=stored_rebuild_service,
            method_descriptor=method_descriptor,
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

    def save_shared_adapter_state(self, state: SharedAdapterState) -> str:
        """simulation용 state를 저장하고 server-owned opaque ref를 반환한다."""

        self.state_repository.save_shared_adapter_state(state)
        return self.state_repository.ref_for_revision(state.model_revision)

    def activate_manifest(self, manifest: ModelManifest) -> ModelManifest:
        """simulation bootstrap manifest를 main_server current로 활성화한다."""

        return self.lifecycle_service.active_manifest_service.save_and_activate(
            manifest,
            activated_at=manifest.published_at,
        )

    def open_round(self, request: RoundOpenDraftRequest) -> RoundRecord:
        """main_server round lifecycle을 통해 round를 연다."""

        return self.lifecycle_service.open_round(request)

    def accept_update(
        self,
        round_id: str,
        update_envelope: TrainingUpdateEnvelope,
        update_payload: SharedAdapterUpdatePayload,
    ) -> None:
        """main_server round lifecycle에 client update를 제출한다."""

        self.lifecycle_service.accept_update_submission(
            round_id,
            make_training_update_submission(
                envelope=update_envelope,
                update_payload=update_payload,
            ),
        )

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
    aggregation_backend_overrides: dict[str, str] | None = None,
    output_dir: Path,
):
    """simulation이 사용할 round family 조합을 만든다."""
    return build_shared_adapter_round_family(
        adapter_family_name,
        aggregation_backend_name=aggregation_backend_name,
        aggregation_backend_overrides=aggregation_backend_overrides,
        aggregation_artifact_store=AggregationArtifactStore(
            state_root=output_dir / "main_server" / "aggregation_artifacts"
        ),
    )


def resolve_simulation_aggregation_backend_name(
    *,
    adapter_family_name: str,
    aggregation_backend_name: str,
    capability_plan: FederatedSslCapabilityPlan | None,
) -> str:
    """server update policy를 simulation aggregation backend 이름으로 해석한다."""

    return resolve_federated_ssl_server_update_backend_name(
        adapter_family_name=adapter_family_name,
        server_update_policy_name=(
            None
            if capability_plan is None
            else capability_plan.server_update_policy_name
        ),
        aggregation_backend_name=aggregation_backend_name,
    )
