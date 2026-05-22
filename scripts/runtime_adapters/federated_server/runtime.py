"""Federated simulation server runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from main_server.src.services.federation.rounds.round_state_exchange.executor import (
    RoundStateExchangeResult,
)
from main_server.src.services.federation.rounds.server_policy.executor import (
    ROUND_ACTIVE_PAIR_ONLY_POLICY_NAME,
    ROUND_RUNTIME_AGGREGATION_BACKEND_POLICY_NAME,
    ServerPolicyExecutionSummary,
)
from methods.adaptation.federated_ssl_server_update import (
    resolve_federated_ssl_server_update_backend_name,
)
from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.capability_plan import FederatedSslCapabilityPlan
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingUpdateEnvelope,
    make_training_update_submission,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState

from .repositories import (
    build_model_manifest_repository,
    build_round_repository,
    build_shared_adapter_state_repository,
    build_shared_adapter_update_repository,
)


class SimulationServerPolicyExecutor:
    """simulation round lifecycle에서 method server policy guard를 적용한다."""

    def prepare_finalize(
        self,
        *,
        method_descriptor: FederatedSslMethodDescriptor,
        round_id: str,
        update_count: int,
    ) -> ServerPolicyExecutionSummary:
        """live 지원 여부가 아니라 simulation 지원 여부로 server policy를 검증한다."""

        if not method_descriptor.runtime_capabilities.simulation_supported:
            raise ValueError(
                "Configured FL SSL method does not support simulation runtime: "
                f"{method_descriptor.name}."
            )
        if method_descriptor.requires_custom_server_runtime:
            raise ValueError(
                "Configured FL SSL method requires a custom server runtime "
                "capability that is not wired in simulation: "
                f"{method_descriptor.name}."
            )
        server_step = method_descriptor.server_step
        if (
            server_step.server_aggregator_name
            != ROUND_RUNTIME_AGGREGATION_BACKEND_POLICY_NAME
        ):
            raise ValueError(
                "Unsupported server aggregation policy for simulation runtime: "
                f"{server_step.server_aggregator_name}."
            )
        if server_step.round_policy_name != ROUND_ACTIVE_PAIR_ONLY_POLICY_NAME:
            raise ValueError(
                "Unsupported round policy for simulation runtime: "
                f"{server_step.round_policy_name}."
            )
        return ServerPolicyExecutionSummary(
            method_name=method_descriptor.name,
            round_id=round_id,
            server_aggregator_name=server_step.server_aggregator_name,
            round_policy_name=server_step.round_policy_name,
            server_aggregate_hint=server_step.server_aggregate_hint,
            update_count=update_count,
        )


class SimulationRoundStateExchangeExecutor:
    """simulation이 별도로 수행한 round state exchange를 finalize에서 요약한다."""

    def summarize(
        self,
        *,
        method_descriptor: FederatedSslMethodDescriptor,
        record: RoundRecord,
    ) -> RoundStateExchangeResult:
        spec = method_descriptor.round_state_exchange
        if spec is None:
            return RoundStateExchangeResult(exchange_name="none")
        summary_metrics = {
            f"{spec.summary_metric_prefix}.update_count": float(len(record.updates)),
            f"{spec.summary_metric_prefix}.example_count": float(
                sum(update.example_count for update in record.updates)
            ),
        }
        for metric_key in spec.required_client_metric_keys:
            present_updates = [
                update
                for update in record.updates
                if metric_key in update.client_metrics
            ]
            if not present_updates:
                continue
            total_examples = sum(update.example_count for update in present_updates)
            if total_examples <= 0:
                continue
            summary_metrics[f"{spec.summary_metric_prefix}.{metric_key}.mean"] = (
                sum(
                    update.client_metrics[metric_key] * update.example_count
                    for update in present_updates
                )
                / total_examples
            )
        return RoundStateExchangeResult(
            exchange_name=spec.exchange_name,
            summary_metrics=summary_metrics,
        )


@dataclass(frozen=True, slots=True)
class SimulationServerRuntime:
    """main_server federation runtime을 simulation용 저장소로 조립한 adapter."""

    output_dir: Path
    state_repository: adapter_state_repo.SharedAdapterStateRepository
    round_manager: RoundManagerService
    lifecycle_service: RoundLifecycleService

    @classmethod
    def build(
        cls,
        *,
        output_dir: Path,
        round_runtime_config: Any,
        method_descriptor: FederatedSslMethodDescriptor | None = None,
        capability_plan: FederatedSslCapabilityPlan | None = None,
    ) -> "SimulationServerRuntime":
        """simulation output root 기준 shared-adapter runtime adapter를 만든다."""

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
        lifecycle_service = RoundLifecycleService(
            round_repository=build_round_repository(output_dir),
            update_payload_repository=build_shared_adapter_update_repository(
                output_dir
            ),
            active_manifest_service=ActiveModelManifestService(
                manifest_repository=build_model_manifest_repository(output_dir)
            ),
            round_manager_service=round_manager,
            server_policy_executor=SimulationServerPolicyExecutor(),
            round_state_exchange_executor=SimulationRoundStateExchangeExecutor(),
            method_descriptor=method_descriptor,
        )
        return cls(
            output_dir=output_dir,
            state_repository=state_repository,
            round_manager=round_manager,
            lifecycle_service=lifecycle_service,
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
        next_auxiliary_artifact_versions: dict[str, str] | None = None,
    ) -> RoundRecord:
        """main_server round lifecycle을 통해 aggregate/finalize를 실행한다."""

        return self.lifecycle_service.finalize_round(
            round_id,
            RoundFinalizeRequest(
                next_model_revision=next_model_revision,
                next_auxiliary_artifact_versions=(
                    next_auxiliary_artifact_versions or {}
                ),
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
