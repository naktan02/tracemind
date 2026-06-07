"""FL round lifecycle orchestration service."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as shared_adapter_update_repository_module,
)
from main_server.src.services.federation.rounds.acceptance.errors import (
    RoundConflictError,
    RoundValidationError,
)
from main_server.src.services.federation.rounds.acceptance.models import (
    RoundUpdateAcceptancePolicy,
)
from main_server.src.services.federation.rounds.acceptance.policies import (
    StrictRoundUpdateAcceptancePolicy,
)
from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from main_server.src.services.federation.rounds.boundary.models import (
    InitialSharedArtifactPublicationRequest,
    RoundFinalizeRequest,
    RoundOpenDraftRequest,
    RoundOpenRequest,
    RoundPublicationSummary,
    RoundRecord,
    RoundStatus,
    RoundStrategyConfig,
    RoundUpdateAcceptance,
)
from main_server.src.services.federation.rounds.initial_publication_service import (
    InitialSharedArtifactPublicationService,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    RoundManagerService,
    RoundPublicationRequest,
)
from main_server.src.services.federation.rounds.round_state_exchange.executor import (
    NO_ROUND_STATE_EXCHANGE_NAME,
    DefaultRoundStateExchangeExecutor,
    RoundStateExchangeExecutor,
    RoundStateExchangeResult,
    build_peer_context_task_payload,
)
from main_server.src.services.federation.rounds.server_policy.executor import (
    DefaultServerPolicyExecutor,
    ServerPolicyExecutor,
)
from main_server.src.services.federation.strategy.active_strategy_service import (
    ActiveStrategyService,
)
from methods.adaptation.server_update_compatibility import (
    require_server_compatible_update_payload,
)
from methods.adaptation.server_update_materialization import (
    require_server_materializable_update_payload,
)
from shared.src.contracts.adapter_contract_families.base import (
    CurrentSharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_current_shared_adapter_state_payload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingUpdateEnvelope,
    TrainingUpdateSubmission,
)
from shared.src.domain.services.clock import Clock, SystemUtcClock
from shared.src.services.secure_update_codec import (
    NoOpSecureUpdateCodec,
    SecureUpdateCodec,
)

if TYPE_CHECKING:
    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )
from methods.federated_ssl.base import FederatedSslMethodDescriptor
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor

SharedAdapterUpdateRepository = (
    shared_adapter_update_repository_module.SharedAdapterUpdateRepository
)
_PRIVATE_UPDATE_METADATA_FIELDS = ("mean_confidence", "mean_margin")


def _build_round_repository() -> RoundRepository:
    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )

    return RoundRepository()


def _build_update_payload_repository() -> SharedAdapterUpdateRepository:
    return SharedAdapterUpdateRepository()


def _build_active_manifest_service() -> ActiveModelManifestService:
    return ActiveModelManifestService()


def _runtime_update_family_name(round_runtime_config: object | None) -> str:
    value = getattr(round_runtime_config, "update_family_name", None)
    if value is None:
        raise RoundValidationError(
            "Initial shared artifact publication requires round runtime "
            "update_family_name."
        )
    normalized = str(value).strip()
    if not normalized:
        raise RoundValidationError(
            "round runtime update_family_name must not be empty."
        )
    return normalized


@dataclass(slots=True)
class RoundLifecycleService:
    """active round open/update/finalize 전이를 조정한다."""

    round_repository: RoundRepository = field(default_factory=_build_round_repository)
    update_payload_repository: SharedAdapterUpdateRepository = field(
        default_factory=_build_update_payload_repository
    )
    active_manifest_service: ActiveModelManifestService = field(
        default_factory=_build_active_manifest_service
    )
    round_manager_service: RoundManagerService = field(
        default_factory=RoundManagerService
    )
    update_acceptance_policy: RoundUpdateAcceptancePolicy = field(
        default_factory=StrictRoundUpdateAcceptancePolicy
    )
    secure_update_codec: SecureUpdateCodec = field(
        default_factory=NoOpSecureUpdateCodec
    )
    server_policy_executor: ServerPolicyExecutor = field(
        default_factory=DefaultServerPolicyExecutor
    )
    round_state_exchange_executor: RoundStateExchangeExecutor = field(
        default_factory=DefaultRoundStateExchangeExecutor
    )
    method_descriptor: FederatedSslMethodDescriptor | None = None
    round_runtime_config: object | None = None
    active_strategy_service: ActiveStrategyService = field(
        default_factory=ActiveStrategyService
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def __post_init__(self) -> None:
        """수락 저장소와 finalize 로딩 저장소를 같은 서버 저장소로 맞춘다."""

        self.round_manager_service.update_payload_repository = (
            self.update_payload_repository
        )

    def publish_initial_shared_artifact(
        self,
        request: InitialSharedArtifactPublicationRequest,
    ) -> ModelManifest:
        """선택된 family initial state를 active manifest로 publish한다."""

        return InitialSharedArtifactPublicationService(
            artifact_repository=self.round_manager_service.artifact_repository,
            active_manifest_service=self.active_manifest_service,
            payload_adapter_kind=self.round_manager_service.payload_adapter.adapter_kind,
            update_family_name=_runtime_update_family_name(self.round_runtime_config),
            round_runtime_config=self.round_runtime_config,
            clock=self.clock,
        ).publish(request)

    def open_round(self, request: RoundOpenDraftRequest) -> RoundRecord:
        active_pointer = self.round_repository.load_active_pointer()
        if active_pointer is not None:
            active_round = self.round_repository.load_round(active_pointer.round_id)
            if active_round.status == RoundStatus.OPEN:
                raise RoundConflictError(
                    f"An active round is already open: {active_round.round_id}"
                )
            self.round_repository.clear_active(expected_round_id=active_round.round_id)

        # strategy가 없거나 ssl_method가 없으면 active_strategy에서 자동 적용한다.
        effective_request = self._apply_fssl_context(
            self._apply_active_strategy(request)
        )

        active_manifest = self.active_manifest_service.get_active_manifest()
        round_id = effective_request.round_id or f"round_{uuid4().hex[:8]}"
        if self.round_repository.has_round(round_id):
            raise RoundConflictError(f"Round already exists: {round_id}")
        resolved_request = effective_request.to_round_open_request(
            active_manifest=active_manifest,
            round_id=round_id,
            task_id=effective_request.task_id,
        )
        self._validate_open_request(resolved_request)

        created_at = self.clock.now()
        training_task = self.round_manager_service.create_training_task(
            resolved_request
        )
        record = RoundRecord(
            round_id=round_id,
            status=RoundStatus.OPEN,
            active_manifest=active_manifest,
            training_task=training_task,
            created_at=created_at,
            updated_at=created_at,
        )
        self.round_repository.save_round(record)
        self.round_repository.set_active(round_id, activated_at=created_at)
        return record

    def get_round(self, round_id: str) -> RoundRecord:
        return self.round_repository.load_round(round_id)

    def get_current_round(self) -> RoundRecord:
        active_pointer = self.round_repository.load_active_pointer()
        if active_pointer is None:
            raise FileNotFoundError("No active round is registered.")
        return self.round_repository.load_round(active_pointer.round_id)

    def get_current_shared_adapter_state(self) -> CurrentSharedAdapterStatePayload:
        """서버 current manifest와 실제 shared adapter state를 함께 반환한다."""

        active_manifest = self.active_manifest_service.get_active_manifest()
        artifact_repository = self.round_manager_service.artifact_repository
        state_payload = artifact_repository.load_shared_adapter_state_from_ref(
            active_manifest.artifact_ref
        )
        return make_current_shared_adapter_state_payload(
            manifest=active_manifest,
            state=state_payload,
        )

    def load_aggregation_json_artifact(self, artifact_ref: str) -> dict[str, object]:
        """server-owned aggregation JSON artifact를 materialize한다."""

        loader = self._require_aggregation_artifact_loader()
        return dict(loader.load_json_artifact(artifact_ref=artifact_ref))

    def load_aggregation_safetensors_artifact(
        self,
        artifact_ref: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """server-owned aggregation safetensors artifact를 materialize한다."""

        loader = self._require_aggregation_artifact_loader()
        tensor_loader = getattr(loader, "load_safetensors_artifact", None)
        if tensor_loader is None:
            raise FileNotFoundError(
                "Current aggregation backend does not expose safetensors artifacts."
            )
        tensors, metadata = tensor_loader(artifact_ref=artifact_ref)
        return dict(tensors), dict(metadata)

    def accept_update_submission(
        self,
        round_id: str,
        submission: TrainingUpdateSubmission,
    ) -> RoundUpdateAcceptance:
        """inline update payload submission을 서버 저장소에 저장하고 수락한다."""

        record = self.round_repository.load_round(round_id)
        if record.status != RoundStatus.OPEN:
            raise RoundConflictError(f"Round is not open: {round_id}")

        accepted_at = self.clock.now()
        decoded_envelope = self.secure_update_codec.decode_submission(
            envelope=submission.envelope,
            training_task=record.training_task,
        )
        server_visible_payload = _server_visible_update_payload(
            submission.update_payload
        )
        self._validate_update_payload_matches_active_payload_adapter(
            envelope=decoded_envelope,
            update_payload=server_visible_payload,
        )
        server_payload_ref = self.update_payload_repository.ref_for_update(
            decoded_envelope.update_id
        )
        server_owned_envelope = decoded_envelope.model_copy(
            update={
                "payload_ref": server_payload_ref,
                "example_count": server_visible_payload.example_count,
                "client_metrics": {},
            }
        )
        decision = self.update_acceptance_policy.evaluate(
            record=record,
            update=server_owned_envelope,
            accepted_at=accepted_at,
        )
        try:
            require_server_materializable_update_payload(server_visible_payload)
            self._validate_update_payload_matches_active_state(
                record=record,
                update_payload=server_visible_payload,
            )
        except ValueError as error:
            raise RoundValidationError(str(error)) from error
        updated_record = record
        if not decision.is_idempotent:
            self.update_payload_repository.save_shared_adapter_update(
                decision.update_envelope.update_id,
                server_visible_payload,
            )
            updated_record = replace(
                record,
                updates=record.updates + (decision.update_envelope,),
                updated_at=accepted_at,
            )
        self.round_repository.save_round(updated_record)
        return RoundUpdateAcceptance(
            round_id=round_id,
            update_id=decision.update_envelope.update_id,
            update_count=len(updated_record.updates),
            accepted_at=accepted_at,
            idempotent=decision.is_idempotent,
        )

    def _require_aggregation_artifact_loader(self) -> Any:
        aggregation_backend = (
            self.round_manager_service.payload_adapter.aggregation_backend
        )
        loader = getattr(aggregation_backend, "artifact_loader", None)
        if loader is None:
            raise FileNotFoundError(
                "Current aggregation backend does not expose artifact materializer."
            )
        return loader

    def finalize_round(
        self,
        round_id: str,
        request: RoundFinalizeRequest,
    ) -> RoundRecord:
        record = self.round_repository.load_round(round_id)
        if record.status != RoundStatus.OPEN:
            raise RoundConflictError(f"Round is not open: {round_id}")

        self._prepare_method_server_policy(record)
        round_state_exchange = self._summarize_round_state_exchange(record)
        publication = self.round_manager_service.publish_next_pair(
            RoundPublicationRequest(
                base_manifest=record.active_manifest,
                updates=record.updates,
                next_model_revision=request.next_model_revision,
                next_auxiliary_artifact_versions=(
                    request.next_auxiliary_artifact_versions
                ),
                published_at=request.published_at,
            )
        )
        finalized_at = publication.next_manifest.published_at
        auxiliary_artifact_refs: dict[str, str] = {}
        auxiliary_artifact_metadata: dict[str, str] = {}
        finalized_record = replace(
            record,
            status=RoundStatus.FINALIZED,
            updated_at=finalized_at,
            finalized_at=finalized_at,
            publication=RoundPublicationSummary(
                next_manifest=publication.next_manifest,
                aggregated_metrics=dict(publication.aggregated_metrics),
                update_count=publication.update_count,
                finalized_at=finalized_at,
                round_state_summary_metrics=dict(round_state_exchange.summary_metrics),
                auxiliary_artifact_refs=auxiliary_artifact_refs,
                auxiliary_artifact_metadata=auxiliary_artifact_metadata,
            ),
        )
        self.round_repository.save_round(finalized_record)
        self.round_repository.clear_active(expected_round_id=round_id)
        self.active_manifest_service.save_and_activate(
            publication.next_manifest,
            activated_at=finalized_at,
        )
        return finalized_record

    def _prepare_method_server_policy(self, record: RoundRecord) -> None:
        if self.method_descriptor is None:
            return
        self.server_policy_executor.prepare_finalize(
            method_descriptor=self.method_descriptor,
            round_id=record.round_id,
            update_count=len(record.updates),
        )

    def _summarize_round_state_exchange(
        self,
        record: RoundRecord,
    ) -> RoundStateExchangeResult:
        if self.method_descriptor is None:
            return RoundStateExchangeResult(exchange_name=NO_ROUND_STATE_EXCHANGE_NAME)
        return self.round_state_exchange_executor.summarize(
            method_descriptor=self.method_descriptor,
            record=record,
        )

    @staticmethod
    def _validate_open_request(request: RoundOpenRequest) -> None:
        if not request.active_manifest.training_enabled:
            raise RoundValidationError("Training is disabled on the active manifest.")
        compatible_task_types = request.active_manifest.compatible_task_types
        if compatible_task_types and request.task_type not in compatible_task_types:
            raise RoundValidationError(
                "Requested task_type is not compatible with the active manifest: "
                f"{request.task_type}"
            )

    def _validate_update_payload_matches_active_payload_adapter(
        self,
        *,
        envelope: TrainingUpdateEnvelope,
        update_payload: SharedAdapterUpdatePayload,
    ) -> None:
        payload_adapter = self.round_manager_service.payload_adapter
        if envelope.payload_format not in payload_adapter.accepted_update_formats:
            raise RoundValidationError(
                "Update payload_format is not accepted by the active payload "
                f"adapter {payload_adapter.adapter_kind}: {envelope.payload_format}"
            )
        if update_payload.adapter_kind != payload_adapter.adapter_kind:
            raise RoundValidationError(
                "Update payload adapter_kind does not match the active payload "
                f"adapter: {update_payload.adapter_kind} != "
                f"{payload_adapter.adapter_kind}"
            )

    def _validate_update_payload_matches_active_state(
        self,
        *,
        record: RoundRecord,
        update_payload: SharedAdapterUpdatePayload,
    ) -> None:
        artifact_repository = self.round_manager_service.artifact_repository
        active_payload = artifact_repository.load_shared_adapter_state_from_ref(
            record.active_manifest.artifact_ref,
        )
        active_state = self.round_manager_service.payload_adapter.state_from_payload(
            active_payload
        )
        require_server_compatible_update_payload(
            update_payload=update_payload,
            active_state=active_state,
        )

    def _apply_active_strategy(
        self, request: RoundOpenDraftRequest
    ) -> RoundOpenDraftRequest:
        """strategy가 없거나 ssl_method가 없으면 active_strategy를 자동으로 적용한다.

        caller가 strategy를 명시한 경우 그 값을 유지하고,
        strategy가 None이거나 ssl_method가 None인 경우에만 active_strategy_service의
        현재 config로 채운다.
        objective_config가 명시된 경우에는 strategy를 건드리지 않는다.
        """
        # objective_config가 명시된 요청은 strategy와 함께 쓸 수 없으므로 그대로 반환
        if request.objective_config is not None:
            return request

        active = self.active_strategy_service.get_active_strategy()
        current_strategy = request.strategy

        # strategy 자체가 없으면 active strategy로 새로 만든다
        if current_strategy is None:
            new_strategy = RoundStrategyConfig(
                mode=("method_owned" if active.fssl_method is not None else "composed"),
                ssl_method=active.ssl_method,
                fssl_method=active.fssl_method,
                aggregation_backend=active.aggregation_backend,
            )
            return replace(request, strategy=new_strategy)

        # strategy가 있지만 ssl_method가 없으면 active에서 채운다.
        if current_strategy.ssl_method is None:
            new_strategy = RoundStrategyConfig(
                mode=current_strategy.mode,
                local_update_profile=current_strategy.local_update_profile,
                ssl_method=active.ssl_method,
                fssl_method=current_strategy.fssl_method,
                scenario=current_strategy.scenario,
                server_update_policy=current_strategy.server_update_policy,
                aggregation_backend=(
                    current_strategy.aggregation_backend or active.aggregation_backend
                ),
                parameter_overrides=current_strategy.parameter_overrides,
            )
            return replace(request, strategy=new_strategy)

        return request

    def _apply_fssl_context(
        self,
        request: RoundOpenDraftRequest,
    ) -> RoundOpenDraftRequest:
        if request.objective_config is not None:
            return request
        if request.fssl_context is not None:
            return request
        strategy = request.strategy
        if strategy is None or strategy.fssl_method is None:
            return request
        descriptor = resolve_federated_ssl_method_descriptor(strategy.fssl_method)
        context = {
            "schema_version": "fssl_context.v1",
            "method_name": descriptor.name,
            "context_kind": "peer_context",
            "peer_context": build_peer_context_task_payload(
                method_descriptor=descriptor,
                source_round=self.round_repository.load_latest_finalized_round(),
            ),
        }
        return replace(request, fssl_context=context)


def _server_visible_update_payload(
    payload: SharedAdapterUpdatePayload,
) -> SharedAdapterUpdatePayload:
    updates: dict[str, object] = {
        "example_count": _server_visible_example_count(payload.example_count)
    }
    for field_name in _PRIVATE_UPDATE_METADATA_FIELDS:
        if hasattr(payload, field_name):
            updates[field_name] = None
    return payload.model_copy(update=updates)


def _server_visible_example_count(raw_count: int) -> int:
    return 0 if raw_count <= 0 else 1
