"""FL round lifecycle orchestration service."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING
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
    RoundFinalizeRequest,
    RoundOpenDraftRequest,
    RoundOpenRequest,
    RoundPublicationSummary,
    RoundRecord,
    RoundStatus,
    RoundUpdateAcceptance,
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
)
from main_server.src.services.federation.rounds.server_policy.executor import (
    DefaultServerPolicyExecutor,
    ServerPolicyExecutor,
)
from methods.adaptation.server_update_materialization import (
    require_server_materializable_update_payload,
)
from shared.src.contracts.adapter_contract_families.base import (
    CurrentSharedAdapterStatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_current_shared_adapter_state_payload,
)
from shared.src.contracts.training_contracts import TrainingUpdateSubmission
from shared.src.domain.services.clock import Clock, SystemUtcClock
from shared.src.services.secure_update_codec import (
    NoOpSecureUpdateCodec,
    SecureUpdateCodec,
)

from ..prototypes.models import (
    StoredReferencePrototypeRebuildRequest,
)
from ..prototypes.stored_input_rebuild_service import (
    StoredReferencePrototypeRebuildService,
)

if TYPE_CHECKING:
    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )
    from methods.federated_ssl.base import FederatedSslMethodDescriptor

SharedAdapterUpdateRepository = (
    shared_adapter_update_repository_module.SharedAdapterUpdateRepository
)


def _build_round_repository() -> RoundRepository:
    from main_server.src.infrastructure.repositories.round_repository import (
        RoundRepository,
    )

    return RoundRepository()


def _build_update_payload_repository() -> SharedAdapterUpdateRepository:
    return SharedAdapterUpdateRepository()


def _build_active_manifest_service() -> ActiveModelManifestService:
    return ActiveModelManifestService()


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
    prototype_rebuild_runtime_service: StoredReferencePrototypeRebuildService | None = (
        None
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
    clock: Clock = field(default_factory=SystemUtcClock)

    def __post_init__(self) -> None:
        """수락 저장소와 finalize 로딩 저장소를 같은 서버 저장소로 맞춘다."""

        self.round_manager_service.update_payload_repository = (
            self.update_payload_repository
        )

    def open_round(self, request: RoundOpenDraftRequest) -> RoundRecord:
        active_pointer = self.round_repository.load_active_pointer()
        if active_pointer is not None:
            active_round = self.round_repository.load_round(active_pointer.round_id)
            if active_round.status == RoundStatus.OPEN:
                raise RoundConflictError(
                    f"An active round is already open: {active_round.round_id}"
                )
            self.round_repository.clear_active(expected_round_id=active_round.round_id)

        active_manifest = self.active_manifest_service.get_active_manifest()
        round_id = request.round_id or f"round_{uuid4().hex[:8]}"
        if self.round_repository.has_round(round_id):
            raise RoundConflictError(f"Round already exists: {round_id}")
        resolved_request = request.to_round_open_request(
            active_manifest=active_manifest,
            round_id=round_id,
            task_id=request.task_id,
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
        server_payload_ref = self.update_payload_repository.ref_for_update(
            decoded_envelope.update_id
        )
        server_owned_envelope = decoded_envelope.model_copy(
            update={"payload_ref": server_payload_ref}
        )
        decision = self.update_acceptance_policy.evaluate(
            record=record,
            update=server_owned_envelope,
            accepted_at=accepted_at,
        )
        try:
            require_server_materializable_update_payload(submission.update_payload)
        except ValueError as error:
            raise RoundValidationError(str(error)) from error
        updated_record = record
        if not decision.is_idempotent:
            self.update_payload_repository.save_shared_adapter_update(
                decision.update_envelope.update_id,
                submission.update_payload,
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
                next_prototype_version=request.next_prototype_version,
                next_model_revision=request.next_model_revision,
                published_at=request.published_at,
            )
        )
        finalized_at = publication.next_manifest.published_at
        rebuild_result = None
        if self.prototype_rebuild_runtime_service is not None:
            rebuild_result = self.prototype_rebuild_runtime_service.rebuild(
                StoredReferencePrototypeRebuildRequest(
                    adapter_state=publication.next_state,
                    prototype_version=publication.next_manifest.prototype_version,
                    embedding_model_id=publication.next_manifest.model_id,
                    embedding_model_revision=publication.next_manifest.model_revision,
                    built_at=publication.next_manifest.published_at,
                )
            )
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
                prototype_pack_ref=(
                    None
                    if rebuild_result is None
                    or rebuild_result.published_pack_path is None
                    else str(rebuild_result.published_pack_path)
                ),
                prototype_build_state_ref=(
                    None
                    if rebuild_result is None
                    or rebuild_result.published_build_state_path is None
                    else str(rebuild_result.published_build_state_path)
                ),
                prototype_rebuild_input_id=(
                    None if rebuild_result is None else rebuild_result.source_input_id
                ),
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
