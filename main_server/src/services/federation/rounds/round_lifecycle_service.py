"""FL round lifecycle orchestration service."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4

from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.federation.assets.prototypes import (
    StoredReferencePrototypeRebuildRequest,
    StoredReferencePrototypeRebuildService,
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
from main_server.src.services.federation.rounds.models import (
    RoundFinalizeRequest,
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
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope
from shared.src.domain.services.clock import Clock, SystemUtcClock


@dataclass(slots=True)
class RoundLifecycleService:
    """active round open/update/finalize 전이를 조정한다."""

    round_repository: RoundRepository = field(default_factory=RoundRepository)
    round_manager_service: RoundManagerService = field(
        default_factory=RoundManagerService
    )
    prototype_rebuild_runtime_service: StoredReferencePrototypeRebuildService | None = (
        None
    )
    update_acceptance_policy: RoundUpdateAcceptancePolicy = field(
        default_factory=StrictRoundUpdateAcceptancePolicy
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def open_round(self, request: RoundOpenRequest) -> RoundRecord:
        active_pointer = self.round_repository.load_active_pointer()
        if active_pointer is not None:
            active_round = self.round_repository.load_round(active_pointer.round_id)
            if active_round.status == RoundStatus.OPEN:
                raise RoundConflictError(
                    f"An active round is already open: {active_round.round_id}"
                )
            self.round_repository.clear_active(expected_round_id=active_round.round_id)

        self._validate_open_request(request)
        round_id = request.round_id or f"round_{uuid4().hex[:8]}"
        if self.round_repository.has_round(round_id):
            raise RoundConflictError(f"Round already exists: {round_id}")

        created_at = self.clock.now()
        training_task = self.round_manager_service.create_training_task(
            replace(request, round_id=round_id)
        )
        record = RoundRecord(
            round_id=round_id,
            status=RoundStatus.OPEN,
            active_manifest=request.active_manifest,
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

    def accept_update(
        self,
        round_id: str,
        update: TrainingUpdateEnvelope,
    ) -> RoundUpdateAcceptance:
        record = self.round_repository.load_round(round_id)
        if record.status != RoundStatus.OPEN:
            raise RoundConflictError(f"Round is not open: {round_id}")

        accepted_at = self.clock.now()
        decision = self.update_acceptance_policy.evaluate(
            record=record,
            update=update,
            accepted_at=accepted_at,
        )
        updated_record = record
        if not decision.is_idempotent:
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
        return finalized_record

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
