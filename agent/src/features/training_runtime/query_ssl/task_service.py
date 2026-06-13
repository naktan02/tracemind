"""Agent Query SSL current-task 실행 서비스."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.src.features.captured_text.storage.repository import (
    CapturedTextRepository,
)
from agent.src.features.federation.rounds.round_client import RoundClient
from agent.src.features.training_runtime.current_task.result import (
    TrainingTaskRunResult,
    TrainingTaskRunStatus,
)
from agent.src.features.training_runtime.query_ssl.method_request_builder import (
    MethodOwnedPeftEncoderTrainingCore,
    run_query_ssl_local_update,
)
from agent.src.features.training_runtime.query_ssl.source_selection import (
    select_query_ssl_training_sources,
)
from agent.src.features.training_runtime.query_ssl.upload_flow import (
    upload_query_ssl_update,
)
from agent.src.features.training_runtime.query_ssl.usage_recording import (
    record_query_ssl_training_usage,
)
from agent.src.features.training_runtime.query_ssl_peft.local_training_service import (
    QuerySslLocalTrainingService,
)
from agent.src.features.training_runtime.storage.training_artifact_repository import (  # noqa: E501
    TrainingArtifactRepository,
)
from agent.src.features.training_runtime.storage.training_usage_ledger_repository import (  # noqa: E501
    TrainingUsageLedgerRepository,
)
from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
)
from methods.adaptation.local_update_backend import SharedAdapterTrainingBackend
from methods.adaptation.peft_text_encoder.federated_ssl.method_owned_training import (
    run_method_owned_peft_encoder_training_request,
)
from methods.adaptation.peft_text_encoder.training.query_ssl_local_training import (
    PeftEncoderTrainerRuntimeConfig,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    materialize_base_peft_encoder_state,
)
from methods.adaptation.peft_text_encoder.update_family_runtime import (
    build_training_backend_for_peft_encoder_state,
)
from methods.federated.aggregation.base import (
    AggregationJsonArtifactLoader,
    FederatedAggregationContext,
)
from methods.ssl.runtime.objective_config import QuerySslObjectiveRuntimeConfig
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask
from shared.src.domain.services.clock import Clock, SystemUtcClock


@dataclass(slots=True, frozen=True)
class AgentQuerySslTrainingTaskRunRequest:
    """Agent Query SSL current-task 실행 입력."""

    training_task: TrainingTask
    model_manifest: ModelManifest
    active_state: PeftClassifierState
    round_client: RoundClient
    analysis_event_repository: AnalysisEventRepository
    captured_text_repository: CapturedTextRepository | None
    analysis_event_days: int
    agent_id: str | None = None
    artifact_loader: AggregationJsonArtifactLoader | None = None


@dataclass(frozen=True, slots=True)
class AgentQuerySslTrainerRuntimeConfig:
    """live agent Query SSL trainer 기본 runtime 설정."""

    device: str = "cpu"
    local_files_only: bool = True
    cache_dir: str | None = "hf_cache"
    trust_remote_code: bool = False
    classifier_dropout: float = 0.1


@dataclass(slots=True)
class AgentQuerySslTrainingTaskService:
    """Query SSL task를 raw row 기반 local trainer로 실행하고 update를 업로드한다."""

    repository: TrainingArtifactRepository = field(
        default_factory=TrainingArtifactRepository
    )
    usage_ledger_repository: TrainingUsageLedgerRepository = field(
        default_factory=TrainingUsageLedgerRepository
    )
    backend: SharedAdapterTrainingBackend | None = None
    method_owned_training_core: MethodOwnedPeftEncoderTrainingCore = (
        run_method_owned_peft_encoder_training_request
    )
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig = field(
        default_factory=AgentQuerySslTrainerRuntimeConfig
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def run_current_task(
        self,
        request: AgentQuerySslTrainingTaskRunRequest,
    ) -> TrainingTaskRunResult:
        """Query SSL objective task를 실행하고 서버에 update를 업로드한다."""

        if not isinstance(request.active_state, PeftClassifierState):
            raise ValueError("Query SSL live training requires PEFT classifier state.")
        query_ssl_config = QuerySslObjectiveRuntimeConfig.from_objective_config(
            request.training_task.objective_config
        )
        if query_ssl_config is None:
            raise ValueError("Query SSL objective extras are required.")
        if request.captured_text_repository is None:
            return _insufficient_query_ssl_examples(
                request.training_task,
                message="Query SSL unlabeled generated view 저장소가 없습니다.",
            )

        selected_sources = select_query_ssl_training_sources(
            analysis_event_repository=request.analysis_event_repository,
            captured_text_repository=request.captured_text_repository,
            analysis_event_days=request.analysis_event_days,
            labels=request.active_state.label_schema,
            max_unlabeled_rows=(
                request.training_task.selection_policy.max_examples or 100
            ),
        )
        if not selected_sources.labeled_rows or not selected_sources.unlabeled_rows:
            return _insufficient_query_ssl_examples(
                request.training_task,
                example_count=len(selected_sources.unlabeled_rows),
                accepted_count=len(selected_sources.labeled_rows),
                message=(
                    "Query SSL 실행에 필요한 labeled anchor 또는 unlabeled "
                    "generated view가 부족합니다."
                ),
            )

        created_at = self.clock.now()
        local_service = QuerySslLocalTrainingService(
            repository=self.repository,
            backend=self.backend
            or build_training_backend_for_peft_encoder_state(
                active_adapter_state=request.active_state,
                objective_config=request.training_task.objective_config,
            ),
            clock=self.clock,
        )
        base_parameters = materialize_base_peft_encoder_state(
            base_state=request.active_state,
            context=FederatedAggregationContext(
                next_model_revision=request.active_state.model_revision,
                aggregated_at=created_at,
                artifact_loader=request.artifact_loader,
            ),
        )
        local_result = run_query_ssl_local_update(
            training_task=request.training_task,
            model_manifest=request.model_manifest,
            active_state=request.active_state,
            agent_id=request.agent_id,
            local_service=local_service,
            method_owned_training_core=self.method_owned_training_core,
            trainer_runtime_config=self.trainer_runtime_config,
            labels=selected_sources.labels,
            labeled_rows=selected_sources.labeled_rows,
            unlabeled_rows=selected_sources.unlabeled_rows,
            base_parameters=base_parameters,
            query_ssl_config=query_ssl_config,
            created_at=created_at,
        )
        upload_query_ssl_update(
            round_client=request.round_client,
            training_task=request.training_task,
            local_result=local_result,
        )
        recorded_at = self.clock.now()
        record_query_ssl_training_usage(
            repository=self.usage_ledger_repository,
            training_task=request.training_task,
            agent_id=request.agent_id,
            query_ssl_config=query_ssl_config,
            local_result=local_result,
            labeled_rows=selected_sources.labeled_rows,
            unlabeled_rows=selected_sources.unlabeled_rows,
            recorded_at=recorded_at,
        )
        return TrainingTaskRunResult(
            status=TrainingTaskRunStatus.UPLOADED,
            round_id=request.training_task.round_id,
            task_id=request.training_task.task_id,
            update_id=local_result.update_envelope.update_id,
            example_count=local_result.candidate_count,
            accepted_count=local_result.accepted_count,
            message="Query SSL update 업로드 완료.",
        )


def _insufficient_query_ssl_examples(
    training_task: TrainingTask,
    *,
    example_count: int = 0,
    accepted_count: int = 0,
    message: str,
) -> TrainingTaskRunResult:
    return TrainingTaskRunResult(
        status=TrainingTaskRunStatus.INSUFFICIENT_EXAMPLES,
        round_id=training_task.round_id,
        task_id=training_task.task_id,
        example_count=example_count,
        accepted_count=accepted_count,
        message=message,
    )
