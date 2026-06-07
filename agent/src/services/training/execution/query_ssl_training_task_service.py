"""Agent Query SSL current-task 실행 서비스."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256

from agent.src.infrastructure.repositories.analysis_event_repository import (
    AnalysisEventRepository,
    StoredAnalysisEvent,
)
from agent.src.infrastructure.repositories.captured_text_repository import (
    CapturedTextRepository,
)
from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.infrastructure.repositories.training_usage_ledger_repository import (
    TRAINING_USAGE_ROLE_LABELED_ANCHOR,
    TRAINING_USAGE_ROLE_UNLABELED_GENERATED_VIEW,
    TRAINING_USAGE_STAGE_QUERY_SSL_INPUT,
    TRAINING_USAGE_STATUS_UPLOADED,
    TrainingUsageLedgerRepository,
    TrainingUsageRowRecord,
    TrainingUsageRunRecord,
)
from agent.src.services.federation.rounds.round_client import RoundClient
from agent.src.services.federation.rounds.runtime_service import (
    FederationRunResult,
    FederationRunStatus,
)
from agent.src.services.training.datasets.captured_text_training_source_service import (
    CapturedTextTrainingSourceService,
)
from agent.src.services.training.execution.query_ssl_local_training_service import (
    QuerySslLocalTrainingService,
    QuerySslPeftEncoderClientTrainingResult,
    QuerySslPeftEncoderLocalTrainingRequest,
)
from methods.adaptation.local_update_backend import SharedAdapterTrainingBackend
from methods.adaptation.peft_text_encoder.training.query_ssl_local_training import (
    PeftEncoderTrainerRuntimeConfig,
)
from methods.adaptation.peft_text_encoder.update.delta_artifacts import (
    PeftEncoderDeltaMaterializer,
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
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingTask,
    make_training_update_submission,
)
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
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig = field(
        default_factory=AgentQuerySslTrainerRuntimeConfig
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def run_current_task(
        self,
        request: AgentQuerySslTrainingTaskRunRequest,
    ) -> FederationRunResult:
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

        labels = tuple(str(label) for label in request.active_state.label_schema)
        labeled_rows = _build_labeled_rows_from_stored_events(
            stored_events=request.analysis_event_repository.get_recent_stored(
                days=request.analysis_event_days
            ),
            labels=labels,
        )
        unlabeled_rows = CapturedTextTrainingSourceService(
            repository=request.captured_text_repository
        ).get_recent_query_ssl_unlabeled_rows(
            days=request.analysis_event_days,
            limit=request.training_task.selection_policy.max_examples or 100,
        )
        if not labeled_rows or not unlabeled_rows:
            return _insufficient_query_ssl_examples(
                request.training_task,
                example_count=len(unlabeled_rows),
                accepted_count=len(labeled_rows),
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
        local_result = local_service.run_peft_encoder(
            QuerySslPeftEncoderLocalTrainingRequest(
                client_id=request.agent_id or "agent",
                seed=_seed_from_task(request.training_task),
                labeled_rows=labeled_rows,
                unlabeled_rows=unlabeled_rows,
                labels=labels,
                base_parameters=base_parameters,
                training_task=request.training_task,
                model_manifest=request.model_manifest,
                query_ssl_config=query_ssl_config,
                trainer_runtime_config=self.trainer_runtime_config,
                created_at=created_at,
                delta_materializer=PeftEncoderDeltaMaterializer(
                    artifact_store=_InlineOnlyPeftEncoderArtifactStore()
                ),
                agent_id=request.agent_id,
            )
        )
        request.round_client.upload_update(
            request.training_task.round_id,
            make_training_update_submission(
                envelope=local_result.update_envelope,
                update_payload=local_result.update_payload,
            ),
        )
        recorded_at = self.clock.now()
        self.usage_ledger_repository.save_run(
            _build_usage_run_record(
                request=request,
                query_ssl_config=query_ssl_config,
                local_result=local_result,
                recorded_at=recorded_at,
            ),
            rows=_build_usage_row_records(
                update_id=local_result.update_envelope.update_id,
                training_task=request.training_task,
                labeled_rows=labeled_rows,
                unlabeled_rows=unlabeled_rows,
                recorded_at=recorded_at,
            ),
        )
        return FederationRunResult(
            status=FederationRunStatus.UPLOADED,
            round_id=request.training_task.round_id,
            task_id=request.training_task.task_id,
            update_id=local_result.update_envelope.update_id,
            example_count=local_result.candidate_count,
            accepted_count=local_result.accepted_count,
            message="Query SSL update 업로드 완료.",
        )


def _build_labeled_rows_from_stored_events(
    *,
    stored_events: Sequence[StoredAnalysisEvent],
    labels: Sequence[str],
) -> tuple[LabeledQueryRow, ...]:
    label_set = {str(label) for label in labels}
    rows: list[LabeledQueryRow] = []
    for stored_event in stored_events:
        event = stored_event.analysis_event
        if event.translated_text is None or not event.category_scores:
            continue
        label = _top_label(event.category_scores)
        if label not in label_set:
            continue
        rows.append(
            {
                "query_id": event.query_id,
                "text": event.translated_text,
                "raw_label_scheme": "agent_local_pseudo_label",
                "raw_label": label,
                "mapped_label_4": label,
                "locale": "en",
                "annotation_source": "agent_local_analysis_event",
                "approved_by": None,
                "created_at": event.occurred_at.isoformat(),
            }
        )
    return tuple(rows)


def _top_label(category_scores: Mapping[str, float]) -> str:
    return max(category_scores.items(), key=lambda item: float(item[1]))[0]


def _seed_from_task(training_task: TrainingTask) -> int:
    source = f"{training_task.round_id}:{training_task.task_id}".encode("utf-8")
    return int.from_bytes(sha256(source).digest()[:4], byteorder="big") % (2**31)


def _build_usage_run_record(
    *,
    request: AgentQuerySslTrainingTaskRunRequest,
    query_ssl_config: QuerySslObjectiveRuntimeConfig,
    local_result: QuerySslPeftEncoderClientTrainingResult,
    recorded_at: datetime,
) -> TrainingUsageRunRecord:
    return TrainingUsageRunRecord(
        update_id=local_result.update_envelope.update_id,
        round_id=request.training_task.round_id,
        task_id=request.training_task.task_id,
        recorded_at=recorded_at,
        agent_id=request.agent_id,
        model_id=request.training_task.model_id,
        model_revision=request.training_task.model_revision,
        objective_method_name=query_ssl_config.method_name,
        objective_algorithm_name=query_ssl_config.algorithm_name,
        status=TRAINING_USAGE_STATUS_UPLOADED,
        candidate_count=local_result.candidate_count,
        accepted_count=local_result.accepted_count,
        metadata={
            "training_scope": str(request.training_task.training_scope),
            "local_epochs": request.training_task.local_epochs,
            "max_steps": request.training_task.max_steps,
            "selection_max_examples": (
                request.training_task.selection_policy.max_examples
            ),
        },
    )


def _build_usage_row_records(
    *,
    update_id: str,
    training_task: TrainingTask,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    recorded_at: datetime,
) -> tuple[TrainingUsageRowRecord, ...]:
    records: list[TrainingUsageRowRecord] = []
    for row in labeled_rows:
        records.append(
            TrainingUsageRowRecord(
                update_id=update_id,
                source_id=str(row["query_id"]),
                role=TRAINING_USAGE_ROLE_LABELED_ANCHOR,
                round_id=training_task.round_id,
                task_id=training_task.task_id,
                recorded_at=recorded_at,
                source_kind="analysis_event",
                stage=TRAINING_USAGE_STAGE_QUERY_SSL_INPUT,
                label=str(row["mapped_label_4"]),
                metadata={
                    "annotation_source": str(row["annotation_source"]),
                    "raw_label_scheme": str(row["raw_label_scheme"]),
                },
            )
        )
    for row in unlabeled_rows:
        records.append(
            TrainingUsageRowRecord(
                update_id=update_id,
                source_id=str(row["query_id"]),
                role=TRAINING_USAGE_ROLE_UNLABELED_GENERATED_VIEW,
                round_id=training_task.round_id,
                task_id=training_task.task_id,
                recorded_at=recorded_at,
                source_kind="captured_text_generated_view",
                stage=TRAINING_USAGE_STAGE_QUERY_SSL_INPUT,
                label=None,
                metadata={
                    "annotation_source": str(row["annotation_source"]),
                    "weak_translated": bool(row.get("weak_translated_text")),
                    "strong_translated": bool(row.get("strong_translated_text")),
                },
            )
        )
    return tuple(records)


def _insufficient_query_ssl_examples(
    training_task: TrainingTask,
    *,
    example_count: int = 0,
    accepted_count: int = 0,
    message: str,
) -> FederationRunResult:
    return FederationRunResult(
        status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
        round_id=training_task.round_id,
        task_id=training_task.task_id,
        example_count=example_count,
        accepted_count=accepted_count,
        message=message,
    )


class _InlineOnlyPeftEncoderArtifactStore:
    """inline delta live runtime에서 호출되지 않아야 하는 artifact store."""

    def ref_for_agent_artifact(self, **_kwargs: object) -> str:
        raise ValueError("live Query SSL run-current-task uses inline deltas.")

    def save_agent_json_artifact(self, **_kwargs: object) -> None:
        raise ValueError("live Query SSL run-current-task uses inline deltas.")

    def ref_for_server_client_update_artifact(self, **_kwargs: object) -> str:
        raise ValueError("live Query SSL run-current-task uses inline deltas.")

    def save_server_safetensors_artifact_ref(self, **_kwargs: object) -> None:
        raise ValueError("live Query SSL run-current-task uses inline deltas.")

    def is_agent_local_ref(self, artifact_ref: str | None) -> bool:
        return artifact_ref is not None and artifact_ref.startswith("agent-local://")

    def upload_agent_local_json_artifact(self, *, agent_local_ref: str) -> str:
        raise ValueError(
            f"live Query SSL run-current-task cannot upload {agent_local_ref!r}."
        )

    def server_artifact_refs_byte_count(
        self,
        *,
        artifact_refs: Sequence[str | None],
    ) -> int:
        return 0
