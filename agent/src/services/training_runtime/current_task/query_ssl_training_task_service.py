"""Agent Query SSL current-task 실행 서비스."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from types import SimpleNamespace

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
from agent.src.services.training_runtime.current_task.result import (
    TrainingTaskRunResult,
    TrainingTaskRunStatus,
)
from agent.src.services.training_runtime.query_ssl_peft.local_training_service import (
    QuerySslLocalTrainingService,
    QuerySslPeftEncoderClientTrainingResult,
    QuerySslPeftEncoderLocalTrainingRequest,
)
from agent.src.services.training_runtime.training_sources.captured_text_source import (
    CapturedTextTrainingSourceService,
)
from methods.adaptation.local_update_backend import SharedAdapterTrainingBackend
from methods.adaptation.peft_text_encoder.federated_ssl.method_owned_training import (
    run_method_owned_peft_encoder_training_core,
)
from methods.adaptation.peft_text_encoder.training.query_ssl_local_training import (
    PeftEncoderTrainerRuntimeConfig,
)
from methods.adaptation.peft_text_encoder.update.delta_artifacts import (
    PeftEncoderDeltaMaterializer,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
    materialize_base_peft_encoder_state,
)
from methods.adaptation.peft_text_encoder.update_family_runtime import (
    build_training_backend_config_for_peft_encoder_state,
    build_training_backend_for_peft_encoder_state,
)
from methods.federated.aggregation.base import (
    AggregationJsonArtifactLoader,
    FederatedAggregationContext,
)
from methods.federated_ssl.hooks.peer_context import FederatedSslPeerContext
from methods.federated_ssl.method_config_surface import (
    DEFAULT_LOCAL_BUDGET_POLICY,
    default_method_local_ssl_policy_name,
    default_method_peer_context_policy_name,
)
from methods.federated_ssl.method_parameters import (
    build_federated_ssl_method_parameter_snapshot,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
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

MethodOwnedPeftEncoderTrainingCore = Callable[
    ...,
    QuerySslPeftEncoderClientTrainingResult,
]


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
        run_method_owned_peft_encoder_training_core
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
        local_result = self._run_local_update(
            request=request,
            local_service=local_service,
            labels=labels,
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            base_parameters=base_parameters,
            query_ssl_config=query_ssl_config,
            created_at=created_at,
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
        return TrainingTaskRunResult(
            status=TrainingTaskRunStatus.UPLOADED,
            round_id=request.training_task.round_id,
            task_id=request.training_task.task_id,
            update_id=local_result.update_envelope.update_id,
            example_count=local_result.candidate_count,
            accepted_count=local_result.accepted_count,
            message="Query SSL update 업로드 완료.",
        )

    def _run_local_update(
        self,
        *,
        request: AgentQuerySslTrainingTaskRunRequest,
        local_service: QuerySslLocalTrainingService,
        labels: Sequence[str],
        labeled_rows: Sequence[LabeledQueryRow],
        unlabeled_rows: Sequence[LabeledQueryRow],
        base_parameters: PeftEncoderMaterializedState,
        query_ssl_config: QuerySslObjectiveRuntimeConfig,
        created_at: datetime,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        fssl_method = _optional_name(request.training_task.fssl_method)
        if fssl_method is None:
            return local_service.run_peft_encoder(
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
        return self._run_method_owned_local_update(
            request=request,
            method_name=fssl_method,
            labels=labels,
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            base_parameters=base_parameters,
            query_ssl_config=query_ssl_config,
            created_at=created_at,
        )

    def _run_method_owned_local_update(
        self,
        *,
        request: AgentQuerySslTrainingTaskRunRequest,
        method_name: str,
        labels: Sequence[str],
        labeled_rows: Sequence[LabeledQueryRow],
        unlabeled_rows: Sequence[LabeledQueryRow],
        base_parameters: PeftEncoderMaterializedState,
        query_ssl_config: QuerySslObjectiveRuntimeConfig,
        created_at: datetime,
    ) -> QuerySslPeftEncoderClientTrainingResult:
        descriptor = resolve_federated_ssl_method_descriptor(method_name)
        if not descriptor.runtime_capabilities.live_agent_supported:
            raise ValueError(
                f"fssl_method={method_name!r}는 live agent runtime을 지원하지 않습니다."
            )
        method_config = _method_config_from_task_context(request.training_task)
        parameter_snapshot = build_federated_ssl_method_parameter_snapshot(
            method_name=descriptor.name,
            method_config=method_config,
        )
        local_ssl_policy = default_method_local_ssl_policy_name(descriptor)
        if local_ssl_policy is None:
            raise ValueError(
                f"fssl_method={method_name!r}의 기본 local SSL policy가 없습니다."
            )
        peft_config = build_training_backend_config_for_peft_encoder_state(
            active_adapter_state=request.active_state,
            objective_config=request.training_task.objective_config,
        )
        client_id = request.agent_id or "agent"
        return self.method_owned_training_core(
            client_id=client_id,
            seed=_seed_from_task(request.training_task),
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            labels=labels,
            base_parameters=base_parameters,
            training_task=request.training_task,
            model_manifest=request.model_manifest,
            ssl_method_config=SimpleNamespace(
                name=descriptor.name,
                scenario=parameter_snapshot.scenario,
                local_budget_policy=str(
                    method_config.get(
                        "local_budget_policy",
                        DEFAULT_LOCAL_BUDGET_POLICY,
                    )
                ),
                effective_parameters=parameter_snapshot.effective_parameters,
            ),
            local_ssl_policy_name=local_ssl_policy,
            query_ssl_config=query_ssl_config,
            strong_view_policy=query_ssl_config.strong_view_policy,
            unlabeled_batch_size=query_ssl_config.unlabeled_batch_size,
            peft_config=peft_config,
            trainer_runtime_config=self.trainer_runtime_config,
            created_at=created_at,
            delta_materializer=PeftEncoderDeltaMaterializer(
                artifact_store=_InlineOnlyPeftEncoderArtifactStore()
            ),
            peer_context=_peer_context_from_task(
                training_task=request.training_task,
                client_id=client_id,
                default_policy_name=default_method_peer_context_policy_name(
                    descriptor,
                    method_config,
                ),
            ),
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


def _method_config_from_task_context(training_task: TrainingTask) -> dict[str, object]:
    context = training_task.fssl_context or {}
    method_name = _optional_name(training_task.fssl_method)
    if method_name is None:
        raise ValueError("fssl_method is required for method-owned local runtime.")
    method_config = {
        "name": method_name,
        "use_original_parameters": True,
        "parameter_overrides": {},
    }
    raw_method_config = context.get("method_config")
    if isinstance(raw_method_config, Mapping):
        method_config.update(dict(raw_method_config))
        method_config["name"] = method_name
    raw_peer_context = context.get("peer_context")
    if isinstance(raw_peer_context, Mapping):
        scenario = _optional_name(raw_peer_context.get("scenario"))
        if scenario is not None:
            method_config["scenario"] = scenario
    return method_config


def _peer_context_from_task(
    *,
    training_task: TrainingTask,
    client_id: str,
    default_policy_name: str | None = None,
) -> FederatedSslPeerContext | None:
    context = training_task.fssl_context or {}
    raw_peer_context = context.get("peer_context")
    if not isinstance(raw_peer_context, Mapping):
        return None
    policy_name = _optional_name(raw_peer_context.get("policy_name"))
    if policy_name is None:
        policy_name = _optional_name(default_policy_name)
    if policy_name is None:
        return None
    client_payload = _find_peer_context_client_payload(
        raw_peer_context=raw_peer_context,
        client_id=client_id,
    )
    return FederatedSslPeerContext(
        client_id=client_id,
        policy_name=policy_name,
        round_index_zero_based=_round_index_zero_based(raw_peer_context),
        helper_client_ids=(
            ()
            if client_payload is None
            else tuple(
                str(helper_id)
                for helper_id in client_payload.get("helper_client_ids", ())
            )
        ),
        refreshed=not bool(raw_peer_context.get("warmup", False)),
        metadata={
            "source_round_id": raw_peer_context.get("source_round_id"),
            "context_kind": context.get("context_kind"),
            "method_name": context.get("method_name"),
            "summary_metrics": dict(raw_peer_context.get("summary_metrics", {})),
        },
    )


def _find_peer_context_client_payload(
    *,
    raw_peer_context: Mapping[str, object],
    client_id: str,
) -> Mapping[str, object] | None:
    raw_client_contexts = raw_peer_context.get("client_contexts", ())
    if not isinstance(raw_client_contexts, Sequence) or isinstance(
        raw_client_contexts,
        (str, bytes),
    ):
        return None
    for item in raw_client_contexts:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("client_id", "")).strip() == client_id:
            return item
    return None


def _round_index_zero_based(raw_peer_context: Mapping[str, object]) -> int:
    raw_round_index = raw_peer_context.get("round_index_zero_based")
    if raw_round_index is None:
        return 0
    return max(0, int(raw_round_index))


def _optional_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _build_usage_run_record(
    *,
    request: AgentQuerySslTrainingTaskRunRequest,
    query_ssl_config: QuerySslObjectiveRuntimeConfig,
    local_result: QuerySslPeftEncoderClientTrainingResult,
    recorded_at: datetime,
) -> TrainingUsageRunRecord:
    fssl_method = _optional_name(request.training_task.fssl_method)
    effective_method_family = (
        "federated_ssl" if fssl_method is not None else "query_ssl"
    )
    return TrainingUsageRunRecord(
        update_id=local_result.update_envelope.update_id,
        round_id=request.training_task.round_id,
        task_id=request.training_task.task_id,
        recorded_at=recorded_at,
        agent_id=request.agent_id,
        model_id=request.training_task.model_id,
        model_revision=request.training_task.model_revision,
        objective_method_name=fssl_method or query_ssl_config.method_name,
        objective_algorithm_name=(
            effective_method_family
            if fssl_method is not None
            else query_ssl_config.algorithm_name
        ),
        status=TRAINING_USAGE_STATUS_UPLOADED,
        candidate_count=local_result.candidate_count,
        accepted_count=local_result.accepted_count,
        metadata={
            "effective_method_family": effective_method_family,
            "fssl_method": fssl_method,
            "query_ssl_method_name": query_ssl_config.method_name,
            "query_ssl_algorithm_name": query_ssl_config.algorithm_name,
            "training_scope": str(request.training_task.training_scope),
            "local_epochs": request.training_task.local_epochs,
            "max_steps": request.training_task.max_steps,
            "batch_size": request.training_task.batch_size,
            "learning_rate": request.training_task.learning_rate,
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
) -> TrainingTaskRunResult:
    return TrainingTaskRunResult(
        status=TrainingTaskRunStatus.INSUFFICIENT_EXAMPLES,
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
