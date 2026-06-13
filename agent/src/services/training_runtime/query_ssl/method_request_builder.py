"""Query SSL/FSSL PEFT local update request builder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from types import SimpleNamespace

from agent.src.services.training_runtime.query_ssl.task_identity import (
    optional_method_name,
    seed_from_task,
)
from agent.src.services.training_runtime.query_ssl_peft.local_training_service import (
    QuerySslLocalTrainingService,
    QuerySslPeftEncoderClientTrainingResult,
    QuerySslPeftEncoderLocalTrainingRequest,
)
from methods.adaptation.peft_text_encoder.federated_ssl.method_training_surface import (
    FsslPeftEncoderMethodTrainingRequest,
)
from methods.adaptation.peft_text_encoder.training.query_ssl_local_training import (
    PeftEncoderTrainerRuntimeConfig,
)
from methods.adaptation.peft_text_encoder.update.delta_artifacts import (
    PeftEncoderDeltaMaterializer,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)
from methods.adaptation.peft_text_encoder.update_family_runtime import (
    build_training_backend_config_for_peft_encoder_state,
)
from methods.federated_ssl.live_task_context import (
    build_method_config_from_live_fssl_context,
    build_peer_context_from_live_fssl_context,
)
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
from shared.src.contracts.training_contracts import TrainingTask

MethodOwnedPeftEncoderTrainingCore = Callable[
    [FsslPeftEncoderMethodTrainingRequest],
    QuerySslPeftEncoderClientTrainingResult,
]


def run_query_ssl_local_update(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    active_state: PeftClassifierState,
    agent_id: str | None,
    local_service: QuerySslLocalTrainingService,
    method_owned_training_core: MethodOwnedPeftEncoderTrainingCore,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    labels: Sequence[str],
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    base_parameters: PeftEncoderMaterializedState,
    query_ssl_config: QuerySslObjectiveRuntimeConfig,
    created_at: datetime,
) -> QuerySslPeftEncoderClientTrainingResult:
    """task method identity에 따라 standard 또는 method-owned update를 실행한다."""

    fssl_method = optional_method_name(training_task.fssl_method)
    if fssl_method is None:
        return local_service.run_peft_encoder(
            build_query_ssl_peft_encoder_request(
                training_task=training_task,
                model_manifest=model_manifest,
                agent_id=agent_id,
                labels=labels,
                labeled_rows=labeled_rows,
                unlabeled_rows=unlabeled_rows,
                base_parameters=base_parameters,
                query_ssl_config=query_ssl_config,
                trainer_runtime_config=trainer_runtime_config,
                created_at=created_at,
            )
        )
    return method_owned_training_core(
        build_fssl_peft_encoder_method_request(
            training_task=training_task,
            model_manifest=model_manifest,
            active_state=active_state,
            method_name=fssl_method,
            agent_id=agent_id,
            labels=labels,
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            base_parameters=base_parameters,
            query_ssl_config=query_ssl_config,
            trainer_runtime_config=trainer_runtime_config,
            created_at=created_at,
        )
    )


def build_query_ssl_peft_encoder_request(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    agent_id: str | None,
    labels: Sequence[str],
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    base_parameters: PeftEncoderMaterializedState,
    query_ssl_config: QuerySslObjectiveRuntimeConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    created_at: datetime,
) -> QuerySslPeftEncoderLocalTrainingRequest:
    """standard Query SSL PEFT local trainer request를 만든다."""

    return QuerySslPeftEncoderLocalTrainingRequest(
        client_id=agent_id or "agent",
        seed=seed_from_task(training_task),
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        labels=labels,
        base_parameters=base_parameters,
        training_task=training_task,
        model_manifest=model_manifest,
        query_ssl_config=query_ssl_config,
        trainer_runtime_config=trainer_runtime_config,
        created_at=created_at,
        delta_materializer=PeftEncoderDeltaMaterializer(
            artifact_store=_InlineOnlyPeftEncoderArtifactStore()
        ),
        agent_id=agent_id,
    )


def build_fssl_peft_encoder_method_request(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    active_state: PeftClassifierState,
    method_name: str,
    agent_id: str | None,
    labels: Sequence[str],
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    base_parameters: PeftEncoderMaterializedState,
    query_ssl_config: QuerySslObjectiveRuntimeConfig,
    trainer_runtime_config: PeftEncoderTrainerRuntimeConfig,
    created_at: datetime,
) -> FsslPeftEncoderMethodTrainingRequest:
    """method-owned FSSL PEFT local trainer request를 만든다."""

    descriptor = resolve_federated_ssl_method_descriptor(method_name)
    if not descriptor.runtime_capabilities.live_agent_supported:
        raise ValueError(
            f"fssl_method={method_name!r}는 live agent runtime을 지원하지 않습니다."
        )
    method_config = build_method_config_from_live_fssl_context(
        fssl_method=training_task.fssl_method,
        fssl_context=training_task.fssl_context,
    )
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
        active_adapter_state=active_state,
        objective_config=training_task.objective_config,
    )
    client_id = agent_id or "agent"
    return FsslPeftEncoderMethodTrainingRequest(
        client_id=client_id,
        seed=seed_from_task(training_task),
        labeled_rows=labeled_rows,
        unlabeled_rows=unlabeled_rows,
        labels=labels,
        base_parameters=base_parameters,
        training_task=training_task,
        model_manifest=model_manifest,
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
        trainer_runtime_config=trainer_runtime_config,
        created_at=created_at,
        delta_materializer=PeftEncoderDeltaMaterializer(
            artifact_store=_InlineOnlyPeftEncoderArtifactStore()
        ),
        peer_context=build_peer_context_from_live_fssl_context(
            fssl_context=training_task.fssl_context,
            client_id=client_id,
            default_policy_name=default_method_peer_context_policy_name(
                descriptor,
                method_config,
            ),
        ),
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
