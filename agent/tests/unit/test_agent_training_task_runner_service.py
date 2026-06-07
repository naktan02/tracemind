"""Agent training task runner service tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.src.services.federation.rounds.runtime_service import (
    FederationRunResult,
    FederationRunStatus,
)
from agent.src.services.training.execution.agent_training_task_runner_service import (
    AgentTrainingTaskRunnerService,
    AgentTrainingTaskRunRequest,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_state_payload,
)
from shared.src.contracts.model_contracts import make_embedding_manifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
)


def _build_query_ssl_task_payload() -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_query_ssl",
        round_id="round_query_ssl",
        model_id="tracemind-embed",
        model_revision="rev_query_ssl",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=4,
        objective_config=TrainingObjectiveConfigPayload(
            algorithm_profile_name="peft_classifier_update_v1",
            training_backend_name="peft_classifier_trainer",
            example_generation_backend_name="weak_strong_pair",
            evidence_backend_name="analysis_score_evidence",
            scorer_backend_name="classifier_head_logits",
            acceptance_policy_name="top1_margin_threshold",
            privacy_guard_name="noop",
            extras={
                "selection.confidence_threshold": 0.6,
                "selection.margin_threshold": 0.02,
                "query_ssl.method_name": "fixmatch_usb_v1",
                "query_ssl.algorithm_name": "fixmatch",
                "query_ssl.strong_view_policy": "first_aug",
                "query_ssl.unlabeled_batch_size": 8,
                "query_ssl.temperature": 0.5,
                "query_ssl.p_cutoff": 0.95,
                "query_ssl.hard_label": True,
                "query_ssl.lambda_u": 1.0,
                "query_ssl.supervised_loss_weight": 1.0,
            },
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
    )


def _build_legacy_task_payload() -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_legacy",
        round_id="round_legacy",
        model_id="tracemind-embed",
        model_revision="rev_legacy",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=4,
        objective_config=TrainingObjectiveConfigPayload(
            training_backend_name="peft_classifier_trainer",
            example_generation_backend_name="peft_classifier_raw_rows",
            evidence_backend_name="analysis_score_evidence",
            scorer_backend_name="classifier_head_logits",
            acceptance_policy_name="top1_margin_threshold",
            privacy_guard_name="noop",
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
    )


def _build_peft_state(*, model_revision: str):
    return make_peft_classifier_state_payload(
        model_id="tracemind-embed",
        model_revision=model_revision,
        backbone={
            "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "backbone_revision": "main",
            "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
            "tokenizer_revision": "main",
            "pooling": "mean",
            "max_length": 256,
            "task_prefix": "",
        },
        peft_adapter_config={
            "peft_adapter_name": "lora",
            "parameters": {"rank": 8},
        },
        label_schema=["anxiety", "normal"],
    )


def _build_service(
    *,
    repo: MagicMock,
    shared_adapter_runtime_service: MagicMock,
    shared_adapter_sync_service: MagicMock,
    round_client_factory: MagicMock,
    runtime_factory: MagicMock,
    query_ssl_task_service: object | None = None,
) -> AgentTrainingTaskRunnerService:
    kwargs = {}
    if query_ssl_task_service is not None:
        kwargs["query_ssl_task_service"] = query_ssl_task_service
    return AgentTrainingTaskRunnerService(
        analysis_event_repository=repo,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        federation_runtime_service_factory=runtime_factory,
        **kwargs,
    )


def test_runner_routes_query_ssl_task_to_query_ssl_service() -> None:
    repo = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_query_ssl",
        artifact_ref="/server/state/rev_query_ssl.json",
    )
    active_state = _build_peft_state(model_revision="rev_query_ssl")
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = active_state
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_query_ssl_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    runtime_factory = MagicMock()
    query_ssl_task_service = MagicMock()
    query_ssl_task_service.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.UPLOADED,
        round_id="round_query_ssl",
        task_id="task_query_ssl",
        update_id="update_query_ssl",
        example_count=3,
        accepted_count=2,
        message="Query SSL update 업로드 완료.",
    )
    service = _build_service(
        repo=repo,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
        query_ssl_task_service=query_ssl_task_service,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == str(FederationRunStatus.UPLOADED)
    assert response.round_id == "round_query_ssl"
    assert response.task_id == "task_query_ssl"
    assert response.update_id == "update_query_ssl"
    shared_adapter_sync_service.pull_current.assert_called_once_with(
        server_base_url="http://server.test"
    )
    query_ssl_request = query_ssl_task_service.run_current_task.call_args.args[0]
    assert query_ssl_request.training_task.task_id == "task_query_ssl"
    assert query_ssl_request.model_manifest is active_manifest
    assert query_ssl_request.active_state is active_state
    runtime_factory.assert_not_called()


def test_runner_rejects_legacy_non_query_ssl_task_without_runtime_contract() -> None:
    repo = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_legacy",
        artifact_ref="/server/state/rev_legacy.json",
    )
    active_state = _build_peft_state(model_revision="rev_legacy")
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = active_state
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_legacy_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    federation_runtime = MagicMock()
    federation_runtime.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
        round_id="round_legacy",
        task_id="task_legacy",
    )
    runtime_factory = MagicMock(return_value=federation_runtime)
    service = _build_service(
        repo=repo,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == "unsupported_runtime"
    shared_adapter_sync_service.pull_current.assert_not_called()
    runtime_factory.assert_not_called()
    federation_runtime.run_current_task.assert_not_called()
