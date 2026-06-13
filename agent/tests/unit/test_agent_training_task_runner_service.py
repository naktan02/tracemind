"""Agent training task runner service tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.src.features.training_runtime.current_task.result import (
    TrainingTaskRunResult,
    TrainingTaskRunStatus,
)
from agent.src.features.training_runtime.current_task.runner import (  # noqa: E501
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


def _build_query_ssl_task_payload(
    *,
    algorithm_profile_name: str | None = "peft_classifier_update_v1",
    fssl_method: str | None = None,
    fssl_execution: dict[str, object] | None = None,
    fssl_capability_plan: dict[str, object] | None = None,
    fssl_context: dict[str, object] | None = None,
) -> TrainingTaskPayload:
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
            algorithm_profile_name=algorithm_profile_name,
            training_backend_name="peft_classifier_trainer",
            privacy_guard_name="noop",
            extras={
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
        fssl_method=fssl_method,
        fssl_execution=fssl_execution,
        fssl_capability_plan=fssl_capability_plan,
        fssl_context=fssl_context,
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
    round_client.fetch_current_task.return_value = _build_query_ssl_task_payload(
        fssl_method="fedmatch",
        fssl_context={
            "schema_version": "fssl_context.v1",
            "method_name": "fedmatch",
            "context_kind": "peer_context",
            "peer_context": {
                "schema_version": "peer_context_task.v1",
                "policy_name": "previous_round_metric_summary",
            },
        },
    )
    round_client_factory = MagicMock(return_value=round_client)
    query_ssl_task_service = MagicMock()
    query_ssl_task_service.run_current_task.return_value = TrainingTaskRunResult(
        status=TrainingTaskRunStatus.UPLOADED,
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
        query_ssl_task_service=query_ssl_task_service,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == TrainingTaskRunStatus.UPLOADED
    assert response.round_id == "round_query_ssl"
    assert response.task_id == "task_query_ssl"
    assert response.update_id == "update_query_ssl"
    shared_adapter_sync_service.pull_current.assert_called_once_with(
        server_base_url="http://server.test"
    )
    query_ssl_request = query_ssl_task_service.run_current_task.call_args.args[0]
    assert query_ssl_request.training_task.task_id == "task_query_ssl"
    assert query_ssl_request.training_task.fssl_method == "fedmatch"
    assert query_ssl_request.training_task.fssl_context["method_name"] == "fedmatch"
    assert query_ssl_request.model_manifest is active_manifest
    assert query_ssl_request.active_state is active_state


def test_runner_validates_fssl_runtime_snapshot_before_local_training() -> None:
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
    round_client.fetch_current_task.return_value = _build_query_ssl_task_payload(
        fssl_method="fedmatch",
        fssl_execution={
            "composition_mode": "method_owned",
            "execution_role": "method_owned",
            "method_name": "fedmatch",
            "descriptor_name": "fedmatch",
        },
        fssl_capability_plan=_fedmatch_capability_plan_payload(),
    )
    round_client_factory = MagicMock(return_value=round_client)
    query_ssl_task_service = MagicMock()
    query_ssl_task_service.run_current_task.return_value = TrainingTaskRunResult(
        status=TrainingTaskRunStatus.UPLOADED,
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
        query_ssl_task_service=query_ssl_task_service,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == TrainingTaskRunStatus.UPLOADED
    query_ssl_task_service.run_current_task.assert_called_once()


def test_runner_rejects_drifted_fssl_runtime_snapshot() -> None:
    repo = MagicMock()
    shared_adapter_sync_service = MagicMock()
    shared_adapter_runtime_service = MagicMock()
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_query_ssl_task_payload(
        fssl_method="fedmatch",
        fssl_execution={
            "composition_mode": "method_owned",
            "execution_role": "method_owned",
            "method_name": "other_method",
            "descriptor_name": "other_method",
        },
        fssl_capability_plan=_fedmatch_capability_plan_payload(),
    )
    round_client_factory = MagicMock(return_value=round_client)
    service = _build_service(
        repo=repo,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == TrainingTaskRunStatus.UNSUPPORTED_RUNTIME
    assert "fssl_method와 fssl_execution.method_name" in str(response.message)
    shared_adapter_sync_service.pull_current.assert_not_called()


def test_runner_rejects_unsupported_query_ssl_runtime_profile_before_sync() -> None:
    repo = MagicMock()
    shared_adapter_sync_service = MagicMock()
    shared_adapter_runtime_service = MagicMock()
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_query_ssl_task_payload(
        algorithm_profile_name="unsupported_profile"
    )
    round_client_factory = MagicMock(return_value=round_client)
    service = _build_service(
        repo=repo,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == TrainingTaskRunStatus.UNSUPPORTED_RUNTIME
    assert "peft_classifier_update_v1" in str(response.message)
    shared_adapter_sync_service.pull_current.assert_not_called()


def test_runner_rejects_unsupported_update_family_before_sync() -> None:
    repo = MagicMock()
    shared_adapter_sync_service = MagicMock()
    shared_adapter_runtime_service = MagicMock()
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_query_ssl_task_payload(
        fssl_method="fedmatch",
        fssl_execution={
            "composition_mode": "method_owned",
            "execution_role": "method_owned",
            "method_name": "fedmatch",
            "descriptor_name": "fedmatch",
            "runtime_surface": {
                "payload_adapter_kind": "linear_head",
                "update_family_name": "linear_head",
                "aggregation_backend_name": "fedavg",
            },
        },
        fssl_capability_plan=_fedmatch_capability_plan_payload(),
    )
    round_client_factory = MagicMock(return_value=round_client)
    service = _build_service(
        repo=repo,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == TrainingTaskRunStatus.UNSUPPORTED_RUNTIME
    assert "peft_text_encoder" in str(response.message)
    shared_adapter_sync_service.pull_current.assert_not_called()


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
    service = _build_service(
        repo=repo,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == TrainingTaskRunStatus.UNSUPPORTED_RUNTIME
    shared_adapter_sync_service.pull_current.assert_not_called()


def _fedmatch_capability_plan_payload() -> dict[str, object]:
    return {
        "client_participation_policy": {"name": "all_clients"},
        "aggregation_weight_policy": {"name": "uniform"},
        "labeled_exposure_policy": {"name": "shared_client_seed"},
        "local_supervision_regime": {"name": "client_labeled_and_unlabeled"},
        "server_step_policy": {"name": "none"},
        "server_update_policy": {"name": "fedmatch_partitioned"},
        "peer_context_policy": {"name": "fixed_probe_output_knn"},
        "update_partition_policy": {"name": "partitioned"},
        "local_ssl_policy": {"name": "fedmatch_agreement"},
        "query_multiview_source": {"name": "materialized_rows"},
    }
