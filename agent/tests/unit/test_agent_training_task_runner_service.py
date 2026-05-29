"""Agent training task runner service tests."""

from __future__ import annotations

from datetime import datetime, timezone
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
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
)


def _build_task_payload() -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_multiview",
        round_id="round_multiview",
        model_id="tracemind-embed",
        model_revision="rev_multiview",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=4,
        objective_config=TrainingObjectiveConfigPayload(
            algorithm_profile_name="prototype_pseudo_label_v1",
            training_backend_name="peft_classifier_trainer",
            confidence_threshold=0.6,
            margin_threshold=0.02,
            example_generation_backend_name="weak_strong_pair",
            evidence_backend_name="prototype_similarity_evidence",
            scorer_backend_name="prototype_similarity",
            acceptance_policy_name="top1_margin_threshold",
            privacy_guard_name="noop",
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
    )


def _build_supported_task_payload() -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id="task_001",
        round_id="round_0001",
        model_id="tracemind-embed",
        model_revision="rev_001",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=4,
        objective_config=TrainingObjectiveConfigPayload(
            training_backend_name="peft_classifier_trainer",
            example_generation_backend_name="prototype_rescore",
            evidence_backend_name="prototype_similarity_evidence",
            scorer_backend_name="prototype_similarity",
            acceptance_policy_name="top1_margin_threshold",
            privacy_guard_name="noop",
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
    )


def _build_prototype_pack() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_001",
            "embedding_model_id": "tracemind-embed",
            "embedding_model_revision": "rev_001",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "built_at": datetime(2026, 4, 2, tzinfo=timezone.utc),
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:single",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    }
                ]
            },
        }
    )


def _peft_backbone() -> dict[str, object]:
    return {
        "backbone_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "backbone_revision": "main",
        "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
        "tokenizer_revision": "main",
        "pooling": "mean",
        "max_length": 256,
        "task_prefix": "",
    }


def _peft_adapter_config() -> dict[str, object]:
    return {
        "peft_adapter_name": "lora",
        "parameters": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "bias": "none",
            "target_modules": "all-linear",
            "use_rslora": False,
        },
    }


def _build_peft_state(*, model_revision: str):
    return make_peft_classifier_state_payload(
        model_id="tracemind-embed",
        model_revision=model_revision,
        backbone=_peft_backbone(),
        peft_adapter_config=_peft_adapter_config(),
        label_schema=["anxiety", "normal"],
    )


def _build_service(
    *,
    repo: MagicMock,
    proto_service: MagicMock,
    proto_sync_service: MagicMock,
    shared_adapter_runtime_service: MagicMock,
    shared_adapter_sync_service: MagicMock,
    round_client_factory: MagicMock,
    runtime_factory: MagicMock,
) -> AgentTrainingTaskRunnerService:
    return AgentTrainingTaskRunnerService(
        scored_event_repository=repo,
        prototype_runtime_service=proto_service,
        prototype_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        federation_runtime_service_factory=runtime_factory,
    )


def test_runner_returns_unsupported_runtime_for_multiview_live_path() -> None:
    repo = MagicMock()
    proto_service = MagicMock()
    proto_sync_service = MagicMock()
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    runtime_factory = MagicMock()
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == "unsupported_runtime"
    assert response.round_id == "round_multiview"
    assert response.task_id == "task_multiview"
    assert "stored-event" in response.message
    repo.get_recent_stored.assert_not_called()
    proto_service.get_active_pack.assert_not_called()
    proto_sync_service.pull_version.assert_not_called()
    shared_adapter_sync_service.pull_current.assert_not_called()
    runtime_factory.assert_not_called()


def test_runner_syncs_shared_state_and_uses_matching_manifest() -> None:
    repo = MagicMock()
    repo.get_recent_stored.return_value = ()
    proto_service = MagicMock()
    proto_service.get_active_pack.return_value = _build_prototype_pack()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_001",
        auxiliary_artifact_versions={"prototype_pack": "proto_001"},
        artifact_ref="/server/state/rev_001.json",
    )
    active_state = _build_peft_state(model_revision="rev_001")
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = active_state
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_supported_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    federation_runtime = MagicMock()
    federation_runtime.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
        round_id="round_0001",
        task_id="task_001",
    )
    runtime_factory = MagicMock(return_value=federation_runtime)
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == str(FederationRunStatus.INSUFFICIENT_EXAMPLES)
    shared_adapter_sync_service.pull_current.assert_called_once_with(
        server_base_url="http://server.test"
    )
    proto_sync_service.pull_version.assert_called_once_with(
        server_base_url="http://server.test",
        prototype_version="proto_001",
    )
    call_kwargs = federation_runtime.run_current_task.call_args.kwargs
    assert call_kwargs["model_manifest"].model_revision == "rev_001"
    assert call_kwargs["task_payload"].model_revision == "rev_001"


def test_runner_does_not_pull_prototype_when_manifest_has_no_auxiliary_pack() -> None:
    repo = MagicMock()
    repo.get_recent_stored.return_value = ()
    proto_service = MagicMock()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_001",
        artifact_ref="/server/state/rev_001.json",
    )
    active_state = _build_peft_state(model_revision="rev_001")
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = active_manifest
    shared_adapter_runtime_service.get_active_state.return_value = active_state
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_supported_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    federation_runtime = MagicMock()
    federation_runtime.run_current_task.return_value = FederationRunResult(
        status=FederationRunStatus.INSUFFICIENT_EXAMPLES,
        round_id="round_0001",
        task_id="task_001",
    )
    runtime_factory = MagicMock(return_value=federation_runtime)
    service = _build_service(
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
    )

    response = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert response.status == str(FederationRunStatus.INSUFFICIENT_EXAMPLES)
    proto_sync_service.pull_version.assert_not_called()
    proto_service.get_active_pack.assert_not_called()
    call_kwargs = federation_runtime.run_current_task.call_args.kwargs
    assert call_kwargs["training_examples"] == ()
