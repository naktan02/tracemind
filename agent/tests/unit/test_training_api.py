"""Training API unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from agent.src.api import training as training_api
from agent.src.services.federation.rounds.runtime_service import (
    FederationRunResult,
    FederationRunStatus,
)
from shared.src.contracts.adapter_contracts import make_identity_state_payload
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
            training_backend_name="diagonal_scale_heuristic",
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
            training_backend_name="diagonal_scale_heuristic",
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


def test_run_current_task_returns_unsupported_runtime_for_multiview_live_path() -> None:
    repo = MagicMock()
    proto_service = MagicMock()
    proto_sync_service = MagicMock()
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    runtime_factory = MagicMock()

    response = training_api.run_current_task(
        training_api.RunCurrentTaskRequest(server_base_url="http://server.test"),
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
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


def test_run_current_task_syncs_shared_state_and_uses_matching_manifest() -> None:
    repo = MagicMock()
    repo.get_recent_stored.return_value = ()
    proto_service = MagicMock()
    proto_service.get_active_pack.return_value = _build_prototype_pack()
    proto_sync_service = MagicMock()
    shared_adapter_sync_service = MagicMock()
    active_manifest = make_embedding_manifest(
        model_id="tracemind-embed",
        model_revision="rev_001",
        prototype_version="proto_001",
        artifact_ref="/server/state/rev_001.json",
    )
    active_state = make_identity_state_payload(
        model_id="tracemind-embed",
        model_revision="rev_001",
        embedding_dim=2,
    )
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

    response = training_api.run_current_task(
        training_api.RunCurrentTaskRequest(server_base_url="http://server.test"),
        repo=repo,
        proto_service=proto_service,
        proto_sync_service=proto_sync_service,
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=shared_adapter_sync_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
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
