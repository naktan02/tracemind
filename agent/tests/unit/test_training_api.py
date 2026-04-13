"""Training API unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent.src.api import training as training_api
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


def test_run_current_task_returns_unsupported_runtime_for_multiview_live_path() -> None:
    repo = MagicMock()
    proto_service = MagicMock()
    round_client = MagicMock()
    round_client.fetch_current_task.return_value = _build_task_payload()
    round_client_factory = MagicMock(return_value=round_client)
    runtime_factory = MagicMock()

    response = training_api.run_current_task(
        training_api.RunCurrentTaskRequest(server_base_url="http://server.test"),
        repo=repo,
        proto_service=proto_service,
        round_client_factory=round_client_factory,
        runtime_factory=runtime_factory,
    )

    assert response.status == "unsupported_runtime"
    assert response.round_id == "round_multiview"
    assert response.task_id == "task_multiview"
    assert "stored-event" in response.message
    repo.get_recent_stored.assert_not_called()
    proto_service.get_active_pack.assert_not_called()
    runtime_factory.assert_not_called()
