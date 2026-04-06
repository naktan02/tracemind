"""RoundClient, FederationRuntimeService 단위 테스트."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import httpx

from agent.src.services.federation.round_client import RoundClient
from agent.src.services.federation.runtime_service import (
    FederationRunStatus,
    FederationRuntimeService,
)
from agent.src.services.training.local_training_service import (
    LocalTrainingResult,
)
from agent.src.services.training.pseudo_label_service import PseudoLabelSelectionResult
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfigPayload,
    TrainingSelectionPolicyPayload,
    TrainingTaskPayload,
    TrainingUpdateEnvelope,
)

# ─── 공통 fixture ────────────────────────────────────────────────────────────


def _build_manifest(revision: str = "rev_000") -> ModelManifest:
    return ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision=revision,
        published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref="/tmp/rev_000.json",
        prototype_version="proto_000",
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )


def _build_task_payload(
    *,
    task_id: str = "task_001",
    round_id: str = "round_0001",
    model_revision: str = "rev_000",
) -> TrainingTaskPayload:
    return TrainingTaskPayload(
        schema_version="training_task.v1",
        task_id=task_id,
        round_id=round_id,
        model_id="tracemind-embed",
        model_revision=model_revision,
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=10,
        objective_config=TrainingObjectiveConfigPayload(
            loss="diagonal_scale_heuristic",
            confidence_threshold=0.6,
            margin_threshold=0.02,
        ),
        selection_policy=TrainingSelectionPolicyPayload(),
    )


def _empty_selection_result() -> PseudoLabelSelectionResult:
    return PseudoLabelSelectionResult(
        candidates=(),
        accepted_candidates=(),
        feedback_signals=(),
    )


def _transport_with_json(
    payload: dict[str, object] | None,
    *,
    status_code: int = 200,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if payload is None:
            return httpx.Response(status_code=status_code, request=request)
        return httpx.Response(
            status_code=status_code,
            request=request,
            content=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    return httpx.MockTransport(handler)


# ─── RoundClient 테스트 ───────────────────────────────────────────────────────


def test_round_client_fetch_current_task_returns_none_on_404() -> None:
    client = RoundClient(
        server_base_url="http://localhost:8000",
        _transport=_transport_with_json(None, status_code=404),
    )
    result = client.fetch_current_task()

    assert result is None


def test_round_client_fetch_current_task_returns_none_when_status_not_open() -> None:
    """active round 상태가 open이 아니면 task를 None으로 반환한다."""
    finalized_record = {
        "round_id": "round_0001",
        "status": "finalized",
        "training_task": {
            "schema_version": "training_task.v1",
            "task_id": "task_001",
            "round_id": "round_0001",
            "model_id": "tracemind-embed",
            "model_revision": "rev_000",
            "task_type": "pseudo_label_self_training",
            "training_scope": "adapter_only",
            "local_epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-2,
            "max_steps": 10,
            "objective_config": {"loss": "diagonal_scale_heuristic"},
            "selection_policy": {},
        },
        "created_at": "2026-03-29T00:00:00Z",
        "updated_at": "2026-03-29T00:00:00Z",
    }
    client = RoundClient(
        server_base_url="http://localhost:8000",
        _transport=_transport_with_json(finalized_record),
    )
    result = client.fetch_current_task()

    assert result is None


# ─── FederationRuntimeService 테스트 ─────────────────────────────────────────


def test_federation_runtime_returns_no_active_task_when_no_round() -> None:
    client = MagicMock(spec=RoundClient)
    client.fetch_current_task.return_value = None
    service = FederationRuntimeService(round_client=client)

    result = service.run_current_task(
        training_examples=(),
        model_manifest=_build_manifest(),
    )

    assert result.status == FederationRunStatus.NO_ACTIVE_TASK
    assert result.round_id is None
    assert result.update_id is None


def test_federation_runtime_returns_already_completed_on_duplicate() -> None:
    client = MagicMock(spec=RoundClient)
    client.fetch_current_task.return_value = _build_task_payload()
    local_service = MagicMock()
    local_service.run_task.return_value = LocalTrainingResult(
        selection_result=_empty_selection_result()
    )
    service = FederationRuntimeService(
        round_client=client,
        local_training_service=local_service,
    )

    # 첫 번째 실행에서 INSUFFICIENT_EXAMPLES가 되어도 task_id를 기록하지 않는다.
    # (update가 없으면 완료 처리하지 않는다.)
    first = service.run_current_task(
        training_examples=(),
        model_manifest=_build_manifest(),
    )
    assert first.status == FederationRunStatus.INSUFFICIENT_EXAMPLES

    # completed_task_ids에 수동으로 추가해서 중복 테스트
    service._completed_task_ids.add("task_001")
    second = service.run_current_task(
        training_examples=(),
        model_manifest=_build_manifest(),
    )
    assert second.status == FederationRunStatus.ALREADY_COMPLETED
    assert second.task_id == "task_001"


def test_federation_runtime_returns_insufficient_examples_when_no_accepted() -> None:
    client = MagicMock(spec=RoundClient)
    client.fetch_current_task.return_value = _build_task_payload()
    local_service = MagicMock()
    local_service.run_task.return_value = LocalTrainingResult(
        selection_result=_empty_selection_result(),
        update_envelope=None,
    )
    service = FederationRuntimeService(
        round_client=client,
        local_training_service=local_service,
    )

    result = service.run_current_task(
        training_examples=(),
        model_manifest=_build_manifest(),
    )

    assert result.status == FederationRunStatus.INSUFFICIENT_EXAMPLES
    assert result.round_id == "round_0001"
    assert result.task_id == "task_001"
    assert result.update_id is None
    # update가 없으므로 task_id를 완료 목록에 추가하지 않는다.
    assert "task_001" not in service._completed_task_ids


def test_federation_runtime_uploads_update_and_marks_completed(
    tmp_path,
) -> None:
    task_payload = _build_task_payload()
    client = MagicMock(spec=RoundClient)
    client.fetch_current_task.return_value = task_payload
    client.upload_update.return_value = {"update_id": "update_abc", "update_count": 1}

    update_file = tmp_path / "update_abc.json"
    update_file.write_text("{}", encoding="utf-8")

    envelope = TrainingUpdateEnvelope(
        schema_version="training_update_envelope.v1",
        update_id="update_abc",
        round_id="round_0001",
        task_id="task_001",
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        payload_ref=str(update_file),
        payload_format="diagonal_scale_update",
        example_count=3,
        client_metrics={
            "accepted_ratio": 0.75,
            "mean_confidence": 0.85,
            "mean_margin": 0.03,
            "delta_l2_norm": 0.01,
            "selected_examples": 3.0,
        },
        created_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        clipped=False,
        dp_applied=False,
    )
    selection = PseudoLabelSelectionResult(
        candidates=(),
        accepted_candidates=(),
        feedback_signals=(),
    )
    local_service = MagicMock()
    local_service.run_task.return_value = LocalTrainingResult(
        selection_result=selection,
        update_envelope=envelope,
    )
    service = FederationRuntimeService(
        round_client=client,
        local_training_service=local_service,
    )

    result = service.run_current_task(
        training_examples=(),
        model_manifest=_build_manifest(),
    )

    assert result.status == FederationRunStatus.UPLOADED
    assert result.update_id == "update_abc"
    assert result.round_id == "round_0001"
    # 완료 task_id가 기록돼야 한다.
    assert "task_001" in service._completed_task_ids
    client.upload_update.assert_called_once()


def test_federation_runtime_clear_completed_resets_state() -> None:
    client = MagicMock(spec=RoundClient)
    service = FederationRuntimeService(round_client=client)
    service._completed_task_ids.add("task_001")
    service._completed_task_ids.add("task_002")

    service.clear_completed()

    assert len(service._completed_task_ids) == 0
