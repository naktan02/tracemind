"""FL round API tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException

import main_server.src.api.fl_rounds as fl_rounds_api
from main_server.src.api.main import app
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.rounds.mappers import (
    model_manifest_to_payload,
    training_update_to_payload,
)
from main_server.src.services.rounds.payloads import (
    RoundFinalizeRequestPayload,
    RoundOpenRequestPayload,
)
from main_server.src.services.rounds.round_lifecycle_service import (
    RoundLifecycleService,
)
from main_server.src.services.rounds.round_manager_service import RoundManagerService
from shared.src.contracts.adapter_contracts import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    dump_shared_adapter_update_payload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope
from shared.src.domain.services.clock import FixedClock


def _build_service(
    *,
    tmp_path: Path,
    fixed_time: datetime,
) -> tuple[RoundLifecycleService, ModelManifest]:
    round_repository = RoundRepository(state_root=tmp_path / "rounds")
    state_repository = (
        shared_adapter_state_repository_module.SharedAdapterStateRepository(
            state_root=tmp_path / "shared_states"
        )
    )
    state_path = state_repository.save_shared_adapter_state(
        DiagonalScaleAdapterStatePayload(
            schema_version="vector_adapter_state.v1",
            adapter_kind="diagonal_scale",
            model_id="tracemind-embed",
            model_revision="rev_000",
            training_scope="adapter_only",
            dimension_scales=[1.0, 1.0],
            updated_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
    )
    active_manifest = ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_000",
        published_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref=str(state_path),
        prototype_version="proto_000",
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )
    service = RoundLifecycleService(
        round_repository=round_repository,
        round_manager_service=RoundManagerService(
            artifact_repository=state_repository,
            clock=FixedClock(fixed_time),
        ),
        clock=FixedClock(fixed_time),
    )
    return service, active_manifest


def _build_update(
    *,
    tmp_path: Path,
    round_id: str,
    task_id: str,
    update_id: str = "update_001",
) -> TrainingUpdateEnvelope:
    payload_path = tmp_path / "updates" / f"{update_id}.json"
    dump_shared_adapter_update_payload(
        payload_path,
        DiagonalScaleAdapterUpdatePayload(
            schema_version="vector_adapter_delta.v1",
            adapter_kind="diagonal_scale",
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            dimension_deltas=[0.05, -0.02],
            example_count=3,
            mean_confidence=0.8,
            mean_margin=0.15,
        ),
    )
    return TrainingUpdateEnvelope(
        schema_version="training_update_envelope.v1",
        update_id=update_id,
        round_id=round_id,
        task_id=task_id,
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        payload_ref=str(payload_path),
        payload_format="diagonal_scale_update",
        example_count=3,
        client_metrics={"mean_loss": 0.2},
    )


def test_fl_rounds_api_runs_open_update_finalize_flow(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    open_response = fl_rounds_api.open_round(
        RoundOpenRequestPayload(
            active_manifest=model_manifest_to_payload(active_manifest),
            round_id="round_0001",
        ),
        service=service,
    )

    assert open_response.status == "open"
    task_id = open_response.training_task.task_id

    update = _build_update(
        tmp_path=tmp_path,
        round_id="round_0001",
        task_id=task_id,
    )
    update_response = fl_rounds_api.accept_update(
        "round_0001",
        training_update_to_payload(update),
        service=service,
    )

    assert update_response.update_count == 1

    current_response = fl_rounds_api.get_current_round(service=service)
    assert current_response.round_id == "round_0001"

    finalize_response = fl_rounds_api.finalize_round(
        "round_0001",
        RoundFinalizeRequestPayload(
            next_prototype_version="proto_001",
            next_model_revision="rev_001",
        ),
        service=service,
    )

    assert finalize_response.status == "finalized"
    assert finalize_response.publication is not None
    assert finalize_response.publication.next_manifest.model_revision == "rev_001"

    with pytest.raises(HTTPException) as error_info:
        fl_rounds_api.get_current_round(service=service)
    assert error_info.value.status_code == 404


def test_fl_rounds_api_rejects_duplicate_update_id(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    open_response = fl_rounds_api.open_round(
        RoundOpenRequestPayload(
            active_manifest=model_manifest_to_payload(active_manifest),
            round_id="round_0001",
        ),
        service=service,
    )
    task_id = open_response.training_task.task_id
    update = _build_update(
        tmp_path=tmp_path,
        round_id="round_0001",
        task_id=task_id,
    )
    payload = training_update_to_payload(update)

    first_response = fl_rounds_api.accept_update(
        "round_0001",
        payload,
        service=service,
    )
    assert first_response.update_count == 1

    with pytest.raises(HTTPException) as error_info:
        fl_rounds_api.accept_update("round_0001", payload, service=service)
    assert error_info.value.status_code == 409


def test_fl_rounds_router_is_registered_on_main_app() -> None:
    route_paths = {route.path for route in app.routes}

    assert "/api/v1/fl/rounds/current" in route_paths
    assert "/api/v1/fl/rounds" in route_paths
    assert "/api/v1/fl/rounds/{round_id}" in route_paths
    assert "/api/v1/fl/rounds/{round_id}/updates" in route_paths
    assert "/api/v1/fl/rounds/{round_id}/finalize" in route_paths
