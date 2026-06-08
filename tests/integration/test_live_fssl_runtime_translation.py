"""Live FL SSL task producer/consumer compatibility tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from agent.src.services.training_runtime.current_task.agent_training_task_runner_service import (  # noqa: E501
    AgentTrainingTaskRunnerService,
    AgentTrainingTaskRunRequest,
)
from agent.src.services.training_runtime.current_task.result import (
    TrainingTaskRunResult,
    TrainingTaskRunStatus,
)
from main_server.src.services.federation.rounds.boundary.models import (
    RoundOpenRequest,
    RoundStrategyConfig,
)
from main_server.src.services.federation.rounds.payload_adapters.registry import (
    build_shared_adapter_round_payload_adapter,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    RoundManagerService,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_state_payload,
)
from shared.src.contracts.model_contracts import ModelManifest


def test_live_method_owned_fssl_task_snapshot_routes_through_agent_runner() -> None:
    manifest = ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_000",
        published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref="/server/state/rev_000.json",
        auxiliary_artifact_versions={},
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )
    task = RoundManagerService(
        payload_adapter=build_shared_adapter_round_payload_adapter(
            "peft_classifier",
            aggregation_backend_name="fedavg",
        )
    ).create_training_task(
        RoundOpenRequest(
            active_manifest=manifest,
            round_id="round_fedmatch",
            strategy=RoundStrategyConfig(
                mode="method_owned",
                fssl_method="fedmatch",
            ),
        )
    )
    assert task.fssl_execution is not None
    assert task.fssl_execution["runtime_surface"] == {
        "payload_adapter_kind": "peft_classifier",
        "update_family_name": "peft_text_encoder",
        "aggregation_backend_name": "fedavg",
    }
    assert task.fssl_capability_plan is not None

    round_client = MagicMock()
    round_client.fetch_current_task.return_value = task
    query_ssl_task_service = MagicMock()
    query_ssl_task_service.run_current_task.return_value = TrainingTaskRunResult(
        status=TrainingTaskRunStatus.UPLOADED,
        round_id=task.round_id,
        task_id=task.task_id,
        update_id="update_001",
        example_count=3,
        accepted_count=2,
        message="uploaded",
    )
    shared_adapter_runtime_service = MagicMock()
    shared_adapter_runtime_service.get_active_manifest.return_value = manifest
    shared_adapter_runtime_service.get_active_state.return_value = (
        make_peft_classifier_state_payload(
            model_id=manifest.model_id,
            model_revision=manifest.model_revision,
            backbone={
                "backbone_model_id": "mxbai",
                "backbone_revision": "main",
                "tokenizer_model_id": "mxbai",
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
    )
    service = AgentTrainingTaskRunnerService(
        analysis_event_repository=MagicMock(),
        shared_adapter_runtime_service=shared_adapter_runtime_service,
        shared_adapter_sync_service=MagicMock(),
        round_client_factory=MagicMock(return_value=round_client),
        query_ssl_task_service=query_ssl_task_service,
    )

    result = service.run_current_task(
        AgentTrainingTaskRunRequest(server_base_url="http://server.test")
    )

    assert result.status == TrainingTaskRunStatus.UPLOADED
    query_ssl_task_service.run_current_task.assert_called_once()
