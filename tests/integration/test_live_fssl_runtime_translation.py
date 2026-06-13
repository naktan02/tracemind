"""Live FL SSL task producer/consumer compatibility tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from agent.src.services.training_runtime.current_task.result import (
    TrainingTaskRunResult,
    TrainingTaskRunStatus,
)
from agent.src.services.training_runtime.current_task.runner import (  # noqa: E501
    AgentTrainingTaskRunnerService,
    AgentTrainingTaskRunRequest,
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


def test_live_manual_query_ssl_task_routes_through_agent_runner() -> None:
    manifest = _manifest()
    task = _round_manager().create_training_task(
        RoundOpenRequest(
            active_manifest=manifest,
            round_id="round_query_ssl",
            strategy=RoundStrategyConfig(
                mode="composed",
                ssl_method="fixmatch_usb_v1",
            ),
        )
    )

    assert task.fssl_method is None
    assert task.fssl_execution is None
    assert task.fssl_capability_plan is None

    result, query_ssl_task_service = _run_agent_with_task(
        task=task,
        manifest=manifest,
    )

    assert result.status == TrainingTaskRunStatus.UPLOADED
    query_ssl_task_service.run_current_task.assert_called_once()


def test_live_method_owned_no_peer_task_routes_through_agent_runner() -> None:
    manifest = _manifest()
    task = _round_manager().create_training_task(
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

    result, query_ssl_task_service = _run_agent_with_task(
        task=task,
        manifest=manifest,
    )

    assert result.status == TrainingTaskRunStatus.UPLOADED
    query_ssl_task_service.run_current_task.assert_called_once()


def test_live_method_owned_peer_context_task_routes_through_agent_runner() -> None:
    manifest = _manifest()
    task = _round_manager().create_training_task(
        RoundOpenRequest(
            active_manifest=manifest,
            round_id="round_fedmatch_peer",
            strategy=RoundStrategyConfig(
                mode="method_owned",
                fssl_method="fedmatch",
            ),
            fssl_context={
                "schema_version": "fssl_context.v1",
                "method_name": "fedmatch",
                "context_kind": "peer_context",
                "peer_context": {
                    "schema_version": "peer_context_task.v1",
                    "policy_name": "fixed_probe_output_knn",
                    "source_round_id": "round_prev",
                    "client_contexts": [
                        {
                            "client_id": "agent_01",
                            "helper_client_ids": ["agent_02"],
                        }
                    ],
                },
            },
        )
    )

    result, query_ssl_task_service = _run_agent_with_task(
        task=task,
        manifest=manifest,
        agent_id="agent_01",
    )

    assert result.status == TrainingTaskRunStatus.UPLOADED
    query_ssl_request = query_ssl_task_service.run_current_task.call_args.args[0]
    assert query_ssl_request.training_task.fssl_context["context_kind"] == (
        "peer_context"
    )


def _manifest() -> ModelManifest:
    return ModelManifest(
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


def _round_manager() -> RoundManagerService:
    return RoundManagerService(
        payload_adapter=build_shared_adapter_round_payload_adapter(
            "peft_classifier",
            aggregation_backend_name="fedavg",
        )
    )


def _run_agent_with_task(
    *,
    task,
    manifest: ModelManifest,
    agent_id: str | None = None,
) -> tuple[TrainingTaskRunResult, MagicMock]:
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
        AgentTrainingTaskRunRequest(
            server_base_url="http://server.test",
            agent_id=agent_id,
        )
    )

    return result, query_ssl_task_service
