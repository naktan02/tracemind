"""round state exchange runtime capability 테스트."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from main_server.src.services.federation.rounds.acceptance.errors import (
    RoundValidationError,
)
from main_server.src.services.federation.rounds.boundary.models import (
    RoundRecord,
    RoundStatus,
)
from main_server.src.services.federation.rounds.round_state_exchange.executor import (
    DefaultRoundStateExchangeExecutor,
)
from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslRequiredViews,
    FederatedSslRoundStateExchangeSpec,
    FederatedSslRuntimeCapabilities,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.fedavg_pseudo_label.descriptor import (
    FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingSelectionPolicy,
    TrainingTask,
    TrainingUpdateEnvelope,
)


def _build_descriptor(
    exchange: FederatedSslRoundStateExchangeSpec,
) -> FederatedSslMethodDescriptor:
    return FederatedSslMethodDescriptor(
        name="test_round_state_method",
        implementation_status="test_only",
        required_views=FederatedSslRequiredViews(
            view_names=("single_view",),
            view_generator_name="training_example_backend",
        ),
        local_step=FederatedSslLocalStepSpec(
            step_name="pseudo_label_self_training",
            client_trainer_name="local_training_service",
            pseudo_labeler_name="ssl_pseudo_label_selection_hook",
        ),
        server_step=FederatedSslServerStepSpec(
            server_aggregator_name="round_runtime_aggregation_backend",
            round_policy_name="round_active_pair_only",
            server_aggregate_hint="use_round_runtime_aggregation_backend",
        ),
        round_state_exchange=exchange,
        runtime_capabilities=FederatedSslRuntimeCapabilities(
            simulation_supported=True,
            live_agent_supported=True,
            live_server_supported=True,
        ),
    )


def _build_record() -> RoundRecord:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    manifest = ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_000",
        published_at=fixed_time,
        artifact_kind="shared_adapter_state",
        artifact_ref="shared_adapter_state::rev_000",
        prototype_version="proto_000",
        training_scope="adapter_only",
        training_enabled=True,
    )
    task = TrainingTask(
        schema_version="training_task.v1",
        task_id="task_001",
        round_id="round_001",
        model_id=manifest.model_id,
        model_revision=manifest.model_revision,
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=2,
        learning_rate=0.01,
        max_steps=1,
        objective_config={"training_backend_name": "diagonal_scale_heuristic"},
        selection_policy=TrainingSelectionPolicy.from_mapping({"min_confidence": 0.5}),
    )
    return RoundRecord(
        round_id="round_001",
        status=RoundStatus.OPEN,
        active_manifest=manifest,
        training_task=task,
        created_at=fixed_time,
        updated_at=fixed_time,
        updates=(
            _build_update(
                update_id="update_001",
                example_count=2,
                client_metrics={"mean_confidence": 0.6},
            ),
            _build_update(
                update_id="update_002",
                example_count=4,
                client_metrics={"mean_confidence": 0.9},
            ),
        ),
    )


def _build_update(
    *,
    update_id: str,
    example_count: int,
    client_metrics: dict[str, float],
) -> TrainingUpdateEnvelope:
    return TrainingUpdateEnvelope(
        schema_version="training_update_envelope.v1",
        update_id=update_id,
        round_id="round_001",
        task_id="task_001",
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        payload_ref=f"server-update://{update_id}",
        payload_format="diagonal_scale_update",
        example_count=example_count,
        client_metrics=client_metrics,
    )


def test_default_round_state_exchange_accepts_noop_method() -> None:
    result = DefaultRoundStateExchangeExecutor().summarize(
        method_descriptor=FEDAVG_PSEUDO_LABEL_DESCRIPTOR,
        record=_build_record(),
    )

    assert result.exchange_name == "none"
    assert result.summary_metrics == {}


def test_default_round_state_exchange_summarizes_required_client_metrics() -> None:
    descriptor = _build_descriptor(
        FederatedSslRoundStateExchangeSpec(
            exchange_name="client_metric_summary",
            required_client_metric_keys=("mean_confidence",),
        )
    )

    result = DefaultRoundStateExchangeExecutor().summarize(
        method_descriptor=descriptor,
        record=_build_record(),
    )

    assert result.exchange_name == "client_metric_summary"
    assert result.summary_metrics == {
        "round_state.update_count": 2.0,
        "round_state.example_count": 6.0,
        "round_state.mean_confidence.mean": pytest.approx(0.8),
    }


def test_default_round_state_exchange_rejects_missing_required_metric() -> None:
    descriptor = _build_descriptor(
        FederatedSslRoundStateExchangeSpec(
            exchange_name="client_metric_summary",
            required_client_metric_keys=("mean_margin",),
        )
    )

    with pytest.raises(RoundValidationError, match="requires client metric"):
        DefaultRoundStateExchangeExecutor().summarize(
            method_descriptor=descriptor,
            record=_build_record(),
        )


def test_default_round_state_exchange_rejects_custom_exchange() -> None:
    descriptor = _build_descriptor(
        FederatedSslRoundStateExchangeSpec(
            exchange_name="custom_exchange",
            requires_custom_exchange=True,
        )
    )

    with pytest.raises(RoundValidationError, match="custom round state exchange"):
        DefaultRoundStateExchangeExecutor().summarize(
            method_descriptor=descriptor,
            record=_build_record(),
        )
