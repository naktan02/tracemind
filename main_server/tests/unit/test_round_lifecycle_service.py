"""RoundLifecycleService unit tests."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from main_server.src.infrastructure.repositories import (
    model_manifest_repository as model_manifest_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as shared_adapter_update_repository_module,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.federation.rounds.acceptance.policies import (
    IdempotentRoundUpdateAcceptancePolicy,
    StrictRoundUpdateAcceptancePolicy,
)
from main_server.src.services.federation.rounds.acceptance.trust_policies import (
    SingleSubmissionPerAgentTrustPolicy,
)
from main_server.src.services.federation.rounds.active_manifest_service import (
    ActiveModelManifestService,
)
from main_server.src.services.federation.rounds.aggregation.models import (
    AggregationResult,
    SharedAdapterAggregationBackend,
)
from main_server.src.services.federation.rounds.aggregation.registry import (
    build_shared_adapter_aggregation_backend,
    register_shared_adapter_aggregation_backend,
)
from main_server.src.services.federation.rounds.boundary.models import (
    RoundFinalizeRequest,
    RoundOpenDraftRequest,
    RoundStatus,
    RoundStrategyConfig,
)
from main_server.src.services.federation.rounds.payload_adapters.models import (
    SharedAdapterRoundPayloadAdapter,
)
from main_server.src.services.federation.rounds.payload_adapters.registry import (
    build_shared_adapter_round_payload_adapter,
    register_shared_adapter_round_payload_adapter,
)
from main_server.src.services.federation.rounds.round_lifecycle_service import (
    RoundConflictError,
    RoundLifecycleService,
    RoundValidationError,
)
from main_server.src.services.federation.rounds.round_manager_service import (
    RoundManagerService,
)
from main_server.src.services.federation.rounds.round_state_exchange.executor import (
    RoundStateExchangeResult,
)
from main_server.src.services.federation.rounds.runtime.config import (
    ServerRoundRuntimeConfig,
)
from main_server.src.services.federation.rounds.runtime.factory import (
    build_round_manager_service_from_config,
)
from main_server.src.services.federation.rounds.server_policy.executor import (
    ServerPolicyExecutionSummary,
)
from methods.federated_ssl.base import (
    FederatedSslLocalStepSpec,
    FederatedSslMethodDescriptor,
    FederatedSslRequiredViews,
    FederatedSslRuntimeCapabilities,
    FederatedSslServerStepSpec,
)
from methods.federated_ssl.runtime_fallbacks import RUNTIME_FALLBACK_TRAINING_PROFILE
from shared.src.contracts.adapter_contract_families.base import (
    SharedAdapterStatePayload,
    SharedAdapterUpdatePayload,
)
from shared.src.contracts.adapter_contract_families.factories import (
    make_peft_classifier_delta_payload,
    make_peft_classifier_state_payload,
)
from shared.src.contracts.adapter_contract_families.registry import (
    register_shared_adapter_payload_family,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import (
    TrainingUpdateEnvelope,
    TrainingUpdateSubmission,
    make_training_update_submission,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.entities.training.shared_adapter_update import (
    SharedAdapterUpdate,
)
from shared.src.domain.services.clock import FixedClock

ModelManifestRepository = model_manifest_repository_module.ModelManifestRepository
SharedAdapterUpdateRepository = (
    shared_adapter_update_repository_module.SharedAdapterUpdateRepository
)
peft_encoder_partitioned_projection = importlib.import_module(
    "methods.adaptation.peft_text_encoder.aggregation."
    "peft_encoder_partitioned_projection"
)
TEST_METHOD_DESCRIPTOR = FederatedSslMethodDescriptor(
    name="test_round_runtime_method",
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
    runtime_capabilities=FederatedSslRuntimeCapabilities(
        simulation_supported=True,
        live_agent_supported=True,
        live_server_supported=True,
    ),
)


@dataclass(slots=True)
class _RecordingServerPolicyExecutor:
    calls: list[tuple[str, str, int]]

    def __init__(self) -> None:
        self.calls = []

    def prepare_finalize(
        self,
        *,
        method_descriptor: FederatedSslMethodDescriptor,
        round_id: str,
        update_count: int,
    ) -> ServerPolicyExecutionSummary:
        self.calls.append((method_descriptor.name, round_id, update_count))
        return ServerPolicyExecutionSummary(
            method_name=method_descriptor.name,
            round_id=round_id,
            server_aggregator_name=method_descriptor.server_step.server_aggregator_name,
            round_policy_name=method_descriptor.server_step.round_policy_name,
            server_aggregate_hint=method_descriptor.server_step.server_aggregate_hint,
            update_count=update_count,
        )


@dataclass(slots=True)
class _StaticRoundStateExchangeExecutor:
    summary_metrics: dict[str, float]

    def summarize(
        self,
        *,
        method_descriptor: FederatedSslMethodDescriptor,
        record,
    ) -> RoundStateExchangeResult:
        del method_descriptor, record
        return RoundStateExchangeResult(
            exchange_name="client_metric_summary",
            summary_metrics=dict(self.summary_metrics),
        )


TEST_SHIFT_ADAPTER_KIND = "test_shift_round_payload_adapter"
TEST_SHIFT_PAYLOAD_ADAPTER_KIND = "test_shift_round_payload_adapter"
TEST_SHIFT_BACKEND_NAME = "test_shift_avg"
TEST_SHIFT_PAYLOAD_FORMAT = "test_shift_update"


class _TestShiftStatePayload(SharedAdapterStatePayload):
    shift_bias: float

    @property
    def embedding_dim(self) -> int:
        return 1

    def apply(self, embedding) -> list[float]:
        if len(embedding) != 1:
            raise ValueError("Expected a one-dimensional embedding.")
        return [float(embedding[0]) + float(self.shift_bias)]


class _TestShiftUpdatePayload(SharedAdapterUpdatePayload):
    shift_delta: float


@dataclass(slots=True)
class _TestShiftAggregationBackend:
    adapter_kind: str = TEST_SHIFT_ADAPTER_KIND

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: tuple[SharedAdapterUpdate, ...] | list[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult:
        if not isinstance(base_state, _TestShiftStatePayload):
            raise TypeError("Expected _TestShiftStatePayload as base state.")
        valid_updates = [
            payload
            for payload in update_payloads
            if (
                isinstance(payload, _TestShiftUpdatePayload)
                and payload.example_count > 0
            )
        ]
        if not valid_updates:
            raise ValueError("At least one test-shift update is required.")

        total_examples = sum(payload.example_count for payload in valid_updates)
        weighted_shift = (
            sum(
                payload.shift_delta * payload.example_count for payload in valid_updates
            )
            / total_examples
        )
        next_state = _TestShiftStatePayload(
            schema_version="test_shift_state.v1",
            model_id=base_state.model_id,
            model_revision=next_model_revision,
            training_scope=base_state.training_scope,
            updated_at=aggregated_at,
            adapter_kind=self.adapter_kind,
            shift_bias=base_state.shift_bias + weighted_shift,
        )
        return AggregationResult(
            next_state=next_state,
            aggregated_metrics={
                "example_count": float(total_examples),
                "mean_shift_delta": float(weighted_shift),
            },
            update_count=len(valid_updates),
        )


@dataclass(slots=True)
class _TestShiftRoundPayloadAdapter:
    adapter_kind: str = TEST_SHIFT_ADAPTER_KIND
    accepted_update_formats: tuple[str, ...] = (TEST_SHIFT_PAYLOAD_FORMAT,)
    aggregation_backend: SharedAdapterAggregationBackend | None = None

    def state_from_payload(
        self,
        payload: SharedAdapterStatePayload,
    ) -> SharedAdapterState:
        if not isinstance(payload, _TestShiftStatePayload):
            raise TypeError("Expected _TestShiftStatePayload.")
        return payload

    def update_from_payload(
        self,
        payload: SharedAdapterUpdatePayload,
    ) -> SharedAdapterUpdate:
        if not isinstance(payload, _TestShiftUpdatePayload):
            raise TypeError("Expected _TestShiftUpdatePayload.")
        return payload

    def state_to_payload(
        self,
        state: SharedAdapterState,
    ) -> SharedAdapterStatePayload:
        if not isinstance(state, _TestShiftStatePayload):
            raise TypeError("Expected _TestShiftStatePayload.")
        return state


def _build_test_shift_round_payload_adapter(
    aggregation_backend_name: str,
    aggregation_backend_overrides,
) -> SharedAdapterRoundPayloadAdapter:
    del aggregation_backend_overrides
    return _TestShiftRoundPayloadAdapter(
        aggregation_backend=build_shared_adapter_aggregation_backend(
            adapter_kind=TEST_SHIFT_ADAPTER_KIND,
            backend_name=aggregation_backend_name,
        )
    )


register_shared_adapter_payload_family(
    TEST_SHIFT_ADAPTER_KIND,
    state_payload_type=_TestShiftStatePayload,
    update_payload_type=_TestShiftUpdatePayload,
)
register_shared_adapter_aggregation_backend(
    TEST_SHIFT_ADAPTER_KIND,
    TEST_SHIFT_BACKEND_NAME,
    factory=lambda _overrides: _TestShiftAggregationBackend(),
    catalog_entry=RegistryCatalogEntry(
        item_name=f"{TEST_SHIFT_ADAPTER_KIND}.{TEST_SHIFT_BACKEND_NAME}",
        display_name=TEST_SHIFT_BACKEND_NAME,
        implementation_module=__name__,
        core_method_name=TEST_SHIFT_BACKEND_NAME,
        family_name=TEST_SHIFT_ADAPTER_KIND,
        supported_adapter_kinds=(TEST_SHIFT_ADAPTER_KIND,),
    ),
)
register_shared_adapter_round_payload_adapter(
    TEST_SHIFT_PAYLOAD_ADAPTER_KIND,
    factory=_build_test_shift_round_payload_adapter,
)


def _build_service(
    *,
    tmp_path: Path,
    fixed_time: datetime,
) -> tuple[RoundLifecycleService, ModelManifest, RoundRepository]:
    round_repository = RoundRepository(state_root=tmp_path / "rounds")
    state_repository = (
        shared_adapter_state_repository_module.SharedAdapterStateRepository(
            state_root=tmp_path / "shared_states"
        )
    )
    state_repository.save_shared_adapter_state(
        _TestShiftStatePayload(
            schema_version="test_shift_state.v1",
            adapter_kind=TEST_SHIFT_ADAPTER_KIND,
            model_id="test-shift-model",
            model_revision="rev_000",
            training_scope="adapter_only",
            shift_bias=0.0,
            updated_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
    )
    active_manifest = ModelManifest(
        schema_version="model_manifest.v1",
        model_id="test-shift-model",
        model_revision="rev_000",
        published_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref=state_repository.ref_for_revision("rev_000"),
        auxiliary_artifact_versions={"calibration_set": "calib_000"},
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )
    service = RoundLifecycleService(
        round_repository=round_repository,
        update_payload_repository=SharedAdapterUpdateRepository(
            state_root=tmp_path / "server_updates"
        ),
        active_manifest_service=ActiveModelManifestService(
            manifest_repository=ModelManifestRepository(
                state_root=tmp_path / "model_manifests"
            ),
            clock=FixedClock(fixed_time),
        ),
        round_manager_service=RoundManagerService(
            payload_adapter=build_shared_adapter_round_payload_adapter(
                TEST_SHIFT_PAYLOAD_ADAPTER_KIND,
                aggregation_backend_name=TEST_SHIFT_BACKEND_NAME,
            ),
            artifact_repository=state_repository,
            clock=FixedClock(fixed_time),
        ),
        clock=FixedClock(fixed_time),
    )
    service.active_manifest_service.save_and_activate(
        active_manifest,
        activated_at=fixed_time,
    )
    return service, active_manifest, round_repository


def _build_peft_service(
    *,
    tmp_path: Path,
    fixed_time: datetime,
) -> tuple[RoundLifecycleService, ModelManifest, RoundRepository]:
    round_repository = RoundRepository(state_root=tmp_path / "rounds")
    state_repository = (
        shared_adapter_state_repository_module.SharedAdapterStateRepository(
            state_root=tmp_path / "shared_states"
        )
    )
    state_repository.save_shared_adapter_state(
        make_peft_classifier_state_payload(
            model_id="tracemind-embed",
            model_revision="rev_000",
            training_scope="adapter_only",
            backbone=_peft_backbone_payload(),
            peft_adapter_config=_peft_adapter_config_payload(),
            label_schema=["anxiety", "normal"],
        )
    )
    active_manifest = ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_000",
        published_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref=state_repository.ref_for_revision("rev_000"),
        auxiliary_artifact_versions={"calibration_set": "calib_000"},
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )
    service = RoundLifecycleService(
        round_repository=round_repository,
        update_payload_repository=SharedAdapterUpdateRepository(
            state_root=tmp_path / "server_updates"
        ),
        active_manifest_service=ActiveModelManifestService(
            manifest_repository=ModelManifestRepository(
                state_root=tmp_path / "model_manifests"
            ),
            clock=FixedClock(fixed_time),
        ),
        round_manager_service=RoundManagerService(
            payload_adapter=build_shared_adapter_round_payload_adapter(
                "peft_classifier",
                aggregation_backend_name="fedavg",
            ),
            artifact_repository=state_repository,
            clock=FixedClock(fixed_time),
        ),
        clock=FixedClock(fixed_time),
    )
    service.active_manifest_service.save_and_activate(
        active_manifest,
        activated_at=fixed_time,
    )
    return service, active_manifest, round_repository


def _peft_backbone_payload() -> dict[str, object]:
    return {
        "backbone_model_id": "mxbai",
        "backbone_revision": "main",
        "tokenizer_model_id": "mxbai",
        "tokenizer_revision": "main",
        "pooling": "mean",
        "max_length": 256,
        "task_prefix": "",
    }


def _peft_adapter_config_payload() -> dict[str, object]:
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


def _build_update(
    *,
    tmp_path: Path,
    round_id: str,
    task_id: str,
    update_id: str = "update_001",
    base_model_revision: str = "rev_000",
    agent_id: str | None = None,
) -> TrainingUpdateSubmission:
    del tmp_path
    update_payload = _TestShiftUpdatePayload(
        schema_version="test_shift_delta.v1",
        adapter_kind=TEST_SHIFT_ADAPTER_KIND,
        model_id="test-shift-model",
        base_model_revision=base_model_revision,
        training_scope="adapter_only",
        shift_delta=0.05,
        example_count=3,
    )
    envelope = TrainingUpdateEnvelope(
        schema_version="training_update_envelope.v1",
        update_id=update_id,
        round_id=round_id,
        task_id=task_id,
        model_id="test-shift-model",
        base_model_revision=base_model_revision,
        training_scope="adapter_only",
        payload_ref=f"client-submission::{update_id}",
        payload_format=TEST_SHIFT_PAYLOAD_FORMAT,
        example_count=3,
        client_metrics={"mean_loss": 0.2},
        agent_id=agent_id,
    )
    return make_training_update_submission(
        envelope=envelope,
        update_payload=update_payload,
    )


def test_round_lifecycle_opens_round_and_sets_active_pointer(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, round_repository = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )

    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))

    assert record.status == RoundStatus.OPEN
    assert record.training_task.round_id == "round_0001"
    assert (
        record.training_task.local_epochs
        == RUNTIME_FALLBACK_TRAINING_PROFILE.local_epochs
    )
    assert (
        record.training_task.batch_size == RUNTIME_FALLBACK_TRAINING_PROFILE.batch_size
    )
    assert (
        record.training_task.learning_rate
        == RUNTIME_FALLBACK_TRAINING_PROFILE.learning_rate
    )
    assert record.training_task.max_steps == RUNTIME_FALLBACK_TRAINING_PROFILE.max_steps
    assert record.updated_at == fixed_time
    assert service.get_current_round().round_id == "round_0001"
    active_pointer = round_repository.load_active_pointer()
    assert active_pointer is not None
    assert active_pointer.round_id == "round_0001"


def test_round_lifecycle_rejects_duplicate_update_id(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )

    acceptance = service.accept_update_submission(record.round_id, update)

    assert acceptance.update_count == 1
    with pytest.raises(RoundConflictError):
        service.accept_update_submission(record.round_id, update)


def test_round_lifecycle_uses_partitioned_backend_for_fedmatch_finalize(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, _round_repository = _build_peft_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(
        RoundOpenDraftRequest(
            round_id="round_fedmatch",
            strategy=RoundStrategyConfig(
                mode="method_owned",
                fssl_method="fedmatch",
            ),
        )
    )

    round_manager = service._round_manager_for_finalize(record)

    assert (
        round_manager.payload_adapter.aggregation_backend.strategy.method_name
        == peft_encoder_partitioned_projection.PARTITIONED_DELTA_AVERAGE_BACKEND_NAME
    )


def test_round_lifecycle_rejects_agent_local_lora_artifact_refs_at_accept(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, round_repository = _build_peft_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update_payload = make_peft_classifier_delta_payload(
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
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
            "parameters": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "bias": "none",
                "target_modules": "all-linear",
                "use_rslora": False,
            },
        },
        label_schema=["anxiety", "normal"],
        example_count=2,
        peft_adapter_delta_artifact_ref=("agent-local://agent_001/peft_adapter_delta"),
        classifier_head_delta_artifact_ref=(
            "agent-local://agent_001/classifier_head_delta"
        ),
        mean_confidence=0.8,
        mean_margin=0.2,
    )
    update = make_training_update_submission(
        envelope=TrainingUpdateEnvelope(
            schema_version="training_update_envelope.v1",
            update_id="peft_update_001",
            round_id=record.round_id,
            task_id=record.training_task.task_id,
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            payload_ref="client-submission::peft_update_001",
            payload_format="peft_classifier_update",
            example_count=2,
            client_metrics={"mean_loss": 0.2},
        ),
        update_payload=update_payload,
    )

    with pytest.raises(RoundValidationError, match="agent-local artifact"):
        service.accept_update_submission(record.round_id, update)
    assert round_repository.load_round(record.round_id).updates == ()


def test_round_lifecycle_rejects_peft_payload_manifest_drift_at_accept(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, round_repository = _build_peft_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update_payload = make_peft_classifier_delta_payload(
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        backbone=_peft_backbone_payload(),
        peft_adapter_config={
            **_peft_adapter_config_payload(),
            "parameters": {
                **_peft_adapter_config_payload()["parameters"],
                "rank": 4,
            },
        },
        label_schema=["anxiety", "normal"],
        example_count=2,
        peft_parameter_deltas={"encoder.q_proj.lora_A": [0.1]},
        classifier_head_weight_deltas={
            "anxiety": [0.2],
            "normal": [-0.2],
        },
        classifier_head_bias_deltas={"anxiety": 0.01, "normal": -0.01},
        delta_format="inline_delta",
        mean_confidence=0.8,
        mean_margin=0.2,
    )
    update = make_training_update_submission(
        envelope=TrainingUpdateEnvelope(
            schema_version="training_update_envelope.v1",
            update_id="peft_update_001",
            round_id=record.round_id,
            task_id=record.training_task.task_id,
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            payload_ref="client-submission::peft_update_001",
            payload_format="peft_classifier_update",
            example_count=2,
            client_metrics={"mean_loss": 0.2},
        ),
        update_payload=update_payload,
    )

    with pytest.raises(RoundValidationError, match="peft_adapter_config"):
        service.accept_update_submission(record.round_id, update)
    assert round_repository.load_round(record.round_id).updates == ()


def test_round_lifecycle_can_accept_idempotent_duplicate_update(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, round_repository = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    service.update_acceptance_policy = IdempotentRoundUpdateAcceptancePolicy()
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )

    first = service.accept_update_submission(record.round_id, update)
    second = service.accept_update_submission(record.round_id, update)

    assert first.idempotent is False
    assert second.idempotent is True
    assert second.update_count == 1
    assert len(round_repository.load_round(record.round_id).updates) == 1


def test_round_lifecycle_can_split_agent_trust_policy_from_network_policy(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    service.update_acceptance_policy = StrictRoundUpdateAcceptancePolicy(
        trust_policy=SingleSubmissionPerAgentTrustPolicy()
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    first_update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
        update_id="update_001",
        agent_id="agent_alpha",
    )
    second_update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
        update_id="update_002",
        agent_id="agent_alpha",
    )

    service.accept_update_submission(record.round_id, first_update)

    with pytest.raises(RoundConflictError):
        service.accept_update_submission(record.round_id, second_update)


def test_round_lifecycle_rejects_base_revision_mismatch(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    bad_update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
        update_id="update_bad",
        base_model_revision="rev_999",
    )

    with pytest.raises(RoundValidationError):
        service.accept_update_submission(record.round_id, bad_update)


def test_round_lifecycle_rejects_payload_format_outside_active_payload_adapter(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, round_repository = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update_payload = make_peft_classifier_delta_payload(
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
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
            "parameters": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "bias": "none",
                "target_modules": "all-linear",
                "use_rslora": False,
            },
        },
        label_schema=["anxiety", "normal"],
        example_count=2,
        peft_parameter_deltas={"encoder.q_proj.lora_A": [0.1]},
        classifier_head_weight_deltas={
            "anxiety": [0.2],
            "normal": [-0.2],
        },
        classifier_head_bias_deltas={"anxiety": 0.01, "normal": -0.01},
        delta_format="inline_delta",
        mean_confidence=0.8,
        mean_margin=0.2,
    )
    update = make_training_update_submission(
        envelope=TrainingUpdateEnvelope(
            schema_version="training_update_envelope.v1",
            update_id="peft_update_001",
            round_id=record.round_id,
            task_id=record.training_task.task_id,
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            payload_ref="client-submission::peft_update_001",
            payload_format="peft_classifier_update",
            example_count=2,
            client_metrics={"mean_loss": 0.2},
        ),
        update_payload=update_payload,
    )

    with pytest.raises(RoundValidationError, match="payload_format"):
        service.accept_update_submission(record.round_id, update)
    assert round_repository.load_round(record.round_id).updates == ()


def test_round_lifecycle_finalizes_round_and_activates_next_manifest(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, round_repository = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )
    service.accept_update_submission(record.round_id, update)
    accepted_update = round_repository.load_round(record.round_id).updates[0]
    assert (
        accepted_update.payload_ref
        == service.update_payload_repository.ref_for_update("update_001")
    )

    finalized = service.finalize_round(
        record.round_id,
        RoundFinalizeRequest(
            next_auxiliary_artifact_versions={"calibration_set": "calib_001"},
            next_model_revision="rev_001",
        ),
    )

    assert finalized.status == RoundStatus.FINALIZED
    assert finalized.finalized_at == fixed_time
    assert finalized.publication is not None
    assert finalized.publication.next_manifest.model_revision == "rev_001"
    assert (
        finalized.publication.next_manifest.artifact_ref
        == service.round_manager_service.artifact_repository.ref_for_revision("rev_001")
    )
    assert finalized.publication.next_manifest.auxiliary_artifact_versions == {
        "calibration_set": "calib_001"
    }
    assert round_repository.load_active_pointer() is None
    assert (
        service.active_manifest_service.get_active_manifest().model_revision
        == "rev_001"
    )
    with pytest.raises(FileNotFoundError):
        service.get_current_round()


def test_round_lifecycle_runs_method_server_policy_before_finalize(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    policy_executor = _RecordingServerPolicyExecutor()
    service.method_descriptor = TEST_METHOD_DESCRIPTOR
    service.server_policy_executor = policy_executor
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )
    service.accept_update_submission(record.round_id, update)

    service.finalize_round(
        record.round_id,
        RoundFinalizeRequest(
            next_auxiliary_artifact_versions={"calibration_set": "calib_001"},
            next_model_revision="rev_001",
        ),
    )

    assert policy_executor.calls == [("test_round_runtime_method", record.round_id, 1)]


def test_round_lifecycle_stores_round_state_exchange_summary(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    service.method_descriptor = TEST_METHOD_DESCRIPTOR
    service.round_state_exchange_executor = _StaticRoundStateExchangeExecutor(
        summary_metrics={"round_state.mean_confidence.mean": 0.8}
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )
    service.accept_update_submission(record.round_id, update)

    finalized = service.finalize_round(
        record.round_id,
        RoundFinalizeRequest(
            next_auxiliary_artifact_versions={"calibration_set": "calib_001"},
            next_model_revision="rev_001",
        ),
    )

    assert finalized.publication is not None
    assert finalized.publication.round_state_summary_metrics == {
        "round_state.mean_confidence.mean": 0.8
    }


def test_round_lifecycle_rejects_update_after_finalize(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, _active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )
    service.accept_update_submission(record.round_id, update)
    service.finalize_round(
        record.round_id,
        RoundFinalizeRequest(
            next_auxiliary_artifact_versions={"calibration_set": "calib_001"},
            next_model_revision="rev_001",
        ),
    )

    with pytest.raises(RoundConflictError):
        service.accept_update_submission(record.round_id, update)


def test_round_lifecycle_finalizes_registered_custom_payload_adapter(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    round_repository = RoundRepository(state_root=tmp_path / "rounds")
    state_repository = (
        shared_adapter_state_repository_module.SharedAdapterStateRepository(
            state_root=tmp_path / "shared_states"
        )
    )
    state_repository.save_shared_adapter_state(
        _TestShiftStatePayload(
            schema_version="test_shift_state.v1",
            model_id="tracemind-embed",
            model_revision="rev_000",
            training_scope="adapter_only",
            updated_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            adapter_kind=TEST_SHIFT_ADAPTER_KIND,
            shift_bias=1.0,
        )
    )
    active_manifest = ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_000",
        published_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref=state_repository.ref_for_revision("rev_000"),
        auxiliary_artifact_versions={"calibration_set": "calib_000"},
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )
    round_manager_service = build_round_manager_service_from_config(
        ServerRoundRuntimeConfig(
            payload_adapter_kind=TEST_SHIFT_PAYLOAD_ADAPTER_KIND,
            aggregation_backend_name=TEST_SHIFT_BACKEND_NAME,
        ),
        artifact_repository=state_repository,
        clock=FixedClock(fixed_time),
    )
    service = RoundLifecycleService(
        round_repository=round_repository,
        update_payload_repository=SharedAdapterUpdateRepository(
            state_root=tmp_path / "server_updates"
        ),
        active_manifest_service=ActiveModelManifestService(
            manifest_repository=ModelManifestRepository(
                state_root=tmp_path / "model_manifests"
            ),
            clock=FixedClock(fixed_time),
        ),
        round_manager_service=round_manager_service,
        clock=FixedClock(fixed_time),
    )
    service.active_manifest_service.save_and_activate(
        active_manifest,
        activated_at=fixed_time,
    )

    record = service.open_round(RoundOpenDraftRequest(round_id="round_0001"))
    update_payload = _TestShiftUpdatePayload(
        schema_version="test_shift_update.v1",
        model_id="tracemind-embed",
        base_model_revision="rev_000",
        training_scope="adapter_only",
        example_count=4,
        created_at=fixed_time,
        adapter_kind=TEST_SHIFT_ADAPTER_KIND,
        shift_delta=0.3,
    )
    update = make_training_update_submission(
        envelope=TrainingUpdateEnvelope(
            schema_version="training_update_envelope.v1",
            update_id="custom_update_001",
            round_id=record.round_id,
            task_id=record.training_task.task_id,
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            payload_ref="client-submission::custom_update_001",
            payload_format=TEST_SHIFT_PAYLOAD_FORMAT,
            example_count=4,
            client_metrics={"test_shift_norm": 0.3},
        ),
        update_payload=update_payload,
    )
    service.accept_update_submission(record.round_id, update)

    finalized = service.finalize_round(
        record.round_id,
        RoundFinalizeRequest(
            next_auxiliary_artifact_versions={"calibration_set": "calib_001"},
            next_model_revision="rev_001",
        ),
    )

    assert finalized.status == RoundStatus.FINALIZED
    assert finalized.publication is not None
    assert finalized.publication.aggregated_metrics["mean_shift_delta"] == 0.3
    assert finalized.publication.next_manifest.model_revision == "rev_001"
    assert round_repository.load_active_pointer() is None
    assert (
        service.active_manifest_service.get_active_manifest().model_revision
        == "rev_001"
    )

    next_state_payload = state_repository.load_shared_adapter_state_from_ref(
        finalized.publication.next_manifest.artifact_ref
    )
    assert isinstance(next_state_payload, _TestShiftStatePayload)
    assert next_state_payload.adapter_kind == TEST_SHIFT_ADAPTER_KIND
    assert next_state_payload.shift_bias == 1.3
