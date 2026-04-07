"""RoundLifecycleService unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from main_server.src.infrastructure.repositories import (
    prototype_build_state_repository as prototype_build_state_repository_module,
)
from main_server.src.infrastructure.repositories import (
    prototype_pack_repository as prototype_pack_repository_module,
)
from main_server.src.infrastructure.repositories import (
    prototype_rebuild_input_repository as prototype_rebuild_input_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.infrastructure.repositories.round_repository import RoundRepository
from main_server.src.services.prototypes import (
    PrototypeBuildStateService,
    PrototypePackService,
    PrototypeRebuildInputRecord,
    PrototypeRebuildService,
    ReferencePrototypeSourceRow,
    ReferenceRebuildPrototypePublicationStrategy,
    StoredReferencePrototypeRebuildService,
)
from main_server.src.services.rounds.models import (
    RoundFinalizeRequest,
    RoundOpenRequest,
    RoundStatus,
)
from main_server.src.services.rounds.round_lifecycle_service import (
    RoundConflictError,
    RoundLifecycleService,
    RoundValidationError,
)
from main_server.src.services.rounds.round_manager_service import RoundManagerService
from main_server.src.services.rounds.update_acceptance_policy import (
    IdempotentRoundUpdateAcceptancePolicy,
    SingleSubmissionPerAgentTrustPolicy,
    StrictRoundUpdateAcceptancePolicy,
)
from shared.src.contracts.adapter_contracts import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    dump_shared_adapter_update_payload,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingUpdateEnvelope
from shared.src.domain.services.clock import FixedClock
from shared.src.domain.value_objects import EmbeddingAdapterSpec
from shared.src.services.prototypes.build_strategies import SinglePrototypeBuildStrategy


class _StaticEmbeddingAdapter:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors[text]) for text in texts]


class _StaticEmbeddingAdapterFactory:
    _vectors: dict[str, list[float]] = {}

    @classmethod
    def create(cls, spec: EmbeddingAdapterSpec) -> _StaticEmbeddingAdapter:
        del spec
        return _StaticEmbeddingAdapter(cls._vectors)


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
    return service, active_manifest, round_repository


def _build_update(
    *,
    tmp_path: Path,
    round_id: str,
    task_id: str,
    update_id: str = "update_001",
    base_model_revision: str = "rev_000",
    agent_id: str | None = None,
) -> TrainingUpdateEnvelope:
    payload_path = tmp_path / "updates" / f"{update_id}.json"
    dump_shared_adapter_update_payload(
        payload_path,
        DiagonalScaleAdapterUpdatePayload(
            schema_version="vector_adapter_delta.v1",
            adapter_kind="diagonal_scale",
            model_id="tracemind-embed",
            base_model_revision=base_model_revision,
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
        base_model_revision=base_model_revision,
        training_scope="adapter_only",
        payload_ref=str(payload_path),
        payload_format="diagonal_scale_update",
        example_count=3,
        client_metrics={"mean_loss": 0.2},
        agent_id=agent_id,
    )


def test_round_lifecycle_opens_round_and_sets_active_pointer(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest, round_repository = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )

    record = service.open_round(
        RoundOpenRequest(
            active_manifest=active_manifest,
            round_id="round_0001",
        )
    )

    assert record.status == RoundStatus.OPEN
    assert record.training_task.round_id == "round_0001"
    assert record.updated_at == fixed_time
    assert service.get_current_round().round_id == "round_0001"
    active_pointer = round_repository.load_active_pointer()
    assert active_pointer is not None
    assert active_pointer.round_id == "round_0001"


def test_round_lifecycle_rejects_duplicate_update_id(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(
        RoundOpenRequest(
            active_manifest=active_manifest,
            round_id="round_0001",
        )
    )
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )

    acceptance = service.accept_update(record.round_id, update)

    assert acceptance.update_count == 1
    with pytest.raises(RoundConflictError):
        service.accept_update(record.round_id, update)


def test_round_lifecycle_can_accept_idempotent_duplicate_update(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest, round_repository = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    service.update_acceptance_policy = IdempotentRoundUpdateAcceptancePolicy()
    record = service.open_round(
        RoundOpenRequest(
            active_manifest=active_manifest,
            round_id="round_0001",
        )
    )
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )

    first = service.accept_update(record.round_id, update)
    second = service.accept_update(record.round_id, update)

    assert first.idempotent is False
    assert second.idempotent is True
    assert second.update_count == 1
    assert len(round_repository.load_round(record.round_id).updates) == 1


def test_round_lifecycle_can_split_agent_trust_policy_from_network_policy(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    service.update_acceptance_policy = StrictRoundUpdateAcceptancePolicy(
        trust_policy=SingleSubmissionPerAgentTrustPolicy()
    )
    record = service.open_round(
        RoundOpenRequest(
            active_manifest=active_manifest,
            round_id="round_0001",
        )
    )
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

    service.accept_update(record.round_id, first_update)

    with pytest.raises(RoundConflictError):
        service.accept_update(record.round_id, second_update)


def test_round_lifecycle_rejects_base_revision_mismatch(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(
        RoundOpenRequest(
            active_manifest=active_manifest,
            round_id="round_0001",
        )
    )
    bad_update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
        update_id="update_bad",
        base_model_revision="rev_999",
    )

    with pytest.raises(RoundValidationError):
        service.accept_update(record.round_id, bad_update)


def test_round_lifecycle_finalizes_and_clears_active_pointer(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest, round_repository = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(
        RoundOpenRequest(
            active_manifest=active_manifest,
            round_id="round_0001",
        )
    )
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )
    service.accept_update(record.round_id, update)

    finalized = service.finalize_round(
        record.round_id,
        RoundFinalizeRequest(
            next_prototype_version="proto_001",
            next_model_revision="rev_001",
        ),
    )

    assert finalized.status == RoundStatus.FINALIZED
    assert finalized.finalized_at == fixed_time
    assert finalized.publication is not None
    assert finalized.publication.next_manifest.model_revision == "rev_001"
    assert finalized.publication.next_manifest.prototype_version == "proto_001"
    assert round_repository.load_active_pointer() is None
    with pytest.raises(FileNotFoundError):
        service.get_current_round()


def test_round_lifecycle_rejects_update_after_finalize(tmp_path: Path) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
    service, active_manifest, _ = _build_service(
        tmp_path=tmp_path,
        fixed_time=fixed_time,
    )
    record = service.open_round(
        RoundOpenRequest(
            active_manifest=active_manifest,
            round_id="round_0001",
        )
    )
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )
    service.accept_update(record.round_id, update)
    service.finalize_round(
        record.round_id,
        RoundFinalizeRequest(
            next_prototype_version="proto_001",
            next_model_revision="rev_001",
        ),
    )

    with pytest.raises(RoundConflictError):
        service.accept_update(record.round_id, update)


def test_round_lifecycle_finalizes_with_prototype_rebuild_runtime(
    tmp_path: Path,
) -> None:
    fixed_time = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)
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
    input_repository = (
        prototype_rebuild_input_repository_module.PrototypeRebuildInputRepository(
            state_root=tmp_path / "prototype_rebuild_inputs"
        )
    )
    input_repository.save_input(
        PrototypeRebuildInputRecord(
            input_id="bootstrap_v1",
            embedding_spec=EmbeddingAdapterSpec(
                backend="hash_debug",
                model_id="hash_debug",
                revision="seed",
                hash_dim=8,
            ),
            rows=(
                ReferencePrototypeSourceRow(text="cluster_a_1", category="anxiety"),
                ReferencePrototypeSourceRow(text="cluster_a_2", category="anxiety"),
                ReferencePrototypeSourceRow(text="normal_1", category="normal"),
            ),
            mapping_version="ourafla_to_4cat.v1",
        )
    )
    input_repository.set_active(
        "bootstrap_v1",
        activated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    prototype_pack_repository = (
        prototype_pack_repository_module.PrototypePackRepository(
            state_root=tmp_path / "prototype_packs"
        )
    )
    prototype_build_state_repository = (
        prototype_build_state_repository_module.PrototypeBuildStateRepository(
            state_root=tmp_path / "prototype_build_states"
        )
    )
    _StaticEmbeddingAdapterFactory._vectors = {
        "cluster_a_1": [1.0, 0.0],
        "cluster_a_2": [1.0, 0.1],
        "normal_1": [0.0, 1.0],
    }
    prototype_rebuild_runtime_service = StoredReferencePrototypeRebuildService(
        input_repository=input_repository,
        prototype_rebuild_service=PrototypeRebuildService(
            build_strategy=SinglePrototypeBuildStrategy(),
            publication_strategy=ReferenceRebuildPrototypePublicationStrategy(
                prototype_pack_service=PrototypePackService(
                    repository=prototype_pack_repository
                ),
                prototype_build_state_service=PrototypeBuildStateService(
                    repository=prototype_build_state_repository
                ),
            ),
            clock=FixedClock(fixed_time),
        ),
        adapter_factory=_StaticEmbeddingAdapterFactory,
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
        prototype_rebuild_runtime_service=prototype_rebuild_runtime_service,
        clock=FixedClock(fixed_time),
    )
    record = service.open_round(
        RoundOpenRequest(
            active_manifest=active_manifest,
            round_id="round_0001",
        )
    )
    update = _build_update(
        tmp_path=tmp_path,
        round_id=record.round_id,
        task_id=record.training_task.task_id,
    )
    service.accept_update(record.round_id, update)

    finalized = service.finalize_round(
        record.round_id,
        RoundFinalizeRequest(
            next_prototype_version="proto_001",
            next_model_revision="rev_001",
        ),
    )

    assert finalized.publication is not None
    assert finalized.publication.prototype_rebuild_input_id == "bootstrap_v1"
    assert finalized.publication.prototype_pack_ref is not None
    assert Path(finalized.publication.prototype_pack_ref).exists()
    assert finalized.publication.prototype_build_state_ref is not None
    assert Path(finalized.publication.prototype_build_state_ref).exists()
