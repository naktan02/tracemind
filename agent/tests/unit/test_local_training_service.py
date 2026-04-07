"""로컬 pseudo-label update 생성 서비스 tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.inference.scoring_backends import register_scoring_backend
from agent.src.services.training.local_training_service import (
    EmbeddedTrainingExample,
    LocalTrainingRequest,
    LocalTrainingService,
)
from agent.src.services.training.training_backends import (
    register_shared_adapter_training_backend,
)
from shared.src.contracts.adapter_contracts import (
    SharedAdapterUpdatePayload,
    register_shared_adapter_update_payload_type,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
)
from shared.src.domain.entities.inference.events import ScoredEvent
from shared.src.domain.services.clock import FixedClock


def _build_manifest() -> ModelManifest:
    return ModelManifest(
        schema_version="model_manifest.v1",
        model_id="tracemind-embed",
        model_revision="rev_000",
        published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        artifact_kind="shared_adapter_state",
        artifact_ref="/tmp/rev_000.json",
        prototype_version="proto_000",
        training_scope="adapter_only",
        training_enabled=True,
        compatible_task_types=("pseudo_label_self_training",),
    )


def _build_task(
    *,
    min_required_examples: int = 1,
    gradient_clip_norm: float | None = 0.05,
    acceptance_policy_name: str | None = None,
    privacy_guard_name: str | None = "diagonal_scale_clip_only",
    scorer_backend_name: str | None = None,
    loss: str = "diagonal_scale_heuristic",
) -> TrainingTask:
    return TrainingTask(
        schema_version="training_task.v1",
        task_id="task_001",
        round_id="round_0001",
        model_id="tracemind-embed",
        model_revision="rev_000",
        task_type="pseudo_label_self_training",
        training_scope="adapter_only",
        local_epochs=1,
        batch_size=8,
        learning_rate=1e-2,
        max_steps=10,
        objective_config=TrainingObjectiveConfig(
            loss=loss,
            confidence_threshold=0.6,
            margin_threshold=0.02,
            scorer_backend_name=scorer_backend_name,
            acceptance_policy_name=acceptance_policy_name,
            privacy_guard_name=privacy_guard_name,
        ),
        selection_policy=TrainingSelectionPolicy(max_examples=1),
        min_required_examples=min_required_examples,
        gradient_clip_norm=gradient_clip_norm,
    )


def _make_example(
    *,
    query_id: str,
    scores: dict[str, float],
    embedding: list[float],
) -> EmbeddedTrainingExample:
    return EmbeddedTrainingExample(
        scored_event=ScoredEvent(
            query_id=query_id,
            occurred_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
            translated_text=None,
            embedding_model_id="tracemind-embed",
            translation_model_id=None,
            category_scores=scores,
        ),
        embedding=embedding,
    )


def test_local_training_service_creates_update_from_top_candidates(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.91, "depression": 0.2, "normal": 0.1},
                    embedding=[1.0, 0.0],
                ),
                _make_example(
                    query_id="q2",
                    scores={"depression": 0.88, "anxiety": 0.5, "normal": 0.1},
                    embedding=[0.0, 1.0],
                ),
                _make_example(
                    query_id="q3",
                    scores={"anxiety": 0.62, "depression": 0.58, "normal": 0.1},
                    embedding=[0.7, 0.7],
                ),
                _make_example(
                    query_id="q4",
                    scores={"normal": 0.59, "anxiety": 0.58, "depression": 0.1},
                    embedding=[0.2, 0.2],
                ),
            ),
            training_task=_build_task(),
            model_manifest=_build_manifest(),
        )
    )

    candidates = {
        candidate.source_event_ref: candidate
        for candidate in result.selection_result.candidates
    }

    assert result.selection_result.total_count == 4
    assert result.selection_result.accepted_count == 1
    assert result.update_envelope is not None
    assert result.update_payload is not None
    assert result.update_envelope.example_count == 1
    assert result.update_payload.label_counts == {"anxiety": 1}
    assert result.update_payload.adapter_kind == "diagonal_scale"
    assert result.update_envelope.payload_format == "diagonal_scale_update"
    assert Path(result.update_envelope.payload_ref).exists()
    assert (
        Path(result.update_envelope.payload_ref).parent.name
        == "shared_adapter_updates"
    )
    assert result.update_envelope.clipped is False
    assert candidates["q1"].metadata["selection_stage"] == "accepted"
    assert candidates["q2"].metadata["selection_stage"] == "dropped_by_cap"
    assert candidates["q3"].metadata["selection_stage"] == "dropped_by_cap"
    assert candidates["q4"].metadata["selection_stage"] == "threshold_rejected"
    assert candidates["q2"].metadata["threshold_accepted"] is True
    assert candidates["q4"].metadata["threshold_accepted"] is False


def test_local_training_service_skips_update_when_examples_are_insufficient(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.91, "depression": 0.2},
                    embedding=[1.0, 0.0],
                ),
            ),
            training_task=_build_task(min_required_examples=2),
            model_manifest=_build_manifest(),
        )
    )

    assert result.selection_result.accepted_count == 1
    assert result.update_envelope is None
    assert result.update_payload is None


def test_local_training_service_marks_update_as_clipped_when_privacy_guard_scales_delta(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.91, "depression": 0.2, "normal": 0.1},
                    embedding=[1.0, 0.0],
                ),
            ),
            training_task=_build_task(gradient_clip_norm=0.01),
            model_manifest=_build_manifest(),
        )
    )

    assert result.update_envelope is not None
    assert result.update_envelope.clipped is True
    assert result.update_payload is not None
    assert result.update_payload.l2_norm() == 0.01


def test_local_training_service_uses_acceptance_policy_from_task_config(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.62, "depression": 0.61, "normal": 0.1},
                    embedding=[1.0, 0.0],
                ),
            ),
            training_task=_build_task(acceptance_policy_name="top1_confidence_only"),
            model_manifest=_build_manifest(),
        )
    )

    candidate = result.selection_result.candidates[0]
    assert candidate.accepted is True
    assert candidate.metadata["acceptance_policy_name"] == "top1_confidence_only"
    assert result.update_envelope is not None


def test_local_training_service_can_switch_privacy_guard_from_task_config(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.91, "depression": 0.2, "normal": 0.1},
                    embedding=[1.0, 0.0],
                ),
            ),
            training_task=_build_task(
                gradient_clip_norm=0.01,
                privacy_guard_name="noop",
            ),
            model_manifest=_build_manifest(),
        )
    )

    assert result.update_envelope is not None
    assert result.update_envelope.clipped is False
    assert result.update_payload is not None
    assert result.update_payload.l2_norm() == 0.05


def test_local_training_service_uses_injected_clock_when_created_at_missing(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    fixed_time = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
    service = LocalTrainingService(
        repository=repository,
        clock=FixedClock(fixed_time),
    )

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.91, "depression": 0.2, "normal": 0.1},
                    embedding=[1.0, 0.0],
                ),
            ),
            training_task=_build_task(),
            model_manifest=_build_manifest(),
        )
    )

    assert result.update_envelope is not None
    assert result.update_envelope.created_at == fixed_time
    assert result.update_payload is not None
    assert result.update_payload.created_at == fixed_time


def test_local_training_service_can_use_registered_non_diagonal_backend(
    tmp_path: Path,
) -> None:
    class TestShiftUpdatePayload(SharedAdapterUpdatePayload):
        shift_norm: float

    @dataclass(slots=True)
    class TestShiftBackend:
        backend_name: str = "test_shift_backend"
        payload_format: str = "test_shift_update"
        adapter_kind: str = "test_shift"

        def build_update(
            self,
            *,
            training_task: TrainingTask,
            model_manifest: ModelManifest,
            accepted_examples,
            created_at: datetime,
        ) -> TestShiftUpdatePayload:
            del training_task
            return TestShiftUpdatePayload(
                model_id=model_manifest.model_id,
                base_model_revision=model_manifest.model_revision,
                training_scope=model_manifest.training_scope,
                example_count=len(accepted_examples),
                created_at=created_at,
                adapter_kind=self.adapter_kind,
                shift_norm=1.25,
            )

        def to_payload(
            self,
            update: TestShiftUpdatePayload,
        ) -> SharedAdapterUpdatePayload:
            return update

        def build_client_metrics(
            self,
            update,
        ) -> dict[str, float]:
            return {"test_shift_norm": float(update.shift_norm)}

    register_shared_adapter_update_payload_type("test_shift", TestShiftUpdatePayload)
    register_shared_adapter_training_backend(
        "test_shift_backend",
        factory=TestShiftBackend,
    )

    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    result = service.run(
        LocalTrainingRequest(
            training_examples=(
                _make_example(
                    query_id="q1",
                    scores={"anxiety": 0.91, "depression": 0.2, "normal": 0.1},
                    embedding=[1.0, 0.0],
                ),
            ),
            training_task=_build_task(
                loss="test_shift_backend",
                privacy_guard_name="noop",
            ),
            model_manifest=_build_manifest(),
        )
    )

    assert result.update_envelope is not None
    assert result.update_payload is not None
    assert result.update_payload.adapter_kind == "test_shift"
    assert result.update_envelope.payload_format == "test_shift_update"
    assert result.update_envelope.client_metrics["test_shift_norm"] == 1.25
    assert "mean_confidence" not in result.update_envelope.client_metrics

    loaded_payload = repository.load_shared_adapter_update(
        result.update_envelope.update_id
    )
    assert isinstance(loaded_payload, TestShiftUpdatePayload)
    assert loaded_payload.adapter_kind == "test_shift"


def test_local_training_service_rejects_incompatible_privacy_guard(
    tmp_path: Path,
) -> None:
    class TestShiftUpdatePayload(SharedAdapterUpdatePayload):
        shift_norm: float

    @dataclass(slots=True)
    class TestShiftBackend:
        backend_name: str = "test_shift_backend_incompatible_guard"
        payload_format: str = "test_shift_update_incompatible_guard"
        adapter_kind: str = "test_shift"

        def build_update(
            self,
            *,
            training_task: TrainingTask,
            model_manifest: ModelManifest,
            accepted_examples,
            created_at: datetime,
        ) -> TestShiftUpdatePayload:
            del training_task, model_manifest, accepted_examples, created_at
            raise AssertionError("호환성 검증 전에 update 생성이 호출되면 안 됩니다.")

        def to_payload(
            self,
            update: TestShiftUpdatePayload,
        ) -> SharedAdapterUpdatePayload:
            return update

        def build_client_metrics(
            self,
            update,
        ) -> dict[str, float]:
            del update
            return {}

    register_shared_adapter_update_payload_type(
        "test_shift_incompatible_guard",
        TestShiftUpdatePayload,
    )
    register_shared_adapter_training_backend(
        "test_shift_backend_incompatible_guard",
        factory=TestShiftBackend,
    )

    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    try:
        service.run(
            LocalTrainingRequest(
                training_examples=(
                    _make_example(
                        query_id="q1",
                        scores={"anxiety": 0.91, "depression": 0.2, "normal": 0.1},
                        embedding=[1.0, 0.0],
                    ),
                ),
                training_task=_build_task(
                    loss="test_shift_backend_incompatible_guard",
                    privacy_guard_name="diagonal_scale_clip_only",
                ),
                model_manifest=_build_manifest(),
            )
        )
    except ValueError as error:
        assert "Incompatible privacy guard" in str(error)
    else:
        raise AssertionError("호환되지 않는 privacy guard 조합이 허용되었습니다.")


def test_local_training_service_rejects_incompatible_scoring_backend(
    tmp_path: Path,
) -> None:
    class TestShiftUpdatePayload(SharedAdapterUpdatePayload):
        shift_norm: float

    @dataclass(slots=True)
    class TestShiftBackend:
        backend_name: str = "test_shift_backend_incompatible_scorer"
        payload_format: str = "test_shift_update_incompatible_scorer"
        adapter_kind: str = "test_shift"

        def build_update(
            self,
            *,
            training_task: TrainingTask,
            model_manifest: ModelManifest,
            accepted_examples,
            created_at: datetime,
        ) -> TestShiftUpdatePayload:
            del training_task, model_manifest, accepted_examples, created_at
            raise AssertionError("호환성 검증 전에 update 생성이 호출되면 안 됩니다.")

        def to_payload(
            self,
            update: TestShiftUpdatePayload,
        ) -> SharedAdapterUpdatePayload:
            return update

        def build_client_metrics(
            self,
            update,
        ) -> dict[str, float]:
            del update
            return {}

    @dataclass(slots=True)
    class DiagonalOnlyScoringBackend:
        backend_name: str = "diagonal_only_test_scorer"
        supported_adapter_kinds: tuple[str, ...] = ("diagonal_scale",)

        def score(self, embedding, prototypes):
            del embedding, prototypes
            return {"anxiety": 1.0}

    register_shared_adapter_update_payload_type(
        "test_shift_incompatible_scorer",
        TestShiftUpdatePayload,
    )
    register_shared_adapter_training_backend(
        "test_shift_backend_incompatible_scorer",
        factory=TestShiftBackend,
    )
    register_scoring_backend(
        "diagonal_only_test_scorer",
        factory=lambda _objective_config, _similarity_name: (
            DiagonalOnlyScoringBackend()
        ),
    )

    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    try:
        service.run(
            LocalTrainingRequest(
                training_examples=(
                    _make_example(
                        query_id="q1",
                        scores={"anxiety": 0.91, "depression": 0.2, "normal": 0.1},
                        embedding=[1.0, 0.0],
                    ),
                ),
                training_task=_build_task(
                    loss="test_shift_backend_incompatible_scorer",
                    privacy_guard_name="noop",
                    scorer_backend_name="diagonal_only_test_scorer",
                ),
                model_manifest=_build_manifest(),
            )
        )
    except ValueError as error:
        assert "Incompatible scoring backend" in str(error)
    else:
        raise AssertionError("호환되지 않는 scoring backend 조합이 허용되었습니다.")
