"""로컬 pseudo-label update 생성 서비스 tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.src.infrastructure.repositories.training_artifact_repository import (
    TrainingArtifactRepository,
)
from agent.src.services.inference.scoring_backends.registry import (
    register_scoring_backend,
)
from agent.src.services.training.backends.inputs import (
    registry as training_example_backend_registry,
)
from agent.src.services.training.backends.training.registry import (
    register_shared_adapter_training_backend,
)
from agent.src.services.training.execution.local_training_service import (
    EmbeddedTrainingExample,
    LocalTrainingRequest,
    LocalTrainingService,
)
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.adapter_contracts import (
    SharedAdapterUpdatePayload,
    VectorAdapterDelta,
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


def _registry_catalog_entry(
    *,
    item_name: str,
    family_name: str,
    supported_adapter_kinds: tuple[str, ...] = ("*",),
    accepted_payload_formats: tuple[str, ...] = (),
) -> RegistryCatalogEntry:
    return RegistryCatalogEntry(
        item_name=item_name,
        display_name=item_name,
        implementation_module=__name__,
        core_method_name=item_name,
        family_name=family_name,
        supported_adapter_kinds=supported_adapter_kinds,
        accepted_payload_formats=accepted_payload_formats,
    )


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
    pseudo_label_algorithm_name: str | None = None,
    privacy_guard_name: str | None = "diagonal_scale_clip_only",
    scorer_backend_name: str | None = None,
    loss: str = "diagonal_scale_heuristic",
    extras: dict[str, str | int | float | bool] | None = None,
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
            pseudo_label_algorithm_name=pseudo_label_algorithm_name,
            privacy_guard_name=privacy_guard_name,
            extras={} if extras is None else extras,
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
    assert (
        result.update_envelope.payload_ref
        == f"client-submission::{result.update_envelope.update_id}"
    )
    assert repository.path_for_update(result.update_envelope.update_id).exists()
    assert result.update_envelope.clipped is False
    assert candidates["q1"].selection_context is not None
    assert candidates["q2"].selection_context is not None
    assert candidates["q3"].selection_context is not None
    assert candidates["q4"].selection_context is not None
    assert candidates["q1"].selection_context.selection_stage.value == "accepted"
    assert candidates["q2"].selection_context.selection_stage.value == "dropped_by_cap"
    assert candidates["q3"].selection_context.selection_stage.value == "dropped_by_cap"
    assert (
        candidates["q4"].selection_context.selection_stage.value == "threshold_rejected"
    )
    assert candidates["q2"].selection_context.threshold_accepted is True
    assert candidates["q4"].selection_context.threshold_accepted is False


def test_local_training_service_applies_training_backend_extra_overrides(
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
                privacy_guard_name="noop",
                gradient_clip_norm=None,
                extras={
                    "training_backend.delta_scale_multiplier": 0.1,
                    "training_backend.max_abs_delta": 0.2,
                    "training_backend.minimum_effective_scale": 0.0,
                },
            ),
            model_manifest=_build_manifest(),
        )
    )

    assert result.update_payload is not None
    assert result.update_payload.dimension_deltas == pytest.approx([0.01, 0.0])


def test_local_training_service_rejects_unknown_training_backend_extra_key(
    tmp_path: Path,
) -> None:
    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(repository=repository)

    with pytest.raises(
        ValueError,
        match="Unsupported diagonal-scale heuristic training backend config key",
    ):
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
                    privacy_guard_name="noop",
                    gradient_clip_norm=None,
                    extras={
                        "training_backend.max_abs_dleta": 0.2,
                    },
                ),
                model_manifest=_build_manifest(),
            )
        )


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
            training_task=_build_task(
                acceptance_policy_name="top1_confidence_only",
                pseudo_label_algorithm_name="top1_confidence_only",
            ),
            model_manifest=_build_manifest(),
        )
    )

    candidate = result.selection_result.candidates[0]
    assert candidate.accepted is True
    assert candidate.selection_context is not None
    assert candidate.selection_context.pseudo_label_algorithm_name == (
        "top1_confidence_only"
    )
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


def test_local_training_service_reuses_injected_backend_on_matching_objective(
    tmp_path: Path,
) -> None:
    @dataclass(slots=True)
    class ReusableTestBackend:
        backend_name: str = "reusable_test_backend"
        payload_format: str = "reusable_test_update"
        adapter_kind: str = "diagonal_scale"

        def build_update(
            self,
            *,
            training_task: TrainingTask,
            model_manifest: ModelManifest,
            accepted_examples,
            created_at: datetime,
        ) -> VectorAdapterDelta:
            del training_task
            return VectorAdapterDelta(
                schema_version="vector_adapter_delta.v1",
                model_id=model_manifest.model_id,
                base_model_revision=model_manifest.model_revision,
                training_scope=model_manifest.training_scope,
                dimension_deltas=[0.01, 0.0],
                example_count=len(accepted_examples),
                mean_confidence=0.91,
                mean_margin=0.71,
                created_at=created_at,
                adapter_kind=self.adapter_kind,
            )

        def to_payload(self, update: VectorAdapterDelta) -> SharedAdapterUpdatePayload:
            return update

        def build_client_metrics(
            self,
            update: VectorAdapterDelta,
        ) -> dict[str, float]:
            return {"reused_backend": float(update.example_count)}

        def matches_objective_config(
            self,
            objective_config: TrainingObjectiveConfig | None,
        ) -> bool:
            del objective_config
            return True

    repository = TrainingArtifactRepository(state_root=tmp_path / "agent_state")
    service = LocalTrainingService(
        repository=repository,
        backend=ReusableTestBackend(),
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
            training_task=_build_task(
                loss="reusable_test_backend",
                privacy_guard_name="noop",
            ),
            model_manifest=_build_manifest(),
        )
    )

    assert result.update_envelope is not None
    assert result.update_envelope.payload_format == "reusable_test_update"
    assert result.update_envelope.client_metrics["reused_backend"] == 1.0
    assert result.update_payload is not None
    assert result.update_payload.adapter_kind == "diagonal_scale"


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
        factory=lambda _objective_config: TestShiftBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="test_shift_backend",
            family_name="training_backend",
            supported_adapter_kinds=("test_shift",),
            accepted_payload_formats=("test_shift_update",),
        ),
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
        factory=lambda _objective_config: TestShiftBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="test_shift_backend_incompatible_guard",
            family_name="training_backend",
            supported_adapter_kinds=("test_shift",),
            accepted_payload_formats=("test_shift_update_incompatible_guard",),
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
        factory=lambda _objective_config: TestShiftBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="test_shift_backend_incompatible_scorer",
            family_name="training_backend",
            supported_adapter_kinds=("test_shift",),
            accepted_payload_formats=("test_shift_update_incompatible_scorer",),
        ),
    )
    register_scoring_backend(
        "diagonal_only_test_scorer",
        factory=lambda _objective_config, _similarity_name: (
            DiagonalOnlyScoringBackend()
        ),
        catalog_entry=_registry_catalog_entry(
            item_name="diagonal_only_test_scorer",
            family_name="scoring_backend",
            supported_adapter_kinds=("diagonal_scale",),
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


def test_local_training_service_rejects_incompatible_training_example_backend(
    tmp_path: Path,
) -> None:
    class TestShiftUpdatePayload(SharedAdapterUpdatePayload):
        shift_norm: float

    @dataclass(slots=True)
    class TestShiftBackend:
        backend_name: str = "test_shift_backend_incompatible_examples"
        payload_format: str = "test_shift_update_incompatible_examples"
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
    class DiagonalOnlyTrainingExampleBackend:
        backend_name: str = "diagonal_only_training_examples"
        supported_adapter_kinds: tuple[str, ...] = ("diagonal_scale",)

        def build_examples(self, request) -> tuple:
            del request
            return ()

        def build_examples_from_stored_events(self, request) -> tuple:
            del request
            return ()

    register_shared_adapter_update_payload_type(
        "test_shift_incompatible_examples",
        TestShiftUpdatePayload,
    )
    register_shared_adapter_training_backend(
        "test_shift_backend_incompatible_examples",
        factory=lambda _objective_config: TestShiftBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="test_shift_backend_incompatible_examples",
            family_name="training_backend",
            supported_adapter_kinds=("test_shift",),
            accepted_payload_formats=("test_shift_update_incompatible_examples",),
        ),
    )
    training_example_backend_registry.register_training_example_backend(
        "diagonal_only_training_examples",
        factory=lambda _objective_config: DiagonalOnlyTrainingExampleBackend(),
        catalog_entry=_registry_catalog_entry(
            item_name="diagonal_only_training_examples",
            family_name="example_generation",
            supported_adapter_kinds=("diagonal_scale",),
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
                training_task=TrainingTask(
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
                        loss="test_shift_backend_incompatible_examples",
                        confidence_threshold=0.6,
                        margin_threshold=0.02,
                        example_generation_backend_name=(
                            "diagonal_only_training_examples"
                        ),
                        privacy_guard_name="noop",
                    ),
                    selection_policy=TrainingSelectionPolicy(max_examples=1),
                    min_required_examples=1,
                    gradient_clip_norm=0.05,
                ),
                model_manifest=_build_manifest(),
            )
        )
    except ValueError as error:
        assert "Incompatible training example backend" in str(error)
    else:
        raise AssertionError(
            "호환되지 않는 training example backend 조합이 허용되었습니다."
        )
