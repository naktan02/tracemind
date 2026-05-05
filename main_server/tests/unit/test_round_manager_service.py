"""RoundManagerService unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from main_server.src.infrastructure.repositories import (  # noqa: E402
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.infrastructure.repositories import (  # noqa: E402
    shared_adapter_update_repository as shared_adapter_update_repository_module,
)
from main_server.src.services.federation.rounds.boundary.models import (  # noqa: E402
    RoundOpenRequest,
)
from main_server.src.services.federation.rounds.round_manager_service import (  # noqa: E402
    RoundManagerService,
    RoundPublicationRequest,
)
from shared.src.config.diagonal_scale_defaults import (  # noqa: E402
    DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG,
)
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.adapter_contracts import (  # noqa: E402
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
)
from shared.src.contracts.model_contracts import (  # noqa: E402
    ModelManifest,
)
from shared.src.contracts.training_contracts import (  # noqa: E402
    TrainingUpdateEnvelope,
)
from shared.src.domain.services.clock import FixedClock


def test_round_manager_publishes_next_model_and_prototype_pair(tmp_path: Path) -> None:
    repository = shared_adapter_state_repository_module.SharedAdapterStateRepository(
        state_root=tmp_path / "shared_states"
    )
    update_repository = (
        shared_adapter_update_repository_module.SharedAdapterUpdateRepository(
            state_root=tmp_path / "updates"
        )
    )
    repository.save_shared_adapter_state(
        DiagonalScaleAdapterStatePayload(
            schema_version="vector_adapter_state.v1",
            adapter_kind="diagonal_scale",
            model_id="tracemind-embed",
            model_revision="rev_000",
            training_scope="adapter_only",
            dimension_scales=[1.0, 1.0],
            updated_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        )
    )
    update_repository.save_shared_adapter_update(
        "u1",
        DiagonalScaleAdapterUpdatePayload(
            schema_version="vector_adapter_delta.v1",
            adapter_kind="diagonal_scale",
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            dimension_deltas=[0.10, -0.05],
            example_count=2,
            mean_confidence=0.9,
            mean_margin=0.2,
        ),
    )
    update_repository.save_shared_adapter_update(
        "u2",
        DiagonalScaleAdapterUpdatePayload(
            schema_version="vector_adapter_delta.v1",
            adapter_kind="diagonal_scale",
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            dimension_deltas=[0.04, 0.02],
            example_count=1,
            mean_confidence=0.8,
            mean_margin=0.1,
        ),
    )

    service = RoundManagerService(
        artifact_repository=repository,
        update_payload_repository=update_repository,
    )
    publication = service.publish_next_pair(
        RoundPublicationRequest(
            base_manifest=ModelManifest(
                schema_version="model_manifest.v1",
                model_id="tracemind-embed",
                model_revision="rev_000",
                published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
                artifact_kind="shared_adapter_state",
                artifact_ref=repository.ref_for_revision("rev_000"),
                prototype_version="proto_000",
                training_scope="adapter_only",
                training_enabled=True,
                compatible_task_types=("pseudo_label_self_training",),
            ),
            updates=[
                TrainingUpdateEnvelope(
                    schema_version="training_update_envelope.v1",
                    update_id="u1",
                    round_id="round_0001",
                    task_id="task_001",
                    model_id="tracemind-embed",
                    base_model_revision="rev_000",
                    training_scope="adapter_only",
                    payload_ref=update_repository.ref_for_update("u1"),
                    payload_format="diagonal_scale_update",
                    example_count=2,
                    client_metrics={"mean_loss": 0.1},
                ),
                TrainingUpdateEnvelope(
                    schema_version="training_update_envelope.v1",
                    update_id="u2",
                    round_id="round_0001",
                    task_id="task_001",
                    model_id="tracemind-embed",
                    base_model_revision="rev_000",
                    training_scope="adapter_only",
                    payload_ref=update_repository.ref_for_update("u2"),
                    payload_format="diagonal_scale_update",
                    example_count=1,
                    client_metrics={"mean_loss": 0.2},
                ),
            ],
            next_model_revision="rev_001",
            next_prototype_version="proto_001",
        )
    )

    assert publication.next_manifest.model_revision == "rev_001"
    assert publication.next_manifest.prototype_version == "proto_001"
    assert publication.next_manifest.artifact_kind == "shared_adapter_state"
    assert publication.next_manifest.artifact_ref == repository.ref_for_revision(
        "rev_001"
    )
    assert (
        repository.load_shared_adapter_state_from_ref(
            publication.next_manifest.artifact_ref
        ).model_revision
        == "rev_001"
    )
    assert publication.next_state.dimension_scales[0] == 1.08
    assert publication.next_state.dimension_scales[1] == 0.9733333333333334
    assert publication.aggregated_metrics["example_count"] == 3.0


def test_round_manager_sets_default_policy_names_on_training_task() -> None:
    service = RoundManagerService()

    task = service.create_training_task(
        RoundOpenRequest(
            active_manifest=ModelManifest(
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
            ),
            round_id="round_0001",
        )
    )

    assert task.local_epochs == DEFAULT_TRAINING_PROFILE.local_epochs
    assert task.batch_size == DEFAULT_TRAINING_PROFILE.batch_size
    assert task.learning_rate == DEFAULT_TRAINING_PROFILE.learning_rate
    assert task.max_steps == DEFAULT_TRAINING_PROFILE.max_steps
    assert (
        task.objective_config.training_backend_name
        == DEFAULT_TRAINING_PROFILE.training_backend_name
    )
    assert task.objective_config.algorithm_profile_name == (
        DEFAULT_TRAINING_PROFILE.algorithm_profile_name
    )
    assert (
        task.objective_config.example_generation_backend_name
        == DEFAULT_TRAINING_PROFILE.example_generation_backend_name
    )
    assert task.objective_config.evidence_backend_name == (
        DEFAULT_TRAINING_PROFILE.evidence_backend_name
    )
    assert task.objective_config.scorer_backend_name == (
        DEFAULT_TRAINING_PROFILE.scorer_backend_name
    )
    assert task.objective_config.score_policy_name == (
        DEFAULT_TRAINING_PROFILE.score_policy_name
    )
    assert task.objective_config.acceptance_policy_name == (
        DEFAULT_TRAINING_PROFILE.acceptance_policy_name
    )
    assert task.objective_config.privacy_guard_name == (
        DEFAULT_TRAINING_PROFILE.privacy_guard_name
    )
    assert task.objective_config.confidence_threshold == (
        DEFAULT_TRAINING_PROFILE.confidence_threshold
    )
    assert task.objective_config.margin_threshold == (
        DEFAULT_TRAINING_PROFILE.margin_threshold
    )
    assert task.objective_config.extras == {
        "training_backend.delta_scale_multiplier": (
            DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.delta_scale_multiplier
        ),
        "training_backend.max_abs_delta": (
            DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.max_abs_delta
        ),
        "training_backend.minimum_effective_scale": (
            DEFAULT_DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_CONFIG.minimum_effective_scale
        ),
    }
    assert task.secure_aggregation.required is False


def test_round_manager_accepts_secure_aggregation_config_on_training_task() -> None:
    service = RoundManagerService()

    task = service.create_training_task(
        RoundOpenRequest(
            active_manifest=ModelManifest(
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
            ),
            round_id="round_0001",
            secure_aggregation={
                "required": True,
                "aggregation_backend_name": "he_ckks",
                "encryption_scheme_name": "ckks",
                "key_ref": "keys/tenant_a_pubkey",
            },
        )
    )

    assert task.secure_aggregation.required is True
    assert task.secure_aggregation.aggregation_backend_name == "he_ckks"
    assert task.secure_aggregation.encryption_scheme_name == "ckks"
    assert task.secure_aggregation.key_ref == "keys/tenant_a_pubkey"


def test_round_manager_uses_injected_clock_for_publication_time(
    tmp_path: Path,
) -> None:
    repository = shared_adapter_state_repository_module.SharedAdapterStateRepository(
        state_root=tmp_path / "shared_states"
    )
    update_repository = (
        shared_adapter_update_repository_module.SharedAdapterUpdateRepository(
            state_root=tmp_path / "updates"
        )
    )
    repository.save_shared_adapter_state(
        DiagonalScaleAdapterStatePayload(
            schema_version="vector_adapter_state.v1",
            adapter_kind="diagonal_scale",
            model_id="tracemind-embed",
            model_revision="rev_000",
            training_scope="adapter_only",
            dimension_scales=[1.0, 1.0],
            updated_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        )
    )
    update_repository.save_shared_adapter_update(
        "u1",
        DiagonalScaleAdapterUpdatePayload(
            schema_version="vector_adapter_delta.v1",
            adapter_kind="diagonal_scale",
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            dimension_deltas=[0.10, -0.05],
            example_count=2,
            mean_confidence=0.9,
            mean_margin=0.2,
        ),
    )
    fixed_time = datetime(2026, 3, 30, 15, 0, tzinfo=timezone.utc)
    service = RoundManagerService(
        artifact_repository=repository,
        update_payload_repository=update_repository,
        clock=FixedClock(fixed_time),
    )

    publication = service.publish_next_pair(
        RoundPublicationRequest(
            base_manifest=ModelManifest(
                schema_version="model_manifest.v1",
                model_id="tracemind-embed",
                model_revision="rev_000",
                published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
                artifact_kind="shared_adapter_state",
                artifact_ref=repository.ref_for_revision("rev_000"),
                prototype_version="proto_000",
                training_scope="adapter_only",
                training_enabled=True,
                compatible_task_types=("pseudo_label_self_training",),
            ),
            updates=[
                TrainingUpdateEnvelope(
                    schema_version="training_update_envelope.v1",
                    update_id="u1",
                    round_id="round_0001",
                    task_id="task_001",
                    model_id="tracemind-embed",
                    base_model_revision="rev_000",
                    training_scope="adapter_only",
                    payload_ref=update_repository.ref_for_update("u1"),
                    payload_format="diagonal_scale_update",
                    example_count=2,
                    client_metrics={"mean_loss": 0.1},
                )
            ],
            next_model_revision="rev_001",
            next_prototype_version="proto_001",
        )
    )

    assert publication.next_manifest.published_at == fixed_time
    assert publication.next_state.updated_at == fixed_time
