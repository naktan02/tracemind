"""RoundManagerService unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from main_server.src.infrastructure.repositories import (  # noqa: E402
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.services.rounds.round_manager_service import (  # noqa: E402
    RoundManagerService,
    RoundPublicationRequest,
    TrainingTaskRequest,
)
from shared.src.config.training_defaults import DEFAULT_TRAINING_PROFILE
from shared.src.contracts.adapter_contracts import (  # noqa: E402
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    dump_shared_adapter_update_payload,
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
    base_state_path = repository.save_shared_adapter_state(
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
    update_one_path = tmp_path / "updates" / "u1.json"
    update_two_path = tmp_path / "updates" / "u2.json"
    dump_shared_adapter_update_payload(
        update_one_path,
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
    dump_shared_adapter_update_payload(
        update_two_path,
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

    service = RoundManagerService(artifact_repository=repository)
    publication = service.publish_next_pair(
        RoundPublicationRequest(
            base_manifest=ModelManifest(
                schema_version="model_manifest.v1",
                model_id="tracemind-embed",
                model_revision="rev_000",
                published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
                artifact_kind="shared_adapter_state",
                artifact_ref=str(base_state_path),
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
                    payload_ref=str(update_one_path),
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
                    payload_ref=str(update_two_path),
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
    assert Path(publication.next_manifest.artifact_ref).exists()
    assert publication.next_state.dimension_scales[0] == 1.08
    assert publication.next_state.dimension_scales[1] == 0.9733333333333334
    assert publication.aggregated_metrics["example_count"] == 3.0


def test_round_manager_sets_default_policy_names_on_training_task() -> None:
    service = RoundManagerService()

    task = service.create_training_task(
        TrainingTaskRequest(
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
    assert (
        task.objective_config.example_generation_backend_name
        == DEFAULT_TRAINING_PROFILE.example_generation_backend_name
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
    assert task.secure_aggregation.required is False


def test_round_manager_accepts_secure_aggregation_config_on_training_task() -> None:
    service = RoundManagerService()

    task = service.create_training_task(
        TrainingTaskRequest(
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
    base_state_path = repository.save_shared_adapter_state(
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
    update_path = tmp_path / "updates" / "u1.json"
    dump_shared_adapter_update_payload(
        update_path,
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
                artifact_ref=str(base_state_path),
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
                    payload_ref=str(update_path),
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
