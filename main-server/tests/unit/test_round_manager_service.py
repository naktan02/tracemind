"""RoundManagerService unit tests."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_SERVER_ROOT))

from src.infrastructure.repositories.vector_adapter_state_repository import (  # noqa: E402
    VectorAdapterStateRepository,
)
from src.services.round_manager_service import RoundManagerService  # noqa: E402

from shared.src.contracts.adapter_contracts import (  # noqa: E402
    VectorAdapterDeltaPayload,
    VectorAdapterStatePayload,
    dump_vector_adapter_delta_payload,
)
from shared.src.domain.entities.model_manifest import ModelManifest  # noqa: E402
from shared.src.domain.entities.training_update import (  # noqa: E402
    TrainingUpdateEnvelope,  # noqa: E402
)


def test_round_manager_publishes_next_model_and_prototype_pair(tmp_path: Path) -> None:
    repository = VectorAdapterStateRepository(state_root=tmp_path / "vector_states")
    base_state_path = repository.save_state(
        VectorAdapterStatePayload(
            schema_version="vector_adapter_state.v1",
            model_id="tracemind-embed",
            model_revision="rev_000",
            training_scope="adapter_only",
            dimension_scales=[1.0, 1.0],
            updated_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        )
    )
    update_one_path = tmp_path / "updates" / "u1.json"
    update_two_path = tmp_path / "updates" / "u2.json"
    dump_vector_adapter_delta_payload(
        update_one_path,
        VectorAdapterDeltaPayload(
            schema_version="vector_adapter_delta.v1",
            model_id="tracemind-embed",
            base_model_revision="rev_000",
            training_scope="adapter_only",
            dimension_deltas=[0.10, -0.05],
            example_count=2,
            mean_confidence=0.9,
            mean_margin=0.2,
        ),
    )
    dump_vector_adapter_delta_payload(
        update_two_path,
        VectorAdapterDeltaPayload(
            schema_version="vector_adapter_delta.v1",
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
        base_manifest=ModelManifest(
            schema_version="model_manifest.v1",
            model_id="tracemind-embed",
            model_revision="rev_000",
            published_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
            artifact_kind="vector_adapter_state",
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
                payload_format="vector_adapter_delta",
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
                payload_format="vector_adapter_delta",
                example_count=1,
                client_metrics={"mean_loss": 0.2},
            ),
        ],
        next_model_revision="rev_001",
        next_prototype_version="proto_001",
    )

    assert publication.next_manifest.model_revision == "rev_001"
    assert publication.next_manifest.prototype_version == "proto_001"
    assert Path(publication.next_manifest.artifact_ref).exists()
    assert publication.next_state.dimension_scales[0] == 1.08
    assert publication.next_state.dimension_scales[1] == 0.9733333333333334
    assert publication.aggregated_metrics["example_count"] == 3.0
