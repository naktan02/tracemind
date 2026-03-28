"""PrototypeBuildStateService unit tests."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

MAIN_SERVER_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_SERVER_ROOT))

from shared.src.contracts.prototype_build_state_contracts import (
    PrototypeBuildStatePayload,
)
from src.infrastructure.repositories.prototype_build_state_repository import (
    PrototypeBuildStateRepository,
)
from src.services.prototype_build_state_service import PrototypeBuildStateService


def test_publish_and_get_state(tmp_path: Path) -> None:
    repository = PrototypeBuildStateRepository(
        state_root=tmp_path / "prototype_build_states"
    )
    service = PrototypeBuildStateService(repository=repository)
    payload = PrototypeBuildStatePayload(
        schema_version="prototype_build_state.v1",
        prototype_version="proto_test_v1",
        embedding_backend="hash_debug",
        embedding_model_id="hash_debug",
        embedding_model_revision="v1",
        normalize_embeddings=True,
        task_prefix="",
        translation_model_id=None,
        translation_model_revision=None,
        translation_direction=None,
        mapping_version="ourafla_to_4cat.v1",
        build_method="mean_centroid_l2_normalized",
        distance_metric="cosine",
        built_at=datetime(2026, 3, 29, tzinfo=timezone.utc),
        categories={
            "anxiety": {
                "embedding_sum": [4.0, 8.0],
                "sample_count": 2,
            },
            "normal": {
                "embedding_sum": [6.0, 6.0],
                "sample_count": 2,
            },
        },
    )

    state_path = service.publish_state(payload)
    loaded = service.get_state(payload.prototype_version)

    assert state_path.exists()
    assert loaded.prototype_version == payload.prototype_version
    assert loaded.embedding_backend == "hash_debug"
    assert sorted(loaded.categories) == ["anxiety", "normal"]
