"""Shared adapter 저장소 ref 검증."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as state_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as update_repository_module,
)
from shared.src.contracts.adapter_contract_families.classifier_head import (
    ClassifierHeadAdapterStatePayload,
    ClassifierHeadAdapterUpdatePayload,
)


def _state_payload() -> ClassifierHeadAdapterStatePayload:
    return ClassifierHeadAdapterStatePayload(
        model_id="tracemind-embed",
        model_revision="rev_001",
        training_scope="head_only",
        label_weights={"anxiety": [0.1, 0.2], "normal": [-0.1, -0.2]},
        label_biases={"anxiety": 0.01, "normal": -0.01},
        updated_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )


def _update_payload() -> ClassifierHeadAdapterUpdatePayload:
    return ClassifierHeadAdapterUpdatePayload(
        model_id="tracemind-embed",
        base_model_revision="rev_001",
        training_scope="head_only",
        label_weight_deltas={"anxiety": [0.05, -0.02]},
        label_bias_deltas={"anxiety": 0.01},
        example_count=3,
        mean_confidence=0.8,
        mean_margin=0.15,
    )


def test_shared_adapter_state_repository_uses_opaque_refs(tmp_path: Path) -> None:
    repository = state_repository_module.SharedAdapterStateRepository(
        state_root=tmp_path / "states"
    )
    path = repository.save_shared_adapter_state(_state_payload())

    ref = repository.ref_for_revision("rev_001")
    loaded_from_ref = repository.load_shared_adapter_state_from_ref(ref)
    loaded_from_legacy_path = repository.load_shared_adapter_state_from_ref(str(path))

    assert ref == "shared_adapter_state::rev_001"
    assert loaded_from_ref.model_revision == "rev_001"
    assert loaded_from_legacy_path.model_revision == "rev_001"


def test_shared_adapter_update_repository_uses_opaque_refs(tmp_path: Path) -> None:
    repository = update_repository_module.SharedAdapterUpdateRepository(
        state_root=tmp_path / "updates"
    )
    path = repository.save_shared_adapter_update("update_001", _update_payload())

    ref = repository.ref_for_update("update_001")
    loaded_from_ref = repository.load_shared_adapter_update_from_ref(ref)
    loaded_from_legacy_path = repository.load_shared_adapter_update_from_ref(str(path))

    assert ref == "shared_adapter_update::update_001"
    assert loaded_from_ref.base_model_revision == "rev_001"
    assert loaded_from_legacy_path.base_model_revision == "rev_001"
