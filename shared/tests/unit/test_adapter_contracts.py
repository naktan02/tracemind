from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from shared.src.contracts.adapter_contracts import (
    DiagonalScaleAdapterStatePayload,
    DiagonalScaleAdapterUpdatePayload,
    dump_shared_adapter_state_payload,
    dump_shared_adapter_update_payload,
    load_shared_adapter_state_payload,
    load_shared_adapter_update_payload,
)


def test_shared_adapter_payloads_capture_revision_and_scales() -> None:
    state = DiagonalScaleAdapterStatePayload(
        schema_version="vector_adapter_state.v1",
        model_id="tracemind-embed",
        model_revision="rev_001",
        training_scope="adapter_only",
        dimension_scales=[1.0, 0.95],
        updated_at=datetime.now(tz=timezone.utc),
    )
    delta = DiagonalScaleAdapterUpdatePayload(
        schema_version="vector_adapter_delta.v1",
        model_id="tracemind-embed",
        base_model_revision="rev_001",
        training_scope="adapter_only",
        dimension_deltas=[0.01, -0.02],
        example_count=8,
        mean_confidence=0.84,
        mean_margin=0.12,
        label_counts={"anxiety": 5, "depression": 3},
    )

    assert state.dimension_scales[1] == 0.95
    assert state.adapter_kind == "diagonal_scale"
    assert delta.base_model_revision == "rev_001"
    assert delta.adapter_kind == "diagonal_scale"
    assert delta.label_counts["anxiety"] == 5


def test_generic_shared_adapter_loader_dispatches_diagonal_scale_payloads(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    update_path = tmp_path / "update.json"
    dump_shared_adapter_state_payload(
        state_path,
        DiagonalScaleAdapterStatePayload(
            schema_version="vector_adapter_state.v1",
            model_id="tracemind-embed",
            model_revision="rev_002",
            training_scope="adapter_only",
            dimension_scales=[1.0, 1.1],
            updated_at=datetime.now(tz=timezone.utc),
        ),
    )
    dump_shared_adapter_update_payload(
        update_path,
        DiagonalScaleAdapterUpdatePayload(
            schema_version="vector_adapter_delta.v1",
            model_id="tracemind-embed",
            base_model_revision="rev_002",
            training_scope="adapter_only",
            dimension_deltas=[0.01, 0.02],
            example_count=3,
            mean_confidence=0.8,
            mean_margin=0.1,
        ),
    )

    loaded_state = load_shared_adapter_state_payload(state_path)
    loaded_update = load_shared_adapter_update_payload(update_path)

    assert isinstance(loaded_state, DiagonalScaleAdapterStatePayload)
    assert isinstance(loaded_update, DiagonalScaleAdapterUpdatePayload)
    assert loaded_state.adapter_kind == "diagonal_scale"
    assert loaded_update.adapter_kind == "diagonal_scale"
