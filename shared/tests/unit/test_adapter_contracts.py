from __future__ import annotations

from datetime import datetime, timezone

from shared.src.contracts.adapter_contracts import (
    VectorAdapterDeltaPayload,
    VectorAdapterStatePayload,
)


def test_vector_adapter_payloads_capture_revision_and_scales() -> None:
    state = VectorAdapterStatePayload(
        schema_version="vector_adapter_state.v1",
        model_id="tracemind-embed",
        model_revision="rev_001",
        training_scope="adapter_only",
        dimension_scales=[1.0, 0.95],
        updated_at=datetime.now(tz=timezone.utc),
    )
    delta = VectorAdapterDeltaPayload(
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
    assert delta.base_model_revision == "rev_001"
    assert delta.label_counts["anxiety"] == 5
