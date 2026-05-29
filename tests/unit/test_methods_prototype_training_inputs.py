"""Prototype training input method core tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import pytest

from methods.prototype.training_inputs.examples import (
    build_prototype_rescore_inputs,
    build_prototype_rescore_inputs_from_stored_events,
    build_prototype_weak_strong_inputs,
    require_weak_strong_texts,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.domain.entities.inference.events import ScoredEvent


@dataclass(slots=True)
class _SourceRow:
    query_id: str
    text: str
    occurred_at: datetime
    translated_text: str | None = None
    weak_text: str | None = None
    strong_text: str | None = None
    weak_translated_text: str | None = None
    strong_translated_text: str | None = None


@dataclass(slots=True)
class _StoredEvent:
    scored_event: ScoredEvent
    base_embedding: Sequence[float] | None


class _ScaleAdapterState:
    def apply(self, embedding: Sequence[float]) -> list[float]:
        return [float(value) * 2.0 for value in embedding]


class _NearestAxisScorer:
    def score(self, embedding, prototypes):
        del prototypes
        return {
            "anxiety": float(embedding[0]),
            "normal": float(embedding[1]),
        }


def _pack_payload() -> PrototypePackPayload:
    return PrototypePackPayload.model_validate(
        {
            "schema_version": "prototype_pack.v1",
            "prototype_version": "proto_test_v1",
            "embedding_model_id": "hash_debug",
            "embedding_model_revision": "main",
            "mapping_version": "ourafla_to_4cat.v1",
            "build_method": "mean_centroid_l2_normalized",
            "distance_metric": "cosine",
            "translation_model_id": "nllb",
            "built_at": "2026-05-04T00:00:00+00:00",
            "categories": {
                "anxiety": [
                    {
                        "prototype_id": "anxiety:single",
                        "centroid": [1.0, 0.0],
                        "sample_count": 2,
                    }
                ],
                "normal": [
                    {
                        "prototype_id": "normal:single",
                        "centroid": [0.0, 1.0],
                        "sample_count": 2,
                    }
                ],
            },
        }
    )


def test_build_prototype_rescore_inputs_applies_adapter_and_scores() -> None:
    inputs = build_prototype_rescore_inputs(
        source_rows=(
            _SourceRow(
                query_id="q1",
                text="panic",
                translated_text="panic translated",
                occurred_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
            ),
        ),
        base_embeddings=([0.5, 0.25],),
        adapter_state=_ScaleAdapterState(),
        prototype_pack=_pack_payload(),
        model_id="hash_debug",
        scorer=_NearestAxisScorer(),
    )

    assert len(inputs) == 1
    assert inputs[0].base_embedding == [0.5, 0.25]
    assert inputs[0].embedding == [1.0, 0.5]
    assert inputs[0].scored_event.query_id == "q1"
    assert inputs[0].scored_event.translated_text == "panic translated"
    assert inputs[0].scored_event.translation_model_id == "nllb"
    assert inputs[0].scored_event.category_scores == {
        "anxiety": 1.0,
        "normal": 0.5,
    }


def test_rescore_inputs_from_stored_events_skips_missing_embedding() -> None:
    inputs = build_prototype_rescore_inputs_from_stored_events(
        stored_events=(
            _StoredEvent(
                scored_event=ScoredEvent(
                    query_id="q1",
                    occurred_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
                    translated_text=None,
                    embedding_model_id="hash_debug",
                    translation_model_id=None,
                    category_scores={"anxiety": 0.1, "normal": 0.9},
                ),
                base_embedding=[0.5, 0.25],
            ),
            _StoredEvent(
                scored_event=ScoredEvent(
                    query_id="q2",
                    occurred_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
                    translated_text=None,
                    embedding_model_id="hash_debug",
                    translation_model_id=None,
                    category_scores={"anxiety": 0.1, "normal": 0.9},
                ),
                base_embedding=None,
            ),
        ),
        adapter_state=_ScaleAdapterState(),
        prototype_pack=_pack_payload(),
        scorer=_NearestAxisScorer(),
    )

    assert len(inputs) == 1
    assert inputs[0].scored_event.query_id == "q1"
    assert inputs[0].scored_event.category_scores == {
        "anxiety": 1.0,
        "normal": 0.5,
    }


def test_require_weak_strong_texts_rejects_missing_view() -> None:
    with pytest.raises(ValueError, match="requires both weak_text and strong_text"):
        require_weak_strong_texts(
            (
                _SourceRow(
                    query_id="q1",
                    text="panic",
                    weak_text="panic weak",
                    occurred_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
                ),
            )
        )


def test_build_prototype_weak_strong_inputs_keeps_selection_and_update_views() -> None:
    row = _SourceRow(
        query_id="q1",
        text="panic",
        translated_text="base translated",
        weak_text="panic weak",
        strong_text="panic strong",
        weak_translated_text="weak translated",
        strong_translated_text="strong translated",
        occurred_at=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    weak_texts, strong_texts = require_weak_strong_texts((row,))

    inputs = build_prototype_weak_strong_inputs(
        source_rows=(row,),
        weak_base_embeddings=([0.5, 0.25],),
        strong_base_embeddings=([0.1, 0.9],),
        adapter_state=_ScaleAdapterState(),
        prototype_pack=_pack_payload(),
        model_id="hash_debug",
        scorer=_NearestAxisScorer(),
        backend_name="weak_strong_pair",
    )

    assert weak_texts == ["panic weak"]
    assert strong_texts == ["panic strong"]
    assert len(inputs) == 1
    assert inputs[0].weak_embedding == [1.0, 0.5]
    assert inputs[0].strong_embedding == [0.2, 1.8]
    assert inputs[0].weak_scored_event.translated_text == "weak translated"
    assert inputs[0].strong_scored_event.translated_text == "strong translated"
    assert inputs[0].weak_scored_event.category_scores["anxiety"] == 1.0
    assert inputs[0].strong_scored_event.category_scores["normal"] == 1.8
    assert inputs[0].metadata == {
        "training_input_backend_name": "weak_strong_pair",
        "selection_view": "weak",
        "update_view": "strong",
    }
