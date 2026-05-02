"""Multiview query adaptation export tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.src.services.training.backends.inputs.models import TrainingExampleSource
from agent.src.services.training.datasets.query_adaptation_dataset_service import (
    QueryAdaptationDataset,
    QueryAdaptationDatasetExample,
    QueryAdaptationDatasetProvenance,
)
from agent.src.services.training.datasets.query_adaptation_multiview_service import (
    IdentityQueryAdaptationMultiviewAugmenter,
    QueryAdaptationMultiviewService,
)
from scripts.experiments.lora_classifier.query_adaptation_multiview_io import (
    QUERY_ADAPTATION_MULTIVIEW_SUMMARY_SCHEMA_VERSION,
    build_labeled_rows_from_query_adaptation_multiview_dataset,
    write_query_adaptation_multiview_dataset,
)


def _build_multiview_dataset():
    occurred_at = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    dataset = QueryAdaptationDataset(
        examples=(
            QueryAdaptationDatasetExample(
                source_row=TrainingExampleSource(
                    query_id="q1",
                    text="불안해서 잠이 안 와요",
                    occurred_at=occurred_at,
                    translated_text="I cannot sleep because I am anxious",
                ),
                label="anxiety",
                provenance=QueryAdaptationDatasetProvenance(
                    locale="ko-KR",
                    source_type="user_message",
                    model_revision="rev_001",
                    selection_confidence_kind="prototype_similarity_top1",
                    translated_text_present=True,
                    candidate_id="round_001:q1",
                ),
                confidence=0.94,
                margin=0.48,
            ),
            QueryAdaptationDatasetExample(
                source_row=TrainingExampleSource(
                    query_id="q2",
                    text="요즘 계속 가라앉아요",
                    occurred_at=occurred_at,
                ),
                label="depression",
                provenance=QueryAdaptationDatasetProvenance(
                    locale="ko-KR",
                    source_type="user_message",
                    model_revision="rev_001",
                    selection_confidence_kind="prototype_similarity_top1",
                    translated_text_present=False,
                    candidate_id="round_001:q2",
                ),
                confidence=0.91,
                margin=0.42,
            ),
        )
    )
    return QueryAdaptationMultiviewService(
        augmenter=IdentityQueryAdaptationMultiviewAugmenter()
    ).build_dataset(dataset=dataset)


def test_build_labeled_rows_from_query_adaptation_multiview_dataset() -> None:
    dataset = _build_multiview_dataset()

    rows = build_labeled_rows_from_query_adaptation_multiview_dataset(dataset)

    assert len(rows) == 2
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["weak_text"] == "불안해서 잠이 안 와요"
    assert rows[0]["strong_text"] == "불안해서 잠이 안 와요"
    assert rows[0]["weak_translated_text"] == "I cannot sleep because I am anxious"
    assert rows[0]["strong_translated_text"] == "I cannot sleep because I am anxious"
    assert rows[1]["query_id"] == "q2"
    assert "weak_translated_text" not in rows[1]


def test_write_query_adaptation_multiview_dataset_writes_summary_and_manifest(
    tmp_path,
) -> None:
    dataset = _build_multiview_dataset()

    outputs = write_query_adaptation_multiview_dataset(
        dataset=dataset,
        output_path=tmp_path / "train_multiview.jsonl",
        generated_at=datetime(2026, 4, 12, 13, 0, tzinfo=timezone.utc),
    )

    assert outputs.jsonl_path.exists()
    assert outputs.manifest_path.exists()
    assert outputs.summary_path.exists()

    summary = json.loads(outputs.summary_path.read_text(encoding="utf-8"))
    assert (
        summary["schema_version"]
        == QUERY_ADAPTATION_MULTIVIEW_SUMMARY_SCHEMA_VERSION
    )
    assert summary["row_count"] == 2
    assert summary["augmenter_name_counts"] == {"identity_multiview": 2}
    assert summary["label_counts"] == {"anxiety": 1, "depression": 1}
    assert summary["weak_translated_text_present_counts"] == {
        "false": 1,
        "true": 1,
    }
