from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.src.services.training.backends.inputs.models import (
    TrainingExampleSource,
)
from agent.src.services.training.datasets.query_adaptation_dataset_service import (
    QueryAdaptationDataset,
    QueryAdaptationDatasetExample,
    QueryAdaptationDatasetProvenance,
)
from scripts.experiments.query_lora_ssl.io.query_adaptation import (
    QUERY_ADAPTATION_EXPORT_SCHEMA_VERSION,
    QUERY_ADAPTATION_SUMMARY_SCHEMA_VERSION,
    build_labeled_rows_from_query_adaptation_dataset,
    write_query_adaptation_lora_dataset,
)
from shared.src.contracts.labeled_query_row_contracts import load_labeled_query_rows
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelSelectionContext,
    PseudoLabelSelectionStage,
)


def _build_dataset() -> QueryAdaptationDataset:
    occurred_at = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    return QueryAdaptationDataset(
        examples=(
            QueryAdaptationDatasetExample(
                source_row=TrainingExampleSource(
                    query_id="q1",
                    text="불안이 심해요",
                    occurred_at=occurred_at,
                    translated_text="I feel anxious",
                ),
                label="anxiety",
                provenance=QueryAdaptationDatasetProvenance(
                    locale="ko-KR",
                    source_type="user_message",
                    model_revision="rev_001",
                    selection_confidence_kind="prototype_similarity_top1",
                    translated_text_present=True,
                    candidate_id="round_1:q1",
                    selection_context=PseudoLabelSelectionContext(
                        threshold_accepted=True,
                        selected_by_cap=True,
                        final_accepted=True,
                        selection_stage=PseudoLabelSelectionStage.ACCEPTED,
                    ),
                    query_buffer_metadata={"was_translated": True},
                ),
                label_source="pseudo_label",
                confidence=0.91,
                margin=0.62,
            ),
            QueryAdaptationDatasetExample(
                source_row=TrainingExampleSource(
                    query_id="q2",
                    text="너무 우울해요",
                    occurred_at=occurred_at,
                ),
                label="depression",
                provenance=QueryAdaptationDatasetProvenance(
                    locale="ko-KR",
                    source_type="user_message",
                    model_revision="rev_001",
                    selection_confidence_kind="prototype_similarity_top1",
                    translated_text_present=False,
                    candidate_id="round_1:q2",
                ),
                label_source="manual_label",
                confidence=1.0,
                margin=1.0,
            ),
        )
    )


def test_build_labeled_rows_from_query_adaptation_dataset_maps_fields() -> None:
    dataset = _build_dataset()

    rows = build_labeled_rows_from_query_adaptation_dataset(dataset)

    assert len(rows) == 2
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["text"] == "불안이 심해요"
    assert rows[0]["raw_label_scheme"] == "pseudo_label"
    assert rows[0]["raw_label"] == "anxiety"
    assert rows[0]["mapped_label_4"] == "anxiety"
    assert rows[0]["locale"] == "ko-KR"
    assert rows[0]["annotation_source"] == "query_adaptation_dataset"
    assert rows[0]["approved_by"] is None
    assert rows[1]["approved_by"] == "manual_override"


def test_write_query_adaptation_lora_dataset_writes_jsonl_and_manifest(
    tmp_path: Path,
) -> None:
    dataset = _build_dataset()

    outputs = write_query_adaptation_lora_dataset(
        dataset=dataset,
        output_path=tmp_path / "query_adaptation.jsonl",
        generated_at=datetime(2026, 4, 12, 13, 0, tzinfo=timezone.utc),
    )

    jsonl_path = outputs.jsonl_path
    manifest_path = outputs.manifest_path
    summary_path = outputs.summary_path

    assert jsonl_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()

    rows = load_labeled_query_rows(jsonl_path)
    assert [row["query_id"] for row in rows] == ["q1", "q2"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == QUERY_ADAPTATION_EXPORT_SCHEMA_VERSION
    assert manifest["row_count"] == 2
    assert manifest["label_counts"] == {
        "anxiety": 1,
        "depression": 1,
    }
    assert manifest["label_source_counts"] == {
        "manual_label": 1,
        "pseudo_label": 1,
    }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == QUERY_ADAPTATION_SUMMARY_SCHEMA_VERSION
    assert summary["row_count"] == 2
    assert summary["locale_counts"] == {"ko-KR": 2}
    assert summary["selection_stage_counts"] == {"accepted": 1}
    assert summary["translated_text_present_counts"] == {"false": 1, "true": 1}
