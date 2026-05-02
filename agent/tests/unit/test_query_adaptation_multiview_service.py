"""Query adaptation multiview preparation tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent.src.services.training.backends.inputs.models import TrainingExampleSource
from agent.src.services.training.datasets.query_adaptation_dataset_service import (
    QueryAdaptationDataset,
    QueryAdaptationDatasetExample,
    QueryAdaptationDatasetProvenance,
)
from agent.src.services.training.datasets.query_adaptation_multiview_service import (
    IdentityQueryAdaptationMultiviewAugmenter,
    QueryAdaptationMultiviewService,
    QueryAdaptationMultiviewViews,
)


def _build_dataset() -> QueryAdaptationDataset:
    occurred_at = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    return QueryAdaptationDataset(
        examples=(
            QueryAdaptationDatasetExample(
                source_row=TrainingExampleSource(
                    query_id="q1",
                    text="불안해서 숨이 가빠요",
                    occurred_at=occurred_at,
                    translated_text="I feel anxious and short of breath",
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
                label_source="pseudo_label",
                confidence=0.93,
                margin=0.52,
            ),
        )
    )


def test_query_adaptation_multiview_service_builds_identity_views() -> None:
    dataset = _build_dataset()

    multiview = QueryAdaptationMultiviewService(
        augmenter=IdentityQueryAdaptationMultiviewAugmenter()
    ).build_dataset(dataset=dataset)

    assert multiview.count == 1
    example = multiview.examples[0]
    assert example.query_id == "q1"
    assert example.views.augmenter_name == "identity_multiview"
    assert example.views.weak_text == "불안해서 숨이 가빠요"
    assert example.views.strong_text == "불안해서 숨이 가빠요"
    assert example.views.weak_translated_text == (
        "I feel anxious and short of breath"
    )
    assert example.source_row.weak_text == "불안해서 숨이 가빠요"
    assert example.source_row.strong_text == "불안해서 숨이 가빠요"


class _CustomAugmenter:
    augmenter_name = "custom_aug_v1"

    def build_views(
        self,
        *,
        example: QueryAdaptationDatasetExample,
    ) -> QueryAdaptationMultiviewViews:
        return QueryAdaptationMultiviewViews(
            augmenter_name=self.augmenter_name,
            weak_text=f"weak::{example.source_row.text}",
            strong_text=f"strong::{example.source_row.text}",
            metadata={"strength": "asymmetric"},
        )


def test_query_adaptation_multiview_service_accepts_custom_augmenter() -> None:
    dataset = _build_dataset()

    multiview = QueryAdaptationMultiviewService(
        augmenter=_CustomAugmenter()
    ).build_dataset(dataset=dataset)

    example = multiview.examples[0]
    assert example.views.augmenter_name == "custom_aug_v1"
    assert example.views.metadata == {"strength": "asymmetric"}
    assert example.source_row.weak_text == "weak::불안해서 숨이 가빠요"
    assert example.source_row.strong_text == "strong::불안해서 숨이 가빠요"


class _MismatchedAugmenter:
    augmenter_name = "declared_aug"

    def build_views(
        self,
        *,
        example: QueryAdaptationDatasetExample,
    ) -> QueryAdaptationMultiviewViews:
        del example
        return QueryAdaptationMultiviewViews(
            augmenter_name="returned_aug",
            weak_text="weak",
            strong_text="strong",
        )


def test_query_adaptation_multiview_service_rejects_mismatched_augmenter_name() -> None:
    dataset = _build_dataset()

    with pytest.raises(ValueError, match="augmenter_name"):
        QueryAdaptationMultiviewService(
            augmenter=_MismatchedAugmenter()
        ).build_dataset(dataset=dataset)
