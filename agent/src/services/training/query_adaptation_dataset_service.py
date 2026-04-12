"""Accepted pseudo-label을 raw-text adaptation dataset으로 조립한다."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRecord,
)
from agent.src.services.training.input_backends.models import (
    TrainingExampleSource,
)
from agent.src.services.training.pseudo_label_service import (
    PseudoLabelSelectionResult,
)
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)
from shared.src.domain.entities.inference.events import ScoredEvent

_T = TypeVar("_T")
_SUPPORTED_LABEL_POLICY_NAMES = frozenset(
    {"pseudo_label_only", "prefer_manual_label"}
)


@dataclass(slots=True)
class QueryAdaptationDatasetConfig:
    """adaptation dataset label source 정책."""

    label_policy_name: str = "pseudo_label_only"

    def __post_init__(self) -> None:
        if self.label_policy_name not in _SUPPORTED_LABEL_POLICY_NAMES:
            raise ValueError(
                "Unsupported query adaptation dataset label policy: "
                f"{self.label_policy_name}."
            )


@dataclass(slots=True)
class QueryAdaptationDatasetExample:
    """LoRA/query 적응에 넘길 단일 raw-text pseudo-labeled row."""

    query_id: str
    source_row: TrainingExampleSource
    label: str
    label_source: str = "pseudo_label"
    confidence: float = 0.0
    margin: float = 0.0
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass(slots=True)
class QueryAdaptationDataset:
    """query-domain 적응용 raw-text dataset 묶음."""

    examples: tuple[QueryAdaptationDatasetExample, ...]

    @property
    def count(self) -> int:
        return len(self.examples)

    @property
    def source_rows(self) -> tuple[TrainingExampleSource, ...]:
        return tuple(example.source_row for example in self.examples)

    @property
    def label_by_query_id(self) -> dict[str, str]:
        return {example.query_id: example.label for example in self.examples}


@dataclass(slots=True)
class QueryAdaptationDatasetService:
    """selection 결과를 future LoRA/scaffold 입력셋으로 조립한다."""

    config: QueryAdaptationDatasetConfig = field(
        default_factory=QueryAdaptationDatasetConfig
    )

    def build_dataset(
        self,
        *,
        selection_result: PseudoLabelSelectionResult,
        records: tuple[QueryBufferRecord, ...] | list[QueryBufferRecord],
        scored_events: tuple[ScoredEvent, ...] | list[ScoredEvent] | None = None,
        manual_label_by_query_id: Mapping[str, str] | None = None,
    ) -> QueryAdaptationDataset:
        if (
            manual_label_by_query_id
            and self.config.label_policy_name == "pseudo_label_only"
        ):
            raise ValueError(
                "manual_label_by_query_id requires "
                "label_policy_name='prefer_manual_label'."
            )
        record_by_query_id = _index_unique(
            items=records,
            key_fn=lambda record: record.query_id,
            item_name="QueryBufferRecord",
        )
        scored_event_by_query_id = _index_unique(
            items=() if scored_events is None else scored_events,
            key_fn=lambda scored_event: scored_event.query_id,
            item_name="ScoredEvent",
        )

        dataset_examples: list[QueryAdaptationDatasetExample] = []
        for candidate in selection_result.accepted_candidates:
            record = record_by_query_id.get(candidate.source_event_ref)
            if record is None:
                raise ValueError(
                    "Missing QueryBufferRecord for accepted candidate: "
                    f"{candidate.source_event_ref}."
                )
            scored_event = scored_event_by_query_id.get(candidate.source_event_ref)
            label, label_source = self._resolve_label(
                candidate=candidate,
                manual_label_by_query_id=manual_label_by_query_id,
            )
            dataset_examples.append(
                QueryAdaptationDatasetExample(
                    query_id=record.query_id,
                    source_row=TrainingExampleSource(
                        query_id=record.query_id,
                        text=record.raw_text,
                        occurred_at=record.occurred_at,
                        translated_text=(
                            None
                            if scored_event is None
                            else scored_event.translated_text
                        ),
                    ),
                    label=label,
                    label_source=label_source,
                    confidence=candidate.confidence,
                    margin=candidate.margin,
                    metadata=_build_dataset_metadata(
                        record=record,
                        candidate=candidate,
                        scored_event=scored_event,
                    ),
                )
            )
        return QueryAdaptationDataset(examples=tuple(dataset_examples))

    def _resolve_label(
        self,
        *,
        candidate: PseudoLabelCandidate,
        manual_label_by_query_id: Mapping[str, str] | None,
    ) -> tuple[str, str]:
        if self.config.label_policy_name == "prefer_manual_label":
            manual_label = (
                None
                if manual_label_by_query_id is None
                else manual_label_by_query_id.get(candidate.source_event_ref)
            )
            if manual_label is not None:
                return str(manual_label), "manual_label"
        return candidate.label, "pseudo_label"


def _build_dataset_metadata(
    *,
    record: QueryBufferRecord,
    candidate: PseudoLabelCandidate,
    scored_event: ScoredEvent | None,
) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "candidate_id": str(candidate.candidate_id),
        "source_event_ref": str(candidate.source_event_ref),
        "query_buffer_model_revision": record.model_revision,
        "query_buffer_locale": record.locale,
        "query_buffer_source_type": record.source_type,
        "selection_confidence_kind": (
            "unknown"
            if candidate.confidence_kind is None
            else str(candidate.confidence_kind)
        ),
        "translated_text_present": (
            False if scored_event is None else scored_event.translated_text is not None
        ),
    }
    if candidate.evidence_ref is not None:
        metadata["evidence_ref"] = str(candidate.evidence_ref)
    for key, value in candidate.metadata.items():
        metadata[f"candidate.{key}"] = _coerce_metadata_scalar(value)
    for key, value in record.metadata.items():
        metadata[f"query_buffer.{key}"] = _coerce_metadata_scalar(value)
    return metadata


def _index_unique(
    *,
    items: tuple[_T, ...] | list[_T],
    key_fn: Callable[[_T], object],
    item_name: str,
) -> dict[str, _T]:
    indexed: dict[str, _T] = {}
    for item in items:
        key = str(key_fn(item))
        if key in indexed:
            raise ValueError(f"Duplicate {item_name} key: {key}.")
        indexed[key] = item
    return indexed


def _coerce_metadata_scalar(value: object) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    return str(value)


__all__ = [
    "QueryAdaptationDataset",
    "QueryAdaptationDatasetConfig",
    "QueryAdaptationDatasetExample",
    "QueryAdaptationDatasetService",
]
