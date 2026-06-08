"""Accepted pseudo-label을 source-row adaptation dataset으로 조립한다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeVar

from agent.src.services.training.backends.inputs.models import (
    TrainingExampleSource,
)
from agent.src.services.training.selection.pseudo_label_service import (
    PseudoLabelSelectionResult,
)
from shared.src.domain.entities.inference.events import AnalysisEvent
from shared.src.domain.entities.training.pseudo_label_candidate import (
    SELECTION_CONTEXT_COMPATIBILITY_METADATA_KEYS,
    PseudoLabelCandidate,
    PseudoLabelSelectionContext,
)

_T = TypeVar("_T")
_MetadataScalar = str | int | float | bool
_SUPPORTED_LABEL_POLICY_NAMES = frozenset({"pseudo_label_only", "prefer_manual_label"})


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
class QueryAdaptationDatasetProvenance:
    """query adaptation 예시의 canonical provenance."""

    locale: str
    source_type: str
    model_revision: str
    translated_text_present: bool
    candidate_id: str
    evidence_ref: str | None = None
    selection_context: PseudoLabelSelectionContext | None = None
    candidate_metadata: dict[str, _MetadataScalar] = field(default_factory=dict)
    source_metadata: dict[str, _MetadataScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.locale.strip():
            raise ValueError("locale must not be empty.")
        if not self.source_type.strip():
            raise ValueError("source_type must not be empty.")
        if not self.model_revision.strip():
            raise ValueError("model_revision must not be empty.")
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty.")


@dataclass(slots=True)
class QueryAdaptationDatasetExample:
    """query-domain 적응에 넘길 단일 raw-text pseudo-labeled row."""

    source_row: TrainingExampleSource
    label: str
    provenance: QueryAdaptationDatasetProvenance
    label_source: str = "pseudo_label"
    confidence: float = 0.0
    margin: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_row.query_id.strip():
            raise ValueError("source_row.query_id must not be empty.")
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        if not self.label_source.strip():
            raise ValueError("label_source must not be empty.")

    @property
    def query_id(self) -> str:
        return self.source_row.query_id


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
    """selection 결과를 source-row adaptation 입력셋으로 조립한다."""

    config: QueryAdaptationDatasetConfig = field(
        default_factory=QueryAdaptationDatasetConfig
    )

    def build_dataset(
        self,
        *,
        selection_result: PseudoLabelSelectionResult,
        source_rows: tuple[TrainingExampleSource, ...] | list[TrainingExampleSource],
        analysis_events: tuple[AnalysisEvent, ...] | list[AnalysisEvent] | None = None,
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
        source_row_by_query_id = _index_unique(
            items=source_rows,
            key_fn=lambda row: row.query_id,
            item_name="TrainingExampleSource",
        )
        analysis_event_by_query_id = _index_unique(
            items=() if analysis_events is None else analysis_events,
            key_fn=lambda analysis_event: analysis_event.query_id,
            item_name="AnalysisEvent",
        )

        dataset_examples: list[QueryAdaptationDatasetExample] = []
        for candidate in selection_result.accepted_candidates:
            source_row = source_row_by_query_id.get(candidate.source_event_ref)
            if source_row is None:
                raise ValueError(
                    "Missing TrainingExampleSource for accepted candidate: "
                    f"{candidate.source_event_ref}."
                )
            analysis_event = analysis_event_by_query_id.get(candidate.source_event_ref)
            label, label_source = self._resolve_label(
                candidate=candidate,
                manual_label_by_query_id=manual_label_by_query_id,
            )
            dataset_examples.append(
                QueryAdaptationDatasetExample(
                    source_row=_merge_analysis_translation(
                        source_row=source_row,
                        analysis_event=analysis_event,
                    ),
                    label=label,
                    provenance=_build_dataset_provenance(
                        source_row=source_row,
                        candidate=candidate,
                        analysis_event=analysis_event,
                    ),
                    label_source=label_source,
                    confidence=candidate.confidence,
                    margin=candidate.margin,
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


def _build_dataset_provenance(
    *,
    source_row: TrainingExampleSource,
    candidate: PseudoLabelCandidate,
    analysis_event: AnalysisEvent | None,
) -> QueryAdaptationDatasetProvenance:
    return QueryAdaptationDatasetProvenance(
        locale=str(candidate.metadata.get("locale", "unknown")),
        source_type=str(candidate.metadata.get("source_type", "unknown")),
        model_revision=str(candidate.metadata.get("model_revision", "unknown")),
        translated_text_present=(
            False
            if analysis_event is None
            else analysis_event.translated_text is not None
        ),
        candidate_id=str(candidate.candidate_id),
        evidence_ref=(
            None if candidate.evidence_ref is None else str(candidate.evidence_ref)
        ),
        selection_context=_require_selection_context(candidate),
        candidate_metadata={
            str(key): _coerce_metadata_scalar(value)
            for key, value in candidate.metadata.items()
            if str(key) not in SELECTION_CONTEXT_COMPATIBILITY_METADATA_KEYS
        },
        source_metadata={
            "source_query_id": source_row.query_id,
            "source_has_weak_view": source_row.weak_text is not None,
            "source_has_strong_view": source_row.strong_text is not None,
        },
    )


def _merge_analysis_translation(
    *,
    source_row: TrainingExampleSource,
    analysis_event: AnalysisEvent | None,
) -> TrainingExampleSource:
    if source_row.translated_text is not None or analysis_event is None:
        return source_row
    return TrainingExampleSource(
        query_id=source_row.query_id,
        text=source_row.text,
        occurred_at=source_row.occurred_at,
        translated_text=analysis_event.translated_text,
        weak_text=source_row.weak_text,
        strong_text=source_row.strong_text,
        weak_translated_text=source_row.weak_translated_text,
        strong_translated_text=source_row.strong_translated_text,
    )


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


def _require_selection_context(
    candidate: PseudoLabelCandidate,
) -> PseudoLabelSelectionContext:
    if candidate.selection_context is None:
        raise ValueError(
            "PseudoLabelCandidate.selection_context is required for dataset "
            f"provenance: {candidate.candidate_id}."
        )
    return candidate.selection_context


def _coerce_metadata_scalar(value: object) -> _MetadataScalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    return str(value)
