"""Query buffer selection 결과를 summary/trace 진단 shape로 정리한다."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from math import fsum
from typing import TypeVar

from agent.src.infrastructure.repositories.query_buffer_repository import (
    QueryBufferRecord,
)
from agent.src.services.training.selection.pseudo_label_service import (
    PseudoLabelSelectionResult,
)
from shared.src.domain.entities.training.pseudo_label_candidate import (
    SELECTION_CONTEXT_COMPATIBILITY_METADATA_KEYS,
    PseudoLabelCandidate,
    PseudoLabelSelectionContext,
)

QUERY_BUFFER_SELECTION_TRACE_SCHEMA_VERSION = "query_buffer_selection_trace.v1"
QUERY_BUFFER_SELECTION_SUMMARY_SCHEMA_VERSION = "query_buffer_selection_summary.v1"

_MetadataScalar = str | int | float | bool
_T = TypeVar("_T")


@dataclass(slots=True)
class QueryBufferSelectionScalarStats:
    """Selection scalar 값의 요약 통계."""

    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None

    def to_mapping(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
        }


@dataclass(slots=True)
class QueryBufferSelectionTraceRow:
    """Selection stage 한 줄 진단 정보."""

    schema_version: str
    query_id: str
    occurred_at: datetime
    locale: str
    source_type: str
    model_revision: str
    query_buffer_label: str
    query_buffer_confidence: float
    query_buffer_margin: float
    query_buffer_runner_up_label: str | None
    query_buffer_runner_up_score: float | None
    query_buffer_confidence_kind: str
    pseudo_label: str
    confidence: float
    margin: float
    runner_up_label: str | None
    runner_up_score: float | None
    confidence_kind: str | None
    sample_weight: float
    threshold_accepted: bool
    selected_by_cap: bool
    final_accepted: bool
    selection_stage: str
    pre_cap_rank: int | None
    confidence_threshold: float | None
    margin_threshold: float | None
    max_examples: int | None
    task_id: str | None
    round_id: str | None
    pseudo_label_algorithm_name: str | None
    evidence_backend_name: str | None
    evidence_view_kind: str
    evidence_ref: str | None
    top1_label: str | None
    top1_score: float | None
    top2_label: str | None
    top2_score: float | None
    raw_scores: dict[str, float]
    label_distribution: dict[str, float] | None
    candidate_metadata: dict[str, _MetadataScalar]
    query_buffer_metadata: dict[str, _MetadataScalar]

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_id": self.query_id,
            "occurred_at": self.occurred_at.isoformat(),
            "locale": self.locale,
            "source_type": self.source_type,
            "model_revision": self.model_revision,
            "query_buffer_label": self.query_buffer_label,
            "query_buffer_confidence": self.query_buffer_confidence,
            "query_buffer_margin": self.query_buffer_margin,
            "query_buffer_runner_up_label": self.query_buffer_runner_up_label,
            "query_buffer_runner_up_score": self.query_buffer_runner_up_score,
            "query_buffer_confidence_kind": self.query_buffer_confidence_kind,
            "pseudo_label": self.pseudo_label,
            "confidence": self.confidence,
            "margin": self.margin,
            "runner_up_label": self.runner_up_label,
            "runner_up_score": self.runner_up_score,
            "confidence_kind": self.confidence_kind,
            "sample_weight": self.sample_weight,
            "threshold_accepted": self.threshold_accepted,
            "selected_by_cap": self.selected_by_cap,
            "final_accepted": self.final_accepted,
            "selection_stage": self.selection_stage,
            "pre_cap_rank": self.pre_cap_rank,
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "max_examples": self.max_examples,
            "task_id": self.task_id,
            "round_id": self.round_id,
            "pseudo_label_algorithm_name": self.pseudo_label_algorithm_name,
            "evidence_backend_name": self.evidence_backend_name,
            "evidence_view_kind": self.evidence_view_kind,
            "evidence_ref": self.evidence_ref,
            "top1_label": self.top1_label,
            "top1_score": self.top1_score,
            "top2_label": self.top2_label,
            "top2_score": self.top2_score,
            "raw_scores": dict(sorted(self.raw_scores.items())),
            "label_distribution": (
                None
                if self.label_distribution is None
                else dict(sorted(self.label_distribution.items()))
            ),
            "candidate_metadata": dict(sorted(self.candidate_metadata.items())),
            "query_buffer_metadata": dict(sorted(self.query_buffer_metadata.items())),
        }


@dataclass(slots=True)
class QueryBufferSelectionSummary:
    """Selection 결과 전체 요약."""

    schema_version: str
    total_candidates: int
    final_accepted_count: int
    accepted_ratio: float
    stage_counts: dict[str, int]
    pseudo_label_counts: dict[str, int]
    accepted_label_counts: dict[str, int]
    locale_counts: dict[str, int]
    source_type_counts: dict[str, int]
    model_revision_counts: dict[str, int]
    confidence_kind_counts: dict[str, int]
    evidence_backend_name_counts: dict[str, int]
    evidence_view_kind_counts: dict[str, int]
    pseudo_label_algorithm_name_counts: dict[str, int]
    confidence_threshold_counts: dict[str, int]
    margin_threshold_counts: dict[str, int]
    max_examples_counts: dict[str, int]
    task_id_counts: dict[str, int]
    round_id_counts: dict[str, int]
    confidence_stats: QueryBufferSelectionScalarStats
    margin_stats: QueryBufferSelectionScalarStats

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "total_candidates": self.total_candidates,
            "final_accepted_count": self.final_accepted_count,
            "accepted_ratio": self.accepted_ratio,
            "stage_counts": dict(sorted(self.stage_counts.items())),
            "pseudo_label_counts": dict(sorted(self.pseudo_label_counts.items())),
            "accepted_label_counts": dict(sorted(self.accepted_label_counts.items())),
            "locale_counts": dict(sorted(self.locale_counts.items())),
            "source_type_counts": dict(sorted(self.source_type_counts.items())),
            "model_revision_counts": dict(sorted(self.model_revision_counts.items())),
            "confidence_kind_counts": dict(sorted(self.confidence_kind_counts.items())),
            "evidence_backend_name_counts": dict(
                sorted(self.evidence_backend_name_counts.items())
            ),
            "evidence_view_kind_counts": dict(
                sorted(self.evidence_view_kind_counts.items())
            ),
            "pseudo_label_algorithm_name_counts": dict(
                sorted(self.pseudo_label_algorithm_name_counts.items())
            ),
            "confidence_threshold_counts": dict(
                sorted(self.confidence_threshold_counts.items())
            ),
            "margin_threshold_counts": dict(
                sorted(self.margin_threshold_counts.items())
            ),
            "max_examples_counts": dict(sorted(self.max_examples_counts.items())),
            "task_id_counts": dict(sorted(self.task_id_counts.items())),
            "round_id_counts": dict(sorted(self.round_id_counts.items())),
            "confidence_stats": self.confidence_stats.to_mapping(),
            "margin_stats": self.margin_stats.to_mapping(),
        }


@dataclass(slots=True)
class QueryBufferSelectionDiagnostics:
    """Selection diagnostics bundle."""

    summary: QueryBufferSelectionSummary
    trace_rows: tuple[QueryBufferSelectionTraceRow, ...]


@dataclass(slots=True)
class QueryBufferSelectionDiagnosticsService:
    """Query buffer selection 결과를 관측 가능한 진단 shape로 정리한다."""

    def build(
        self,
        *,
        selection_result: PseudoLabelSelectionResult,
        records: tuple[QueryBufferRecord, ...] | list[QueryBufferRecord],
    ) -> QueryBufferSelectionDiagnostics:
        record_by_query_id = _index_unique(
            items=records,
            key_fn=lambda record: record.query_id,
            item_name="QueryBufferRecord",
        )
        evidence_by_query_id = _index_unique(
            items=selection_result.evidences,
            key_fn=lambda evidence: evidence.source_event_ref,
            item_name="PseudoLabelEvidence",
        )

        trace_rows: list[QueryBufferSelectionTraceRow] = []
        stage_counts: Counter[str] = Counter()
        pseudo_label_counts: Counter[str] = Counter()
        accepted_label_counts: Counter[str] = Counter()
        locale_counts: Counter[str] = Counter()
        source_type_counts: Counter[str] = Counter()
        model_revision_counts: Counter[str] = Counter()
        confidence_kind_counts: Counter[str] = Counter()
        evidence_backend_name_counts: Counter[str] = Counter()
        evidence_view_kind_counts: Counter[str] = Counter()
        pseudo_label_algorithm_name_counts: Counter[str] = Counter()
        confidence_threshold_counts: Counter[str] = Counter()
        margin_threshold_counts: Counter[str] = Counter()
        max_examples_counts: Counter[str] = Counter()
        task_id_counts: Counter[str] = Counter()
        round_id_counts: Counter[str] = Counter()
        confidence_values: list[float] = []
        margin_values: list[float] = []

        for candidate in selection_result.candidates:
            query_id = candidate.source_event_ref
            record = record_by_query_id.get(query_id)
            if record is None:
                raise ValueError(
                    f"Missing QueryBufferRecord for selection diagnostics: {query_id}."
                )
            evidence = evidence_by_query_id.get(query_id)

            selection_context = _require_selection_context(candidate)
            stage = selection_context.selection_stage.value
            threshold_accepted = selection_context.threshold_accepted
            selected_by_cap = selection_context.selected_by_cap
            final_accepted = selection_context.final_accepted
            pre_cap_rank = selection_context.pre_cap_rank
            confidence_threshold = selection_context.confidence_threshold
            margin_threshold = selection_context.margin_threshold
            max_examples = selection_context.max_examples
            pseudo_label_algorithm_name = selection_context.pseudo_label_algorithm_name
            evidence_backend_name = selection_context.evidence_backend_name

            trace_rows.append(
                QueryBufferSelectionTraceRow(
                    schema_version=QUERY_BUFFER_SELECTION_TRACE_SCHEMA_VERSION,
                    query_id=query_id,
                    occurred_at=record.occurred_at,
                    locale=record.locale,
                    source_type=record.source_type,
                    model_revision=record.model_revision,
                    query_buffer_label=record.predicted_label,
                    query_buffer_confidence=record.confidence,
                    query_buffer_margin=record.margin,
                    query_buffer_runner_up_label=record.runner_up_label,
                    query_buffer_runner_up_score=record.runner_up_score,
                    query_buffer_confidence_kind=record.confidence_kind,
                    pseudo_label=candidate.label,
                    confidence=candidate.confidence,
                    margin=candidate.margin,
                    runner_up_label=candidate.runner_up_label,
                    runner_up_score=candidate.runner_up_score,
                    confidence_kind=(
                        None
                        if candidate.confidence_kind is None
                        else str(candidate.confidence_kind)
                    ),
                    sample_weight=candidate.sample_weight,
                    threshold_accepted=threshold_accepted,
                    selected_by_cap=selected_by_cap,
                    final_accepted=final_accepted,
                    selection_stage=stage,
                    pre_cap_rank=pre_cap_rank,
                    confidence_threshold=confidence_threshold,
                    margin_threshold=margin_threshold,
                    max_examples=(
                        None
                        if max_examples is not None and max_examples < 0
                        else max_examples
                    ),
                    task_id=(
                        None if candidate.task_id is None else str(candidate.task_id)
                    ),
                    round_id=(
                        None if candidate.round_id is None else str(candidate.round_id)
                    ),
                    pseudo_label_algorithm_name=pseudo_label_algorithm_name,
                    evidence_backend_name=evidence_backend_name,
                    evidence_view_kind=(
                        "unknown" if evidence is None else evidence.view_kind
                    ),
                    evidence_ref=(
                        None
                        if candidate.evidence_ref is None
                        else str(candidate.evidence_ref)
                    ),
                    top1_label=None if evidence is None else evidence.top1_label,
                    top1_score=None if evidence is None else evidence.top1_score,
                    top2_label=None if evidence is None else evidence.top2_label,
                    top2_score=None if evidence is None else evidence.top2_score,
                    raw_scores=(
                        {}
                        if evidence is None
                        else dict(sorted(evidence.raw_scores.items()))
                    ),
                    label_distribution=(
                        None
                        if evidence is None or evidence.label_distribution is None
                        else dict(sorted(evidence.label_distribution.items()))
                    ),
                    candidate_metadata={
                        str(key): _coerce_metadata_scalar(value)
                        for key, value in candidate.metadata.items()
                        if str(key) not in SELECTION_CONTEXT_COMPATIBILITY_METADATA_KEYS
                    },
                    query_buffer_metadata={
                        str(key): _coerce_metadata_scalar(value)
                        for key, value in record.metadata.items()
                    },
                )
            )

            stage_counts[stage] += 1
            pseudo_label_counts[candidate.label] += 1
            if final_accepted:
                accepted_label_counts[candidate.label] += 1
            locale_counts[record.locale] += 1
            source_type_counts[record.source_type] += 1
            model_revision_counts[record.model_revision] += 1
            confidence_kind_counts[
                (
                    record.confidence_kind
                    if candidate.confidence_kind is None
                    else str(candidate.confidence_kind)
                )
            ] += 1
            evidence_backend_name_counts[
                "unknown" if evidence_backend_name is None else evidence_backend_name
            ] += 1
            evidence_view_kind_counts[
                "unknown" if evidence is None else evidence.view_kind
            ] += 1
            pseudo_label_algorithm_name_counts[
                "unknown"
                if pseudo_label_algorithm_name is None
                else pseudo_label_algorithm_name
            ] += 1
            confidence_threshold_counts[_stringify_count_key(confidence_threshold)] += 1
            margin_threshold_counts[_stringify_count_key(margin_threshold)] += 1
            max_examples_counts[_stringify_count_key(max_examples)] += 1
            task_id_counts[_stringify_count_key(candidate.task_id)] += 1
            round_id_counts[_stringify_count_key(candidate.round_id)] += 1
            confidence_values.append(float(candidate.confidence))
            margin_values.append(float(candidate.margin))

        return QueryBufferSelectionDiagnostics(
            summary=QueryBufferSelectionSummary(
                schema_version=QUERY_BUFFER_SELECTION_SUMMARY_SCHEMA_VERSION,
                total_candidates=selection_result.total_count,
                final_accepted_count=selection_result.accepted_count,
                accepted_ratio=selection_result.accepted_ratio,
                stage_counts=dict(stage_counts),
                pseudo_label_counts=dict(pseudo_label_counts),
                accepted_label_counts=dict(accepted_label_counts),
                locale_counts=dict(locale_counts),
                source_type_counts=dict(source_type_counts),
                model_revision_counts=dict(model_revision_counts),
                confidence_kind_counts=dict(confidence_kind_counts),
                evidence_backend_name_counts=dict(evidence_backend_name_counts),
                evidence_view_kind_counts=dict(evidence_view_kind_counts),
                pseudo_label_algorithm_name_counts=dict(
                    pseudo_label_algorithm_name_counts
                ),
                confidence_threshold_counts=dict(confidence_threshold_counts),
                margin_threshold_counts=dict(margin_threshold_counts),
                max_examples_counts=dict(max_examples_counts),
                task_id_counts=dict(task_id_counts),
                round_id_counts=dict(round_id_counts),
                confidence_stats=_summarize_scalar_values(confidence_values),
                margin_stats=_summarize_scalar_values(margin_values),
            ),
            trace_rows=tuple(trace_rows),
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
            "PseudoLabelCandidate.selection_context is required for diagnostics: "
            f"{candidate.candidate_id}."
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
    raise TypeError(
        f"Selection diagnostics metadata must be a scalar value, got {type(value)!r}."
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def _stringify_count_key(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def _summarize_scalar_values(
    values: list[float],
) -> QueryBufferSelectionScalarStats:
    if not values:
        return QueryBufferSelectionScalarStats(
            count=0,
            minimum=None,
            maximum=None,
            mean=None,
        )
    return QueryBufferSelectionScalarStats(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=fsum(values) / len(values),
    )
