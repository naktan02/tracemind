"""Query buffer selection 결과를 summary/trace 진단 shape로 정리한다."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
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
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PseudoLabelEvidence,
)

QUERY_BUFFER_SELECTION_TRACE_SCHEMA_VERSION = "query_buffer_selection_trace.v1"
QUERY_BUFFER_SELECTION_SUMMARY_SCHEMA_VERSION = "query_buffer_selection_summary.v1"

_MetadataScalar = str | int | float | bool
_T = TypeVar("_T")


@dataclass(slots=True)
class QueryBufferSelectionDiagnostics:
    """Selection diagnostics bundle."""

    summary: dict[str, object]
    trace_rows: tuple[dict[str, object], ...]


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

        trace_rows: list[dict[str, object]] = []
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
            final_accepted = selection_context.final_accepted
            confidence_threshold = selection_context.confidence_threshold
            margin_threshold = selection_context.margin_threshold
            max_examples = selection_context.max_examples
            pseudo_label_algorithm_name = selection_context.pseudo_label_algorithm_name
            evidence_backend_name = selection_context.evidence_backend_name

            trace_rows.append(
                _build_trace_row(
                    candidate=candidate,
                    record=record,
                    evidence=evidence,
                    selection_context=selection_context,
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
            summary={
                "schema_version": QUERY_BUFFER_SELECTION_SUMMARY_SCHEMA_VERSION,
                "total_candidates": selection_result.total_count,
                "final_accepted_count": selection_result.accepted_count,
                "accepted_ratio": selection_result.accepted_ratio,
                "stage_counts": _counter_to_mapping(stage_counts),
                "pseudo_label_counts": _counter_to_mapping(pseudo_label_counts),
                "accepted_label_counts": _counter_to_mapping(accepted_label_counts),
                "locale_counts": _counter_to_mapping(locale_counts),
                "source_type_counts": _counter_to_mapping(source_type_counts),
                "model_revision_counts": _counter_to_mapping(model_revision_counts),
                "confidence_kind_counts": _counter_to_mapping(confidence_kind_counts),
                "evidence_backend_name_counts": _counter_to_mapping(
                    evidence_backend_name_counts
                ),
                "evidence_view_kind_counts": _counter_to_mapping(
                    evidence_view_kind_counts
                ),
                "pseudo_label_algorithm_name_counts": _counter_to_mapping(
                    pseudo_label_algorithm_name_counts
                ),
                "confidence_threshold_counts": _counter_to_mapping(
                    confidence_threshold_counts
                ),
                "margin_threshold_counts": _counter_to_mapping(margin_threshold_counts),
                "max_examples_counts": _counter_to_mapping(max_examples_counts),
                "task_id_counts": _counter_to_mapping(task_id_counts),
                "round_id_counts": _counter_to_mapping(round_id_counts),
                "confidence_stats": _summarize_scalar_values(confidence_values),
                "margin_stats": _summarize_scalar_values(margin_values),
            },
            trace_rows=tuple(trace_rows),
        )


def _build_trace_row(
    *,
    candidate: PseudoLabelCandidate,
    record: QueryBufferRecord,
    evidence: PseudoLabelEvidence | None,
    selection_context: PseudoLabelSelectionContext,
) -> dict[str, object]:
    max_examples = selection_context.max_examples
    normalized_max_examples = (
        None if max_examples is not None and max_examples < 0 else max_examples
    )

    return {
        "schema_version": QUERY_BUFFER_SELECTION_TRACE_SCHEMA_VERSION,
        "query_id": candidate.source_event_ref,
        "occurred_at": record.occurred_at.isoformat(),
        "locale": record.locale,
        "source_type": record.source_type,
        "model_revision": record.model_revision,
        "query_buffer_label": record.predicted_label,
        "query_buffer_confidence": record.confidence,
        "query_buffer_margin": record.margin,
        "query_buffer_runner_up_label": record.runner_up_label,
        "query_buffer_runner_up_score": record.runner_up_score,
        "query_buffer_confidence_kind": record.confidence_kind,
        "pseudo_label": candidate.label,
        "confidence": candidate.confidence,
        "margin": candidate.margin,
        "runner_up_label": candidate.runner_up_label,
        "runner_up_score": candidate.runner_up_score,
        "confidence_kind": (
            None
            if candidate.confidence_kind is None
            else str(candidate.confidence_kind)
        ),
        "sample_weight": candidate.sample_weight,
        "threshold_accepted": selection_context.threshold_accepted,
        "selected_by_cap": selection_context.selected_by_cap,
        "final_accepted": selection_context.final_accepted,
        "selection_stage": selection_context.selection_stage.value,
        "pre_cap_rank": selection_context.pre_cap_rank,
        "confidence_threshold": selection_context.confidence_threshold,
        "margin_threshold": selection_context.margin_threshold,
        "max_examples": normalized_max_examples,
        "task_id": None if candidate.task_id is None else str(candidate.task_id),
        "round_id": None if candidate.round_id is None else str(candidate.round_id),
        "pseudo_label_algorithm_name": selection_context.pseudo_label_algorithm_name,
        "evidence_backend_name": selection_context.evidence_backend_name,
        "evidence_view_kind": "unknown" if evidence is None else evidence.view_kind,
        "evidence_ref": (
            None if candidate.evidence_ref is None else str(candidate.evidence_ref)
        ),
        "top1_label": None if evidence is None else evidence.top1_label,
        "top1_score": None if evidence is None else evidence.top1_score,
        "top2_label": None if evidence is None else evidence.top2_label,
        "top2_score": None if evidence is None else evidence.top2_score,
        "raw_scores": {} if evidence is None else _float_mapping(evidence.raw_scores),
        "label_distribution": (
            None
            if evidence is None or evidence.label_distribution is None
            else _float_mapping(evidence.label_distribution)
        ),
        "candidate_metadata": _metadata_mapping(
            candidate.metadata,
            excluded_keys=SELECTION_CONTEXT_COMPATIBILITY_METADATA_KEYS,
        ),
        "query_buffer_metadata": _metadata_mapping(record.metadata),
    }


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


def _stringify_count_key(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def _counter_to_mapping(counter: Counter[str]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _float_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
    }


def _metadata_mapping(
    values: Mapping[str, object],
    *,
    excluded_keys: Collection[str] = frozenset(),
) -> dict[str, _MetadataScalar]:
    return {
        str(key): _coerce_metadata_scalar(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        if str(key) not in excluded_keys
    }


def _summarize_scalar_values(
    values: list[float],
) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": fsum(values) / len(values),
    }
