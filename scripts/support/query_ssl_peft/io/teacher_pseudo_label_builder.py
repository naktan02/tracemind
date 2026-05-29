"""Teacher prediction을 pseudo-label payload로 변환한다."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from methods.ssl.hooks.registry import build_pseudo_label_selection_hook
from methods.ssl.hooks.teacher import TeacherPrediction
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.domain.entities.training.pseudo_label_evidence import (
    PSEUDO_LABEL_EVIDENCE_V1,
    PseudoLabelEvidence,
)

TEACHER_PREDICTION_TRACE_SCHEMA_VERSION = "teacher_prediction_trace.v1"
TEACHER_PREDICTION_SUMMARY_SCHEMA_VERSION = "teacher_prediction_summary.v1"


@dataclass(frozen=True, slots=True)
class TeacherPseudoLabelExport:
    """Teacher prediction을 pseudo-label rows와 diagnostics payload로 변환한 결과."""

    pseudo_label_rows: list[LabeledQueryRow]
    prediction_trace_rows: list[dict[str, Any]]
    prediction_summary: dict[str, Any]


class TeacherPseudoLabelBuilder:
    """teacher prediction에서 pseudo-label row, trace, summary payload를 조립한다."""

    def build_export(
        self,
        *,
        rows: Sequence[LabeledQueryRow],
        predictions: Sequence[TeacherPrediction],
        pseudo_label_algorithm,
        generated_at: datetime,
        run_id: str,
    ) -> TeacherPseudoLabelExport:
        if len(rows) != len(predictions):
            raise ValueError("rows and predictions must have the same length.")

        selection_hook = build_pseudo_label_selection_hook(
            pseudo_label_algorithm.algorithm_name
        )
        accepted_rows: list[LabeledQueryRow] = []
        trace_rows: list[dict[str, Any]] = []
        accepted_label_counts: Counter[str] = Counter()
        hidden_label_counts: Counter[str] = Counter()
        accepted_hidden_label_counts: Counter[str] = Counter()
        accepted_correct = 0

        for row, prediction in zip(rows, predictions, strict=True):
            hidden_label = str(row["mapped_label_4"])
            decision = selection_hook.evaluate(
                evidence=_build_teacher_evidence(
                    row=row,
                    prediction=prediction,
                    generated_at=generated_at,
                    run_id=run_id,
                ),
                config=pseudo_label_algorithm.config,
            )
            hidden_label_counts[hidden_label] += 1
            if decision.accepted:
                accepted_label_counts[decision.label] += 1
                accepted_hidden_label_counts[hidden_label] += 1
                if decision.label == hidden_label:
                    accepted_correct += 1
                accepted_rows.append(
                    LabeledQueryRow(
                        query_id=str(row["query_id"]),
                        text=str(row["text"]),
                        raw_label_scheme="pseudo_label",
                        raw_label=decision.label,
                        mapped_label_4=decision.label,
                        locale=str(row["locale"]),
                        annotation_source="teacher_bootstrap",
                        approved_by=None,
                        created_at=generated_at.isoformat(),
                    )
                )
            trace_rows.append(
                {
                    "schema_version": TEACHER_PREDICTION_TRACE_SCHEMA_VERSION,
                    "query_id": str(row["query_id"]),
                    "hidden_true_label": hidden_label,
                    "predicted_label": prediction.predicted_label,
                    "confidence": round(prediction.confidence, 6),
                    "margin": round(prediction.margin, 6),
                    "runner_up_label": prediction.runner_up_label,
                    "runner_up_score": round(
                        0.0
                        if prediction.runner_up_score is None
                        else float(prediction.runner_up_score),
                        6,
                    ),
                    "threshold_accepted": decision.accepted,
                    "final_accepted": decision.accepted,
                    "is_prediction_correct": (
                        prediction.predicted_label == hidden_label
                    ),
                    "category_scores": {
                        label: round(score, 6)
                        for label, score in sorted(prediction.raw_scores.items())
                    },
                }
            )

        summary = {
            "schema_version": TEACHER_PREDICTION_SUMMARY_SCHEMA_VERSION,
            "bootstrap_version": run_id,
            "total_rows": len(rows),
            "accepted_count": len(accepted_rows),
            "accepted_ratio": round(
                len(accepted_rows) / len(rows) if rows else 0.0,
                6,
            ),
            "pseudo_label_algorithm": (pseudo_label_algorithm.to_manifest_entry()),
            "accepted_label_counts": dict(sorted(accepted_label_counts.items())),
            "hidden_label_counts": dict(sorted(hidden_label_counts.items())),
            "accepted_hidden_label_counts": dict(
                sorted(accepted_hidden_label_counts.items())
            ),
            "accepted_hidden_label_accuracy": round(
                accepted_correct / len(accepted_rows) if accepted_rows else 0.0,
                6,
            ),
        }
        return TeacherPseudoLabelExport(
            pseudo_label_rows=accepted_rows,
            prediction_trace_rows=trace_rows,
            prediction_summary=summary,
        )


def _build_teacher_evidence(
    *,
    row: LabeledQueryRow,
    prediction: TeacherPrediction,
    generated_at: datetime,
    run_id: str,
) -> PseudoLabelEvidence:
    return PseudoLabelEvidence(
        schema_version=PSEUDO_LABEL_EVIDENCE_V1,
        evidence_id=f"{run_id}:{row['query_id']}",
        source_event_ref=str(row["query_id"]),
        occurred_at=_parse_row_timestamp(str(row["created_at"]), generated_at),
        label=prediction.predicted_label,
        confidence=prediction.confidence,
        confidence_kind="classifier_posterior_top1",
        margin=prediction.margin,
        top1_label=prediction.predicted_label,
        top1_score=prediction.confidence,
        top2_label=prediction.runner_up_label,
        top2_score=(
            0.0
            if prediction.runner_up_score is None
            else float(prediction.runner_up_score)
        ),
        raw_scores=dict(prediction.raw_scores),
    )


def _parse_row_timestamp(value: str, fallback: datetime) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return fallback
