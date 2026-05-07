"""FL simulation selection diagnostics writer."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedDiagnosticsConfig,
)
from scripts.io.labeled_query_rows import LabeledQueryRow


class SelectionDiagnosticsWriter:
    """row별 selection 원인과 요약 artifact 생성을 담당한다."""

    def save(
        self,
        *,
        output_dir: Path,
        round_id: str,
        client_id: str,
        rows: list[LabeledQueryRow],
        training_examples: tuple[Any, ...],
        selection_result: Any,
        diagnostics_config: FederatedDiagnosticsConfig,
    ) -> tuple[Path, Path]:
        diagnostics_dir = (
            output_dir / "agents" / client_id / diagnostics_config.dump_dir_name
        )
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = diagnostics_dir / f"{round_id}.candidates.jsonl"
        summary_path = diagnostics_dir / f"{round_id}.summary.json"

        rows_by_query_id = {str(row["query_id"]): row for row in rows}
        examples_by_query_id = {
            example.selection_key: example for example in training_examples
        }
        evidences_by_query_id = {
            evidence.source_event_ref: evidence
            for evidence in selection_result.evidences
        }
        stage_counts: dict[str, int] = defaultdict(int)
        by_true_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_predicted_label: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        lines: list[str] = []

        for candidate in selection_result.candidates:
            query_id = candidate.source_event_ref
            row = rows_by_query_id[query_id]
            example = examples_by_query_id[query_id]
            evidence = evidences_by_query_id.get(query_id)
            selection_context = candidate.selection_context
            if selection_context is None:
                raise ValueError(
                    "PseudoLabelCandidate.selection_context is required for "
                    f"federated selection artifacts: {candidate.candidate_id}."
                )
            selection_stage = selection_context.selection_stage.value
            raw_scores = (
                example.evidence_scored_event.category_scores
                if evidence is None
                else evidence.raw_scores
            )
            label_distribution = (
                None if evidence is None else evidence.label_distribution
            )

            stage_counts[selection_stage] += 1
            by_true_label[str(row["mapped_label_4"])]["total"] += 1
            by_true_label[str(row["mapped_label_4"])][selection_stage] += 1
            by_predicted_label[candidate.label]["total"] += 1
            by_predicted_label[candidate.label][selection_stage] += 1

            lines.append(
                json.dumps(
                    {
                        "round_id": round_id,
                        "client_id": client_id,
                        "query_id": query_id,
                        "true_label": row["mapped_label_4"],
                        "predicted_label": candidate.label,
                        "confidence": candidate.confidence,
                        "margin": candidate.margin,
                        "runner_up_label": candidate.runner_up_label,
                        "runner_up_score": candidate.runner_up_score,
                        "threshold_accepted": (selection_context.threshold_accepted),
                        "selected_by_cap": selection_context.selected_by_cap,
                        "final_accepted": candidate.accepted,
                        "selection_stage": selection_stage,
                        "pre_cap_rank": selection_context.pre_cap_rank,
                        "is_prediction_correct": (
                            candidate.label == row["mapped_label_4"]
                        ),
                        "view_kind": example.view_kind,
                        "confidence_kind": candidate.confidence_kind,
                        "category_scores": raw_scores,
                        "label_distribution": label_distribution,
                        "evidence_view_kind": (
                            example.view_kind
                            if evidence is None
                            else evidence.view_kind
                        ),
                        "evidence_confidence": (
                            candidate.confidence
                            if evidence is None
                            else evidence.top1_score
                        ),
                        "evidence_margin": (
                            candidate.margin if evidence is None else evidence.margin
                        ),
                    },
                    ensure_ascii=True,
                )
            )

        candidates_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        summary = {
            "round_id": round_id,
            "client_id": client_id,
            "total_candidates": selection_result.total_count,
            "final_accepted_count": selection_result.accepted_count,
            "stage_counts": dict(sorted(stage_counts.items())),
            "by_true_label": {
                label: dict(sorted(counts.items()))
                for label, counts in sorted(by_true_label.items())
            },
            "by_predicted_label": {
                label: dict(sorted(counts.items()))
                for label, counts in sorted(by_predicted_label.items())
            },
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return candidates_path, summary_path
