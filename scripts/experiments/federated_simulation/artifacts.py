"""Federated simulation 산출물 저장 유틸리티."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from agent.src.services.training.examples.models import (
    EmbeddedTrainingExample,
)
from agent.src.services.training.selection.pseudo_label_service import (
    PseudoLabelSelectionResult,
)
from scripts.experiments.federated_simulation.models import (
    ClientEvaluationSummary,
    FederatedDiagnosticsConfig,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedShardPolicyConfig,
    FederatedSslMethodConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationResult,
)
from scripts.labeled_query_rows import LabeledQueryRow
from shared.src.contracts.model_contracts import (
    ModelManifest,
    dump_model_manifest_payload,
)
from shared.src.contracts.prototype_contracts import (
    PrototypePackPayload,
    dump_prototype_pack_payload,
)


def save_selection_diagnostics(
    *,
    output_dir: Path,
    round_id: str,
    client_id: str,
    rows: list[LabeledQueryRow],
    training_examples: tuple[EmbeddedTrainingExample, ...],
    selection_result: PseudoLabelSelectionResult,
    diagnostics_config: FederatedDiagnosticsConfig,
) -> tuple[Path, Path]:
    """row별 selection 원인과 요약을 저장한다."""
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
        evidence.source_event_ref: evidence for evidence in selection_result.evidences
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
        threshold_accepted = selection_context.threshold_accepted
        selected_by_cap = selection_context.selected_by_cap
        pre_cap_rank = selection_context.pre_cap_rank
        raw_scores = (
            example.evidence_scored_event.category_scores
            if evidence is None
            else evidence.raw_scores
        )
        label_distribution = None if evidence is None else evidence.label_distribution

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
                    "threshold_accepted": threshold_accepted,
                    "selected_by_cap": selected_by_cap,
                    "final_accepted": candidate.accepted,
                    "selection_stage": selection_stage,
                    "pre_cap_rank": pre_cap_rank,
                    "is_prediction_correct": candidate.label == row["mapped_label_4"],
                    "view_kind": example.view_kind,
                    "confidence_kind": candidate.confidence_kind,
                    "category_scores": raw_scores,
                    "label_distribution": label_distribution,
                    "evidence_view_kind": (
                        example.view_kind if evidence is None else evidence.view_kind
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


def save_prototype_pack(output_dir: Path, payload: PrototypePackPayload) -> Path:
    """prototype pack payload를 output_dir 아래에 저장한다."""
    path = (
        output_dir
        / "main_server"
        / "prototype_packs"
        / f"{payload.prototype_version}.json"
    )
    dump_prototype_pack_payload(path, payload)
    return path


def save_model_manifest(output_dir: Path, manifest: ModelManifest) -> Path:
    """model manifest entity를 output_dir 아래 JSON으로 저장한다."""
    path = (
        output_dir
        / "main_server"
        / "model_manifests"
        / f"{manifest.model_revision}.json"
    )
    dump_model_manifest_payload(path, manifest)
    return path


def save_simulation_report(
    *,
    output_dir: Path,
    result: SimulationResult,
    report_config: FederatedReportConfig,
    client_count: int,
    round_budget: int,
    bootstrap_ratio: float,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
    ssl_method_config: FederatedSslMethodConfig,
    training_task_config: FederatedTrainingTaskConfig,
    validation_config: FederatedValidationConfig,
    round_runtime_config: FederatedRoundRuntimeConfig,
) -> Path:
    """FL SSL main comparison 전용 report를 저장한다."""
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{report_config.track}.report.json"
    path.write_text(
        json.dumps(
            _build_simulation_report_payload(
                result=result,
                report_config=report_config,
                client_count=client_count,
                round_budget=round_budget,
                bootstrap_ratio=bootstrap_ratio,
                seed=seed,
                shard_policy=shard_policy,
                ssl_method_config=ssl_method_config,
                training_task_config=training_task_config,
                validation_config=validation_config,
                round_runtime_config=round_runtime_config,
            ),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _build_simulation_report_payload(
    *,
    result: SimulationResult,
    report_config: FederatedReportConfig,
    client_count: int,
    round_budget: int,
    bootstrap_ratio: float,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
    ssl_method_config: FederatedSslMethodConfig,
    training_task_config: FederatedTrainingTaskConfig,
    validation_config: FederatedValidationConfig,
    round_runtime_config: FederatedRoundRuntimeConfig,
) -> dict[str, object]:
    client_metric_summary = _build_client_metric_summary(result.client_evaluations)
    communication_cost = _build_communication_cost_summary(result)
    return {
        "schema_version": report_config.schema_version,
        "track": report_config.track,
        "table_role": report_config.table_role,
        "must_not_merge_with": ["central_ssl_control"],
        "protocol": {
            "client_count": client_count,
            "round_budget": round_budget,
            "completed_rounds": len(result.rounds),
            "seed": seed,
            "seed_count": report_config.seed_count,
            "bootstrap_ratio": bootstrap_ratio,
            "shard_policy": _shard_policy_to_payload(shard_policy),
            "ssl_method": _ssl_method_to_payload(ssl_method_config),
            "labeled_unlabeled_split": {
                "labeled_ratio": report_config.labeled_ratio,
                "unlabeled_ratio": report_config.unlabeled_ratio,
                "status": "protocol_metadata_pending_method_runner_enforcement",
            },
            "local_update_budget": {
                "local_epochs": training_task_config.local_epochs,
                "batch_size": training_task_config.batch_size,
                "learning_rate": training_task_config.learning_rate,
                "max_steps": training_task_config.max_steps,
                "min_required_examples": (training_task_config.min_required_examples),
                "gradient_clip_norm": training_task_config.gradient_clip_norm,
                "selection_policy": _config_to_mapping(
                    training_task_config.selection_policy
                ),
            },
            "round_runtime": {
                "adapter_family_name": round_runtime_config.adapter_family_name,
                "aggregation_backend_name": (
                    round_runtime_config.aggregation_backend_name
                ),
                "classifier_head_bootstrap_logit_scale": (
                    round_runtime_config.classifier_head_bootstrap_logit_scale
                ),
            },
            "objective": _config_to_mapping(training_task_config.objective_config),
            "validation": {
                "similarity_name": validation_config.similarity_name,
                "scorer_backend_name": validation_config.scorer_backend_name,
                "score_policy_name": validation_config.score_policy_name,
                "score_top_k": validation_config.score_top_k,
                "confidence_threshold": validation_config.confidence_threshold,
                "margin_threshold": validation_config.margin_threshold,
            },
        },
        "metrics": {
            "primary": {
                "macro_f1": result.final_validation.macro_f1,
                "worst_client_macro_f1": (
                    client_metric_summary["worst_client_macro_f1"]
                ),
            },
            "secondary": {
                "expected_calibration_error": (
                    result.final_validation.expected_calibration_error
                ),
                "communication_cost": communication_cost,
                "per_client_macro_f1_variance": (
                    client_metric_summary["macro_f1_variance"]
                ),
            },
            "primary_metric_names": list(report_config.primary_metrics),
            "secondary_metric_names": list(report_config.secondary_metrics),
            "initial_validation": _evaluation_to_payload(result.initial_validation),
            "final_validation": _evaluation_to_payload(result.final_validation),
            "client_validation": client_metric_summary,
        },
        "rounds": [
            {
                "round_id": round_summary.round_id,
                "model_revision": round_summary.model_revision,
                "prototype_version": round_summary.prototype_version,
                "update_count": round_summary.update_count,
                "validation": _evaluation_to_payload(round_summary.validation),
                "clients": [
                    {
                        "client_id": client.client_id,
                        "candidate_count": client.candidate_count,
                        "accepted_count": client.accepted_count,
                        "update_generated": client.update_generated,
                    }
                    for client in round_summary.clients
                ],
            }
            for round_summary in result.rounds
        ],
    }


def _evaluation_to_payload(evaluation: SimulationEvaluation) -> dict[str, object]:
    return {
        "row_count": evaluation.row_count,
        "top1_accuracy": evaluation.top1_accuracy,
        "accepted_ratio": evaluation.accepted_ratio,
        "macro_f1": evaluation.macro_f1,
        "expected_calibration_error": evaluation.expected_calibration_error,
        "per_label": evaluation.per_label,
        "confusion_matrix": evaluation.confusion_matrix,
    }


def _build_client_metric_summary(
    client_evaluations: tuple[ClientEvaluationSummary, ...],
) -> dict[str, object]:
    evaluated = [
        client for client in client_evaluations if client.validation.row_count > 0
    ]
    macro_f1_values = [client.validation.macro_f1 for client in evaluated]
    return {
        "client_count": len(client_evaluations),
        "evaluated_client_count": len(evaluated),
        "worst_client_macro_f1": (min(macro_f1_values) if macro_f1_values else None),
        "macro_f1_mean": _mean(macro_f1_values),
        "macro_f1_variance": _population_variance(macro_f1_values),
        "clients": [
            {
                "client_id": client.client_id,
                "validation": _evaluation_to_payload(client.validation),
            }
            for client in client_evaluations
        ],
    }


def _build_communication_cost_summary(
    result: SimulationResult,
) -> dict[str, object]:
    total_client_updates = sum(
        round_summary.update_count for round_summary in result.rounds
    )
    total_candidates = sum(
        client.candidate_count
        for round_summary in result.rounds
        for client in round_summary.clients
    )
    total_accepted = sum(
        client.accepted_count
        for round_summary in result.rounds
        for client in round_summary.clients
    )
    return {
        "unit": "client_update_envelopes",
        "value": total_client_updates,
        "total_client_updates": total_client_updates,
        "total_candidates": total_candidates,
        "total_accepted": total_accepted,
        "status": "proxy_until_payload_byte_accounting",
    }


def _shard_policy_to_payload(
    shard_policy: FederatedShardPolicyConfig,
) -> dict[str, object]:
    return {
        "name": shard_policy.name,
        "client_id_prefix": shard_policy.client_id_prefix,
        "dominant_ratio": shard_policy.dominant_ratio,
        "alpha": shard_policy.alpha,
    }


def _ssl_method_to_payload(
    ssl_method_config: FederatedSslMethodConfig,
) -> dict[str, object]:
    return {
        "schema_version": ssl_method_config.schema_version,
        "name": ssl_method_config.name,
        "display_name": ssl_method_config.display_name,
        "method_role": ssl_method_config.method_role,
        "implementation_status": ssl_method_config.implementation_status,
        "client_step": dict(ssl_method_config.client_step),
        "server_step": dict(ssl_method_config.server_step),
        "report_tags": list(ssl_method_config.report_tags),
        "notes": list(ssl_method_config.notes),
    }


def _config_to_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if hasattr(value, "to_mapping"):
        mapping = value.to_mapping()
        if not isinstance(mapping, Mapping):
            raise TypeError("to_mapping() must return a mapping.")
        return dict(mapping)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported config mapping type: {type(value).__name__}")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _population_variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)
