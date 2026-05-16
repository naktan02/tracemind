"""FL simulation report payload builder."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    FederatedClientPoolSplitConfig,
    FederatedClientShard,
    FederatedDatasetSplit,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationResult,
    SimulationRoundSummary,
)


class SimulationReportBuilder:
    """FL SSL main comparison report schema payload를 조립한다."""

    def build_payload(
        self,
        *,
        result: SimulationResult,
        report_config: FederatedReportConfig,
        client_count: int,
        round_budget: int,
        bootstrap_ratio: float,
        seed: int,
        shard_policy: FederatedShardPolicyConfig,
        dataset_split: FederatedDatasetSplit,
        ssl_method_config: FederatedSslMethodConfig,
        client_pool_split_config: FederatedClientPoolSplitConfig | None,
        training_task_config: FederatedTrainingTaskConfig,
        validation_config: FederatedValidationConfig,
        round_runtime_config: FederatedRoundRuntimeConfig,
    ) -> dict[str, object]:
        client_metric_summary = _build_client_metric_summary(
            result=result,
            dataset_split=dataset_split,
        )
        communication_cost = _build_communication_cost_summary(result)
        round_progression = _build_round_progression(result)
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
                "labeled_unlabeled_split": _client_pool_split_to_payload(
                    dataset_split=dataset_split,
                    client_pool_split_config=client_pool_split_config,
                    report_config=report_config,
                ),
                "local_update_budget": {
                    "local_epochs": training_task_config.local_epochs,
                    "batch_size": training_task_config.batch_size,
                    "learning_rate": training_task_config.learning_rate,
                    "max_steps": training_task_config.max_steps,
                    "min_required_examples": (
                        training_task_config.min_required_examples
                    ),
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
            "diagnostics": {
                "round_progression": round_progression,
                "aggregation": _build_aggregation_diagnostics(result),
            },
            "metrics": {
                "primary": {
                    "macro_f1": result.final_validation.macro_f1,
                    "worst_client_macro_f1": (
                        client_metric_summary["worst_client_macro_f1"]
                    ),
                },
                "secondary": {
                    "loss": result.final_validation.loss,
                    "weighted_f1": result.final_validation.weighted_f1,
                    "balanced_accuracy": result.final_validation.balanced_accuracy,
                    "worst_category_f1_value": (
                        result.final_validation.worst_category_f1_value
                    ),
                    "max_calibration_error": (
                        result.final_validation.max_calibration_error
                    ),
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
                "round_progression": round_progression,
            },
            "rounds": _build_round_payloads(result),
        }


def _evaluation_to_payload(evaluation: SimulationEvaluation) -> dict[str, object]:
    payload = dict(evaluation.classification_report)
    payload.update(
        {
            "row_count": evaluation.row_count,
            "rows_total": evaluation.row_count,
            "top1_accuracy": evaluation.top1_accuracy,
            "accuracy_top_1": evaluation.accuracy_top_1,
            "correct_top_1": evaluation.correct_top_1,
            "accepted_ratio": evaluation.accepted_ratio,
            "loss": evaluation.loss,
            "loss_kind": evaluation.loss_kind,
            "macro_f1": evaluation.macro_f1,
            "macro_precision": evaluation.macro_precision,
            "macro_recall": evaluation.macro_recall,
            "weighted_f1": evaluation.weighted_f1,
            "balanced_accuracy": evaluation.balanced_accuracy,
            "worst_category_f1": evaluation.worst_category_f1,
            "worst_category_f1_value": evaluation.worst_category_f1_value,
            "worst_category_recall": evaluation.worst_category_recall,
            "worst_category_precision": evaluation.worst_category_precision,
            "expected_calibration_error": evaluation.expected_calibration_error,
            "max_calibration_error": evaluation.max_calibration_error,
            "overconfidence_gap": evaluation.overconfidence_gap,
            "mean_true_label_probability": evaluation.mean_true_label_probability,
            "mean_top_1_probability": evaluation.mean_top_1_probability,
            "mean_margin_top1_top2": evaluation.mean_margin_top1_top2,
            "mean_correct_top_1_probability": (
                evaluation.mean_correct_top_1_probability
            ),
            "mean_incorrect_top_1_probability": (
                evaluation.mean_incorrect_top_1_probability
            ),
            "score_distribution_kind": evaluation.score_distribution_kind,
            "selection_confidence_kind": evaluation.selection_confidence_kind,
            "mean_selection_confidence": evaluation.mean_selection_confidence,
            "mean_selection_margin": evaluation.mean_selection_margin,
            "per_label": evaluation.per_label,
            "per_category": evaluation.per_label,
            "confusion_matrix": evaluation.confusion_matrix,
        }
    )
    return payload


def _build_round_payloads(result: SimulationResult) -> list[dict[str, object]]:
    previous_validation = result.initial_validation
    payloads: list[dict[str, object]] = []
    for round_index, round_summary in enumerate(result.rounds, start=1):
        validation_payload = _evaluation_to_payload(round_summary.validation)
        payloads.append(
            {
                "round_id": round_summary.round_id,
                "round_index": round_index,
                "model_revision": round_summary.model_revision,
                "prototype_version": round_summary.prototype_version,
                "update_count": round_summary.update_count,
                "validation": validation_payload,
                "delta_from_previous_round": _evaluation_delta(
                    previous=previous_validation,
                    current=round_summary.validation,
                ),
                "delta_from_initial": _evaluation_delta(
                    previous=result.initial_validation,
                    current=round_summary.validation,
                ),
                "clients": [
                    {
                        "client_id": client.client_id,
                        "candidate_count": client.candidate_count,
                        "accepted_count": client.accepted_count,
                        "accepted_ratio": _safe_ratio(
                            client.accepted_count,
                            client.candidate_count,
                        ),
                        "update_generated": client.update_generated,
                        "delta_l2_norm": client.delta_l2_norm,
                    }
                    for client in round_summary.clients
                ],
            }
        )
        previous_validation = round_summary.validation
    return payloads


def _build_round_progression(result: SimulationResult) -> dict[str, object]:
    validation_points = [
        {
            "round_id": "initial",
            "round_index": 0,
            "validation": result.initial_validation,
        }
    ] + [
        {
            "round_id": round_summary.round_id,
            "round_index": index,
            "validation": round_summary.validation,
        }
        for index, round_summary in enumerate(result.rounds, start=1)
    ]
    best_macro_f1 = max(
        validation_points,
        key=lambda point: point["validation"].macro_f1,
    )
    best_loss = min(
        validation_points,
        key=lambda point: point["validation"].loss,
    )
    return {
        "best_macro_f1_round": _round_point_to_payload(best_macro_f1),
        "best_loss_round": _round_point_to_payload(best_loss),
        "early_stop_candidate": _build_early_stop_candidate(result),
        "final_delta_from_initial": _evaluation_delta(
            previous=result.initial_validation,
            current=result.final_validation,
        ),
        "round_count": len(result.rounds),
    }


def _round_point_to_payload(point: dict[str, object]) -> dict[str, object]:
    validation = point["validation"]
    if not isinstance(validation, SimulationEvaluation):
        raise TypeError("round progression validation must be SimulationEvaluation.")
    return {
        "round_id": point["round_id"],
        "round_index": point["round_index"],
        "macro_f1": validation.macro_f1,
        "loss": validation.loss,
        "expected_calibration_error": validation.expected_calibration_error,
    }


def _evaluation_delta(
    *,
    previous: SimulationEvaluation,
    current: SimulationEvaluation,
) -> dict[str, float]:
    return {
        "loss_delta": current.loss - previous.loss,
        "loss_reduction": previous.loss - current.loss,
        "macro_f1_delta": current.macro_f1 - previous.macro_f1,
        "accuracy_top_1_delta": current.accuracy_top_1 - previous.accuracy_top_1,
        "expected_calibration_error_delta": (
            current.expected_calibration_error - previous.expected_calibration_error
        ),
        "accepted_ratio_delta": current.accepted_ratio - previous.accepted_ratio,
    }


def _build_early_stop_candidate(
    result: SimulationResult,
    *,
    patience_rounds: int = 3,
    min_delta: float = 1e-6,
) -> dict[str, object]:
    """macro-F1 정체 여부를 report에서 빠르게 확인하기 위한 진단값."""

    if not result.rounds:
        return {
            "status": "no_rounds",
            "is_candidate": False,
            "metric": "macro_f1",
            "mode": "maximize",
            "patience_rounds": patience_rounds,
            "min_delta": min_delta,
            "rounds_without_improvement": 0,
        }

    best_value = result.initial_validation.macro_f1
    rounds_without_improvement = 0
    for round_summary in result.rounds:
        current_value = round_summary.validation.macro_f1
        if current_value > best_value + min_delta:
            best_value = current_value
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

    if len(result.rounds) < patience_rounds:
        status = "insufficient_rounds"
        is_candidate = False
    else:
        is_candidate = rounds_without_improvement >= patience_rounds
        status = "candidate" if is_candidate else "not_candidate"
    return {
        "status": status,
        "is_candidate": is_candidate,
        "metric": "macro_f1",
        "mode": "maximize",
        "patience_rounds": patience_rounds,
        "min_delta": min_delta,
        "rounds_without_improvement": rounds_without_improvement,
        "best_value": best_value,
    }


def _build_aggregation_diagnostics(result: SimulationResult) -> dict[str, object]:
    return {
        "weight_basis": "accepted_count_proxy",
        "client_weight_details_excluded": True,
        "rounds": [
            _round_aggregation_diagnostics(round_summary)
            for round_summary in result.rounds
        ],
    }


def _round_aggregation_diagnostics(
    round_summary: SimulationRoundSummary,
) -> dict[str, object]:
    accepted_counts = [
        client.accepted_count
        for client in round_summary.clients
        if client.update_generated
    ]
    total_accepted = sum(accepted_counts)
    normalized_weights = [
        accepted_count / total_accepted
        for accepted_count in accepted_counts
        if total_accepted > 0
    ]
    delta_l2_norms = [
        client.delta_l2_norm
        for client in round_summary.clients
        if client.update_generated and client.delta_l2_norm is not None
    ]
    zero_update_client_count = sum(
        1 for client in round_summary.clients if not client.update_generated
    )
    aggregation_weight_summary = _numeric_summary(normalized_weights)
    delta_l2_norm_summary = _numeric_summary(delta_l2_norms)
    return {
        "round_id": round_summary.round_id,
        "participating_client_count": len(round_summary.clients),
        "update_count": round_summary.update_count,
        "zero_update_client_count": zero_update_client_count,
        "total_candidate_count": sum(
            client.candidate_count for client in round_summary.clients
        ),
        "total_accepted_count": total_accepted,
        "total_aggregation_examples": total_accepted,
        "aggregation_example_basis": "accepted_count_proxy",
        "accepted_count_summary": _numeric_summary(accepted_counts),
        "aggregation_weight_summary": aggregation_weight_summary,
        "delta_l2_norm_summary": delta_l2_norm_summary,
        "mean_delta_l2_norm": delta_l2_norm_summary["mean"],
        "max_delta_l2_norm": delta_l2_norm_summary["max"],
        "update_norm_variance": delta_l2_norm_summary["variance"],
    }


def _numeric_summary(values: list[float] | list[int]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": _mean([float(value) for value in values]),
        "variance": _population_variance([float(value) for value in values]),
    }


def _build_client_metric_summary(
    *,
    result: SimulationResult,
    dataset_split: FederatedDatasetSplit,
) -> dict[str, object]:
    client_evaluations = result.client_evaluations
    evaluated = [
        client for client in client_evaluations if client.validation.row_count > 0
    ]
    macro_f1_values = [client.validation.macro_f1 for client in evaluated]
    loss_values = [client.validation.loss for client in evaluated]
    weighted_f1_values = [client.validation.weighted_f1 for client in evaluated]
    split_by_client = {shard.client_id: shard for shard in dataset_split.client_shards}
    round_summary_by_client = _aggregate_client_round_summaries(result)
    worst_client_macro_f1 = min(macro_f1_values) if macro_f1_values else None
    best_client_macro_f1 = max(macro_f1_values) if macro_f1_values else None
    worst_client_loss = max(loss_values) if loss_values else None
    best_client_loss = min(loss_values) if loss_values else None
    return {
        "client_count": len(client_evaluations),
        "evaluated_client_count": len(evaluated),
        "worst_client_macro_f1": worst_client_macro_f1,
        "best_client_macro_f1": best_client_macro_f1,
        "worst_client_loss": worst_client_loss,
        "best_client_loss": best_client_loss,
        "macro_f1_mean": _mean(macro_f1_values),
        "macro_f1_variance": _population_variance(macro_f1_values),
        "macro_f1_std": _population_std(macro_f1_values),
        "loss_mean": _mean(loss_values),
        "loss_variance": _population_variance(loss_values),
        "loss_std": _population_std(loss_values),
        "weighted_f1_mean": _mean(weighted_f1_values),
        "fairness_gap": (
            None
            if best_client_macro_f1 is None or worst_client_macro_f1 is None
            else best_client_macro_f1 - worst_client_macro_f1
        ),
        "clients": [
            _client_validation_payload(
                client=client,
                train_shard=split_by_client.get(client.client_id),
                round_summary=round_summary_by_client.get(client.client_id),
            )
            for client in client_evaluations
        ],
    }


def _aggregate_client_round_summaries(
    result: SimulationResult,
) -> dict[str, dict[str, object]]:
    client_ids: set[str] = set()
    candidate_count_by_client: dict[str, int] = {}
    accepted_count_by_client: dict[str, int] = {}
    update_generated_by_client: dict[str, bool] = {}
    update_generated_round_count_by_client: dict[str, int] = {}
    latest_round_id_by_client: dict[str, str] = {}
    latest_update_generated_by_client: dict[str, bool] = {}
    delta_l2_norms_by_client: dict[str, list[float]] = {}

    for round_summary in result.rounds:
        for client in round_summary.clients:
            client_ids.add(client.client_id)
            candidate_count_by_client[client.client_id] = (
                candidate_count_by_client.get(client.client_id, 0)
                + client.candidate_count
            )
            accepted_count_by_client[client.client_id] = (
                accepted_count_by_client.get(client.client_id, 0)
                + client.accepted_count
            )
            latest_round_id_by_client[client.client_id] = round_summary.round_id
            latest_update_generated_by_client[client.client_id] = (
                client.update_generated
            )
            if client.update_generated:
                update_generated_by_client[client.client_id] = True
                update_generated_round_count_by_client[client.client_id] = (
                    update_generated_round_count_by_client.get(client.client_id, 0) + 1
                )
            if client.delta_l2_norm is not None:
                delta_l2_norms_by_client.setdefault(client.client_id, []).append(
                    client.delta_l2_norm
                )

    return {
        client_id: _client_round_summary_payload(
            candidate_count=candidate_count_by_client.get(client_id, 0),
            accepted_count=accepted_count_by_client.get(client_id, 0),
            client_update_generated=update_generated_by_client.get(client_id, False),
            update_generated_round_count=(
                update_generated_round_count_by_client.get(client_id, 0)
            ),
            latest_round_id=latest_round_id_by_client.get(client_id),
            latest_update_generated=latest_update_generated_by_client.get(
                client_id,
                False,
            ),
            delta_l2_norms=delta_l2_norms_by_client.get(client_id, []),
        )
        for client_id in sorted(client_ids)
    }


def _client_round_summary_payload(
    *,
    candidate_count: int,
    accepted_count: int,
    client_update_generated: bool,
    update_generated_round_count: int,
    latest_round_id: str | None,
    latest_update_generated: bool,
    delta_l2_norms: list[float],
) -> dict[str, object]:
    return {
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "client_accepted_ratio": _safe_ratio(accepted_count, candidate_count),
        "client_update_generated": client_update_generated,
        "latest_round_id": latest_round_id,
        "latest_update_generated": latest_update_generated,
        "update_generated_round_count": update_generated_round_count,
        "client_delta_l2_norm": delta_l2_norms[-1] if delta_l2_norms else None,
        "mean_delta_l2_norm": _mean(delta_l2_norms),
        "max_delta_l2_norm": max(delta_l2_norms) if delta_l2_norms else None,
        "update_norm_variance": _population_variance(delta_l2_norms),
        "delta_l2_norm_status": "available" if delta_l2_norms else "not_available",
    }


def _client_validation_payload(
    *,
    client: ClientEvaluationSummary,
    train_shard: FederatedClientShard | None,
    round_summary: dict[str, object] | None,
) -> dict[str, object]:
    total_count = len(train_shard.rows) if train_shard is not None else None
    labeled_count = len(train_shard.labeled_rows) if train_shard is not None else None
    unlabeled_count = (
        len(train_shard.unlabeled_rows) if train_shard is not None else None
    )
    round_summary = round_summary or {}
    return {
        "client_id": client.client_id,
        "client_train_size": total_count,
        "client_labeled_count": labeled_count,
        "client_unlabeled_count": unlabeled_count,
        "client_label_distribution": (
            _label_distribution(train_shard.rows) if train_shard is not None else {}
        ),
        "client_candidate_count": round_summary.get("candidate_count"),
        "client_accepted_count": round_summary.get("accepted_count"),
        "client_accepted_ratio": round_summary.get("client_accepted_ratio"),
        "client_update_generated": round_summary.get(
            "client_update_generated",
            False,
        ),
        "latest_round_id": round_summary.get("latest_round_id"),
        "latest_update_generated": round_summary.get(
            "latest_update_generated",
            False,
        ),
        "update_generated_round_count": round_summary.get(
            "update_generated_round_count",
            0,
        ),
        "client_delta_l2_norm": round_summary.get("client_delta_l2_norm"),
        "mean_delta_l2_norm": round_summary.get("mean_delta_l2_norm"),
        "max_delta_l2_norm": round_summary.get("max_delta_l2_norm"),
        "update_norm_variance": round_summary.get("update_norm_variance"),
        "delta_l2_norm_status": round_summary.get(
            "delta_l2_norm_status",
            "not_available",
        ),
        "validation": _evaluation_to_payload(client.validation),
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
        "accepted_per_update": _safe_ratio(total_accepted, total_client_updates),
        "acceptance_ratio": _safe_ratio(total_accepted, total_candidates),
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
        "round_state_exchange": dict(ssl_method_config.round_state_exchange),
        "report_tags": list(ssl_method_config.report_tags),
        "notes": list(ssl_method_config.notes),
    }


def _client_pool_split_to_payload(
    *,
    dataset_split: FederatedDatasetSplit,
    client_pool_split_config: FederatedClientPoolSplitConfig | None,
    report_config: FederatedReportConfig,
) -> dict[str, object]:
    all_client_rows = [
        row for shard in dataset_split.client_shards for row in shard.rows
    ]
    label_distribution = _label_distribution(all_client_rows)
    total_rows = sum(len(shard.rows) for shard in dataset_split.client_shards)
    labeled_count = sum(
        len(shard.labeled_rows) for shard in dataset_split.client_shards
    )
    unlabeled_count = sum(
        len(shard.unlabeled_rows) for shard in dataset_split.client_shards
    )
    client_payloads = [
        _client_shard_split_payload(shard) for shard in dataset_split.client_shards
    ]
    client_sizes = [int(payload["total_count"]) for payload in client_payloads]
    dominant_ratios = [
        float(payload["dominant_label_ratio"])
        for payload in client_payloads
        if payload["dominant_label_ratio"] is not None
    ]
    entropies = [
        float(payload["label_distribution_entropy"]) for payload in client_payloads
    ]
    tiny_client_threshold = 1
    return {
        "labeled_ratio": report_config.labeled_ratio,
        "unlabeled_ratio": report_config.unlabeled_ratio,
        "status": (
            "enforced_by_client_pool_split"
            if client_pool_split_config is not None
            else "not_configured"
        ),
        "actual_labeled_count": labeled_count,
        "actual_unlabeled_count": unlabeled_count,
        "actual_labeled_ratio": _safe_ratio(labeled_count, total_rows),
        "actual_unlabeled_ratio": _safe_ratio(unlabeled_count, total_rows),
        "label_distribution": label_distribution,
        "label_distribution_entropy": _label_entropy(label_distribution),
        "min_client_size": min(client_sizes) if client_sizes else None,
        "max_client_size": max(client_sizes) if client_sizes else None,
        "empty_or_tiny_client_count": sum(
            1 for size in client_sizes if size <= tiny_client_threshold
        ),
        "tiny_client_threshold": tiny_client_threshold,
        "label_skew_summary": {
            "dominant_label_ratio": _numeric_summary(dominant_ratios),
            "label_distribution_entropy": _numeric_summary(entropies),
        },
        "clients": client_payloads,
    }


def _client_shard_split_payload(shard: FederatedClientShard) -> dict[str, object]:
    label_distribution = _label_distribution(shard.rows)
    return {
        "client_id": shard.client_id,
        "total_count": len(shard.rows),
        "labeled_count": len(shard.labeled_rows),
        "unlabeled_count": len(shard.unlabeled_rows),
        "label_distribution": label_distribution,
        "labeled_label_distribution": _label_distribution(shard.labeled_rows),
        "unlabeled_label_distribution": _label_distribution(shard.unlabeled_rows),
        "label_distribution_entropy": _label_entropy(label_distribution),
        "dominant_label": _dominant_label(label_distribution),
        "dominant_label_ratio": _dominant_label_ratio(label_distribution),
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


def _population_std(values: list[float]) -> float | None:
    variance = _population_variance(values)
    if variance is None:
        return None
    return math.sqrt(variance)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _label_distribution(rows: list[object]) -> dict[str, int]:
    counter = Counter(str(row["mapped_label_4"]) for row in rows)
    return dict(sorted(counter.items()))


def _label_entropy(distribution: dict[str, int]) -> float:
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total, 2)
        for count in distribution.values()
        if count > 0
    )


def _dominant_label(distribution: dict[str, int]) -> str | None:
    if not distribution:
        return None
    return max(distribution, key=lambda label: (distribution[label], label))


def _dominant_label_ratio(distribution: dict[str, int]) -> float | None:
    total = sum(distribution.values())
    if total <= 0:
        return None
    return max(distribution.values()) / total
