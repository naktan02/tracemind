"""FL simulation report payload builder."""

from __future__ import annotations

from collections.abc import Mapping

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    FederatedClientPoolSplitConfig,
    FederatedDatasetSplit,
    FederatedReportConfig,
    FederatedRoundRuntimeConfig,
    FederatedSslMethodConfig,
    FederatedTrainingTaskConfig,
    FederatedValidationConfig,
    SimulationEvaluation,
    SimulationResult,
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
    total_rows = sum(len(shard.rows) for shard in dataset_split.client_shards)
    labeled_count = sum(
        len(shard.labeled_rows) for shard in dataset_split.client_shards
    )
    unlabeled_count = sum(
        len(shard.unlabeled_rows) for shard in dataset_split.client_shards
    )
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
        "clients": [
            {
                "client_id": shard.client_id,
                "total_count": len(shard.rows),
                "labeled_count": len(shard.labeled_rows),
                "unlabeled_count": len(shard.unlabeled_rows),
            }
            for shard in dataset_split.client_shards
        ],
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


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
