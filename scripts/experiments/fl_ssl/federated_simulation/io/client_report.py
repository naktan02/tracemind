"""FL client validation and update summary payload helpers."""

from __future__ import annotations

from collections import Counter

from scripts.experiments.fl_ssl.federated_simulation.io.aggregation_diagnostics import (
    aggregation_example_count,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_math import (
    mean,
    population_std,
    population_variance,
    safe_ratio,
    weighted_mean,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_metrics import (
    evaluation_to_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.io.split_diagnostics import (
    label_distribution,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    FederatedClientShard,
    FederatedDatasetSplit,
    SimulationResult,
)


def build_client_metric_summary(
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
        "macro_f1_mean": mean(macro_f1_values),
        "macro_f1_variance": population_variance(macro_f1_values),
        "macro_f1_std": population_std(macro_f1_values),
        "loss_mean": mean(loss_values),
        "loss_variance": population_variance(loss_values),
        "loss_std": population_std(loss_values),
        "weighted_f1_mean": mean(weighted_f1_values),
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
    diagnostic_candidate_count_by_client: dict[str, int] = {}
    accepted_count_by_client: dict[str, int] = {}
    aggregation_example_count_by_client: dict[str, int] = {}
    payload_bytes_by_client: dict[str, int] = {}
    update_generated_by_client: dict[str, bool] = {}
    update_generated_round_count_by_client: dict[str, int] = {}
    latest_round_id_by_client: dict[str, str] = {}
    latest_update_generated_by_client: dict[str, bool] = {}
    delta_l2_norms_by_client: dict[str, list[float]] = {}
    train_times_by_client: dict[str, list[float]] = {}
    confidence_means_by_client: dict[str, list[tuple[float, int]]] = {}
    margin_means_by_client: dict[str, list[tuple[float, int]]] = {}
    pseudo_label_correct_count_by_client: dict[str, int] = {}
    pseudo_label_evaluated_count_by_client: dict[str, int] = {}
    accepted_label_distribution_by_client: dict[str, Counter[str]] = {}
    rejected_label_distribution_by_client: dict[str, Counter[str]] = {}

    for round_summary in result.rounds:
        for client in round_summary.clients:
            client_ids.add(client.client_id)
            candidate_count_by_client[client.client_id] = (
                candidate_count_by_client.get(client.client_id, 0)
                + client.candidate_count
            )
            diagnostic_candidate_count_by_client[client.client_id] = (
                diagnostic_candidate_count_by_client.get(client.client_id, 0)
                + client.diagnostic_candidate_count
            )
            accepted_count_by_client[client.client_id] = (
                accepted_count_by_client.get(client.client_id, 0)
                + client.accepted_count
            )
            aggregation_example_count_by_client[client.client_id] = (
                aggregation_example_count_by_client.get(client.client_id, 0)
                + aggregation_example_count(client)
            )
            if client.client_payload_bytes is not None:
                payload_bytes_by_client[client.client_id] = (
                    payload_bytes_by_client.get(client.client_id, 0)
                    + client.client_payload_bytes
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
            if client.client_train_time_seconds is not None:
                train_times_by_client.setdefault(client.client_id, []).append(
                    client.client_train_time_seconds
                )
            if client.pseudo_label_confidence_mean is not None:
                confidence_means_by_client.setdefault(client.client_id, []).append(
                    (
                        client.pseudo_label_confidence_mean,
                        client.diagnostic_candidate_count,
                    )
                )
            if client.pseudo_label_margin_mean is not None:
                margin_means_by_client.setdefault(client.client_id, []).append(
                    (client.pseudo_label_margin_mean, client.diagnostic_candidate_count)
                )
            pseudo_label_correct_count_by_client[client.client_id] = (
                pseudo_label_correct_count_by_client.get(client.client_id, 0)
                + client.pseudo_label_correct_count
            )
            pseudo_label_evaluated_count_by_client[client.client_id] = (
                pseudo_label_evaluated_count_by_client.get(client.client_id, 0)
                + client.pseudo_label_evaluated_count
            )
            accepted_label_distribution_by_client.setdefault(
                client.client_id, Counter()
            ).update(client.accepted_label_distribution)
            rejected_label_distribution_by_client.setdefault(
                client.client_id, Counter()
            ).update(client.rejected_label_distribution)

    return {
        client_id: _client_round_summary_payload(
            candidate_count=candidate_count_by_client.get(client_id, 0),
            diagnostic_candidate_count=diagnostic_candidate_count_by_client.get(
                client_id,
                0,
            ),
            accepted_count=accepted_count_by_client.get(client_id, 0),
            aggregation_examples=aggregation_example_count_by_client.get(client_id, 0),
            payload_bytes=payload_bytes_by_client.get(client_id),
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
            train_times=train_times_by_client.get(client_id, []),
            confidence_means=confidence_means_by_client.get(client_id, []),
            margin_means=margin_means_by_client.get(client_id, []),
            pseudo_label_correct_count=pseudo_label_correct_count_by_client.get(
                client_id,
                0,
            ),
            pseudo_label_evaluated_count=pseudo_label_evaluated_count_by_client.get(
                client_id,
                0,
            ),
            accepted_label_distribution=accepted_label_distribution_by_client.get(
                client_id,
                Counter(),
            ),
            rejected_label_distribution=rejected_label_distribution_by_client.get(
                client_id,
                Counter(),
            ),
        )
        for client_id in sorted(client_ids)
    }


def _client_round_summary_payload(
    *,
    candidate_count: int,
    diagnostic_candidate_count: int,
    accepted_count: int,
    aggregation_examples: int,
    payload_bytes: int | None,
    client_update_generated: bool,
    update_generated_round_count: int,
    latest_round_id: str | None,
    latest_update_generated: bool,
    delta_l2_norms: list[float],
    train_times: list[float],
    confidence_means: list[tuple[float, int]],
    margin_means: list[tuple[float, int]],
    pseudo_label_correct_count: int,
    pseudo_label_evaluated_count: int,
    accepted_label_distribution: Counter[str],
    rejected_label_distribution: Counter[str],
) -> dict[str, object]:
    return {
        "candidate_count": candidate_count,
        "diagnostic_candidate_count": diagnostic_candidate_count,
        "accepted_count": accepted_count,
        "client_accepted_ratio": safe_ratio(accepted_count, candidate_count),
        "aggregation_example_count": aggregation_examples,
        "client_payload_bytes": payload_bytes,
        "client_update_generated": client_update_generated,
        "latest_round_id": latest_round_id,
        "latest_update_generated": latest_update_generated,
        "update_generated_round_count": update_generated_round_count,
        "client_delta_l2_norm": delta_l2_norms[-1] if delta_l2_norms else None,
        "mean_delta_l2_norm": mean(delta_l2_norms),
        "max_delta_l2_norm": max(delta_l2_norms) if delta_l2_norms else None,
        "update_norm_variance": population_variance(delta_l2_norms),
        "delta_l2_norm_status": "available" if delta_l2_norms else "not_available",
        "client_train_time_seconds": train_times[-1] if train_times else None,
        "mean_client_train_time_seconds": mean(train_times),
        "pseudo_label_confidence_mean": _weighted_pairs_mean(confidence_means),
        "pseudo_label_margin_mean": _weighted_pairs_mean(margin_means),
        "pseudo_label_accuracy": safe_ratio(
            pseudo_label_correct_count,
            pseudo_label_evaluated_count,
        ),
        "pseudo_label_correct_count": pseudo_label_correct_count,
        "pseudo_label_evaluated_count": pseudo_label_evaluated_count,
        "accepted_label_distribution": dict(
            sorted(accepted_label_distribution.items())
        ),
        "rejected_label_distribution": dict(
            sorted(rejected_label_distribution.items())
        ),
    }


def _weighted_pairs_mean(values: list[tuple[float, int]]) -> float | None:
    return weighted_mean(values)


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
            label_distribution(train_shard.rows) if train_shard is not None else {}
        ),
        "client_candidate_count": round_summary.get("candidate_count"),
        "client_diagnostic_candidate_count": round_summary.get(
            "diagnostic_candidate_count"
        ),
        "client_accepted_count": round_summary.get("accepted_count"),
        "client_accepted_ratio": round_summary.get("client_accepted_ratio"),
        "aggregation_example_count": round_summary.get("aggregation_example_count"),
        "client_payload_bytes": round_summary.get("client_payload_bytes"),
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
        "client_train_time_seconds": round_summary.get("client_train_time_seconds"),
        "mean_client_train_time_seconds": round_summary.get(
            "mean_client_train_time_seconds"
        ),
        "candidate_confidence_mean": round_summary.get("pseudo_label_confidence_mean"),
        "candidate_margin_mean": round_summary.get("pseudo_label_margin_mean"),
        "pseudo_label_confidence_mean": round_summary.get(
            "pseudo_label_confidence_mean"
        ),
        "pseudo_label_margin_mean": round_summary.get("pseudo_label_margin_mean"),
        "pseudo_label_accuracy": round_summary.get("pseudo_label_accuracy"),
        "pseudo_label_correct_count": round_summary.get(
            "pseudo_label_correct_count",
            0,
        ),
        "pseudo_label_evaluated_count": round_summary.get(
            "pseudo_label_evaluated_count",
            0,
        ),
        "accepted_label_distribution": round_summary.get(
            "accepted_label_distribution",
            {},
        ),
        "rejected_label_distribution": round_summary.get(
            "rejected_label_distribution",
            {},
        ),
        "client_validation_loss": client.validation.loss,
        "client_validation_macro_f1": client.validation.macro_f1,
        "client_validation_ece": client.validation.expected_calibration_error,
        "validation": evaluation_to_payload(client.validation),
    }
