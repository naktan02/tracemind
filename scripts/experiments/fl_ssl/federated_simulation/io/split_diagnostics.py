"""FL client split diagnostics payload helpers."""

from __future__ import annotations

import math
from collections import Counter

from scripts.experiments.fl_ssl.federated_simulation.io.report_math import (
    numeric_summary,
    safe_ratio,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientPoolSplitConfig,
    FederatedClientShard,
    FederatedDatasetSplit,
    FederatedReportConfig,
)


def build_client_pool_split_payload(
    *,
    dataset_split: FederatedDatasetSplit,
    client_pool_split_config: FederatedClientPoolSplitConfig | None,
    report_config: FederatedReportConfig,
) -> dict[str, object]:
    all_client_rows = [
        row for shard in dataset_split.client_shards for row in shard.rows
    ]
    distribution = label_distribution(all_client_rows)
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
        "actual_labeled_ratio": safe_ratio(labeled_count, total_rows),
        "actual_unlabeled_ratio": safe_ratio(unlabeled_count, total_rows),
        "label_distribution": distribution,
        "label_distribution_entropy": label_entropy(distribution),
        "min_client_size": min(client_sizes) if client_sizes else None,
        "max_client_size": max(client_sizes) if client_sizes else None,
        "empty_or_tiny_client_count": sum(
            1 for size in client_sizes if size <= tiny_client_threshold
        ),
        "tiny_client_threshold": tiny_client_threshold,
        "label_skew_summary": {
            "dominant_label_ratio": numeric_summary(dominant_ratios),
            "label_distribution_entropy": numeric_summary(entropies),
        },
        "clients": client_payloads,
    }


def label_distribution(rows: list[object]) -> dict[str, int]:
    counter = Counter(str(row["mapped_label_4"]) for row in rows)
    return dict(sorted(counter.items()))


def label_entropy(distribution: dict[str, int]) -> float:
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total, 2)
        for count in distribution.values()
        if count > 0
    )


def _client_shard_split_payload(shard: FederatedClientShard) -> dict[str, object]:
    distribution = label_distribution(shard.rows)
    return {
        "client_id": shard.client_id,
        "total_count": len(shard.rows),
        "labeled_count": len(shard.labeled_rows),
        "unlabeled_count": len(shard.unlabeled_rows),
        "label_distribution": distribution,
        "labeled_label_distribution": label_distribution(shard.labeled_rows),
        "unlabeled_label_distribution": label_distribution(shard.unlabeled_rows),
        "label_distribution_entropy": label_entropy(distribution),
        "dominant_label": _dominant_label(distribution),
        "dominant_label_ratio": _dominant_label_ratio(distribution),
    }


def _dominant_label(distribution: dict[str, int]) -> str | None:
    if not distribution:
        return None
    return max(distribution, key=lambda label: (distribution[label], label))


def _dominant_label_ratio(distribution: dict[str, int]) -> float | None:
    total = sum(distribution.values())
    if total <= 0:
        return None
    return max(distribution.values()) / total
