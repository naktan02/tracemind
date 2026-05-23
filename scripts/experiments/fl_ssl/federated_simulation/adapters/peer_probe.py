"""FL simulation fixed peer-probe subset selection."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence

from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedPeerProbeConfig,
    FederatedPeerProbeManifest,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    count_labeled_query_rows_by_label,
)


def build_fixed_peer_probe(
    *,
    rows: Sequence[LabeledQueryRow],
    config: FederatedPeerProbeConfig,
    run_seed: int,
    source: str = "validation_rows",
) -> tuple[tuple[LabeledQueryRow, ...], FederatedPeerProbeManifest | None]:
    """validation rows에서 작고 재현 가능한 peer-selection probe를 만든다."""

    if not config.enabled:
        return tuple(rows), None
    selected_rows = _select_label_balanced_rows(
        rows=rows,
        max_rows=config.max_rows,
        seed=run_seed + config.seed_offset,
    )
    query_ids = tuple(str(row["query_id"]) for row in selected_rows)
    manifest = FederatedPeerProbeManifest(
        selection_policy=config.selection_policy,
        seed=run_seed,
        seed_offset=config.seed_offset,
        source=source,
        requested_max_rows=config.max_rows,
        row_count=len(selected_rows),
        query_ids_sha256=_hash_query_ids(query_ids),
        label_distribution=count_labeled_query_rows_by_label(selected_rows),
        query_ids=query_ids,
    )
    return tuple(selected_rows), manifest


def _select_label_balanced_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    max_rows: int,
    seed: int,
) -> list[LabeledQueryRow]:
    rows_by_label: dict[str, list[LabeledQueryRow]] = defaultdict(list)
    for row in rows:
        rows_by_label[str(row["mapped_label_4"])].append(row)
    rng = random.Random(seed)
    for label in sorted(rows_by_label):
        label_rows = rows_by_label[label]
        label_rows.sort(key=lambda row: str(row["query_id"]))
        rng.shuffle(label_rows)

    selected: list[LabeledQueryRow] = []
    labels = sorted(rows_by_label)
    while len(selected) < max_rows and labels:
        next_labels: list[str] = []
        for label in labels:
            label_rows = rows_by_label[label]
            if not label_rows:
                continue
            selected.append(label_rows.pop(0))
            if len(selected) >= max_rows:
                break
            if label_rows:
                next_labels.append(label)
        labels = next_labels
    return selected


def _hash_query_ids(query_ids: Sequence[str]) -> str:
    payload = json.dumps(list(query_ids), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
