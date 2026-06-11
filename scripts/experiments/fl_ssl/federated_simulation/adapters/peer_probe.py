"""FL simulation fixed peer-probe subset selection."""

from __future__ import annotations

from collections.abc import Sequence

from methods.federated_ssl.diagnostics.sampling import (
    hash_query_ids,
    select_label_balanced_probe_rows,
)
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
    selected_rows = select_label_balanced_probe_rows(
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
        query_ids_sha256=hash_query_ids(query_ids),
        label_distribution=count_labeled_query_rows_by_label(selected_rows),
        query_ids=query_ids,
    )
    return tuple(selected_rows), manifest
