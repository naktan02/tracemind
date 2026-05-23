"""FL simulation client-local diagnostic row view selection."""

from __future__ import annotations

import random
from collections.abc import Sequence

from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedDiagnosticViewConfig,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


def build_client_diagnostic_unlabeled_view(
    *,
    rows: Sequence[LabeledQueryRow],
    config: FederatedDiagnosticViewConfig,
    run_seed: int,
    round_index: int,
    client_id: str,
) -> tuple[LabeledQueryRow, ...]:
    """client의 full unlabeled pool에서 diagnostics용 deterministic subset을 고른다."""

    if not config.enabled or len(rows) <= config.max_rows:
        return tuple(rows)
    normalized_policy = config.selection_policy.strip().lower().replace("-", "_")
    if normalized_policy != "deterministic_random":
        raise ValueError(
            "diagnostic_view.selection_policy currently supports "
            "deterministic_random only."
        )
    sorted_rows = sorted(rows, key=lambda row: str(row["query_id"]))
    rng = random.Random(
        f"{int(run_seed) + int(config.seed_offset)}:{int(round_index)}:{client_id}"
    )
    selected_indices = sorted(rng.sample(range(len(sorted_rows)), config.max_rows))
    return tuple(sorted_rows[index] for index in selected_indices)
