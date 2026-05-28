"""FL simulation client-local diagnostic row view selection."""

from __future__ import annotations

from collections.abc import Sequence

from methods.federated_ssl.diagnostic_sampling import (
    DIAGNOSTIC_VIEW_DETERMINISTIC_RANDOM,
    normalize_sampling_policy_name,
    select_deterministic_diagnostic_rows,
)
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
    normalized_policy = normalize_sampling_policy_name(config.selection_policy)
    if normalized_policy != DIAGNOSTIC_VIEW_DETERMINISTIC_RANDOM:
        raise ValueError(
            "diagnostic_view.selection_policy currently supports "
            f"{DIAGNOSTIC_VIEW_DETERMINISTIC_RANDOM} only."
        )
    return select_deterministic_diagnostic_rows(
        rows=rows,
        max_rows=config.max_rows,
        run_seed=run_seed,
        seed_offset=config.seed_offset,
        round_index=round_index,
        client_id=client_id,
    )
