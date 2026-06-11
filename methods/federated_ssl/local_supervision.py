"""FL SSL local supervision regime 해석 helper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from methods.federated_ssl.capabilities.plan import (
    LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED,
    LOCAL_SUPERVISION_REGIMES,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


@dataclass(frozen=True, slots=True)
class FederatedSslLocalSupervisionRegime:
    """client local trainer가 labeled/unlabeled row 노출을 해석한 결과."""

    name: str
    uses_client_labeled_rows: bool


def resolve_local_supervision_regime(
    regime_name: str,
) -> FederatedSslLocalSupervisionRegime:
    """capability 이름을 client local trainer가 쓸 수 있는 regime으로 해석한다."""

    normalized = regime_name.strip().lower().replace("-", "_")
    if normalized not in LOCAL_SUPERVISION_REGIMES:
        raise ValueError(f"Unsupported FL SSL local supervision regime: {regime_name}")
    return FederatedSslLocalSupervisionRegime(
        name=normalized,
        uses_client_labeled_rows=(
            normalized == LOCAL_SUPERVISION_CLIENT_LABELED_AND_UNLABELED
        ),
    )


def require_rows_match_local_supervision_regime(
    *,
    regime: FederatedSslLocalSupervisionRegime,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
    context: str,
) -> None:
    """local supervision regime과 실제 client row 노출이 맞는지 검증한다."""

    if regime.uses_client_labeled_rows and not labeled_rows:
        raise ValueError(f"{context} requires client labeled_rows.")
    if not regime.uses_client_labeled_rows and labeled_rows:
        raise ValueError(f"{context} must not receive client labeled_rows.")
    if not unlabeled_rows:
        raise ValueError(f"{context} requires client unlabeled_rows.")
