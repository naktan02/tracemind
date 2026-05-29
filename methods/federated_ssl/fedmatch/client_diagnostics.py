"""FedMatch client diagnostic metric contract."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

NumericSummaryFn = Callable[[Iterable[float]], Mapping[str, float | int | None]]

FEDMATCH_HELPER_COUNT = "fedmatch_helper_count"
FEDMATCH_PEER_CONTEXT_HELPER_COUNT = "fedmatch_peer_context_helper_count"
FEDMATCH_HELPER_PROVIDER_COUNT = "fedmatch_helper_provider_count"
FEDMATCH_MISSING_HELPER_SNAPSHOT_COUNT = "fedmatch_missing_helper_snapshot_count"
FEDMATCH_MATERIALIZED_HELPER_MODEL_COUNT = "fedmatch_materialized_helper_model_count"
FEDMATCH_PEER_CONTEXT_REFRESHED = "fedmatch_peer_context_refreshed"
FEDMATCH_C2S_SPARSE_UPLOAD_VALUE_COUNT = "fedmatch_c2s_sparse_upload_value_count"
FEDMATCH_S2C_SPARSE_DOWNLOAD_VALUE_COUNT = "fedmatch_s2c_sparse_download_value_count"

_CLIENT_DIAGNOSTIC_METRIC_NAMES = (
    FEDMATCH_HELPER_COUNT,
    FEDMATCH_PEER_CONTEXT_HELPER_COUNT,
    FEDMATCH_HELPER_PROVIDER_COUNT,
    FEDMATCH_MISSING_HELPER_SNAPSHOT_COUNT,
    FEDMATCH_MATERIALIZED_HELPER_MODEL_COUNT,
    FEDMATCH_PEER_CONTEXT_REFRESHED,
    FEDMATCH_C2S_SPARSE_UPLOAD_VALUE_COUNT,
    FEDMATCH_S2C_SPARSE_DOWNLOAD_VALUE_COUNT,
)

_BOOLEAN_COUNT_METRIC_NAMES = frozenset({FEDMATCH_PEER_CONTEXT_REFRESHED})


def client_diagnostic_metric_names() -> tuple[str, ...]:
    """FedMatch가 client round report에 노출하는 method-local metric key."""

    return _CLIENT_DIAGNOSTIC_METRIC_NAMES


def non_negative_client_diagnostic_metric_names() -> frozenset[str]:
    """FedMatch client diagnostic 중 음수가 될 수 없는 metric key."""

    return frozenset(_CLIENT_DIAGNOSTIC_METRIC_NAMES)


def client_diagnostic_summary_payload(
    diagnostics_by_client: Iterable[Mapping[str, float]],
    *,
    numeric_summary: NumericSummaryFn,
) -> dict[str, object]:
    """FedMatch legacy aggregation diagnostic payload를 만든다."""

    diagnostics = tuple(diagnostics_by_client)
    payload: dict[str, object] = {}
    for metric_name in _CLIENT_DIAGNOSTIC_METRIC_NAMES:
        values = [
            float(client_diagnostics[metric_name])
            for client_diagnostics in diagnostics
            if metric_name in client_diagnostics
        ]
        if metric_name in _BOOLEAN_COUNT_METRIC_NAMES:
            payload[f"{metric_name}_count"] = sum(1 for value in values if value)
        else:
            payload[f"{metric_name}_summary"] = numeric_summary(values)
    return payload
