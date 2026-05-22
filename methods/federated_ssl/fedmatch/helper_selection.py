"""FedMatch helper selection core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from methods.federated_ssl.capability_plan import (
    PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN,
    PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK,
)
from methods.federated_ssl.fedmatch.original_spec import (
    FEDMATCH_ORIGINAL_METHOD_DEFAULTS,
)
from methods.federated_ssl.peer_context import (
    NearestPeerClientIndex,
    select_nearest_peer_client_ids,
    should_refresh_peer_context,
)

FEDMATCH_HELPER_SELECTION_NAME = PEER_CONTEXT_FIXED_PROBE_OUTPUT_KNN
FEDMATCH_LEGACY_HELPER_SELECTION_NAME = PEER_CONTEXT_PREDICTION_SIMILARITY_TOPK
FEDMATCH_DEFAULT_NUM_HELPERS = FEDMATCH_ORIGINAL_METHOD_DEFAULTS.num_helpers
FEDMATCH_DEFAULT_REFRESH_INTERVAL = (
    FEDMATCH_ORIGINAL_METHOD_DEFAULTS.helper_refresh_interval
)
FEDMATCH_ORIGINAL_HELPER_PROBE = "fixed_gaussian_noise_model_output"
FEDMATCH_TRACE_HELPER_PROBE = "fixed_probe_text_batch_model_output"


@dataclass(frozen=True, slots=True)
class FedMatchHelperSelectionSpec:
    """원본 KDTree helper selection을 NLP runtime용 vector selection으로 일반화한다."""

    selection_name: str
    num_helpers: int
    refresh_interval: int
    original_probe_source: str
    trace_probe_source: str
    similarity_name: str
    preferred_index_backend: str


helper_selection_spec = FedMatchHelperSelectionSpec(
    selection_name=FEDMATCH_HELPER_SELECTION_NAME,
    num_helpers=FEDMATCH_DEFAULT_NUM_HELPERS,
    refresh_interval=FEDMATCH_DEFAULT_REFRESH_INTERVAL,
    original_probe_source=FEDMATCH_ORIGINAL_HELPER_PROBE,
    trace_probe_source=FEDMATCH_TRACE_HELPER_PROBE,
    similarity_name="euclidean_nearest",
    preferred_index_backend="scipy_kdtree",
)


def should_refresh_helper_context(
    *,
    round_index_zero_based: int,
    refresh_interval: int = FEDMATCH_DEFAULT_REFRESH_INTERVAL,
) -> bool:
    """원본 `(curr_round + 1) % h_interval == 0` refresh 조건."""

    return should_refresh_peer_context(
        round_index_zero_based=round_index_zero_based,
        refresh_interval=refresh_interval,
    )


def select_helper_client_ids(
    *,
    client_id: str,
    client_vectors: Mapping[str, Sequence[float]],
    num_helpers: int = FEDMATCH_DEFAULT_NUM_HELPERS,
) -> tuple[str, ...]:
    """원본 FedMatch처럼 KDTree 우선 index로 최근접 helper client를 고른다."""

    return select_nearest_peer_client_ids(
        client_id=client_id,
        client_vectors=client_vectors,
        peer_count=num_helpers,
        prefer_kdtree=True,
    )


def build_helper_index(
    *,
    client_vectors: Mapping[str, Sequence[float]],
) -> NearestPeerClientIndex:
    """FedMatch helper selection용 KDTree 우선 index를 만든다."""

    return NearestPeerClientIndex(client_vectors=client_vectors, prefer_kdtree=True)
