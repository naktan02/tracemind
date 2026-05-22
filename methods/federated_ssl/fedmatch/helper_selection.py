"""FedMatch helper selection core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

FEDMATCH_HELPER_SELECTION_NAME = "prediction_similarity_topk"
FEDMATCH_DEFAULT_NUM_HELPERS = 2
FEDMATCH_DEFAULT_REFRESH_INTERVAL = 10
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


helper_selection_spec = FedMatchHelperSelectionSpec(
    selection_name=FEDMATCH_HELPER_SELECTION_NAME,
    num_helpers=FEDMATCH_DEFAULT_NUM_HELPERS,
    refresh_interval=FEDMATCH_DEFAULT_REFRESH_INTERVAL,
    original_probe_source=FEDMATCH_ORIGINAL_HELPER_PROBE,
    trace_probe_source=FEDMATCH_TRACE_HELPER_PROBE,
    similarity_name="euclidean_nearest",
)


def should_refresh_helper_context(
    *,
    round_index_zero_based: int,
    refresh_interval: int = FEDMATCH_DEFAULT_REFRESH_INTERVAL,
) -> bool:
    """원본 `(curr_round + 1) % h_interval == 0` refresh 조건."""

    if round_index_zero_based < 0:
        raise ValueError("round_index_zero_based must be non-negative.")
    if refresh_interval <= 0:
        raise ValueError("refresh_interval must be positive.")
    return (round_index_zero_based + 1) % refresh_interval == 0


def select_helper_client_ids(
    *,
    client_id: str,
    client_vectors: Mapping[str, Sequence[float]],
    num_helpers: int = FEDMATCH_DEFAULT_NUM_HELPERS,
) -> tuple[str, ...]:
    """client prediction vector 기준 최근접 helper client를 고른다."""

    if num_helpers < 0:
        raise ValueError("num_helpers must be non-negative.")
    if num_helpers == 0 or client_id not in client_vectors:
        return ()

    target = _validate_vector(client_vectors[client_id], name=client_id)
    distances: list[tuple[float, str]] = []
    for candidate_id, candidate_vector in client_vectors.items():
        if candidate_id == client_id:
            continue
        vector = _validate_vector(candidate_vector, name=candidate_id)
        if len(vector) != len(target):
            raise ValueError("all helper selection vectors must share one dimension.")
        distances.append((_squared_euclidean_distance(target, vector), candidate_id))

    distances.sort(key=lambda item: (item[0], item[1]))
    return tuple(candidate_id for _, candidate_id in distances[:num_helpers])


def _squared_euclidean_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True))


def _validate_vector(vector: Sequence[float], *, name: str) -> tuple[float, ...]:
    if not vector:
        raise ValueError(f"helper selection vector for {name!r} must not be empty.")
    return tuple(float(value) for value in vector)
