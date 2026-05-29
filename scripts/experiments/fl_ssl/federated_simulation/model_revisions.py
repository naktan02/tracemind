"""FL simulation model revision naming policy."""

from __future__ import annotations

SIMULATION_MODEL_REVISION_PREFIX = "sim_rev"
INITIAL_SIMULATION_ROUND_INDEX = 0


def build_simulation_model_revision(round_index: int) -> str:
    """communication round index를 stable simulation model revision으로 변환한다."""

    return f"{SIMULATION_MODEL_REVISION_PREFIX}_{int(round_index):04d}"


def build_active_model_revision_for_round(round_index: int) -> str:
    """round 시작 시 client가 받은 active global model revision을 반환한다."""

    return build_simulation_model_revision(max(0, int(round_index) - 1))


INITIAL_SIMULATION_MODEL_REVISION = build_simulation_model_revision(
    INITIAL_SIMULATION_ROUND_INDEX
)
