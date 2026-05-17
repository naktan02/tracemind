"""FL SSL 실행 budget safety guard."""

from __future__ import annotations

from omegaconf import DictConfig

DEFAULT_LONG_RUN_ACK = "ALLOW_FL_SSL_LONG_RUN"
DEFAULT_MAX_TOTAL_ROUNDS_WITHOUT_ACK = 49


def require_fl_ssl_run_budget_allowed(
    cfg: DictConfig,
    *,
    run_kind: str,
    planned_run_count: int = 1,
) -> None:
    """명시 승인 없이 장시간 FL SSL 실행이 시작되지 않게 막는다."""

    single_run_rounds = int(cfg.federated_run_budget.rounds)
    total_planned_rounds = single_run_rounds * int(planned_run_count)
    safety_cfg = cfg.get("run_safety", {})
    max_total_rounds = int(
        safety_cfg.get(
            "max_total_rounds_without_ack",
            DEFAULT_MAX_TOTAL_ROUNDS_WITHOUT_ACK,
        )
    )
    if total_planned_rounds <= max_total_rounds:
        return

    required_ack = str(
        safety_cfg.get("required_long_run_ack", DEFAULT_LONG_RUN_ACK)
    ).strip()
    allow_long_run = bool(safety_cfg.get("allow_long_run", False))
    long_run_ack = str(safety_cfg.get("long_run_ack") or "").strip()
    if allow_long_run and long_run_ack == required_ack:
        return

    raise ValueError(
        "FL SSL long-run guard blocked this execution. "
        f"run_kind={run_kind} single_run_rounds={single_run_rounds} "
        f"planned_run_count={planned_run_count} "
        f"total_planned_rounds={total_planned_rounds} "
        f"max_total_rounds_without_ack={max_total_rounds}. "
        "To run intentionally, set "
        "run_safety.allow_long_run=true and "
        f"run_safety.long_run_ack={required_ack!r}."
    )
