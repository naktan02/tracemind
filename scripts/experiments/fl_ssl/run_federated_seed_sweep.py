"""FL SSL simulation seed sweep entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig

from scripts.experiments.fl_ssl.federated_simulation.config_request import (
    build_simulation_request_from_config,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_math import (
    mean,
)
from scripts.experiments.fl_ssl.federated_simulation.io.sweep_summary import (
    build_sweep_run_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationResult,
)
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    run_simulation_request,
)
from scripts.experiments.fl_ssl.run_federated_simulation import (
    render_simulation_result_lines,
)
from scripts.experiments.fl_ssl.run_layout import (
    build_fl_ssl_run_dir,
    build_fl_ssl_seed_sweep_member_dir,
)
from scripts.experiments.fl_ssl.run_safety import require_fl_ssl_run_budget_allowed

SUMMARY_SCHEMA_VERSION = "fl_ssl_seed_sweep_summary.v2"


@dataclass(frozen=True, slots=True)
class SeedSweepRunResult:
    """seed 하나의 simulation 결과와 산출물 위치."""

    seed: int
    output_dir: Path
    result: SimulationResult


def resolve_seed_sweep_values(cfg: DictConfig) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in cfg.seed_sweep.seeds)
    if not seeds:
        raise ValueError("seed_sweep.seeds must not be empty.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed_sweep.seeds must be unique.")
    seed_count = int(cfg.report.seed_count)
    if len(seeds) != seed_count:
        raise ValueError(
            "seed_sweep.seeds length must match report.seed_count: "
            f"{len(seeds)} != {seed_count}."
        )
    return seeds


def run_seed_sweep_from_config(
    cfg: DictConfig,
    *,
    created_at: datetime | None = None,
) -> dict[str, object]:
    seeds = resolve_seed_sweep_values(cfg)
    require_fl_ssl_run_budget_allowed(
        cfg,
        run_kind="seed_sweep",
        planned_run_count=len(seeds),
    )
    effective_created_at = created_at or datetime.now(timezone.utc)
    run_id = effective_created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = build_fl_ssl_run_dir(
        cfg.seed_sweep.output_dir,
        cfg=cfg,
        run_id=run_id,
        run_kind="seed_sweep",
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    run_results: list[SeedSweepRunResult] = []
    for seed in seeds:
        seed_output_dir = build_fl_ssl_seed_sweep_member_dir(
            output_dir,
            seed=seed,
        )
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        result = run_simulation_request(
            build_simulation_request_from_config(
                cfg,
                output_dir=seed_output_dir,
                seed=seed,
            )
        )
        run_results.append(
            SeedSweepRunResult(
                seed=seed,
                output_dir=seed_output_dir,
                result=result,
            )
        )
        for line in render_simulation_result_lines(
            output_dir=seed_output_dir,
            result=result,
        ):
            print(f"seed={seed} {line}")

    summary = build_seed_sweep_summary_payload(
        cfg=cfg,
        run_id=run_id,
        output_dir=output_dir,
        run_results=tuple(run_results),
    )
    summary_path = output_dir / "reports" / "fl_ssl_seed_sweep.summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"sweep_output_dir={output_dir}")
    print(f"sweep_summary_json={summary_path}")
    return summary


def build_seed_sweep_summary_payload(
    *,
    cfg: DictConfig,
    run_id: str,
    output_dir: Path,
    run_results: tuple[SeedSweepRunResult, ...],
) -> dict[str, object]:
    run_payloads = tuple(
        _run_result_to_payload(run_result) for run_result in run_results
    )
    macro_f1_values = [
        float(payload["metrics"]["primary"]["macro_f1"]) for payload in run_payloads
    ]
    loss_values = [
        float(payload["metrics"]["secondary"]["loss"]) for payload in run_payloads
    ]
    weighted_f1_values = [
        float(payload["metrics"]["secondary"]["weighted_f1"])
        for payload in run_payloads
    ]
    worst_client_macro_f1_values = [
        float(payload["metrics"]["primary"]["worst_client_macro_f1"])
        for payload in run_payloads
        if payload["metrics"]["primary"]["worst_client_macro_f1"] is not None
    ]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "track": str(cfg.report.track),
        "table_role": "seed_sweep_summary",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "seed_count": int(cfg.report.seed_count),
        "seeds": [run_result.seed for run_result in run_results],
        "protocol": {
            "client_count": int(cfg.federated_run_budget.client_count),
            "round_budget": int(cfg.federated_run_budget.rounds),
            "ssl_method": str(cfg.ssl_method.name),
            "shard_policy": str(cfg.shard_policy.name),
            "labeled_ratio": float(cfg.client_pool_split.labeled_ratio),
            "unlabeled_ratio": float(cfg.client_pool_split.unlabeled_ratio),
        },
        "aggregate": {
            "macro_f1_mean": mean(macro_f1_values),
            "macro_f1_min": min(macro_f1_values) if macro_f1_values else None,
            "macro_f1_max": max(macro_f1_values) if macro_f1_values else None,
            "loss_mean": mean(loss_values),
            "loss_min": min(loss_values) if loss_values else None,
            "loss_max": max(loss_values) if loss_values else None,
            "weighted_f1_mean": mean(weighted_f1_values),
            "worst_client_macro_f1_mean": mean(worst_client_macro_f1_values),
            "worst_client_macro_f1_min": (
                min(worst_client_macro_f1_values)
                if worst_client_macro_f1_values
                else None
            ),
            "completed_rounds_min": (
                min(
                    int(payload["protocol"]["completed_rounds"])
                    for payload in run_payloads
                )
                if run_payloads
                else None
            ),
            "completed_rounds_max": (
                max(
                    int(payload["protocol"]["completed_rounds"])
                    for payload in run_payloads
                )
                if run_payloads
                else None
            ),
        },
        "runs": list(run_payloads),
    }


def _run_result_to_payload(run_result: SeedSweepRunResult) -> dict[str, object]:
    return {
        "seed": run_result.seed,
        **build_sweep_run_payload(
            result=run_result.result,
            output_dir=run_result.output_dir,
        ),
    }


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/fl_ssl/run_federated_simulation",
)
def main(cfg: DictConfig) -> None:
    run_seed_sweep_from_config(cfg)


if __name__ == "__main__":
    main()
