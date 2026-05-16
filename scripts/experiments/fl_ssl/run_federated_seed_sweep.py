"""FL SSL simulation seed sweep entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig

from scripts.artifacts.run_artifacts import build_run_dir
from scripts.experiments.fl_ssl.federated_simulation.io.report_math import (
    mean,
    population_variance,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_metrics import (
    build_communication_cost_summary,
    evaluation_to_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    SimulationResult,
)
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    run_simulation_request,
)
from scripts.experiments.fl_ssl.run_federated_simulation import (
    build_simulation_request_from_config,
    render_simulation_result_lines,
)

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
    effective_created_at = created_at or datetime.now(timezone.utc)
    run_id = effective_created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = build_run_dir(
        cfg.seed_sweep.output_dir,
        run_id=run_id,
        created_at=effective_created_at,
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    run_results: list[SeedSweepRunResult] = []
    for seed in resolve_seed_sweep_values(cfg):
        seed_output_dir = output_dir / f"seed_{seed}"
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
    result = run_result.result
    return {
        "seed": run_result.seed,
        "output_dir": str(run_result.output_dir),
        "report_path": result.report_path,
        "protocol": {
            "completed_rounds": len(result.rounds),
            "initial_model_revision": result.initial_model_revision,
            "initial_prototype_version": result.initial_prototype_version,
        },
        "metrics": {
            "primary": {
                "macro_f1": result.final_validation.macro_f1,
                "worst_client_macro_f1": _worst_client_macro_f1(
                    result.client_evaluations
                ),
            },
            "secondary": {
                "loss": result.final_validation.loss,
                "weighted_f1": result.final_validation.weighted_f1,
                "balanced_accuracy": result.final_validation.balanced_accuracy,
                "worst_category_f1_value": (
                    result.final_validation.worst_category_f1_value
                ),
                "max_calibration_error": result.final_validation.max_calibration_error,
                "expected_calibration_error": (
                    result.final_validation.expected_calibration_error
                ),
                "communication_cost": build_communication_cost_summary(result),
                "per_client_macro_f1_variance": population_variance(
                    [
                        client.validation.macro_f1
                        for client in result.client_evaluations
                        if client.validation.row_count > 0
                    ]
                ),
            },
            "initial_validation": evaluation_to_payload(result.initial_validation),
            "final_validation": evaluation_to_payload(result.final_validation),
        },
    }


def _worst_client_macro_f1(
    client_evaluations: tuple[ClientEvaluationSummary, ...],
) -> float | None:
    values = [
        client.validation.macro_f1
        for client in client_evaluations
        if client.validation.row_count > 0
    ]
    return min(values) if values else None


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/fl_ssl/run_federated_simulation",
)
def main(cfg: DictConfig) -> None:
    run_seed_sweep_from_config(cfg)


if __name__ == "__main__":
    main()
