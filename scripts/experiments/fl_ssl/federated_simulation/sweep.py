"""FL SSL 반복 실행 sweep helper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scripts.experiments.fl_ssl.federated_simulation.config_request import (
    build_simulation_request_from_config,
)
from scripts.experiments.fl_ssl.federated_simulation.io.report_math import mean
from scripts.experiments.fl_ssl.federated_simulation.io.sweep_summary import (
    build_sweep_run_payload,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT,
    FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN,
    SimulationResult,
)
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    run_simulation_request,
)
from scripts.experiments.fl_ssl.support.layout import (
    build_fl_ssl_client_count_sweep_member_dir,
    build_fl_ssl_run_dir,
    build_fl_ssl_seed_sweep_member_dir,
)
from scripts.experiments.fl_ssl.support.safety import require_fl_ssl_run_budget_allowed

SEED_SWEEP_SUMMARY_SCHEMA_VERSION = "fl_ssl_seed_sweep_summary.v2"
CLIENT_COUNT_SWEEP_SUMMARY_SCHEMA_VERSION = "fl_ssl_client_count_sweep_summary.v1"
SWEEP_AXIS_NONE = "none"
SWEEP_AXIS_SEED = "seed"
SWEEP_AXIS_CLIENT_COUNT = "client_count"


@dataclass(frozen=True, slots=True)
class SeedSweepRunResult:
    """seed 하나의 simulation 결과와 산출물 위치."""

    seed: int
    output_dir: Path
    result: SimulationResult


@dataclass(frozen=True, slots=True)
class ClientCountSweepRunResult:
    """client_count 하나의 simulation 결과와 산출물 위치."""

    client_count: int
    output_dir: Path
    result: SimulationResult


def resolve_sweep_axis(cfg: DictConfig) -> str:
    """실행 진입점이 수행할 sweep axis를 정규화한다."""

    axis = str(
        OmegaConf.select(cfg, "sweep.axis", default=SWEEP_AXIS_NONE) or ""
    ).strip()
    if axis in {"", SWEEP_AXIS_NONE}:
        return SWEEP_AXIS_NONE
    if axis in {SWEEP_AXIS_SEED, SWEEP_AXIS_CLIENT_COUNT}:
        return axis
    raise ValueError(
        "sweep.axis must be one of "
        f"{SWEEP_AXIS_NONE!r}, {SWEEP_AXIS_SEED!r}, {SWEEP_AXIS_CLIENT_COUNT!r}. "
        f"Got {axis!r}."
    )


def resolve_seed_sweep_values(cfg: DictConfig) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in cfg.sweep.seed.members)
    if not values:
        raise ValueError("sweep.seed.members must not be empty.")
    if len(set(values)) != len(values):
        raise ValueError("sweep.seed.members must be unique.")
    seed_count = int(cfg.report.seed_count)
    if len(values) != seed_count:
        raise ValueError(
            "sweep.seed.members length must match report.seed_count: "
            f"{len(values)} != {seed_count}."
        )
    return values


def resolve_client_count_sweep_values(cfg: DictConfig) -> tuple[int, ...]:
    values = tuple(int(client_count) for client_count in cfg.sweep.client_count.members)
    if not values:
        raise ValueError("sweep.client_count.members must not be empty.")
    if len(set(values)) != len(values):
        raise ValueError("sweep.client_count.members must be unique.")
    invalid_values = [client_count for client_count in values if client_count < 1]
    if invalid_values:
        raise ValueError(
            "sweep.client_count.members must be positive: "
            f"{invalid_values}."
        )
    return values


def resolve_client_count_sweep_split_manifests(
    cfg: DictConfig,
    *,
    client_counts: tuple[int, ...],
) -> dict[int, str]:
    if _fl_data_source_mode(cfg) != FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT:
        return {}

    mapping_cfg = cfg.sweep.client_count.get("split_manifest_by_client_count")
    if mapping_cfg is None:
        raise ValueError(
            "sweep.client_count.split_manifest_by_client_count is required "
            "when fl_data.source_mode is materialized_client_split."
        )

    mapping = _normalize_client_count_split_manifest_mapping(mapping_cfg)
    missing_counts = [
        client_count for client_count in client_counts if client_count not in mapping
    ]
    if missing_counts:
        raise ValueError(
            "sweep.client_count.split_manifest_by_client_count must include "
            "a split manifest for every client_count when fl_data.source_mode "
            f"is materialized_client_split. Missing client_count keys: "
            f"{missing_counts}."
        )
    return mapping


def run_seed_sweep_from_config(
    cfg: DictConfig,
    *,
    created_at: datetime | None = None,
    line_renderer,
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
        cfg.sweep.output_dir,
        cfg=cfg,
        run_id=run_id,
        run_kind="seed_sweep",
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    run_results: list[SeedSweepRunResult] = []
    for seed in seeds:
        seed_output_dir = build_fl_ssl_seed_sweep_member_dir(output_dir, seed=seed)
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        result = run_simulation_request(
            build_simulation_request_from_config(
                cfg,
                output_dir=seed_output_dir,
                seed=seed,
            )
        )
        run_results.append(
            SeedSweepRunResult(seed=seed, output_dir=seed_output_dir, result=result)
        )
        for line in line_renderer(output_dir=seed_output_dir, result=result):
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


def run_client_count_sweep_from_config(
    cfg: DictConfig,
    *,
    created_at: datetime | None = None,
    line_renderer,
) -> dict[str, object]:
    client_counts = resolve_client_count_sweep_values(cfg)
    split_manifest_by_client_count = resolve_client_count_sweep_split_manifests(
        cfg,
        client_counts=client_counts,
    )
    require_fl_ssl_run_budget_allowed(
        cfg,
        run_kind="client_count_sweep",
        planned_run_count=len(client_counts),
    )

    effective_created_at = created_at or datetime.now(timezone.utc)
    run_id = effective_created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = build_fl_ssl_run_dir(
        cfg.sweep.output_dir,
        cfg=cfg,
        run_id=run_id,
        run_kind="client_count_sweep",
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    run_results: list[ClientCountSweepRunResult] = []
    for client_count in client_counts:
        client_output_dir = build_fl_ssl_client_count_sweep_member_dir(
            output_dir,
            client_count=client_count,
        )
        client_output_dir.mkdir(parents=True, exist_ok=True)
        run_cfg = _copy_config_with_client_count(
            cfg,
            client_count=client_count,
            split_manifest=split_manifest_by_client_count.get(client_count),
        )
        result = run_simulation_request(
            build_simulation_request_from_config(
                run_cfg,
                output_dir=client_output_dir,
            )
        )
        run_results.append(
            ClientCountSweepRunResult(
                client_count=client_count,
                output_dir=client_output_dir,
                result=result,
            )
        )
        for line in line_renderer(output_dir=client_output_dir, result=result):
            print(f"client_count={client_count} {line}")

    summary = build_client_count_sweep_summary_payload(
        cfg=cfg,
        run_id=run_id,
        output_dir=output_dir,
        run_results=tuple(run_results),
    )
    summary_path = output_dir / "reports" / "fl_ssl_client_count_sweep.summary.json"
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
        {
            "seed": run_result.seed,
            **build_sweep_run_payload(
                result=run_result.result,
                output_dir=run_result.output_dir,
            ),
        }
        for run_result in run_results
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
        "schema_version": SEED_SWEEP_SUMMARY_SCHEMA_VERSION,
        "track": str(cfg.report.track),
        "table_role": "seed_sweep_summary",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "seed_count": int(cfg.report.seed_count),
        "seeds": [run_result.seed for run_result in run_results],
        "protocol": {
            "client_count": int(cfg.federated_run_budget.client_count),
            "round_budget": int(cfg.federated_run_budget.rounds),
            "ssl_method": _select_config_value(
                cfg,
                "query_ssl_method.name",
                default="manual",
            ),
            "fl_composition_mode": _select_config_value(
                cfg,
                "fl_method.composition_mode",
                default="manual",
            ),
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


def build_client_count_sweep_summary_payload(
    *,
    cfg: DictConfig,
    run_id: str,
    output_dir: Path,
    run_results: tuple[ClientCountSweepRunResult, ...],
) -> dict[str, object]:
    run_payloads = tuple(
        {
            "client_count": run_result.client_count,
            **build_sweep_run_payload(
                result=run_result.result,
                output_dir=run_result.output_dir,
                protocol={"client_count": run_result.client_count},
            ),
        }
        for run_result in run_results
    )
    macro_f1_values = [
        float(payload["metrics"]["primary"]["macro_f1"]) for payload in run_payloads
    ]
    loss_values = [
        float(payload["metrics"]["secondary"]["loss"]) for payload in run_payloads
    ]
    return {
        "schema_version": CLIENT_COUNT_SWEEP_SUMMARY_SCHEMA_VERSION,
        "track": str(cfg.report.track),
        "table_role": "client_count_sweep_summary",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        "client_counts": [run_result.client_count for run_result in run_results],
        "protocol": {
            "round_budget": int(cfg.federated_run_budget.rounds),
            "ssl_method": _select_config_value(
                cfg,
                "query_ssl_method.name",
                default="manual",
            ),
            "fl_composition_mode": _select_config_value(
                cfg,
                "fl_method.composition_mode",
                default="manual",
            ),
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
            "best_client_count_by_macro_f1": _best_client_count_by_macro_f1(
                run_payloads
            ),
        },
        "runs": list(run_payloads),
    }


def _copy_config_with_client_count(
    cfg: DictConfig,
    *,
    client_count: int,
    split_manifest: str | None = None,
) -> DictConfig:
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)
    run_cfg = OmegaConf.create(raw_cfg)
    run_cfg.federated_run_budget.client_count = client_count
    if _fl_data_source_mode(run_cfg) == FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT:
        if split_manifest is None:
            split_manifest = resolve_client_count_sweep_split_manifests(
                run_cfg,
                client_counts=(client_count,),
            )[client_count]
        run_cfg.fl_data.split_manifest = split_manifest
    return run_cfg


def _fl_data_source_mode(cfg: DictConfig) -> str:
    fl_data_cfg = cfg.get("fl_data", {})
    return str(fl_data_cfg.get("source_mode", FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN))


def _select_config_value(
    cfg: DictConfig,
    key: str,
    *,
    default: object,
) -> str:
    return str(OmegaConf.select(cfg, key, default=default))


def _normalize_client_count_split_manifest_mapping(
    mapping_cfg: object,
) -> dict[int, str]:
    raw_mapping = (
        OmegaConf.to_container(mapping_cfg, resolve=True)
        if isinstance(mapping_cfg, DictConfig)
        else mapping_cfg
    )
    if not isinstance(raw_mapping, dict):
        raise ValueError(
            "sweep.client_count.split_manifest_by_client_count must be a mapping "
            "from client_count to fl_data.split_manifest path."
        )

    mapping: dict[int, str] = {}
    for raw_key, raw_value in raw_mapping.items():
        try:
            client_count = int(raw_key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "sweep.client_count.split_manifest_by_client_count keys must be "
                f"integer client_count values. Got {raw_key!r}."
            ) from error
        if client_count < 1:
            raise ValueError(
                "sweep.client_count.split_manifest_by_client_count keys must be "
                f"positive client_count values. Got {raw_key!r}."
            )
        if client_count in mapping:
            raise ValueError(
                "sweep.client_count.split_manifest_by_client_count contains "
                f"duplicate key for client_count={client_count}."
            )
        if raw_value is None or str(raw_value) == "":
            raise ValueError(
                "sweep.client_count.split_manifest_by_client_count values must be "
                f"non-empty split manifest paths. Got empty value for "
                f"client_count={client_count}."
            )
        mapping[client_count] = str(raw_value)
    return mapping


def _best_client_count_by_macro_f1(
    run_payloads: tuple[dict[str, object], ...],
) -> int | None:
    if not run_payloads:
        return None
    best_payload = max(
        run_payloads,
        key=lambda payload: float(payload["metrics"]["primary"]["macro_f1"]),
    )
    return int(best_payload["client_count"])
