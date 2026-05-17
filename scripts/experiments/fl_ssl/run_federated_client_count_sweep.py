"""FL SSL simulation client-count sweep entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.artifacts.run_artifacts import build_run_dir
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
from scripts.experiments.fl_ssl.run_federated_simulation import (
    build_simulation_request_from_config,
    render_simulation_result_lines,
)
from scripts.experiments.fl_ssl.run_safety import require_fl_ssl_run_budget_allowed

SUMMARY_SCHEMA_VERSION = "fl_ssl_client_count_sweep_summary.v1"


@dataclass(frozen=True, slots=True)
class ClientCountSweepRunResult:
    """client_count 하나의 simulation 결과와 산출물 위치."""

    client_count: int
    output_dir: Path
    result: SimulationResult


def resolve_client_count_sweep_values(cfg: DictConfig) -> tuple[int, ...]:
    client_counts = tuple(
        int(client_count) for client_count in cfg.client_count_sweep.client_counts
    )
    if not client_counts:
        raise ValueError("client_count_sweep.client_counts must not be empty.")
    if len(set(client_counts)) != len(client_counts):
        raise ValueError("client_count_sweep.client_counts must be unique.")
    invalid_counts = [
        client_count for client_count in client_counts if client_count < 1
    ]
    if invalid_counts:
        raise ValueError(
            f"client_count_sweep.client_counts must be positive: {invalid_counts}."
        )
    return client_counts


def resolve_client_count_sweep_split_manifests(
    cfg: DictConfig,
    *,
    client_counts: tuple[int, ...],
) -> dict[int, str]:
    if _fl_data_source_mode(cfg) != FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT:
        return {}

    mapping_cfg = cfg.client_count_sweep.get("split_manifest_by_client_count")
    if mapping_cfg is None:
        raise ValueError(
            "client_count_sweep.split_manifest_by_client_count is required "
            "when fl_data.source_mode is materialized_client_split."
        )

    mapping = _normalize_client_count_split_manifest_mapping(mapping_cfg)
    missing_counts = [
        client_count for client_count in client_counts if client_count not in mapping
    ]
    if missing_counts:
        raise ValueError(
            "client_count_sweep.split_manifest_by_client_count must include "
            "a split manifest for every client_count when fl_data.source_mode "
            f"is materialized_client_split. Missing client_count keys: "
            f"{missing_counts}."
        )
    return mapping


def run_client_count_sweep_from_config(
    cfg: DictConfig,
    *,
    created_at: datetime | None = None,
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
    output_dir = build_run_dir(
        cfg.client_count_sweep.output_dir,
        run_id=run_id,
        created_at=effective_created_at,
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    run_results: list[ClientCountSweepRunResult] = []
    for client_count in client_counts:
        client_output_dir = output_dir / f"clients_{client_count:02d}"
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
        for line in render_simulation_result_lines(
            output_dir=client_output_dir,
            result=result,
        ):
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


def build_client_count_sweep_summary_payload(
    *,
    cfg: DictConfig,
    run_id: str,
    output_dir: Path,
    run_results: tuple[ClientCountSweepRunResult, ...],
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
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "track": str(cfg.report.track),
        "table_role": "client_count_sweep_summary",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "seed": int(cfg.seed),
        "client_counts": [run_result.client_count for run_result in run_results],
        "protocol": {
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
            "client_count_sweep.split_manifest_by_client_count must be a mapping "
            "from client_count to fl_data.split_manifest path."
        )

    mapping: dict[int, str] = {}
    for raw_key, raw_value in raw_mapping.items():
        try:
            client_count = int(raw_key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "client_count_sweep.split_manifest_by_client_count keys must be "
                f"integer client_count values. Got {raw_key!r}."
            ) from error
        if client_count < 1:
            raise ValueError(
                "client_count_sweep.split_manifest_by_client_count keys must be "
                f"positive client_count values. Got {raw_key!r}."
            )
        if client_count in mapping:
            raise ValueError(
                "client_count_sweep.split_manifest_by_client_count contains "
                f"duplicate key for client_count={client_count}."
            )
        if raw_value is None or str(raw_value) == "":
            raise ValueError(
                "client_count_sweep.split_manifest_by_client_count values must be "
                f"non-empty split manifest paths. Got empty value for "
                f"client_count={client_count}."
            )
        mapping[client_count] = str(raw_value)
    return mapping


def _run_result_to_payload(
    run_result: ClientCountSweepRunResult,
) -> dict[str, object]:
    return {
        "client_count": run_result.client_count,
        **build_sweep_run_payload(
            result=run_result.result,
            output_dir=run_result.output_dir,
            protocol={"client_count": run_result.client_count},
        ),
    }


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


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/fl_ssl/run_federated_simulation",
)
def main(cfg: DictConfig) -> None:
    run_client_count_sweep_from_config(cfg)


if __name__ == "__main__":
    main()
