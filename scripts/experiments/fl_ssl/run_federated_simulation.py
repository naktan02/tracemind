"""합성 federation simulation을 실행한다."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig

from scripts.experiments.fl_ssl.federated_simulation.config_request import (
    build_simulation_request_from_config,
)
from scripts.experiments.fl_ssl.federated_simulation.models import SimulationResult
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    run_simulation_request,
)
from scripts.experiments.fl_ssl.federated_simulation.sweep import (
    SWEEP_AXIS_CLIENT_COUNT,
    SWEEP_AXIS_SEED,
    resolve_sweep_axis,
    run_client_count_sweep_from_config,
    run_seed_sweep_from_config,
)
from scripts.experiments.fl_ssl.support.layout import build_fl_ssl_run_dir
from scripts.experiments.fl_ssl.support.safety import require_fl_ssl_run_budget_allowed


def render_simulation_result_lines(
    *,
    output_dir: Path,
    result,
) -> list[str]:
    lines = [
        f"output_dir={output_dir}",
        f"initial_model_revision={result.initial_model_revision}",
        (
            "initial_validation="
            f"accuracy:{result.initial_validation.top1_accuracy:.4f},"
            f"loss:{result.initial_validation.loss:.4f},"
            f"macro_f1:{result.initial_validation.macro_f1:.4f},"
            f"ece:{result.initial_validation.expected_calibration_error:.4f},"
            f"accepted_ratio:{result.initial_validation.accepted_ratio:.4f}"
        ),
    ]
    if result.rounds:
        last_round = result.rounds[-1]
        lines.extend(
            [
                f"final_model_revision={last_round.model_revision}",
                (
                    "final_validation="
                    f"accuracy:{result.final_validation.top1_accuracy:.4f},"
                    f"loss:{result.final_validation.loss:.4f},"
                    f"macro_f1:{result.final_validation.macro_f1:.4f},"
                    f"ece:{result.final_validation.expected_calibration_error:.4f},"
                    f"accepted_ratio:{result.final_validation.accepted_ratio:.4f}"
                ),
                f"round_count={len(result.rounds)}",
            ]
        )
    else:
        lines.extend(
            [
                "round_count=0",
                "note=no client updates satisfied the pseudo-label selection criteria.",
            ]
        )
    if result.report_path is not None:
        lines.append(f"report_json={result.report_path}")
    return lines


def run_sweep_if_requested(cfg: DictConfig) -> bool:
    """Hydra config가 sweep을 요청하면 해당 sweep을 실행한다."""

    sweep_axis = resolve_sweep_axis(cfg)
    if sweep_axis == SWEEP_AXIS_SEED:
        run_seed_sweep_from_config(cfg, line_renderer=render_simulation_result_lines)
        return True
    if sweep_axis == SWEEP_AXIS_CLIENT_COUNT:
        run_client_count_sweep_from_config(
            cfg,
            line_renderer=render_simulation_result_lines,
        )
        return True
    return False


def resolve_single_simulation_output_dir(
    cfg: DictConfig,
    *,
    created_at: datetime | None = None,
) -> Path:
    """단일 FL SSL simulation 산출물 위치를 결정한다."""

    if bool(cfg.resume.enabled):
        if cfg.resume.run_dir is None:
            raise ValueError("resume.run_dir is required when resume.enabled=true.")
        return Path(str(cfg.resume.run_dir))

    effective_created_at = created_at or datetime.now(timezone.utc)
    run_id = effective_created_at.strftime("%Y%m%dT%H%M%SZ")
    return build_fl_ssl_run_dir(
        cfg.federated_run_budget.output_dir,
        cfg=cfg,
        run_id=run_id,
    )


def run_single_simulation_from_config(
    cfg: DictConfig,
    *,
    created_at: datetime | None = None,
) -> tuple[Path, SimulationResult]:
    """단일 FL SSL simulation을 config에서 request로 변환해 실행한다."""

    require_fl_ssl_run_budget_allowed(
        cfg,
        run_kind="single_simulation",
    )
    output_dir = resolve_single_simulation_output_dir(cfg, created_at=created_at)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    request = build_simulation_request_from_config(cfg, output_dir=output_dir)
    return output_dir, run_simulation_request(request)


def print_simulation_result(
    *,
    output_dir: Path,
    result: SimulationResult,
) -> None:
    """단일 simulation 결과 요약을 stdout에 출력한다."""

    for line in render_simulation_result_lines(output_dir=output_dir, result=result):
        print(line)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/fl_ssl/run_federated_simulation",
)
def main(cfg: DictConfig) -> None:
    if run_sweep_if_requested(cfg):
        return

    output_dir, result = run_single_simulation_from_config(cfg)
    print_simulation_result(
        output_dir=output_dir,
        result=result,
    )


if __name__ == "__main__":
    main()
