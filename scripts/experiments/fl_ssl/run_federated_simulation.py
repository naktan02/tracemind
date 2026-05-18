"""합성 federation simulation을 실행한다."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig

from scripts.experiments.fl_ssl.federated_simulation.config_request import (
    build_simulation_request_from_config,
)
from scripts.experiments.fl_ssl.federated_simulation.simulation import (
    run_simulation_request,
)
from scripts.experiments.fl_ssl.run_layout import build_fl_ssl_run_dir
from scripts.experiments.fl_ssl.run_safety import require_fl_ssl_run_budget_allowed


def render_simulation_result_lines(
    *,
    output_dir: Path,
    result,
) -> list[str]:
    lines = [
        f"output_dir={output_dir}",
        f"initial_model_revision={result.initial_model_revision}",
        f"initial_prototype_version={result.initial_prototype_version}",
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
                f"final_prototype_version={last_round.prototype_version}",
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


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/fl_ssl/run_federated_simulation",
)
def main(cfg: DictConfig) -> None:
    require_fl_ssl_run_budget_allowed(
        cfg,
        run_kind="single_simulation",
    )
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = build_fl_ssl_run_dir(
        cfg.federated_run_budget.output_dir,
        cfg=cfg,
        run_id=run_id,
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    result = run_simulation_request(
        build_simulation_request_from_config(
            cfg,
            output_dir=output_dir,
        )
    )
    for line in render_simulation_result_lines(output_dir=output_dir, result=result):
        print(line)


if __name__ == "__main__":
    main()
