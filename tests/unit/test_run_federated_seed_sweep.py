"""FL SSL seed sweep runner tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.experiments.fl_ssl.federated_simulation.models import (
    ClientEvaluationSummary,
    SimulationEvaluation,
    SimulationResult,
    SimulationRoundSummary,
)
from scripts.experiments.fl_ssl.run_federated_seed_sweep import (
    SeedSweepRunResult,
    build_seed_sweep_summary_payload,
    resolve_seed_sweep_values,
)


def _cfg(*, seeds: list[int], seed_count: int = 3):
    return OmegaConf.create(
        {
            "report": {
                "track": "fl_ssl_main_comparison",
                "seed_count": seed_count,
            },
            "federated_run_preset": {
                "client_count": 10,
                "rounds": 50,
            },
            "ssl_method": {"name": "fedavg_pseudo_label"},
            "shard_policy": {"name": "dirichlet_label_skew"},
            "client_pool_split": {
                "labeled_ratio": 0.1,
                "unlabeled_ratio": 0.9,
            },
            "seed_sweep": {"seeds": seeds},
        }
    )


def _evaluation(*, macro_f1: float, ece: float = 0.1) -> SimulationEvaluation:
    return SimulationEvaluation(
        row_count=10,
        top1_accuracy=macro_f1,
        accepted_ratio=0.5,
        macro_f1=macro_f1,
        expected_calibration_error=ece,
    )


def _result(*, macro_f1: float, client_macro_f1: float) -> SimulationResult:
    evaluation = _evaluation(macro_f1=macro_f1)
    return SimulationResult(
        initial_model_revision="sim_rev_0000",
        initial_prototype_version="proto_sim_0000",
        initial_validation=_evaluation(macro_f1=0.1),
        final_validation=evaluation,
        rounds=(
            SimulationRoundSummary(
                round_id="round_0001",
                model_revision="sim_rev_0001",
                prototype_version="proto_sim_0001",
                update_count=2,
                validation=evaluation,
                clients=(),
            ),
        ),
        client_evaluations=(
            ClientEvaluationSummary(
                client_id="agent_01",
                validation=_evaluation(macro_f1=client_macro_f1),
            ),
        ),
        report_path="runs/seed/report.json",
    )


def test_resolve_seed_sweep_values_requires_seed_count_match() -> None:
    with pytest.raises(ValueError, match="report.seed_count"):
        resolve_seed_sweep_values(_cfg(seeds=[42, 43], seed_count=3))


def test_resolve_seed_sweep_values_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        resolve_seed_sweep_values(_cfg(seeds=[42, 42, 43], seed_count=3))


def test_build_seed_sweep_summary_payload_aggregates_runs(tmp_path: Path) -> None:
    payload = build_seed_sweep_summary_payload(
        cfg=_cfg(seeds=[42, 43], seed_count=2),
        run_id="20260506T000000Z",
        output_dir=tmp_path / "sweep",
        run_results=(
            SeedSweepRunResult(
                seed=42,
                output_dir=tmp_path / "sweep" / "seed_42",
                result=_result(macro_f1=0.2, client_macro_f1=0.15),
            ),
            SeedSweepRunResult(
                seed=43,
                output_dir=tmp_path / "sweep" / "seed_43",
                result=_result(macro_f1=0.4, client_macro_f1=0.25),
            ),
        ),
    )

    assert payload["schema_version"] == "fl_ssl_seed_sweep_summary.v1"
    assert payload["seed_count"] == 2
    assert payload["seeds"] == [42, 43]
    assert payload["aggregate"]["macro_f1_mean"] == pytest.approx(0.3)
    assert payload["aggregate"]["worst_client_macro_f1_min"] == pytest.approx(0.15)
    assert payload["aggregate"]["completed_rounds_min"] == 1
