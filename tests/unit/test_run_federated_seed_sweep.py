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
            "federated_run_budget": {
                "client_count": 10,
                "rounds": 50,
            },
            "query_ssl_method": {"name": "fixmatch_usb_v1"},
            "fl_method": {"composition_mode": "manual"},
            "shard_policy": {"name": "dirichlet_label_skew"},
            "client_pool_split": {
                "labeled_ratio": 0.1,
                "unlabeled_ratio": 0.9,
            },
            "seed_sweep": {"seeds": seeds},
        }
    )


def _evaluation(
    *,
    macro_f1: float,
    ece: float = 0.1,
    loss: float | None = None,
) -> SimulationEvaluation:
    effective_loss = (1.0 - macro_f1) if loss is None else loss
    return SimulationEvaluation(
        row_count=10,
        top1_accuracy=macro_f1,
        accepted_ratio=0.5,
        loss=effective_loss,
        loss_kind="negative_log_likelihood_from_score_distribution",
        accuracy_top_1=macro_f1,
        correct_top_1=round(macro_f1 * 10),
        macro_f1=macro_f1,
        macro_precision=macro_f1,
        macro_recall=macro_f1,
        weighted_f1=macro_f1,
        balanced_accuracy=macro_f1,
        expected_calibration_error=ece,
        max_calibration_error=ece,
        score_distribution_kind="softmax_raw_scores_temperature_1.0",
    )


def _result(*, macro_f1: float, client_macro_f1: float) -> SimulationResult:
    evaluation = _evaluation(macro_f1=macro_f1)
    return SimulationResult(
        initial_model_revision="sim_rev_0000",
        initial_validation=_evaluation(macro_f1=0.1),
        final_validation=evaluation,
        rounds=(
            SimulationRoundSummary(
                round_id="round_0001",
                model_revision="sim_rev_0001",
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

    assert payload["schema_version"] == "fl_ssl_seed_sweep_summary.v2"
    assert payload["seed_count"] == 2
    assert payload["seeds"] == [42, 43]
    assert payload["aggregate"]["macro_f1_mean"] == pytest.approx(0.3)
    assert payload["aggregate"]["loss_mean"] == pytest.approx(0.7)
    assert payload["aggregate"]["weighted_f1_mean"] == pytest.approx(0.3)
    assert payload["aggregate"]["worst_client_macro_f1_min"] == pytest.approx(0.15)
    assert payload["aggregate"]["completed_rounds_min"] == 1
    assert payload["runs"][0]["metrics"]["secondary"]["loss"] == pytest.approx(0.8)
    communication_cost = payload["runs"][0]["metrics"]["secondary"][
        "communication_cost"
    ]
    assert communication_cost["status"] == "proxy_until_payload_byte_accounting"
    assert communication_cost["total_candidates"] == 0
    assert payload["runs"][0]["metrics"]["final_validation"]["loss_kind"] == (
        "negative_log_likelihood_from_score_distribution"
    )
    assert "confusion_matrix" in payload["runs"][0]["metrics"]["final_validation"]
