"""FL SSL client-count sweep runner tests."""

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
from scripts.experiments.fl_ssl.run_federated_client_count_sweep import (
    ClientCountSweepRunResult,
    build_client_count_sweep_summary_payload,
    resolve_client_count_sweep_values,
)


def _cfg(*, client_counts: list[int]):
    return OmegaConf.create(
        {
            "seed": 42,
            "report": {
                "track": "fl_ssl_main_comparison",
            },
            "federated_run_budget": {
                "rounds": 50,
            },
            "ssl_method": {"name": "fedavg_pseudo_label"},
            "shard_policy": {"name": "dirichlet_label_skew"},
            "client_pool_split": {
                "labeled_ratio": 0.1,
                "unlabeled_ratio": 0.9,
            },
            "client_count_sweep": {"client_counts": client_counts},
        }
    )


def _evaluation(
    *,
    macro_f1: float,
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
        expected_calibration_error=0.1,
        max_calibration_error=0.1,
        score_distribution_kind="softmax_raw_scores_temperature_1.0",
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
        report_path="runs/client_count/report.json",
    )


def test_resolve_client_count_sweep_values_rejects_duplicate_counts() -> None:
    with pytest.raises(ValueError, match="unique"):
        resolve_client_count_sweep_values(_cfg(client_counts=[1, 2, 2]))


def test_resolve_client_count_sweep_values_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_client_count_sweep_values(_cfg(client_counts=[1, 0, 3]))


def test_build_client_count_sweep_summary_payload_aggregates_runs(
    tmp_path: Path,
) -> None:
    payload = build_client_count_sweep_summary_payload(
        cfg=_cfg(client_counts=[1, 3]),
        run_id="20260506T000000Z",
        output_dir=tmp_path / "sweep",
        run_results=(
            ClientCountSweepRunResult(
                client_count=1,
                output_dir=tmp_path / "sweep" / "clients_01",
                result=_result(macro_f1=0.2, client_macro_f1=0.2),
            ),
            ClientCountSweepRunResult(
                client_count=3,
                output_dir=tmp_path / "sweep" / "clients_03",
                result=_result(macro_f1=0.5, client_macro_f1=0.25),
            ),
        ),
    )

    assert payload["schema_version"] == "fl_ssl_client_count_sweep_summary.v1"
    assert payload["table_role"] == "client_count_sweep_summary"
    assert payload["seed"] == 42
    assert payload["client_counts"] == [1, 3]
    assert payload["aggregate"]["macro_f1_mean"] == pytest.approx(0.35)
    assert payload["aggregate"]["loss_mean"] == pytest.approx(0.65)
    assert payload["aggregate"]["best_client_count_by_macro_f1"] == 3
    assert payload["runs"][0]["client_count"] == 1
    assert payload["runs"][1]["protocol"]["client_count"] == 3
    assert payload["runs"][1]["metrics"]["primary"]["worst_client_macro_f1"] == (
        pytest.approx(0.25)
    )
