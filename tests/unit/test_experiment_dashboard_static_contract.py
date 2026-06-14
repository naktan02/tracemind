"""Experiment dashboard 정적 JS selector 계약 검증."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_central_dashboard_selectors_keep_ssl_and_supervised_tracks_separate() -> None:
    script = """
        const {
          centralEvalSets,
          centralMetricRows,
          isCentralSslResultTrack,
          isCentralSupervisedTrack,
        } = await import(
          "./apps/experiment_dashboard/src/features/central_ssl/logic/selectors.js"
        );

        const run = (run_id, track, method_name, algorithm_name = null) => ({
          run_id,
          track,
          method_name,
          algorithm_name,
        });
        const metric = (run_id, eval_set, macro_f1) => ({
          run_id,
          eval_set,
          macro_f1,
        });

        const bundle = {
          runs: [
            run("ssl_run", "central_peft_ssl", "fixmatch", "fixmatch"),
            run("peft_sup", "central_peft_supervised", "supervised"),
            run("full_sup", "central_full_text_encoder_supervised", "supervised"),
          ],
          eval_metrics: [
            metric("ssl_run", "validation", 0.7),
            metric("peft_sup", "best", 0.8),
            metric("peft_sup", "test", 0.75),
            metric("full_sup", "final", 0.82),
          ],
        };

        const result = {
          sslRows: centralMetricRows(
            bundle,
            "validation",
            "macro_f1",
            isCentralSslResultTrack,
          ).map((row) => row.run_id),
          peftSupIsSsl: isCentralSslResultTrack(bundle.runs[1]),
          fullSupIsSupervised: isCentralSupervisedTrack(bundle.runs[2]),
          supervisedEvalSets: centralEvalSets(
            bundle,
            isCentralSupervisedTrack,
            { hideTest: false },
          ),
        };
        console.log(JSON.stringify(result));
    """

    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result == {
        "sslRows": ["ssl_run"],
        "peftSupIsSsl": False,
        "fullSupIsSupervised": True,
        "supervisedEvalSets": ["best", "final", "test"],
    }


def test_central_dashboard_entrypoint_uses_ssl_only_predicate_for_ssl_track() -> None:
    source = (
        REPO_ROOT / "apps" / "experiment_dashboard" / "src" / "main.js"
    ).read_text(encoding="utf-8")

    assert 'if (state.activeTrack === "central_ssl")' in source
    assert "return isCentralSslResultTrack;" in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_central_dashboard_labels_are_track_aware() -> None:
    script = """
        const {
          evaluationDataLabel,
          labelBudgetLabel,
          runDetail,
          trainingDataLabel,
        } = await import(
          "./apps/experiment_dashboard/src/features/central_ssl/logic/labels.js"
        );

        const supervised = {
          run_id: "peft_clf_2026_06_15_031353",
          track: "central_peft_supervised",
          method_family: "peft_classifier",
          method_name: "supervised",
          labeled_dataset_name: "szegeelim_general4",
          unlabeled_dataset_name: "ourafla_reddit",
          validation_dataset_name: "ourafla_reddit",
          test_dataset_name: "ourafla_reddit",
          label_budget_name: "pc100",
          seed: 42,
        };
        const ssl = {
          ...supervised,
          run_id: "peft_fixmatch_2026_06_04_120231",
          track: "central_peft_ssl",
          method_name: "fixmatch",
          algorithm_name: "fixmatch",
          label_budget_name: "pc1024",
        };

        console.log(JSON.stringify({
          supervisedBudget: labelBudgetLabel(supervised),
          supervisedTrain: trainingDataLabel(supervised),
          supervisedEval: evaluationDataLabel(supervised),
          supervisedDetail: runDetail(supervised),
          sslTrain: trainingDataLabel(ssl),
          sslEval: evaluationDataLabel(ssl),
        }));
    """

    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result["supervisedBudget"] == "pc100"
    assert result["supervisedTrain"] == "labeled=szegeelim_general4 pc100"
    assert result["supervisedEval"] == "test=ourafla_reddit"
    assert "unlabeled=" not in result["supervisedDetail"]
    assert "validation=" not in result["supervisedDetail"]
    assert result["sslTrain"] == (
        "labeled=szegeelim_general4 pc1024 · unlabeled=ourafla_reddit"
    )
    assert result["sslEval"] == "validation=ourafla_reddit · test=ourafla_reddit"


def test_central_dashboard_confusion_matrix_uses_result_index_field_names() -> None:
    source = (
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "central_ssl"
        / "ui"
        / "detail_page.js"
    ).read_text(encoding="utf-8")

    assert "actual_category" in source
    assert "predicted_category" in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_selected_run_card_uses_single_app_tooltip() -> None:
    script = """
        const { renderSelectedRunCard } = await import(
          "./apps/experiment_dashboard/src/ui/controls/selected_run_card.js"
        );
        const html = renderSelectedRunCard({
          id: "run_1",
          label: "run label",
          detail: "long run detail",
          aliasValue: "",
          aliasPlaceholder: "alias",
          aliasDataAttribute: "alias-run-id",
          aliasAriaLabel: "alias label",
          removeDataAttribute: "remove-run-id",
          removeAriaLabel: "remove label",
        });
        console.log(JSON.stringify({
          hasNativeTitle: /\\stitle=/.test(html),
          tooltipCount: (html.match(/selected-run-detail/g) ?? []).length,
        }));
    """

    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result == {"hasNativeTitle": False, "tooltipCount": 1}
