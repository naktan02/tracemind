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
          centralPerClassEvalSets,
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
          sslPerClassEvalSets: centralPerClassEvalSets(
            {
              ...bundle,
              per_class_metrics: [
                { run_id: "ssl_run", eval_set: "test", category: "normal" },
                { run_id: "peft_sup", eval_set: "best", category: "normal" },
              ],
            },
            isCentralSslResultTrack,
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
        "supervisedEvalSets": ["test", "best", "final"],
        "sslPerClassEvalSets": ["test"],
    }


def test_central_dashboard_entrypoint_uses_ssl_only_predicate_for_ssl_track() -> None:
    source = (
        REPO_ROOT / "apps" / "experiment_dashboard" / "src" / "main.js"
    ).read_text(encoding="utf-8")

    assert 'if (state.activeTrack === "central_ssl")' in source
    assert "return isCentralSslResultTrack;" in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_central_overview_column_selection_preserves_removed_meta_columns() -> None:
    script = """
        const { normalizeOverviewSelection } = await import(
          "./apps/experiment_dashboard/src/features/central_ssl/ui/overview_page.js"
        );

        const state = {
          overviewMetricIds: ["macro_f1"],
          overviewTableColumns: {
            visible: ["axis:algorithm", "metric:macro_f1"],
            order: ["axis:algorithm", "metric:macro_f1"],
          },
        };
        const rows = [{
          run_id: "ssl_run",
          track: "central_peft_ssl",
          method_name: "fixmatch",
          algorithm_name: "fixmatch",
          method_family: "peft_classifier",
          labeled_dataset_name: "szegeelim_general4",
          unlabeled_dataset_name: "ourafla_reddit",
          test_dataset_name: "ourafla_reddit",
          label_budget_name: "pc1024",
          initial_checkpoint_name: "none",
          created_at: "2026-06-16T00:00:00",
          macro_f1: 0.77,
          accuracy_top_1: 0.76,
        }];

        normalizeOverviewSelection(rows, state);
        console.log(JSON.stringify(state.overviewTableColumns));
    """

    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    result = json.loads(completed.stdout)

    assert result["visible"] == ["axis:algorithm", "metric:macro_f1"]
    assert "axis:initial_checkpoint" not in result["visible"]


def test_run_option_detail_positioning_uses_viewport_clamped_left_offset() -> None:
    source = (
        REPO_ROOT / "apps" / "experiment_dashboard" / "src" / "main.js"
    ).read_text(encoding="utf-8")

    assert "const optionRect = option.getBoundingClientRect();" in source
    assert "const overlay = runOptionDetailOverlay();" in source
    assert "document.body.appendChild(overlay);" in source
    assert "overlay.style.left = `${Math.round(nextLeft)}px`;" in source
    assert "let nextTop = optionRect.bottom + 8;" in source
    assert "nextTop = optionRect.top - rect.height - 8;" in source


def test_supervised_dashboard_defaults_to_test_eval_set() -> None:
    source = (
        REPO_ROOT / "apps" / "experiment_dashboard" / "src" / "main.js"
    ).read_text(encoding="utf-8")

    assert (
        'activeTrack === "supervised" || activeTrack === "central_ssl"\n'
        '    ? ["test", "best", "final", "validation", "final_validation", '
        '"initial_validation"]'
    ) in source
    assert (
        'hideTest: state.activeTrack !== "supervised" && '
        'state.activeTrack !== "central_ssl"'
    ) in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_central_dashboard_labels_are_track_aware() -> None:
    script = """
        const {
          evaluationDataLabel,
          initialCheckpointLabel,
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
          train_batch_size: 8,
          labeled_batch_size: 8,
          backbone_model_id: "mixedbread-ai/mxbai-embed-large-v1",
          seed: 42,
        };
        const ssl = {
          ...supervised,
          run_id: "peft_fixmatch_2026_06_04_120231",
          track: "central_peft_ssl",
          method_name: "fixmatch",
          algorithm_name: "fixmatch",
          label_budget_name: "pc1024",
          unlabeled_batch_size: 16,
        };

        console.log(JSON.stringify({
          supervisedBudget: labelBudgetLabel(supervised),
          supervisedTrain: trainingDataLabel(supervised),
          supervisedEval: evaluationDataLabel(supervised),
          supervisedCheckpoint: initialCheckpointLabel(supervised),
          supervisedDetail: runDetail(supervised),
          sslTrain: trainingDataLabel(ssl),
          sslEval: evaluationDataLabel(ssl),
          sslCheckpoint: initialCheckpointLabel(ssl),
          sslDetail: runDetail(ssl),
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
    assert result["supervisedCheckpoint"] == "-"
    assert "model=mixedbread-ai/mxbai-embed-large-v1" in result["supervisedDetail"]
    assert "unlabeled=" not in result["supervisedDetail"]
    assert "validation=" not in result["supervisedDetail"]
    assert "labeled_batch=8" in result["supervisedDetail"]
    assert "unlabeled_batch=" not in result["supervisedDetail"]
    assert result["sslTrain"] == (
        "labeled=szegeelim_general4 pc1024 · unlabeled=ourafla_reddit"
    )
    assert result["sslEval"] == "test=ourafla_reddit"
    assert result["sslCheckpoint"] == "unrecorded"
    assert "model=mixedbread-ai/mxbai-embed-large-v1" in result["sslDetail"]
    assert "validation=" not in result["sslDetail"]
    assert "labeled_batch=8" in result["sslDetail"]
    assert "unlabeled_batch=16" in result["sslDetail"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_fl_dashboard_shows_label_budget_in_filters_and_details() -> None:
    script = """
        const { flFilterAxes } = await import(
          "./apps/experiment_dashboard/src/features/fl_ssl/logic/filters.js"
        );
        const {
          compactRunSubLabel,
          runDetailLabel,
          runHoverDetail,
        } = await import(
          "./apps/experiment_dashboard/src/features/fl_ssl/logic/labels.js"
        );

        const row = {
          run_id: "fl_run",
          track: "fl_ssl_main_comparison",
          method_family: "manual_baselines",
          method_name: "fixmatch",
          algorithm_name: "fixmatch",
          label_budget_name: "pc100",
          labeled_dataset_name: "ourafla_reddit",
          unlabeled_dataset_name: "ourafla_reddit",
          peft_adapter_name: "lora",
          peft_adapter_rank: 8,
          peft_adapter_alpha: 16,
          peft_adapter_dropout: 0.1,
          payload_adapter_kind: "peft_classifier",
          aggregation_backend_name: "fedavg",
          initial_checkpoint_name: "central_seed",
          backbone_model_id: "mixedbread-ai/mxbai-embed-large-v1",
          train_batch_size: 12,
          labeled_batch_size: 12,
          unlabeled_batch_size: 12,
          client_count: 10,
          completed_rounds: 30,
          round_budget: 30,
          seed: 42,
        };
        const axes = flFilterAxes({}).map((axis) => axis.id);
        console.log(JSON.stringify({
          hasLabelBudgetFilter: axes.includes("label_budget"),
          hover: runHoverDetail(row),
          detail: runDetailLabel(row),
          sub: compactRunSubLabel(row),
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

    assert result["hasLabelBudgetFilter"] is True
    assert "label_budget=pc100" in result["hover"]
    assert "model=mixedbread-ai/mxbai-embed-large-v1" in result["hover"]
    assert "label_budget=pc100" in result["detail"]
    assert "model=mixedbread-ai/mxbai-embed-large-v1" in result["detail"]
    assert "label_budget=pc100" in result["sub"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_fl_filter_prune_preserves_user_selected_single_option_axes() -> None:
    script = """
        const { applyFlFilters, pruneFlFilters } = await import(
          "./apps/experiment_dashboard/src/features/fl_ssl/logic/filters.js"
        );

        const rows = [
          {
            run_id: "run_1",
            track: "fl_ssl_main_comparison",
            method_family: "fedmatch",
            algorithm_name: "fedmatch",
            label_budget_name: "pc1024",
          },
          {
            run_id: "run_2",
            track: "fl_ssl_main_comparison",
            method_family: "manual_baselines",
            algorithm_name: "fixmatch",
            label_budget_name: "pc1024",
          },
          {
            run_id: "run_3",
            track: "fl_ssl_main_comparison",
            method_family: "manual_baselines",
            algorithm_name: "freematch",
            label_budget_name: "pc100",
          },
        ];
        const state = {
          filterAxisIds: ["algorithm", "method_family"],
          filterValues: {
            algorithm: ["fedmatch"],
            method_family: ["fedmatch"],
          },
        };
        const beforeRows = applyFlFilters({}, rows, state).map((row) => row.run_id);
        pruneFlFilters({}, rows, state);
        const afterRows = applyFlFilters({}, rows, state).map((row) => row.run_id);
        console.log(JSON.stringify({ state, beforeRows, afterRows }));
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
        "state": {
            "filterAxisIds": ["algorithm", "method_family"],
            "filterValues": {
                "algorithm": ["fedmatch"],
                "method_family": ["fedmatch"],
            },
        },
        "beforeRows": ["run_1"],
        "afterRows": ["run_1"],
    }


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
    assert 'class="matrix-grid"' in source
    assert "matrix-category-count" in source
    assert 'class="matrix-cell"' in source

    style_source = (
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "styles"
        / "index.css"
    ).read_text(encoding="utf-8")
    assert ".matrix-grid {" in style_source
    assert ".matrix-cell {" in style_source
    assert "--cell-alpha" in style_source


def test_dashboard_filters_include_labeled_and_unlabeled_batch_size_axes() -> None:
    central_source = (
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "central_ssl"
        / "logic"
        / "filters.js"
    ).read_text(encoding="utf-8")
    fl_source = (
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "fl_ssl"
        / "logic"
        / "filters.js"
    ).read_text(encoding="utf-8")

    for source in (central_source, fl_source):
        assert 'axis("backbone_model", "Model"' in source
        assert 'axis("labeled_batch_size", "Labeled Batch"' in source
        assert 'axis("unlabeled_batch_size", "Unlabeled Batch"' in source


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_selected_run_card_uses_single_app_tooltip() -> None:
    script = """
        const { renderSelectedRunCard } = await import(
          "./apps/experiment_dashboard/src/ui/controls/selected_run_card.js"
        );
        const html = renderSelectedRunCard({
          id: "run_1",
          label: "run label",
          detail: "algo=fixmatch · seed=42",
          peerDetails: [
            "algo=fixmatch · seed=42",
            "algo=adamatch · seed=42",
          ],
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
          diffCount: (html.match(/run-option-detail-diff/g) ?? []).length,
          seedDiff: /seed=<span class="run-option-detail-diff">42<\\/span>/.test(html),
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

    assert result == {
        "hasNativeTitle": False,
        "tooltipCount": 1,
        "diffCount": 1,
        "seedDiff": False,
    }


def test_alias_inputs_do_not_rerender_on_each_keystroke() -> None:
    source = (
        REPO_ROOT / "apps" / "experiment_dashboard" / "src" / "main.js"
    ).read_text(encoding="utf-8")
    alias_block = source.split("function handleLiveInput(event) {", 1)[1].split(
        "if (input.dataset.chartAxisLabelScope)",
        1,
    )[0]

    assert "updateAlias(" in alias_block
    assert "render();" not in alias_block


def test_overview_and_fl_runs_default_to_algorithm_meta_only() -> None:
    central_source = (
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "central_ssl"
        / "ui"
        / "overview_page.js"
    ).read_text(encoding="utf-8")
    fl_source = (
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "fl_ssl"
        / "ui"
        / "runs_page.js"
    ).read_text(encoding="utf-8")

    assert 'const DEFAULT_VISIBLE_COLUMNS = [\n  "axis:algorithm",\n];' in central_source
    assert 'const DEFAULT_VISIBLE_COLUMNS = [\n  "axis:algorithm",\n];' in fl_source


def test_selected_run_card_detail_is_hover_only() -> None:
    source = (
        REPO_ROOT / "apps" / "experiment_dashboard" / "src" / "styles" / "index.css"
    ).read_text(encoding="utf-8")

    assert ".selected-run-label-wrap:hover .selected-run-detail" in source
    assert ".selected-run-card:hover .selected-run-detail" not in source
    assert ".selected-run-card:focus-within .selected-run-detail" not in source


def test_run_selection_blocks_use_single_app_hover_detail() -> None:
    files = [
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "central_ssl"
        / "ui"
        / "overview_page.js",
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "central_ssl"
        / "ui"
        / "compare_page.js",
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "central_ssl"
        / "ui"
        / "projection_page.js",
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "fl_ssl"
        / "ui"
        / "runs_page.js",
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "fl_ssl"
        / "ui"
        / "rounds_page.js",
        REPO_ROOT
        / "apps"
        / "experiment_dashboard"
        / "src"
        / "features"
        / "fl_ssl"
        / "ui"
        / "projection_page.js",
    ]

    for path in files:
        source = path.read_text(encoding="utf-8")
        assert '<label class="run-option" title=' not in source
        assert 'class="run-option-detail"' in source

    style_source = (
        REPO_ROOT / "apps" / "experiment_dashboard" / "src" / "styles" / "index.css"
    ).read_text(encoding="utf-8")
    assert ".run-option .run-option-detail {" in style_source
    assert ".run-option-detail-overlay {" in style_source
    assert "position: fixed;" in style_source
    assert ".run-option:hover .run-option-detail" not in style_source
    assert ".run-option-detail-overlay span" in style_source
    assert "display: inline;" in style_source
    assert ".run-option-detail-diff" in style_source

    main_source = (
        REPO_ROOT / "apps" / "experiment_dashboard" / "src" / "main.js"
    ).read_text(encoding="utf-8")
    assert "bindRunOptionDetailPositioning();" in main_source
    assert "const optionRect = option.getBoundingClientRect();" in main_source
    assert "const overlay = runOptionDetailOverlay();" in main_source
    assert "document.body.appendChild(overlay);" in main_source
    assert "overlay.style.left = `${Math.round(nextLeft)}px`;" in main_source
    assert "let nextTop = optionRect.bottom + 8;" in main_source
    assert "nextTop = optionRect.top - rect.height - 8;" in main_source


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_run_option_detail_highlights_only_different_parts() -> None:
    script = """
        const { renderRunOptionDetail } = await import(
          "./apps/experiment_dashboard/src/ui/controls/run_option_detail.js"
        );
        const html = renderRunOptionDetail(
          "algo=fixmatch · pc=100 · seed=42",
          [
            "algo=fixmatch · pc=100 · seed=42",
            "algo=fixmatch · pc=200 · seed=42",
          ],
        );
        console.log(JSON.stringify({
          diffCount: (html.match(/run-option-detail-diff/g) ?? []).length,
          hasDifferentPc: (
            /pc=<span class="run-option-detail-diff">100<\\/span>/.test(html)
          ),
          hasCommonSeed: /class="run-option-detail-part">seed=42</.test(html),
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

    assert result == {
        "diffCount": 1,
        "hasDifferentPc": True,
        "hasCommonSeed": True,
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="node가 설치되어 있지 않음")
def test_run_option_detail_compares_keyed_parts_without_position_drift() -> None:
    script = """
        const { renderRunOptionDetail } = await import(
          "./apps/experiment_dashboard/src/ui/controls/run_option_detail.js"
        );
        const html = renderRunOptionDetail(
          "algo=fixmatch · seed=42 · model=mxbai · split=a",
          [
            "algo=fixmatch · seed=42 · model=mxbai · split=a",
            "algo=adamatch · model=mxbai · seed=42",
          ],
        );
        console.log(JSON.stringify({
          diffCount: (html.match(/run-option-detail-diff/g) ?? []).length,
          seedDiff: /seed=<span class="run-option-detail-diff">42<\\/span>/.test(html),
          modelDiff: /model=<span class="run-option-detail-diff">mxbai<\\/span>/.test(html),
          algoDiff: /algo=<span class="run-option-detail-diff">fixmatch<\\/span>/.test(html),
          splitDiff: /split=<span class="run-option-detail-diff">a<\\/span>/.test(html),
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

    assert result == {
        "diffCount": 2,
        "seedDiff": False,
        "modelDiff": False,
        "algoDiff": True,
        "splitDiff": True,
    }
