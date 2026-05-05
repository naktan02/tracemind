from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from agent.src.services.training.backends.inputs.models import (
    TrainingExampleSource,
)
from agent.src.services.training.datasets.query_adaptation_dataset_service import (
    QueryAdaptationDataset,
    QueryAdaptationDatasetExample,
    QueryAdaptationDatasetProvenance,
)
from scripts.experiments.query_lora_ssl.pseudo_label_runner import (
    prepare_pseudo_label_self_training_run,
    run_pseudo_label_self_training,
)
from scripts.io.labeled_query_rows import LabeledQueryRow, load_labeled_query_rows
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelSelectionContext,
    PseudoLabelSelectionStage,
)

VALIDATION_JSONL = "data/processed/splits/ourafla_train_split.v1.validation.jsonl"
TEST_JSONL = (
    "data/processed/labeled_query_sets/"
    "ourafla_mental_health_text_classification_test.v1.jsonl"
)


def _build_cfg() -> object:
    return OmegaConf.create(
        {
            "trainer_version": "pseudo_label_run_v1",
            "runtime": {"device": "cpu"},
            "train_jsonl": "",
            "pseudo_label_jsonl": None,
            "pseudo_label_export_root": "",
            "include_seed_train_rows": False,
            "pseudo_label_algorithm": {
                "name": "margin_threshold_v1",
                "confidence_threshold": 0.6,
                "margin_threshold": 0.02,
                "algorithm_name": "top1_margin_threshold",
            },
            "fixed_categories": [
                "anxiety",
                "depression",
                "normal",
                "suicidal",
            ],
            "eval_sets": {
                "validation": VALIDATION_JSONL,
                "test": TEST_JSONL,
            },
            "selection_set": "validation",
        }
    )


def _build_seed_rows() -> list[LabeledQueryRow]:
    return [
        LabeledQueryRow(
            query_id="seed_q1",
            text="불안해요",
            raw_label_scheme="manual_label",
            raw_label="anxiety",
            mapped_label_4="anxiety",
            locale="ko-KR",
            annotation_source="seed_train",
            approved_by="annotator",
            created_at="2026-04-12T12:00:00+00:00",
        )
    ]


def _build_pseudo_label_dataset() -> QueryAdaptationDataset:
    occurred_at = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    return QueryAdaptationDataset(
        examples=(
            QueryAdaptationDatasetExample(
                source_row=TrainingExampleSource(
                    query_id="pl_q1",
                    text="너무 우울해요",
                    occurred_at=occurred_at,
                    translated_text=None,
                ),
                label="depression",
                provenance=QueryAdaptationDatasetProvenance(
                    locale="ko-KR",
                    source_type="user_message",
                    model_revision="rev_001",
                    selection_confidence_kind="prototype_similarity_top1",
                    translated_text_present=False,
                    candidate_id="round_1:pl_q1",
                    selection_context=PseudoLabelSelectionContext(
                        threshold_accepted=True,
                        selected_by_cap=True,
                        final_accepted=True,
                        selection_stage=PseudoLabelSelectionStage.ACCEPTED,
                    ),
                ),
                label_source="pseudo_label",
                confidence=0.91,
                margin=0.42,
            ),
        )
    )


def test_prepare_pseudo_label_self_training_run_excludes_seed_rows_by_default(
    tmp_path: Path,
) -> None:
    prepared = prepare_pseudo_label_self_training_run(
        cfg=_build_cfg(),
        seed_train_rows=_build_seed_rows(),
        pseudo_label_dataset=_build_pseudo_label_dataset(),
        export_root=tmp_path,
        generated_at=datetime(2026, 4, 12, 13, 0, tzinfo=timezone.utc),
    )

    assert prepared.export_dir == tmp_path / "pseudo_label_run_v1"
    assert prepared.seed_train_count == 0
    assert prepared.pseudo_label_count == 1
    assert prepared.combined_train_count == 1
    assert prepared.combined_train_rows[0]["query_id"] == "pl_q1"
    assert prepared.train_jsonl_ref.endswith("combined_train.jsonl")
    assert prepared.pseudo_label_artifacts.jsonl_path.exists()
    assert prepared.pseudo_label_artifacts.manifest_path.exists()
    assert prepared.pseudo_label_artifacts.summary_path.exists()
    assert prepared.combined_train_artifacts.jsonl_path.exists()
    rows = load_labeled_query_rows(prepared.combined_train_artifacts.jsonl_path)
    assert [row["query_id"] for row in rows] == ["pl_q1"]


def test_prepare_pseudo_label_self_training_run_can_opt_in_seed_replay(
    tmp_path: Path,
) -> None:
    prepared = prepare_pseudo_label_self_training_run(
        cfg=_build_cfg(),
        seed_train_rows=_build_seed_rows(),
        pseudo_label_dataset=_build_pseudo_label_dataset(),
        include_seed_train_rows=True,
        export_root=tmp_path,
        generated_at=datetime(2026, 4, 12, 13, 0, tzinfo=timezone.utc),
    )

    assert prepared.seed_train_count == 1
    assert prepared.pseudo_label_count == 1
    assert prepared.combined_train_count == 2
    assert prepared.combined_train_rows[0]["query_id"] == "seed_q1"
    assert prepared.combined_train_rows[1]["query_id"] == "pl_q1"


def test_run_pseudo_label_self_training_calls_baseline_runner_with_combined_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_runner(
        cfg,
        *,
        train_rows,
        eval_rows_by_name=None,
        selection_set_name=None,
        train_jsonl_ref=None,
        eval_set_refs=None,
        trainer_version_override=None,
        extra_manifest=None,
        categories_override=None,
    ) -> dict[str, str]:
        captured["cfg"] = cfg
        captured["train_rows"] = train_rows
        captured["train_jsonl_ref"] = train_jsonl_ref
        captured["trainer_version_override"] = trainer_version_override
        captured["extra_manifest"] = extra_manifest
        captured["categories_override"] = categories_override
        return {
            "output_dir": "runs/fake_pseudo",
            "report_json": "runs/fake_pseudo/report.json",
        }

    monkeypatch.setattr(
        "scripts.experiments.query_lora_ssl.pseudo_label_runner."
        "run_supervised_lora_baseline",
        _fake_runner,
    )

    outputs = run_pseudo_label_self_training(
        cfg=_build_cfg(),
        seed_train_rows=_build_seed_rows(),
        pseudo_label_dataset=_build_pseudo_label_dataset(),
        export_root=tmp_path,
        generated_at=datetime(2026, 4, 12, 13, 0, tzinfo=timezone.utc),
    )

    train_rows = captured["train_rows"]
    extra_manifest = captured["extra_manifest"]
    assert len(train_rows) == 1
    assert train_rows[0]["raw_label_scheme"] == "pseudo_label"
    assert extra_manifest["pseudo_label_row_count"] == 1
    assert extra_manifest["seed_train_row_count"] == 0
    assert extra_manifest["include_seed_train_rows"] is False
    assert extra_manifest["combined_train_row_count"] == 1
    assert extra_manifest["pseudo_label_algorithm"]["preset_name"] == (
        "margin_threshold_v1"
    )
    assert extra_manifest["pseudo_label_algorithm"]["algorithm_name"] == (
        "top1_margin_threshold"
    )
    assert extra_manifest["pseudo_label_algorithm"]["margin_threshold"] == 0.02
    assert captured["train_jsonl_ref"].endswith("combined_train.jsonl")
    assert captured["trainer_version_override"] == "pseudo_label_run_v1"
    assert captured["categories_override"] is None
    assert outputs["output_dir"] == "runs/fake_pseudo"
    assert Path(outputs["pseudo_label_jsonl"]).exists()
    assert Path(outputs["combined_train_jsonl"]).exists()
