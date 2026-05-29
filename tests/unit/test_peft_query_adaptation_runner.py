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
from scripts.support.query_ssl_peft.runners.query_adaptation import (
    prepare_query_adaptation_supervised_run,
    run_query_adaptation_supervised_baseline,
)
from shared.src.contracts.labeled_query_row_contracts import load_labeled_query_rows


def _build_dataset(
    *,
    query_id: str,
    text: str,
    label: str,
) -> QueryAdaptationDataset:
    occurred_at = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    return QueryAdaptationDataset(
        examples=(
            QueryAdaptationDatasetExample(
                source_row=TrainingExampleSource(
                    query_id=query_id,
                    text=text,
                    occurred_at=occurred_at,
                    translated_text=None,
                ),
                label=label,
                provenance=QueryAdaptationDatasetProvenance(
                    locale="ko-KR",
                    source_type="user_message",
                    model_revision="rev_001",
                    selection_confidence_kind="prototype_similarity_top1",
                    translated_text_present=False,
                    candidate_id=f"round_1:{query_id}",
                ),
                label_source="pseudo_label",
                confidence=0.9,
                margin=0.5,
            ),
        )
    )


def _build_cfg() -> object:
    return OmegaConf.create(
        {
            "trainer_version": "adapt_run_v1",
            "runtime": {"device": "cpu"},
            "train_jsonl": "",
            "eval_sets": {},
            "selection_set": "",
        }
    )


def test_prepare_query_adaptation_supervised_run_exports_train_and_selection(
    tmp_path: Path,
) -> None:
    train_dataset = _build_dataset(
        query_id="q1",
        text="불안이 심해요",
        label="anxiety",
    )

    prepared = prepare_query_adaptation_supervised_run(
        cfg=_build_cfg(),
        train_dataset=train_dataset,
        export_root=tmp_path,
        generated_at=datetime(2026, 4, 12, 13, 0, tzinfo=timezone.utc),
    )

    export_dir = prepared.export_dir
    train_jsonl = prepared.train_artifacts.jsonl_path

    assert export_dir == tmp_path / "adapt_run_v1"
    assert train_jsonl.exists()
    assert prepared.train_artifacts.manifest_path.exists()
    assert prepared.train_artifacts.summary_path.exists()
    assert prepared.train_rows[0]["query_id"] == "q1"
    assert prepared.eval_rows_by_name["selection"][0]["mapped_label_4"] == "anxiety"
    assert prepared.train_jsonl_ref == str(train_jsonl)
    assert prepared.selection_set_name == "selection"
    assert prepared.eval_set_refs["selection"] == str(train_jsonl)

    rows = load_labeled_query_rows(train_jsonl)
    assert len(rows) == 1
    assert rows[0]["query_id"] == "q1"
    assert rows[0]["mapped_label_4"] == "anxiety"


def test_run_query_adaptation_supervised_baseline_calls_existing_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_dataset = _build_dataset(
        query_id="q1",
        text="불안이 심해요",
        label="anxiety",
    )
    selection_dataset = _build_dataset(
        query_id="q2",
        text="너무 우울해요",
        label="depression",
    )

    captured: dict[str, object] = {}

    def _fake_runner(
        cfg,
        *,
        train_rows,
        eval_rows_by_name,
        selection_set_name,
        train_jsonl_ref=None,
        eval_set_refs=None,
    ) -> dict[str, str]:
        captured["cfg"] = cfg
        captured["train_rows"] = train_rows
        captured["eval_rows_by_name"] = eval_rows_by_name
        captured["selection_set_name"] = selection_set_name
        captured["train_jsonl_ref"] = train_jsonl_ref
        captured["eval_set_refs"] = eval_set_refs
        return {
            "output_dir": "runs/fake",
            "report_json": "runs/fake/report.json",
        }

    monkeypatch.setattr(
        "scripts.support.query_ssl_peft.runners.query_adaptation."
        "run_supervised_peft_baseline",
        _fake_runner,
    )

    outputs = run_query_adaptation_supervised_baseline(
        cfg=_build_cfg(),
        train_dataset=train_dataset,
        selection_dataset=selection_dataset,
        export_root=tmp_path,
        generated_at=datetime(2026, 4, 12, 13, 0, tzinfo=timezone.utc),
    )

    train_rows = captured["train_rows"]
    eval_rows_by_name = captured["eval_rows_by_name"]
    selection_set_name = captured["selection_set_name"]
    train_jsonl_ref = captured["train_jsonl_ref"]
    eval_set_refs = captured["eval_set_refs"]
    assert str(train_jsonl_ref).endswith("train.jsonl")
    assert selection_set_name == "selection"
    assert train_rows[0]["query_id"] == "q1"
    assert eval_rows_by_name["selection"][0]["query_id"] == "q2"
    assert str(eval_set_refs["selection"]).endswith("selection.jsonl")
    assert outputs["output_dir"] == "runs/fake"
    assert Path(outputs["train_jsonl"]).exists()
    assert Path(outputs["selection_jsonl"]).exists()
    assert Path(outputs["train_summary"]).exists()
    assert Path(outputs["selection_summary"]).exists()
