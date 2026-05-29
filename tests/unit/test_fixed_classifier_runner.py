from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from scripts.experiments.central.fixed_classifier_seed.common import (
    prepare_fixed_classifier_run_context,
)
from scripts.experiments.central.fixed_classifier_seed.runner import (
    run_fixed_embedding_classifier,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
)


def _build_cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "train_jsonl": str(tmp_path / "train.jsonl"),
            "eval_sets": {
                "validation": str(tmp_path / "validation.jsonl"),
                "test": str(tmp_path / "test.jsonl"),
            },
            "selection_set": "validation",
            "embed_chunk_size": 8,
            "train_batch_size": 4,
            "eval_batch_size": 8,
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "output_dir": "runs/train_classifier",
            "model_output_dir": "data/processed/classifier_heads",
            "classifier_version": "",
            "embedding": {"spec": {"_target_": "builtins.dict"}},
        }
    )


def _row(query_id: str, label: str, text: str) -> LabeledQueryRow:
    return LabeledQueryRow(
        query_id=query_id,
        text=text,
        raw_label_scheme="manual_label",
        raw_label=label,
        mapped_label_4=label,
        locale="ko-KR",
        annotation_source="annotated",
        approved_by="annotator",
        created_at="2026-04-22T00:00:00+00:00",
    )


def test_prepare_fixed_classifier_run_context_normalizes_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _build_cfg(tmp_path)
    dump_labeled_query_rows(
        Path(str(cfg.train_jsonl)),
        [_row("seed_q1", "anxiety", "불안해요")],
    )
    dump_labeled_query_rows(
        Path(str(cfg.eval_sets.validation)),
        [_row("v1", "anxiety", "검증")],
    )
    dump_labeled_query_rows(
        Path(str(cfg.eval_sets.test)),
        [_row("t1", "depression", "테스트")],
    )
    monkeypatch.setattr(
        "scripts.experiments.central.fixed_classifier_seed.common.instantiate",
        lambda _spec: SimpleNamespace(
            backend="transformers_mxbai",
            model_id="mixedbread-ai/mxbai-embed-large-v1",
            revision="main",
            task_prefix="query: ",
            device="cpu",
        ),
    )

    context = prepare_fixed_classifier_run_context(
        cfg=cfg,
        train_jsonl_ref="in_memory/train_rows.jsonl",
        output_dir_root="runs/override_classifier",
        model_output_dir="data/override_classifier_heads",
        classifier_version="clf_manual_v1",
    )

    assert len(context.train_rows) == 1
    assert set(context.eval_rows_by_name) == {"validation", "test"}
    assert context.effective_selection_set == "validation"
    assert context.effective_train_jsonl_ref == "in_memory/train_rows.jsonl"
    assert context.output_dir_root == "runs/override_classifier"
    assert context.model_output_dir == "data/override_classifier_heads"
    assert context.classifier_version == "clf_manual_v1"
    assert context.eval_set_map["validation"] == Path(str(cfg.eval_sets.validation))
    assert context.embedding_spec.model_id == "mixedbread-ai/mxbai-embed-large-v1"


def test_run_fixed_embedding_classifier_wires_prepared_context_and_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _build_cfg(tmp_path)
    captured: dict[str, object] = {}
    prepared_context = SimpleNamespace(
        cfg=cfg,
        train_rows=[_row("seed_q1", "anxiety", "불안해요")],
        eval_rows_by_name={
            "validation": [_row("v1", "anxiety", "검증")],
            "test": [_row("t1", "depression", "테스트")],
        },
        eval_set_map={
            "validation": Path("data/validation.jsonl"),
            "test": Path("data/test.jsonl"),
        },
        effective_selection_set="validation",
        effective_train_jsonl_ref="in_memory/train_rows.jsonl",
        output_dir_root="runs/train_classifier_override",
        model_output_dir="data/processed/classifier_heads_override",
        classifier_version="clf_v1",
        created_at=datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc),
        embedding_spec=SimpleNamespace(device="cpu"),
    )

    monkeypatch.setattr(
        "scripts.experiments.central.fixed_classifier_seed.runner."
        "prepare_fixed_classifier_run_context",
        lambda **_kwargs: prepared_context,
    )

    def _fake_train_fixed_embedding_classifier(**kwargs):
        captured["train_kwargs"] = kwargs
        return SimpleNamespace(
            model=object(),
            embedding_spec=SimpleNamespace(
                backend="transformers_mxbai",
                model_id="mixedbread-ai/mxbai-embed-large-v1",
                revision="main",
                task_prefix="",
            ),
            categories=["anxiety", "depression"],
            training_device="cpu",
            best_selection_report={"accuracy_top_1": 0.8},
            history=[{"epoch": 1, "train_loss": 0.1}],
            eval_results={"validation": {"accuracy_top_1": 0.8}},
        )

    def _fake_write_fixed_classifier_artifacts(**kwargs):
        captured["artifact_kwargs"] = kwargs
        return {
            "output_dir": "runs/fake_fixed_classifier",
            "model_path": "data/processed/classifier_heads/fake.pt",
            "manifest": "data/processed/classifier_heads/fake.manifest.json",
            "report_json": "runs/fake_fixed_classifier/reports/report.json",
        }

    monkeypatch.setattr(
        "scripts.experiments.central.fixed_classifier_seed.runner.train_fixed_embedding_classifier",
        _fake_train_fixed_embedding_classifier,
    )
    monkeypatch.setattr(
        "scripts.experiments.central.fixed_classifier_seed.runner.write_fixed_classifier_artifacts",
        _fake_write_fixed_classifier_artifacts,
    )

    outputs = run_fixed_embedding_classifier(cfg=cfg)

    assert outputs["output_dir"] == "runs/fake_fixed_classifier"
    assert captured["train_kwargs"]["train_rows"] == prepared_context.train_rows
    assert (
        captured["train_kwargs"]["selection_set_name"]
        == prepared_context.effective_selection_set
    )
    assert captured["artifact_kwargs"]["classifier_version"] == "clf_v1"
    assert (
        captured["artifact_kwargs"]["train_jsonl_ref"]
        == prepared_context.effective_train_jsonl_ref
    )
    assert (
        captured["artifact_kwargs"]["output_dir_root"]
        == prepared_context.output_dir_root
    )
