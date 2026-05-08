"""Query LoRA run artifact 저장 검증."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.experiments.query_lora_ssl.io.artifacts import write_run_artifacts


class _DummySaver:
    def __init__(self, marker_filename: str) -> None:
        self.marker_filename = marker_filename

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / self.marker_filename).write_text("saved\n", encoding="utf-8")


class _DummyClassifier:
    in_features = 384

    def state_dict(self) -> dict[str, list[float]]:
        return {"weights": [0.1, 0.2, 0.3]}


def test_write_run_artifacts_writes_model_manifest_and_report(
    tmp_path: Path,
) -> None:
    cfg = SimpleNamespace(
        output_dir=tmp_path / "runs",
        adapter_output_dir=tmp_path / "adapters",
        classifier_output_dir=tmp_path / "classifiers",
        train_jsonl=tmp_path / "train.jsonl",
        selection_set="validation",
        seed=7,
        epochs=3,
        train_batch_size=4,
        eval_batch_size=5,
        learning_rate=0.001,
        classifier_learning_rate=0.002,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )
    model = SimpleNamespace(
        backbone=_DummySaver("backbone.txt"),
        classifier=_DummyClassifier(),
    )
    tokenizer = _DummySaver("tokenizer.txt")

    outputs = write_run_artifacts(
        cfg=cfg,
        trainer_version="run-001",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        model=model,
        tokenizer=tokenizer,
        categories=["alpha", "beta"],
        eval_set_map={"validation": tmp_path / "validation.jsonl"},
        training_device="cpu",
        backbone_summary={"name": "dummy-backbone"},
        history=[{"epoch": 1, "loss": 0.5}],
        best_selection_report={"macro_f1": 0.75},
        results={"test": {"accuracy": 0.8}},
        extra_manifest={"ssl_algorithm": "fixmatch"},
    )

    adapter_dir = Path(outputs["adapter_dir"])
    classifier_path = Path(outputs["classifier_path"])
    manifest_path = Path(outputs["manifest"])
    report_path = Path(outputs["report_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    classifier_payload = torch.load(classifier_path, weights_only=False)

    assert Path(outputs["output_dir"]) == tmp_path / "runs" / "run-001"
    assert (Path(outputs["output_dir"]) / "logs").is_dir()
    assert (adapter_dir / "backbone.txt").read_text(encoding="utf-8") == "saved\n"
    assert (adapter_dir / "tokenizer.txt").read_text(encoding="utf-8") == "saved\n"
    assert classifier_payload == {
        "classifier_state_dict": {"weights": [0.1, 0.2, 0.3]},
        "categories": ["alpha", "beta"],
        "hidden_size": 384,
    }
    assert manifest["adapter_dir"] == str(adapter_dir)
    assert manifest["classifier_path"] == str(classifier_path)
    assert manifest["ssl_algorithm"] == "fixmatch"
    assert report["schema_version"] == "central_lora_classifier_eval.v1"
    assert report["manifest"] == manifest
    assert report["results"] == {"test": {"accuracy": 0.8}}
