"""Query PEFT run artifact 저장 검증."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from methods.adaptation.text_encoder_classifier.classifier_head_tensor_artifact import (
    load_classifier_head_state_tensor_artifact,
)
from scripts.support.query_ssl_text_encoder.io.artifacts import write_run_artifacts
from scripts.support.query_ssl_text_encoder.io.full_text_encoder_artifacts import (
    write_full_text_encoder_run_artifacts,
)


class _DummySaver:
    def __init__(self, marker_filename: str) -> None:
        self.marker_filename = marker_filename

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / self.marker_filename).write_text("saved\n", encoding="utf-8")


class _DummyClassifier:
    in_features = 384

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "weight": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "bias": torch.tensor([0.01, -0.01]),
        }


class _ProjectionModel:
    def __init__(self) -> None:
        self.backbone = _DummySaver("backbone.txt")
        self.classifier = nn.Linear(2, 2)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(2))
            self.classifier.bias.zero_()

    def eval(self) -> None:
        self.classifier.eval()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            f"classifier.{key}": value.detach().clone()
            for key, value in self.classifier.state_dict().items()
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.classifier.load_state_dict(
            {
                key.removeprefix("classifier."): value
                for key, value in state_dict.items()
                if key.startswith("classifier.")
            }
        )

    def extract_pooled_features(self, *, input_ids, attention_mask):
        del attention_mask
        return input_ids.float()


def test_write_run_artifacts_writes_model_manifest_and_report(
    tmp_path: Path,
) -> None:
    cfg = SimpleNamespace(
        output_dir=tmp_path / "runs",
        central_ssl_budget=SimpleNamespace(name="smoke", output_root="runs/_smoke"),
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
        backbone_summary={
            "name": "dummy-backbone",
            "trainable_surface": {
                "name": "peft_text_encoder",
                "trainable_state": "peft_adapter_and_classifier_head",
            },
        },
        history=[{"epoch": 1, "loss": 0.5}],
        best_selection_report={"macro_f1": 0.75},
        final_selection_report={"macro_f1": 0.74},
        results={"test": {"accuracy": 0.8}},
        extra_manifest={"ssl_algorithm": "fixmatch"},
    )

    adapter_dir = Path(outputs["adapter_dir"])
    classifier_path = Path(outputs["classifier_path"])
    manifest_path = Path(outputs["manifest"])
    report_path = Path(outputs["report_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    classifier_payload = load_classifier_head_state_tensor_artifact(classifier_path)

    assert Path(outputs["output_dir"]) == tmp_path / "runs" / "run-001"
    assert (Path(outputs["output_dir"]) / "logs").is_dir()
    assert adapter_dir == Path(outputs["output_dir"]) / "artifacts" / "adapter"
    assert classifier_path == (
        Path(outputs["output_dir"]) / "artifacts" / "classifier_head.safetensors"
    )
    assert (adapter_dir / "backbone.txt").read_text(encoding="utf-8") == "saved\n"
    assert (adapter_dir / "tokenizer.txt").read_text(encoding="utf-8") == "saved\n"
    assert classifier_payload.label_schema == ("alpha", "beta")
    assert classifier_payload.classifier_head_weights["alpha"] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
    assert classifier_payload.classifier_head_biases == pytest.approx(
        {"alpha": 0.01, "beta": -0.01}
    )
    assert manifest["adapter_dir"] == str(adapter_dir)
    assert manifest["classifier_path"] == str(classifier_path)
    assert manifest["run_control"] == {
        "track": "central_ssl",
        "budget_name": "smoke",
        "output_root": "runs/_smoke",
    }
    assert manifest["trainable_surface"] == {
        "name": "peft_text_encoder",
        "trainable_state": "peft_adapter_and_classifier_head",
    }
    assert manifest["ssl_algorithm"] == "fixmatch"
    assert report["schema_version"] == "central_peft_classifier_eval.v1"
    assert report["manifest"] == manifest
    assert report["results"]["test"] == {"accuracy": 0.8}
    assert report["results"]["best"] == {"accuracy": 0.8}
    assert report["results"]["final"] == {"macro_f1": 0.74}


def test_write_run_artifacts_writes_projection_artifacts(tmp_path: Path) -> None:
    cfg = SimpleNamespace(
        output_dir=tmp_path / "runs",
        train_jsonl=tmp_path / "train.jsonl",
        selection_set="validation",
        seed=7,
        epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
        learning_rate=0.001,
        classifier_learning_rate=0.002,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )
    eval_loader = DataLoader(
        [
            {
                "input_ids": torch.tensor([2.0, 0.0]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor(0),
            },
            {
                "input_ids": torch.tensor([0.0, 3.0]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor(1),
            },
        ],
        batch_size=2,
    )

    model = _ProjectionModel()
    final_state = model.state_dict()

    outputs = write_run_artifacts(
        cfg=cfg,
        trainer_version="run-002",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        model=model,
        tokenizer=_DummySaver("tokenizer.txt"),
        categories=["alpha", "beta"],
        eval_set_map={"validation": tmp_path / "validation.jsonl"},
        training_device="cpu",
        backbone_summary={"name": "dummy-backbone"},
        history=[],
        best_selection_report={"macro_f1": 1.0},
        final_selection_report=None,
        results={"validation": {"accuracy_top_1": 1.0}},
        final_model_state_dict=final_state,
        eval_loaders={"validation": eval_loader},
    )

    manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
    report = json.loads(Path(outputs["report_json"]).read_text(encoding="utf-8"))
    projection_manifest_path = Path(outputs["projection_manifest"])
    projection_manifest = json.loads(
        projection_manifest_path.read_text(encoding="utf-8")
    )
    best_projection = projection_manifest["datasets"]["best"]

    assert manifest["projection_artifacts"]["enabled"] is True
    assert set(report["results"]) == {"best", "final"}
    assert (
        projection_manifest["projection_space"]
        == "final_peft_encoder_pooled_backbone_features"
    )
    assert projection_manifest["mark_incorrect"] is False
    assert best_projection["row_count"] == 2
    assert Path(best_projection["points_jsonl"]).exists()
    assert Path(best_projection["figure_png"]).exists()
    final_projection_manifest = json.loads(
        Path(outputs["final_projection_manifest"]).read_text(encoding="utf-8")
    )
    final_projection = final_projection_manifest["datasets"]["final"]
    assert Path(outputs["final_adapter_dir"]).is_dir()
    assert Path(outputs["final_classifier_path"]).exists()
    assert final_projection["row_count"] == 2
    assert Path(final_projection["points_jsonl"]).exists()
    assert Path(final_projection["figure_png"]).exists()


def test_write_full_text_encoder_run_artifacts_writes_model_manifest_and_report(
    tmp_path: Path,
) -> None:
    cfg = SimpleNamespace(
        output_dir=tmp_path / "runs",
        central_ssl_budget=SimpleNamespace(name="smoke", output_root="runs/_smoke"),
        train_jsonl=tmp_path / "train.jsonl",
        selection_set="validation",
        seed=7,
        epochs=3,
        train_batch_size=4,
        eval_batch_size=5,
        learning_rate=0.00002,
        classifier_learning_rate=0.0002,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )
    model = SimpleNamespace(
        backbone=_DummySaver("model.txt"),
        classifier=_DummyClassifier(),
    )
    tokenizer = _DummySaver("tokenizer.txt")

    outputs = write_full_text_encoder_run_artifacts(
        cfg=cfg,
        trainer_version="full-run-001",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        model=model,
        tokenizer=tokenizer,
        categories=["alpha", "beta"],
        eval_set_map={"validation": tmp_path / "validation.jsonl"},
        training_device="cpu",
        backbone_summary={
            "name": "dummy-backbone",
            "trainable_surface": {
                "name": "full_text_encoder",
                "trainable_state": "full_encoder_and_classifier_head",
            },
        },
        history=[{"epoch": 1, "loss": 0.5}],
        best_selection_report={"macro_f1": 0.75},
        final_selection_report=None,
        results={"test": {"accuracy": 0.8}},
        extra_manifest={"experiment_family": "full_supervised"},
    )

    model_dir = Path(outputs["model_dir"])
    classifier_path = Path(outputs["classifier_path"])
    manifest_path = Path(outputs["manifest"])
    report_path = Path(outputs["report_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "adapter_dir" not in outputs
    assert Path(outputs["output_dir"]) == tmp_path / "runs" / "full-run-001"
    assert (Path(outputs["output_dir"]) / "logs").is_dir()
    assert model_dir == Path(outputs["output_dir"]) / "artifacts" / "model"
    assert classifier_path == (
        Path(outputs["output_dir"]) / "artifacts" / "classifier_head.safetensors"
    )
    assert (model_dir / "model.txt").read_text(encoding="utf-8") == "saved\n"
    assert (model_dir / "tokenizer.txt").read_text(encoding="utf-8") == "saved\n"
    assert load_classifier_head_state_tensor_artifact(classifier_path).hidden_size == 3
    assert manifest["model_dir"] == str(model_dir)
    assert manifest["classifier_path"] == str(classifier_path)
    assert manifest["trainable_surface"] == {
        "name": "full_text_encoder",
        "trainable_state": "full_encoder_and_classifier_head",
    }
    assert manifest["experiment_family"] == "full_supervised"
    assert report["schema_version"] == "central_full_text_encoder_eval.v1"
    assert report["manifest"] == manifest
    assert report["results"]["test"] == {"accuracy": 0.8}
    assert report["results"]["best"] == {"accuracy": 0.8}


def test_write_full_text_encoder_run_artifacts_writes_projection_artifacts(
    tmp_path: Path,
) -> None:
    cfg = SimpleNamespace(
        output_dir=tmp_path / "runs",
        train_jsonl=tmp_path / "train.jsonl",
        selection_set="test",
        seed=7,
        epochs=1,
        train_batch_size=2,
        eval_batch_size=2,
        learning_rate=0.00002,
        classifier_learning_rate=0.0002,
        weight_decay=0.01,
        max_grad_norm=1.0,
    )
    eval_loader = DataLoader(
        [
            {
                "input_ids": torch.tensor([2.0, 0.0]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor(0),
            },
            {
                "input_ids": torch.tensor([0.0, 3.0]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor(1),
            },
        ],
        batch_size=2,
    )

    model = _ProjectionModel()
    final_state = model.state_dict()

    outputs = write_full_text_encoder_run_artifacts(
        cfg=cfg,
        trainer_version="full-run-002",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        model=model,
        tokenizer=_DummySaver("tokenizer.txt"),
        categories=["alpha", "beta"],
        eval_set_map={"test": tmp_path / "test.jsonl"},
        training_device="cpu",
        backbone_summary={
            "name": "dummy-backbone",
            "trainable_surface": {
                "name": "full_text_encoder",
                "trainable_state": "full_encoder_and_classifier_head",
            },
        },
        history=[],
        best_selection_report={"macro_f1": 1.0},
        final_selection_report={"macro_f1": 1.0},
        results={"test": {"accuracy_top_1": 1.0}},
        final_model_state_dict=final_state,
        eval_loaders={"test": eval_loader},
    )

    manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
    report = json.loads(Path(outputs["report_json"]).read_text(encoding="utf-8"))
    projection_manifest_path = Path(outputs["projection_manifest"])
    projection_manifest = json.loads(
        projection_manifest_path.read_text(encoding="utf-8")
    )
    best_projection = projection_manifest["datasets"]["best"]

    assert manifest["projection_artifacts"]["enabled"] is True
    assert set(report["results"]) == {"best", "final"}
    assert (
        projection_manifest["projection_space"]
        == "final_full_text_encoder_pooled_backbone_features"
    )
    assert projection_manifest["mark_incorrect"] is False
    assert best_projection["row_count"] == 2
    assert Path(best_projection["points_jsonl"]).exists()
    assert Path(best_projection["figure_png"]).exists()
    final_projection_manifest = json.loads(
        Path(outputs["final_projection_manifest"]).read_text(encoding="utf-8")
    )
    final_projection = final_projection_manifest["datasets"]["final"]
    assert Path(outputs["final_model_dir"]).is_dir()
    assert Path(outputs["final_classifier_path"]).exists()
    assert final_projection["row_count"] == 2
    assert Path(final_projection["points_jsonl"]).exists()
    assert Path(final_projection["figure_png"]).exists()
