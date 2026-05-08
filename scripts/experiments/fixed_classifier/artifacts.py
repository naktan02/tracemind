"""Fixed classifier artifact 저장/로드."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from scripts.artifacts.run_artifacts import build_run_dir
from scripts.experiments.fixed_classifier.models import TrainedFixedClassifier
from scripts.runtime_adapters.embedding_runtime import (
    create_embedding_adapter,
    resolve_runtime_device_name,
)
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


def write_fixed_classifier_artifacts(
    *,
    classifier_version: str,
    created_at: datetime,
    train_jsonl_ref: str,
    eval_set_map: dict[str, str],
    selection_set_name: str,
    output_dir_root: str,
    model_output_dir: str,
    epochs: int,
    train_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    trained: TrainedFixedClassifier,
) -> dict[str, str]:
    """fixed classifier 산출물을 저장한다."""

    output_dir = build_run_dir(
        output_dir_root,
        run_id=classifier_version,
        created_at=created_at,
    )
    resolved_model_output_dir = Path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    model_path = resolved_model_output_dir / f"{classifier_version}.pt"
    manifest_path = resolved_model_output_dir / f"{classifier_version}.manifest.json"
    report_path = report_dir / "report.json"

    torch.save(
        {
            "classifier_state_dict": trained.model.state_dict(),
            "categories": trained.categories,
            "embedding_model_id": trained.embedding_spec.model_id,
            "embedding_model_revision": trained.embedding_spec.revision,
            "backend": trained.embedding_spec.backend,
        },
        model_path,
    )
    manifest = {
        "classifier_version": classifier_version,
        "train_jsonl": train_jsonl_ref,
        "eval_sets": dict(eval_set_map),
        "selection_set": selection_set_name,
        "embedding_backend": trained.embedding_spec.backend,
        "embedding_model_id": trained.embedding_spec.model_id,
        "embedding_model_revision": trained.embedding_spec.revision,
        "task_prefix": trained.embedding_spec.task_prefix,
        "device": trained.training_device,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "categories": trained.categories,
        "model_path": str(model_path),
        "best_selection_report": trained.best_selection_report,
        "history": trained.history,
    }
    report = {
        "schema_version": "classifier_head_eval.v1",
        "classifier_version": classifier_version,
        "manifest": manifest,
        "results": trained.eval_results,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "manifest": str(manifest_path),
        "report_json": str(report_path),
    }


def load_fixed_classifier_artifacts(
    *,
    manifest_path: str | Path,
    device: str,
    batch_size: int,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> tuple[TrainedFixedClassifier, dict[str, str]]:
    """저장된 fixed classifier artifact를 teacher/reference로 다시 연다."""

    resolved_manifest_path = Path(str(manifest_path))
    manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    model_path = Path(str(manifest["model_path"]))
    serialized = cast(
        dict[str, Any],
        torch.load(model_path, map_location="cpu"),
    )
    categories = [str(category) for category in serialized["categories"]]
    state_dict = cast(dict[str, torch.Tensor], serialized["classifier_state_dict"])
    hidden_size = int(serialized.get("hidden_size", state_dict["weight"].shape[1]))
    training_device = resolve_runtime_device_name(device)
    model = nn.Linear(hidden_size, len(categories)).to(training_device)
    model.load_state_dict(state_dict)
    model.eval()

    embedding_spec = EmbeddingAdapterSpec(
        backend=str(manifest["embedding_backend"]),
        model_id=str(manifest["embedding_model_id"]),
        revision=str(manifest["embedding_model_revision"]),
        device=training_device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        task_prefix=str(manifest.get("task_prefix", "")),
        local_files_only=local_files_only,
    )
    adapter = create_embedding_adapter(embedding_spec)
    classifier_version = str(manifest["classifier_version"])
    report_path = (
        Path("runs/train_classifier") / classifier_version / "reports" / "report.json"
    )
    outputs = {
        "model_path": str(model_path),
        "manifest": str(resolved_manifest_path),
        "report_json": str(report_path),
    }
    if report_path.exists():
        outputs["output_dir"] = str(report_path.parent.parent)

    return (
        TrainedFixedClassifier(
            model=model,
            adapter=adapter,
            embedding_spec=embedding_spec,
            categories=categories,
            label_to_index={label: index for index, label in enumerate(categories)},
            training_device=training_device,
            history=[],
            best_selection_report=dict(manifest.get("best_selection_report", {})),
            eval_results={},
        ),
        outputs,
    )
