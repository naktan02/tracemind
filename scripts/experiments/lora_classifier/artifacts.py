"""LoRA classifier 실험 산출물 저장 유틸리티."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from scripts.run_artifacts import build_run_dir


def write_run_artifacts(
    *,
    cfg,
    trainer_version: str,
    created_at: datetime,
    model: Any,
    tokenizer: Any,
    categories: list[str],
    eval_set_map: dict[str, Path],
    training_device: str,
    backbone_summary: dict[str, Any],
    history: list[dict[str, Any]],
    best_selection_report: dict[str, Any],
    results: dict[str, Any],
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = build_run_dir(
        cfg.output_dir,
        run_id=trainer_version,
        created_at=created_at,
    )
    adapter_output_dir = Path(str(cfg.adapter_output_dir)) / trainer_version
    classifier_output_dir = Path(str(cfg.classifier_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    classifier_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    classifier_path = classifier_output_dir / f"{trainer_version}.pt"
    classifier_manifest_path = (
        classifier_output_dir / f"{trainer_version}.manifest.json"
    )
    report_path = report_dir / "report.json"

    model.backbone.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)
    torch.save(
        {
            "classifier_state_dict": model.classifier.state_dict(),
            "categories": categories,
            "hidden_size": int(model.classifier.in_features),
        },
        classifier_path,
    )

    manifest = {
        "trainer_version": trainer_version,
        "train_jsonl": str(cfg.train_jsonl),
        "eval_sets": {name: str(path) for name, path in eval_set_map.items()},
        "selection_set": str(cfg.selection_set),
        "device": training_device,
        "seed": int(cfg.seed),
        "epochs": int(cfg.epochs),
        "train_batch_size": int(cfg.train_batch_size),
        "eval_batch_size": int(cfg.eval_batch_size),
        "learning_rate": float(cfg.learning_rate),
        "classifier_learning_rate": float(cfg.classifier_learning_rate),
        "weight_decay": float(cfg.weight_decay),
        "max_grad_norm": float(cfg.max_grad_norm),
        "categories": categories,
        "adapter_dir": str(adapter_output_dir),
        "classifier_path": str(classifier_path),
        "backbone": backbone_summary,
        "best_selection_report": best_selection_report,
        "history": history,
    }
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    report = {
        "schema_version": "central_lora_classifier_eval.v1",
        "trainer_version": trainer_version,
        "manifest": manifest,
        "results": results,
    }
    classifier_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_output_dir),
        "classifier_path": str(classifier_path),
        "manifest": str(classifier_manifest_path),
        "report_json": str(report_path),
    }
