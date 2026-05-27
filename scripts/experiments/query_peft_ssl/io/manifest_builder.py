"""Query-domain PEFT SSL manifest/report payload 조립."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.experiments.query_peft_ssl.io.artifact_paths import (
    QueryPeftRunArtifactPaths,
)


def build_query_peft_run_manifest(
    *,
    cfg: Any,
    trainer_version: str,
    eval_set_map: dict[str, Path],
    training_device: str,
    backbone_summary: dict[str, Any],
    history: list[dict[str, Any]],
    best_selection_report: dict[str, Any],
    categories: list[str],
    paths: QueryPeftRunArtifactPaths,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "trainer_version": trainer_version,
        "train_jsonl": str(cfg.train_jsonl),
        "eval_sets": {name: str(path) for name, path in eval_set_map.items()},
        "selection_set": str(cfg.selection_set),
        "device": training_device,
        "seed": int(cfg.seed),
        "epochs": int(cfg.epochs),
        "max_train_steps": (
            None
            if getattr(cfg, "max_train_steps", None) is None
            else int(cfg.max_train_steps)
        ),
        "train_batch_size": int(cfg.train_batch_size),
        "eval_batch_size": int(cfg.eval_batch_size),
        "learning_rate": float(cfg.learning_rate),
        "classifier_learning_rate": float(cfg.classifier_learning_rate),
        "weight_decay": float(cfg.weight_decay),
        "max_grad_norm": float(cfg.max_grad_norm),
        "categories": categories,
        "adapter_dir": str(paths.adapter_output_dir),
        "classifier_path": str(paths.classifier_path),
        "backbone": backbone_summary,
        "best_selection_report": best_selection_report,
        "history": history,
    }
    run_control = _build_run_control_manifest(cfg)
    if run_control is not None:
        manifest["run_control"] = run_control
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    return manifest


def build_query_peft_eval_report(
    *,
    trainer_version: str,
    manifest: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "central_lora_classifier_eval.v1",
        "trainer_version": trainer_version,
        "manifest": manifest,
        "results": results,
    }


def _build_run_control_manifest(cfg: Any) -> dict[str, object] | None:
    budget = getattr(cfg, "central_ssl_budget", None)
    if budget is None:
        return None
    return {
        "track": "central_ssl",
        "budget_name": _optional_str(getattr(budget, "name", None)),
        "output_root": _optional_str(getattr(budget, "output_root", None)),
    }


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
