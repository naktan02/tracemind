"""중앙 supervised run의 epoch별 warm-start checkpoint 저장."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.support.query_ssl_text_encoder.io.model_artifact_exporter import (
    save_classifier_head_artifact,
)


def write_peft_supervised_epoch_checkpoint(
    *,
    checkpoint_root: Path,
    trainer_version: str,
    epoch: int,
    model: Any,
    tokenizer: Any,
    categories: list[str],
    history: list[dict[str, Any]],
    best_checkpoint_state: Mapping[str, Any],
) -> dict[str, str]:
    """PEFT adapter/head를 epoch별 SSL warm-start artifact로 저장한다."""

    if not history:
        raise ValueError("Cannot write epoch checkpoint without training history.")
    epoch_record = dict(history[-1])
    step = int(epoch_record.get("step") or 0)
    checkpoint_dir = checkpoint_root / f"epoch_{int(epoch):04d}_step_{step:06d}"
    adapter_dir = checkpoint_dir / "adapter"
    classifier_path = checkpoint_dir / "classifier_head.safetensors"
    manifest_path = checkpoint_dir / "manifest.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.backbone.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    save_classifier_head_artifact(
        model=model,
        categories=categories,
        classifier_path=classifier_path,
    )
    manifest = {
        "schema_version": "central_peft_supervised_epoch_checkpoint.v1",
        "trainer_version": trainer_version,
        "epoch": int(epoch),
        "step": step,
        "adapter_dir": str(adapter_dir),
        "classifier_path": str(classifier_path),
        "selection_epoch_record": epoch_record,
        "best_selection_metric": _best_selection_metric(best_checkpoint_state),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    latest_path = checkpoint_root / "latest_epoch_checkpoint.json"
    latest_path.write_text(
        json.dumps(
            {
                "schema_version": "central_peft_supervised_latest_checkpoint.v1",
                "trainer_version": trainer_version,
                "epoch": int(epoch),
                "step": step,
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "supervised_epoch_checkpoint=saved "
        f"manifest={manifest_path} epoch={epoch} step={step}",
        flush=True,
    )
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_path": str(manifest_path),
        "adapter_dir": str(adapter_dir),
        "classifier_path": str(classifier_path),
    }


def _best_selection_metric(best_checkpoint_state: Mapping[str, Any]) -> float | None:
    raw_value = best_checkpoint_state.get("best_metric")
    if raw_value is None:
        return None
    return float(raw_value)
