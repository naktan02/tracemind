"""Query adaptation initial checkpoint 해석 유틸리티."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass(slots=True)
class ResolvedQueryAdaptationInitialCheckpoint:
    """실험 runner가 공통으로 쓰는 initial checkpoint 해석 결과."""

    cfg: DictConfig
    extra_manifest: dict[str, Any]


def resolve_query_adaptation_initial_checkpoint(
    cfg: DictConfig,
) -> ResolvedQueryAdaptationInitialCheckpoint:
    """Hydra initial checkpoint 축을 canonical top-level path로 정규화한다."""

    checkpoint_cfg = getattr(cfg, "query_adaptation_initial_checkpoint", None)
    preset_name = (
        None
        if checkpoint_cfg is None
        else str(getattr(checkpoint_cfg, "name", "") or "").strip() or None
    )
    mode = (
        "none"
        if checkpoint_cfg is None
        else str(getattr(checkpoint_cfg, "mode", "none") or "").strip() or "none"
    )
    manifest_path = (
        ""
        if checkpoint_cfg is None
        else str(getattr(checkpoint_cfg, "manifest_path", "") or "").strip()
    )
    group_adapter_dir = (
        ""
        if checkpoint_cfg is None
        else str(getattr(checkpoint_cfg, "adapter_dir", "") or "").strip()
    )
    group_classifier_path = (
        ""
        if checkpoint_cfg is None
        else str(getattr(checkpoint_cfg, "classifier_path", "") or "").strip()
    )

    legacy_adapter_dir = str(getattr(cfg, "initial_adapter_dir", "") or "").strip()
    legacy_classifier_path = str(
        getattr(cfg, "initial_classifier_path", "") or ""
    ).strip()

    adapter_dir = group_adapter_dir or legacy_adapter_dir
    classifier_path = group_classifier_path or legacy_classifier_path
    manifest_payload: dict[str, Any] | None = None
    source = "none"
    resolved_kind = "none"

    if manifest_path:
        manifest_payload = _load_manifest_payload(Path(manifest_path))
        adapter_dir = (
            adapter_dir or str(manifest_payload.get("adapter_dir", "")).strip()
        )
        classifier_path = (
            classifier_path
            or str(
                manifest_payload.get("classifier_path", "")
                or manifest_payload.get("model_path", "")
            ).strip()
        )
        source = "manifest"
        resolved_kind = _detect_manifest_kind(manifest_payload)
    elif group_adapter_dir or group_classifier_path:
        source = "config_group_paths"
        resolved_kind = "explicit_paths"
    elif legacy_adapter_dir or legacy_classifier_path:
        source = "legacy_top_level"
        resolved_kind = "explicit_paths"

    if mode == "required" and not (adapter_dir or classifier_path):
        raise ValueError(
            "query_adaptation_initial_checkpoint is required for this experiment. "
            "Provide query_adaptation_initial_checkpoint.manifest_path or explicit "
            "adapter/classifier paths. Fresh-start ablation이면 "
            "`strategy_axes/adaptation/initial_checkpoint=none`으로 명시해라."
        )

    if adapter_dir:
        _ensure_path_exists(Path(adapter_dir), path_kind="adapter_dir")
    if classifier_path:
        _ensure_path_exists(Path(classifier_path), path_kind="classifier_path")

    resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    resolved_cfg.initial_adapter_dir = adapter_dir
    resolved_cfg.initial_classifier_path = classifier_path

    reference_id = None
    if manifest_payload is not None:
        reference_id = (
            str(
                manifest_payload.get("trainer_version", "")
                or manifest_payload.get("classifier_version", "")
            ).strip()
            or None
        )

    return ResolvedQueryAdaptationInitialCheckpoint(
        cfg=resolved_cfg,
        extra_manifest={
            "query_adaptation_initial_checkpoint": {
                "preset_name": preset_name,
                "mode": mode,
                "source": source,
                "resolved_kind": resolved_kind,
                "manifest_path": manifest_path or None,
                "reference_id": reference_id,
                "adapter_dir": adapter_dir or None,
                "classifier_path": classifier_path or None,
            }
        },
    )


def _load_manifest_payload(path: Path) -> dict[str, Any]:
    _ensure_path_exists(path, path_kind="manifest_path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("manifest"), dict):
        return dict(payload["manifest"])
    if not isinstance(payload, dict):
        raise ValueError(f"Initial checkpoint manifest must be a JSON object: {path}")
    return payload


def _detect_manifest_kind(payload: dict[str, Any]) -> str:
    if str(payload.get("adapter_dir", "")).strip():
        return "lora_classifier_manifest"
    if str(payload.get("model_path", "")).strip():
        return "fixed_classifier_manifest"
    return "generic_manifest"


def _ensure_path_exists(path: Path, *, path_kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Resolved query adaptation initial checkpoint {path_kind} does not "
            f"exist: {path}"
        )
