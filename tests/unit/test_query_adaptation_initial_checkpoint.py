from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.experiments.query_lora_ssl.config.initial_checkpoint import (
    resolve_query_adaptation_initial_checkpoint,
)


def _base_cfg() -> object:
    return OmegaConf.create(
        {
            "query_adaptation_initial_checkpoint": {
                "name": "none",
                "mode": "none",
                "manifest_path": "",
                "adapter_dir": "",
                "classifier_path": "",
            },
            "initial_adapter_dir": "",
            "initial_classifier_path": "",
        }
    )


def test_initial_checkpoint_none_keeps_fresh_start() -> None:
    cfg = _base_cfg()

    resolved = resolve_query_adaptation_initial_checkpoint(cfg)

    assert resolved.cfg.initial_adapter_dir == ""
    assert resolved.cfg.initial_classifier_path == ""
    assert resolved.extra_manifest["query_adaptation_initial_checkpoint"]["source"] == (
        "none"
    )


def test_initial_checkpoint_resolves_lora_manifest_paths(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    classifier_path = tmp_path / "classifier.pt"
    classifier_path.write_bytes(b"checkpoint")
    manifest_path = tmp_path / "lora.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "trainer_version": "seed_lora_v1",
                "adapter_dir": str(adapter_dir),
                "classifier_path": str(classifier_path),
            }
        ),
        encoding="utf-8",
    )
    cfg = _base_cfg()
    cfg.query_adaptation_initial_checkpoint = {
        "name": "required",
        "mode": "required",
        "manifest_path": str(manifest_path),
        "adapter_dir": "",
        "classifier_path": "",
    }

    resolved = resolve_query_adaptation_initial_checkpoint(cfg)

    assert resolved.cfg.initial_adapter_dir == str(adapter_dir)
    assert resolved.cfg.initial_classifier_path == str(classifier_path)
    assert (
        resolved.extra_manifest["query_adaptation_initial_checkpoint"]["resolved_kind"]
        == "lora_classifier_manifest"
    )
    assert (
        resolved.extra_manifest["query_adaptation_initial_checkpoint"]["reference_id"]
        == "seed_lora_v1"
    )


def test_initial_checkpoint_resolves_fixed_classifier_manifest_model_path(
    tmp_path: Path,
) -> None:
    classifier_path = tmp_path / "fixed_classifier.pt"
    classifier_path.write_bytes(b"classifier")
    manifest_path = tmp_path / "fixed.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classifier_version": "clf_seed_v1",
                "model_path": str(classifier_path),
            }
        ),
        encoding="utf-8",
    )
    cfg = _base_cfg()
    cfg.query_adaptation_initial_checkpoint = {
        "name": "fixed_seed",
        "mode": "required",
        "manifest_path": str(manifest_path),
        "adapter_dir": "",
        "classifier_path": "",
    }

    resolved = resolve_query_adaptation_initial_checkpoint(cfg)

    assert resolved.cfg.initial_adapter_dir == ""
    assert resolved.cfg.initial_classifier_path == str(classifier_path)
    assert (
        resolved.extra_manifest["query_adaptation_initial_checkpoint"]["resolved_kind"]
        == "fixed_classifier_manifest"
    )
    assert (
        resolved.extra_manifest["query_adaptation_initial_checkpoint"]["reference_id"]
        == "clf_seed_v1"
    )


def test_initial_checkpoint_required_without_paths_raises() -> None:
    cfg = _base_cfg()
    cfg.query_adaptation_initial_checkpoint = {
        "name": "required",
        "mode": "required",
        "manifest_path": "",
        "adapter_dir": "",
        "classifier_path": "",
    }

    with pytest.raises(ValueError, match="query_adaptation_initial_checkpoint"):
        resolve_query_adaptation_initial_checkpoint(cfg)
