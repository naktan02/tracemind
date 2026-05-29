from __future__ import annotations

from types import SimpleNamespace

import pytest

from methods.adaptation.peft_text_encoder.training.modeling import (
    build_trainable_surface_manifest,
)


def test_build_trainable_surface_manifest_accepts_peft_text_encoder() -> None:
    cfg = SimpleNamespace(
        trainable_surface=SimpleNamespace(
            name="peft_text_encoder",
            model_artifact_kind="peft_text_encoder_with_linear_head",
            trainable_state="peft_adapter_and_classifier_head",
            requires_peft_adapter=True,
            supports_initial_adapter=True,
        )
    )

    assert build_trainable_surface_manifest(cfg) == {
        "name": "peft_text_encoder",
        "model_artifact_kind": "peft_text_encoder_with_linear_head",
        "trainable_state": "peft_adapter_and_classifier_head",
        "requires_peft_adapter": True,
        "supports_initial_adapter": True,
    }


def test_build_trainable_surface_manifest_rejects_missing_config() -> None:
    with pytest.raises(ValueError, match="trainable_surface config is required"):
        build_trainable_surface_manifest(SimpleNamespace())


def test_build_trainable_surface_manifest_rejects_unimplemented_surface() -> None:
    cfg = SimpleNamespace(
        trainable_surface=SimpleNamespace(
            name="full_text_encoder",
            model_artifact_kind="full_text_encoder_with_linear_head",
            trainable_state="full_encoder_and_classifier_head",
        )
    )

    with pytest.raises(ValueError, match="Unsupported trainable_surface.name"):
        build_trainable_surface_manifest(cfg)
