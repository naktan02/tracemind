"""중앙 supervised epoch checkpoint 저장 검증."""

from __future__ import annotations

import json

import torch
from torch import nn

from scripts.support.query_ssl_text_encoder.io.supervised_epoch_checkpoints import (
    write_peft_supervised_epoch_checkpoint,
)


class _DummyBackbone:
    def save_pretrained(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_text("stub", encoding="utf-8")


class _DummyTokenizer:
    def save_pretrained(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _DummyModel:
    def __init__(self) -> None:
        self.backbone = _DummyBackbone()
        self.classifier = nn.Linear(3, 2)


def test_write_peft_supervised_epoch_checkpoint_exports_warm_start_manifest(
    tmp_path,
) -> None:
    checkpoint = write_peft_supervised_epoch_checkpoint(
        checkpoint_root=tmp_path / "checkpoints",
        trainer_version="peft_clf_test",
        epoch=2,
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        categories=["anxiety", "normal"],
        history=[
            {
                "epoch": 2,
                "step": 4000,
                "selection_accuracy_top_1": 0.8,
            }
        ],
        best_checkpoint_state={"best_metric": 0.8},
    )

    manifest = json.loads(
        (tmp_path / "checkpoints/epoch_0002_step_004000/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint["manifest_path"].endswith(
        "checkpoints/epoch_0002_step_004000/manifest.json"
    )
    assert manifest["adapter_dir"].endswith(
        "checkpoints/epoch_0002_step_004000/adapter"
    )
    assert manifest["classifier_path"].endswith(
        "checkpoints/epoch_0002_step_004000/classifier_head.pt"
    )
    assert manifest["epoch"] == 2
    assert manifest["step"] == 4000
    saved_head = torch.load(manifest["classifier_path"], map_location="cpu")
    assert saved_head["categories"] == ["anxiety", "normal"]
    latest = json.loads(
        (tmp_path / "checkpoints/latest_epoch_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert latest["manifest_path"] == checkpoint["manifest_path"]
