from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from torch import nn

from methods.adaptation.full_text_encoder.training import modeling


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.embedding = nn.Embedding(8, 3)

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return _TinyBackbone()


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*_args, **_kwargs):
        return SimpleNamespace(pad_token=None, eos_token="[PAD]", unk_token=None)


def _build_cfg() -> object:
    return OmegaConf.create(
        {
            "runtime": {
                "local_files_only": True,
            },
            "paper_backbone": {
                "model_id": "mixedbread-ai/mxbai-embed-large-v1",
                "revision": "main",
                "tokenizer_model_id": "mixedbread-ai/mxbai-embed-large-v1",
                "tokenizer_revision": "main",
                "cache_dir": "hf_cache",
                "trust_remote_code": False,
                "pooling": "mean",
                "max_length": 256,
                "task_prefix": "",
                "classifier_dropout": 0.0,
            },
            "trainable_surface": {
                "name": "full_text_encoder",
                "model_artifact_kind": "full_text_encoder_with_linear_head",
                "trainable_state": "full_encoder_and_classifier_head",
                "requires_peft_adapter": False,
                "supports_initial_adapter": False,
            },
            "initial_adapter_dir": "",
            "initial_classifier_path": "",
        }
    )


def test_build_full_text_encoder_model_marks_backbone_trainable(monkeypatch) -> None:
    monkeypatch.setattr(
        modeling,
        "require_transformer_auto_stack",
        lambda: (_FakeAutoModel, _FakeAutoTokenizer),
    )

    model, tokenizer, summary = modeling.build_model(
        cfg=_build_cfg(),
        categories=["anxiety", "normal"],
        device="cpu",
    )

    assert tokenizer.pad_token == "[PAD]"
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.classifier.out_features == 2
    assert summary["trainable_surface"] == {
        "name": "full_text_encoder",
        "model_artifact_kind": "full_text_encoder_with_linear_head",
        "trainable_state": "full_encoder_and_classifier_head",
        "requires_peft_adapter": False,
        "supports_initial_adapter": False,
    }
    assert summary["full_text_encoder_config"] == {
        "encoder_trainable": True,
        "classifier_head_trainable": True,
    }
    assert (
        summary["parameter_counts"]["trainable"] == summary["parameter_counts"]["total"]
    )


def test_build_full_text_encoder_model_rejects_initial_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        modeling,
        "require_transformer_auto_stack",
        lambda: (_FakeAutoModel, _FakeAutoTokenizer),
    )
    cfg = _build_cfg()
    cfg.initial_adapter_dir = "data/artifacts/peft_adapters/example"

    with pytest.raises(ValueError, match="does not accept initial_adapter_dir"):
        modeling.build_model(
            cfg=cfg,
            categories=["anxiety", "normal"],
            device="cpu",
        )
