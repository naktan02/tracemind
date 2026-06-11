"""Query SSL model capability helper tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from methods.ssl.model_capabilities import (
    build_query_ssl_auxiliary_modules,
    forward_logits_and_pooled_features_once,
    query_ssl_auxiliary_trainable_parameters,
    require_pooled_feature_classifier,
)


class _FeatureModel(nn.Module):
    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        return torch.ones((input_ids.shape[0], 2), dtype=torch.float32)

    def extract_pooled_features(self, *, input_ids, attention_mask):
        del attention_mask
        return torch.ones((input_ids.shape[0], 3), dtype=torch.float32)


class _LogitsOnlyModel:
    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        return torch.ones((input_ids.shape[0], 2), dtype=torch.float32)


class _ScaleFeatures(nn.Module):
    def forward(self, features):
        return features * 10.0


class _SinglePassFeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = _ScaleFeatures()
        self.classifier = nn.Linear(2, 2, bias=False)
        self.extract_call_count = 0
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(2))

    def forward(self, *, input_ids, attention_mask):
        del input_ids, attention_mask
        raise AssertionError("single-pass helper must not call model.forward")

    def extract_pooled_features(self, *, input_ids, attention_mask):
        del attention_mask
        self.extract_call_count += 1
        return input_ids.to(dtype=torch.float32)


class _AuxiliaryAlgorithm:
    algorithm_name = "auxiliary"

    def build_auxiliary_modules(self, *, model):
        require_pooled_feature_classifier(model)
        return {"projection": nn.Linear(3, 2)}


def test_require_pooled_feature_classifier_accepts_feature_model() -> None:
    model = _FeatureModel()

    classifier = require_pooled_feature_classifier(model)
    pooled = classifier.extract_pooled_features(
        input_ids=torch.ones((2, 2), dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
    )

    assert tuple(pooled.shape) == (2, 3)


def test_require_pooled_feature_classifier_rejects_logits_only_model() -> None:
    with pytest.raises(TypeError, match="extract_pooled_features"):
        require_pooled_feature_classifier(_LogitsOnlyModel())


def test_forward_logits_and_pooled_features_once_reuses_single_pooled_tensor() -> None:
    model = _SinglePassFeatureModel()

    logits, pooled = forward_logits_and_pooled_features_once(
        model,
        input_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
    )

    assert model.extract_call_count == 1
    assert torch.allclose(
        pooled,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        logits,
        torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32),
    )


def test_build_query_ssl_auxiliary_modules_requires_named_modules() -> None:
    algorithm = _AuxiliaryAlgorithm()

    modules = build_query_ssl_auxiliary_modules(
        algorithm,
        model=_FeatureModel(),
    )
    parameters = query_ssl_auxiliary_trainable_parameters(modules)

    assert list(modules) == ["projection"]
    assert len(parameters) == 2


def test_build_query_ssl_auxiliary_modules_rejects_unnamed_shapes() -> None:
    algorithm = SimpleNamespace(
        algorithm_name="bad_auxiliary",
        build_auxiliary_modules=lambda *, model: [nn.Linear(3, 2)],
    )

    with pytest.raises(TypeError, match="mapping"):
        build_query_ssl_auxiliary_modules(
            algorithm,
            model=_FeatureModel(),
        )
