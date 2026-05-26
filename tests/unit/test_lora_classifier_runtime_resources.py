"""LoRA-classifier runtime resource cache tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from torch import nn

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.federated_ssl import peer_predictions
from methods.adaptation.lora_classifier.training import modeling
from methods.adaptation.lora_classifier.training.delta_extraction import (
    load_lora_classifier_base_parameters_into_model,
)
from methods.federated_ssl.peer_context import (
    FederatedSslPeerClientSnapshot,
    FederatedSslPeerContext,
)


@dataclass(slots=True)
class _RuntimeConfig:
    device: str = "cpu"
    classifier_dropout: float = 0.1
    cache_dir: str | None = "hf_cache"
    local_files_only: bool = True
    trust_remote_code: bool = False


@dataclass(slots=True)
class _Cache:
    resources: dict[str, object] = field(default_factory=dict)

    def get_resource(self, key: str) -> object | None:
        return self.resources.get(key)

    def set_resource(self, key: str, value: object) -> None:
        self.resources[key] = value


def test_lora_classifier_model_builder_reuses_runtime_resources(
    monkeypatch,
) -> None:
    calls = {"tokenizer": 0, "model": 0}

    class _Tokenizer:
        pad_token = None
        eos_token = "<eos>"
        unk_token = "<unk>"

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            calls["tokenizer"] += 1
            return _Tokenizer()

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)

    class _AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            calls["model"] += 1
            return _Backbone()

    class _TaskType:
        FEATURE_EXTRACTION = "FEATURE_EXTRACTION"

    class _LoraConfig:
        def __init__(self, **_kwargs) -> None:
            pass

    def _fake_get_peft_model(backbone_base, _peft_config):
        return backbone_base

    monkeypatch.setattr(
        modeling,
        "require_transformer_stack",
        lambda: (
            _AutoModel,
            _AutoTokenizer,
            _LoraConfig,
            _TaskType,
            _fake_get_peft_model,
            object,
        ),
    )

    cache = _Cache()
    lora_config = LoraClassifierTrainingBackendConfig()
    runtime_config = _RuntimeConfig()

    model_a, tokenizer_a = modeling.build_lora_text_classifier_from_config(
        labels=["anxiety", "normal"],
        lora_config=lora_config,
        runtime_config=runtime_config,
        runtime_resource_cache=cache,
    )
    model_b, tokenizer_b = modeling.build_lora_text_classifier_from_config(
        labels=["anxiety", "normal"],
        lora_config=lora_config,
        runtime_config=runtime_config,
        runtime_resource_cache=cache,
    )

    assert calls == {"tokenizer": 1, "model": 1}
    assert tokenizer_a is tokenizer_b
    assert model_a.backbone is not model_b.backbone


def test_load_lora_classifier_base_parameters_does_not_mutate_snapshot() -> None:
    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)

    model = modeling.LoraTextClassifier(
        backbone=_Backbone(),
        hidden_size=2,
        num_labels=2,
        classifier_dropout=0.0,
    )
    base_parameters = LoraClassifierMaterializedState(
        lora_parameters={},
        classifier_head_weights={
            "anxiety": [0.1, 0.2],
            "normal": [-0.1, -0.2],
        },
        classifier_head_biases={"anxiety": 0.03, "normal": -0.03},
    )
    original_weights = {
        label: list(values)
        for label, values in base_parameters.classifier_head_weights.items()
    }
    original_biases = dict(base_parameters.classifier_head_biases)

    load_lora_classifier_base_parameters_into_model(
        model=model,
        labels=("anxiety", "normal"),
        base_parameters=base_parameters,
        device="cpu",
    )

    assert base_parameters.classifier_head_weights == original_weights
    assert base_parameters.classifier_head_biases == original_biases


def test_lora_classifier_helper_provider_reuses_materialized_helper_model(
    monkeypatch,
) -> None:
    calls = {"build": 0, "load": 0}

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, *, input_ids, attention_mask):
            del attention_mask
            hidden = self.linear(input_ids.float()).unsqueeze(dim=1)
            return SimpleNamespace(last_hidden_state=hidden)

    def _fake_build_model(**_kwargs):
        calls["build"] += 1
        return (
            peer_predictions.LoraTextClassifier(
                backbone=_Backbone(),
                hidden_size=2,
                num_labels=2,
                classifier_dropout=0.1,
            ),
            object(),
        )

    def _fake_load_base_parameters(**_kwargs) -> None:
        calls["load"] += 1

    monkeypatch.setattr(
        peer_predictions,
        "build_lora_text_classifier_from_config",
        _fake_build_model,
    )
    monkeypatch.setattr(
        peer_predictions,
        "load_lora_classifier_base_parameters_into_model",
        _fake_load_base_parameters,
    )

    snapshot = FederatedSslPeerClientSnapshot(
        client_id="agent_02",
        selection_vector=(0.2, 0.8),
        payload_kind=peer_predictions.LORA_CLASSIFIER_PEER_SNAPSHOT_KIND,
        payload=LoraClassifierMaterializedState(
            lora_parameters={"lora.test": [0.1]},
            classifier_head_weights={
                "anxiety": [0.1, 0.0],
                "normal": [0.0, -0.1],
            },
            classifier_head_biases={"anxiety": 0.01, "normal": -0.01},
        ),
    )
    cache = _Cache()
    context = FederatedSslPeerContext(
        client_id="agent_01",
        policy_name="fixed_probe_output_knn",
        round_index_zero_based=1,
        helper_client_ids=("agent_02",),
        refreshed=True,
    )

    provider_a = peer_predictions.build_lora_classifier_helper_probability_provider(
        peer_context=context,
        peer_snapshots={"agent_02": snapshot},
        labels=("anxiety", "normal"),
        lora_config=LoraClassifierTrainingBackendConfig(),
        trainer_runtime_config=_RuntimeConfig(),
        runtime_resource_cache=cache,
    )
    provider_b = peer_predictions.build_lora_classifier_helper_probability_provider(
        peer_context=context,
        peer_snapshots={"agent_02": snapshot},
        labels=("anxiety", "normal"),
        lora_config=LoraClassifierTrainingBackendConfig(),
        trainer_runtime_config=_RuntimeConfig(),
        runtime_resource_cache=cache,
    )

    assert calls == {"build": 0, "load": 0}
    assert provider_a is not None
    assert provider_b is not None
    assert provider_a.helper_count == 1
    assert provider_a.materialized_helper_count == 0

    batch = {
        "weak_input_ids": torch.ones(1, 2),
        "weak_attention_mask": torch.ones(1, 2),
    }
    assert provider_a(unlabeled_batch=batch) is not None
    assert calls == {"build": 1, "load": 1}
    assert provider_a.materialized_helper_count == 1

    assert provider_b(unlabeled_batch=batch) is not None
    assert calls == {"build": 1, "load": 1}
    assert provider_a.helper_models[0] is provider_b.helper_models[0]


def test_lora_classifier_helper_provider_counts_only_materializable_snapshots() -> None:
    valid_snapshot = FederatedSslPeerClientSnapshot(
        client_id="agent_02",
        selection_vector=(0.2, 0.8),
        payload_kind=peer_predictions.LORA_CLASSIFIER_PEER_SNAPSHOT_KIND,
        payload=LoraClassifierMaterializedState(
            lora_parameters={"lora.test": [0.1]},
            classifier_head_weights={
                "anxiety": [0.1, 0.0],
                "normal": [0.0, -0.1],
            },
            classifier_head_biases={"anxiety": 0.01, "normal": -0.01},
        ),
    )
    ignored_snapshot = FederatedSslPeerClientSnapshot(
        client_id="agent_03",
        selection_vector=(0.8, 0.2),
        payload_kind="other_snapshot.v1",
        payload=object(),
    )
    context = FederatedSslPeerContext(
        client_id="agent_01",
        policy_name="fixed_probe_output_knn",
        round_index_zero_based=1,
        helper_client_ids=("agent_02", "agent_03", "agent_04"),
        refreshed=True,
    )

    provider = peer_predictions.build_lora_classifier_helper_probability_provider(
        peer_context=context,
        peer_snapshots={
            "agent_02": valid_snapshot,
            "agent_03": ignored_snapshot,
        },
        labels=("anxiety", "normal"),
        lora_config=LoraClassifierTrainingBackendConfig(),
        trainer_runtime_config=_RuntimeConfig(),
        runtime_resource_cache=None,
    )

    assert context.helper_count == 3
    assert provider is not None
    assert provider.helper_count == 1
    assert provider.materialized_helper_count == 0
