"""PEFT-backed partitioned LoRA-classifier builder tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from methods.adaptation.lora_classifier.aggregation.materialization import (
    LoraClassifierMaterializedState,
)
from methods.adaptation.lora_classifier.config import (
    LoraClassifierTrainingBackendConfig,
)
from methods.adaptation.lora_classifier.federated_ssl.partitioned_model_builder import (
    build_partitioned_lora_text_classifier_from_config,
)


@dataclass(frozen=True, slots=True)
class TinyRuntimeConfig:
    device: str = "cpu"
    classifier_dropout: float = 0.0
    cache_dir: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False


class TinyLoraTextClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_lora = nn.Linear(3, 3, bias=False)
        self.classifier = nn.Linear(3, 2)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        del attention_mask
        return self.classifier(self.encoder_lora(input_ids.float()))


def test_partitioned_lora_builder_loads_base_state_into_each_partition() -> None:
    calls: list[tuple[tuple[str, ...], object | None]] = []
    tokenizer = object()

    def classifier_factory(
        *,
        labels: list[str],
        lora_config: LoraClassifierTrainingBackendConfig,
        runtime_config: TinyRuntimeConfig,
        runtime_resource_cache: object | None = None,
    ) -> tuple[nn.Module, Any]:
        del lora_config, runtime_config
        calls.append((tuple(labels), runtime_resource_cache))
        return TinyLoraTextClassifier(), tokenizer

    cache = object()
    base_parameters = LoraClassifierMaterializedState(
        lora_parameters={
            "encoder_lora.weight": [
                0.2,
                0.0,
                0.1,
                0.0,
                0.3,
                0.1,
                0.1,
                0.0,
                0.4,
            ]
        },
        classifier_head_weights={
            "anxiety": [0.5, -0.2, 0.1],
            "normal": [-0.1, 0.3, 0.2],
        },
        classifier_head_biases={"anxiety": 0.1, "normal": -0.1},
    )

    result = build_partitioned_lora_text_classifier_from_config(
        partition_names=("sigma", "psi"),
        labels=("anxiety", "normal"),
        base_parameters=base_parameters,
        lora_config=LoraClassifierTrainingBackendConfig(),
        runtime_config=TinyRuntimeConfig(),
        runtime_resource_cache=cache,  # type: ignore[arg-type]
        classifier_factory=classifier_factory,  # type: ignore[arg-type]
    )

    assert result.partition_names == ("sigma", "psi")
    assert result.tokenizer is tokenizer
    assert calls == [
        (("anxiety", "normal"), cache),
        (("anxiety", "normal"), cache),
    ]
    sigma_parameters = result.model.partition_parameter_tensors("sigma")
    psi_parameters = result.model.partition_parameter_tensors("psi")
    assert set(sigma_parameters) == {
        "encoder_lora.weight",
        "classifier.weight",
        "classifier.bias",
    }
    assert set(psi_parameters) == set(sigma_parameters)
    torch.testing.assert_close(
        sigma_parameters["encoder_lora.weight"].detach(),
        psi_parameters["encoder_lora.weight"].detach(),
    )
    torch.testing.assert_close(
        sigma_parameters["classifier.bias"].detach(),
        torch.tensor([0.1, -0.1]),
    )


def test_partitioned_lora_builder_rejects_invalid_partition_and_label_inputs() -> None:
    base_parameters = LoraClassifierMaterializedState(
        lora_parameters={},
        classifier_head_weights={},
        classifier_head_biases={},
    )

    with pytest.raises(ValueError, match="duplicates"):
        build_partitioned_lora_text_classifier_from_config(
            partition_names=("sigma", "sigma"),
            labels=("anxiety", "normal"),
            base_parameters=base_parameters,
            lora_config=LoraClassifierTrainingBackendConfig(),
            runtime_config=TinyRuntimeConfig(),
            classifier_factory=_unused_classifier_factory,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="labels must not contain duplicates"):
        build_partitioned_lora_text_classifier_from_config(
            partition_names=("sigma", "psi"),
            labels=("anxiety", "anxiety"),
            base_parameters=base_parameters,
            lora_config=LoraClassifierTrainingBackendConfig(),
            runtime_config=TinyRuntimeConfig(),
            classifier_factory=_unused_classifier_factory,  # type: ignore[arg-type]
        )


def _unused_classifier_factory(
    *,
    labels: list[str],
    lora_config: LoraClassifierTrainingBackendConfig,
    runtime_config: TinyRuntimeConfig,
    runtime_resource_cache: object | None = None,
) -> tuple[nn.Module, Any]:  # pragma: no cover - validation fails first.
    del labels, lora_config, runtime_config, runtime_resource_cache
    raise AssertionError("classifier_factory should not be called.")
