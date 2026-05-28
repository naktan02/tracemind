"""PEFT encoder partitioned model builder tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from methods.adaptation.peft_text_encoder.config import (
    PeftEncoderTrainingBackendConfig,
)
from methods.adaptation.peft_text_encoder.federated_ssl.partitioned import (
    model_builder,
)
from methods.adaptation.peft_text_encoder.update.materialization import (
    PeftEncoderMaterializedState,
)


@dataclass(frozen=True, slots=True)
class TinyRuntimeConfig:
    device: str = "cpu"
    classifier_dropout: float = 0.0
    cache_dir: str | None = None
    local_files_only: bool = True
    trust_remote_code: bool = False


class TinyPeftEncoderTextClassifier(nn.Module):
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


def test_partitioned_peft_encoder_builder_loads_base_state_into_each_partition() -> (
    None
):
    calls: list[tuple[tuple[str, ...], object | None]] = []
    tokenizer = object()

    def classifier_factory(
        *,
        labels: list[str],
        peft_config: PeftEncoderTrainingBackendConfig,
        runtime_config: TinyRuntimeConfig,
        runtime_resource_cache: object | None = None,
    ) -> tuple[nn.Module, Any]:
        del peft_config, runtime_config
        calls.append((tuple(labels), runtime_resource_cache))
        return TinyPeftEncoderTextClassifier(), tokenizer

    cache = object()
    base_parameters = PeftEncoderMaterializedState(
        peft_parameters={
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

    result = model_builder.build_partitioned_peft_encoder_text_classifier_from_config(
        partition_names=("sigma", "psi"),
        labels=("anxiety", "normal"),
        base_parameters=base_parameters,
        peft_config=PeftEncoderTrainingBackendConfig(),
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


def test_partitioned_peft_encoder_builder_rejects_invalid_inputs() -> None:
    base_parameters = PeftEncoderMaterializedState(
        peft_parameters={},
        classifier_head_weights={},
        classifier_head_biases={},
    )

    with pytest.raises(ValueError, match="duplicates"):
        model_builder.build_partitioned_peft_encoder_text_classifier_from_config(
            partition_names=("sigma", "sigma"),
            labels=("anxiety", "normal"),
            base_parameters=base_parameters,
            peft_config=PeftEncoderTrainingBackendConfig(),
            runtime_config=TinyRuntimeConfig(),
            classifier_factory=_unused_classifier_factory,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="labels must not contain duplicates"):
        model_builder.build_partitioned_peft_encoder_text_classifier_from_config(
            partition_names=("sigma", "psi"),
            labels=("anxiety", "anxiety"),
            base_parameters=base_parameters,
            peft_config=PeftEncoderTrainingBackendConfig(),
            runtime_config=TinyRuntimeConfig(),
            classifier_factory=_unused_classifier_factory,  # type: ignore[arg-type]
        )


def test_partitioned_peft_encoder_builder_prefers_partition_base_state() -> None:
    base_parameters = PeftEncoderMaterializedState(
        peft_parameters={
            "encoder_lora.weight": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        },
        classifier_head_weights={
            "anxiety": [1.0, 1.1, 1.2],
            "normal": [1.3, 1.4, 1.5],
        },
        classifier_head_biases={"anxiety": 1.6, "normal": 1.7},
    )
    partition_base_parameters = {
        "sigma": PeftEncoderMaterializedState(
            peft_parameters={
                "encoder_lora.weight": [
                    2.1,
                    2.2,
                    2.3,
                    2.4,
                    2.5,
                    2.6,
                    2.7,
                    2.8,
                    2.9,
                ]
            },
            classifier_head_weights={
                "anxiety": [3.0, 3.1, 3.2],
                "normal": [3.3, 3.4, 3.5],
            },
            classifier_head_biases={"anxiety": 3.6, "normal": 3.7},
        )
    }

    result = model_builder.build_partitioned_peft_encoder_text_classifier_from_config(
        partition_names=("sigma", "psi"),
        labels=("anxiety", "normal"),
        base_parameters=base_parameters,
        base_partition_parameters=partition_base_parameters,
        peft_config=PeftEncoderTrainingBackendConfig(),
        runtime_config=TinyRuntimeConfig(),
        classifier_factory=_tiny_classifier_factory,  # type: ignore[arg-type]
    )

    sigma_parameters = result.model.partition_parameter_tensors("sigma")
    psi_parameters = result.model.partition_parameter_tensors("psi")
    torch.testing.assert_close(
        sigma_parameters["encoder_lora.weight"].detach().flatten(),
        torch.tensor([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]),
    )
    torch.testing.assert_close(
        psi_parameters["encoder_lora.weight"].detach().flatten(),
        torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    )


def _tiny_classifier_factory(
    *,
    labels: list[str],
    peft_config: PeftEncoderTrainingBackendConfig,
    runtime_config: TinyRuntimeConfig,
    runtime_resource_cache: object | None = None,
) -> tuple[nn.Module, Any]:
    del labels, peft_config, runtime_config, runtime_resource_cache
    return TinyPeftEncoderTextClassifier(), object()


def _unused_classifier_factory(
    *,
    labels: list[str],
    peft_config: PeftEncoderTrainingBackendConfig,
    runtime_config: TinyRuntimeConfig,
    runtime_resource_cache: object | None = None,
) -> tuple[nn.Module, Any]:  # pragma: no cover - validation fails first.
    del labels, peft_config, runtime_config, runtime_resource_cache
    raise AssertionError("classifier_factory should not be called.")
