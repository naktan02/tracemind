"""PEFT encoder classifier parameter delta extraction helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from methods.adaptation.peft_text_classifier.update.materialization import (
    PeftEncoderMaterializedState,
)

from .modeling import PeftEncoderTextClassifier


def load_peft_encoder_base_parameters_into_model(
    *,
    model: PeftEncoderTextClassifier,
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    device: str,
) -> None:
    """materialized global PEFT/head state를 local model 파라미터에 반영한다."""

    if base_parameters.peft_parameters:
        name_to_parameter = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("classifier.")
        }
        missing = sorted(set(base_parameters.peft_parameters) - set(name_to_parameter))
        if missing:
            raise ValueError(
                "Base PEFT adapter artifact contains parameters not present in model: "
                f"{missing[:5]}."
            )
        for name, values in base_parameters.peft_parameters.items():
            parameter = name_to_parameter[name]
            tensor = torch.tensor(values, dtype=parameter.dtype, device=device)
            if tensor.numel() != parameter.numel():
                raise ValueError(
                    f"Base PEFT adapter parameter shape mismatch for {name!r}."
                )
            parameter.data.copy_(tensor.reshape_as(parameter))

    if base_parameters.classifier_head_weights:
        weight_rows = []
        for label in labels:
            values = base_parameters.classifier_head_weights.get(str(label))
            if values is None:
                raise ValueError(
                    f"Base classifier head weights missing label {label!r}."
                )
            weight_rows.append(values)
        weight = torch.tensor(
            weight_rows,
            dtype=model.classifier.weight.dtype,
            device=device,
        )
        if tuple(weight.shape) != tuple(model.classifier.weight.shape):
            raise ValueError("Base classifier head weight shape mismatch.")
        model.classifier.weight.data.copy_(weight)

    if base_parameters.classifier_head_biases:
        bias = torch.tensor(
            [
                float(base_parameters.classifier_head_biases.get(str(label), 0.0))
                for label in labels
            ],
            dtype=model.classifier.bias.dtype,
            device=device,
        )
        if tuple(bias.shape) != tuple(model.classifier.bias.shape):
            raise ValueError("Base classifier head bias shape mismatch.")
        model.classifier.bias.data.copy_(bias)


def extract_peft_encoder_parameter_deltas(
    *,
    model: PeftEncoderTextClassifier,
    base_parameters: PeftEncoderMaterializedState,
    labels: Sequence[str],
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, float]]:
    """local model과 base global snapshot의 PEFT/head delta를 추출한다."""

    peft_parameter_deltas = extract_peft_parameter_deltas(
        model=model,
        base_parameters=base_parameters.peft_parameters,
    )
    head_weight_deltas, head_bias_deltas = extract_classifier_head_deltas(
        model=model,
        labels=labels,
        base_weights=base_parameters.classifier_head_weights,
        base_biases=base_parameters.classifier_head_biases,
    )
    return peft_parameter_deltas, head_weight_deltas, head_bias_deltas


def extract_peft_parameter_deltas(
    *,
    model: PeftEncoderTextClassifier,
    base_parameters: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """trainable PEFT adapter parameter delta를 flat vector mapping으로 추출한다."""

    deltas: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith("classifier."):
            continue
        current_values = parameter.detach().cpu().reshape(-1).tolist()
        base_values = [float(value) for value in base_parameters.get(name, [])]
        if base_values and len(base_values) != len(current_values):
            raise ValueError(
                f"Base PEFT adapter parameter dimension mismatch for {name!r}."
            )
        if not base_values:
            base_values = [0.0 for _ in current_values]
        deltas[name] = [
            float(current) - float(base)
            for current, base in zip(current_values, base_values, strict=True)
        ]
    if not deltas:
        raise ValueError("PEFT adapter model did not expose trainable parameters.")
    return deltas


def extract_classifier_head_deltas(
    *,
    model: PeftEncoderTextClassifier,
    labels: Sequence[str],
    base_weights: Mapping[str, Sequence[float]],
    base_biases: Mapping[str, float],
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """classifier head weight/bias delta를 label-keyed mapping으로 추출한다."""

    weight = model.classifier.weight.detach().cpu()
    bias = model.classifier.bias.detach().cpu()
    weight_deltas: dict[str, list[float]] = {}
    bias_deltas: dict[str, float] = {}
    for label_index, label in enumerate(labels):
        key = str(label)
        current_weight = weight[label_index].reshape(-1).tolist()
        base_weight = [float(value) for value in base_weights.get(key, [])]
        if base_weight and len(base_weight) != len(current_weight):
            raise ValueError(
                f"Base classifier head weight dimension mismatch for {key!r}."
            )
        if not base_weight:
            base_weight = [0.0 for _ in current_weight]
        weight_deltas[key] = [
            float(current) - float(base)
            for current, base in zip(current_weight, base_weight, strict=True)
        ]
        bias_deltas[key] = float(bias[label_index].item()) - float(
            base_biases.get(key, 0.0)
        )
    return weight_deltas, bias_deltas


def peft_encoder_delta_l2_norm(
    *,
    peft_parameter_deltas: Mapping[str, Sequence[float]],
    classifier_head_weight_deltas: Mapping[str, Sequence[float]],
    classifier_head_bias_deltas: Mapping[str, float],
) -> float:
    """PEFT/head delta mapping의 L2 norm을 계산한다."""

    squared_norm = 0.0
    for mapping in (peft_parameter_deltas, classifier_head_weight_deltas):
        squared_norm += sum(
            float(value) * float(value)
            for vector in mapping.values()
            for value in vector
        )
    squared_norm += sum(
        float(value) * float(value) for value in classifier_head_bias_deltas.values()
    )
    return math.sqrt(squared_norm)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_float_or_none(value: object) -> float | None:
    """metric 후보 값을 finite float으로 정규화한다."""

    if not is_number(value):
        return None
    normalized = float(value)
    if math.isnan(normalized):
        return None
    return normalized
