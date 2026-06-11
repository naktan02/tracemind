"""PEFT Query SSL trainer의 model extension lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import nn

from methods.ssl.base import QuerySslAlgorithm
from methods.ssl.model_capabilities import (
    build_query_ssl_auxiliary_modules,
    query_ssl_auxiliary_trainable_parameters,
)

from .modeling import PeftTextEncoderWithLinearHead


@dataclass(frozen=True, slots=True)
class PeftQuerySslModelExtensions:
    """PEFT trainer가 model 밖에서 함께 학습/checkpoint할 SSL extension."""

    auxiliary_modules: Mapping[str, nn.Module]
    auxiliary_trainable_parameters: tuple[nn.Parameter, ...]


def build_peft_query_ssl_model_extensions(
    *,
    algorithm: QuerySslAlgorithm,
    model: PeftTextEncoderWithLinearHead,
    device: str,
) -> PeftQuerySslModelExtensions:
    """algorithm-local auxiliary module을 PEFT trainer lifecycle에 연결한다."""

    auxiliary_modules = build_query_ssl_auxiliary_modules(
        algorithm,
        model=model,
    )
    for module in auxiliary_modules.values():
        module.to(device)
    return PeftQuerySslModelExtensions(
        auxiliary_modules=auxiliary_modules,
        auxiliary_trainable_parameters=query_ssl_auxiliary_trainable_parameters(
            auxiliary_modules
        ),
    )


def set_peft_query_ssl_auxiliary_modules_train(
    extensions: PeftQuerySslModelExtensions,
    *,
    training: bool,
) -> None:
    """model.train()/eval()과 분리된 auxiliary module mode를 맞춘다."""

    for module in extensions.auxiliary_modules.values():
        module.train(training)
