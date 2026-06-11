"""PEFT-backed partitioned text-encoder/head model builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from torch import nn

from methods.common.runtime_resources import RuntimeResourceCache

from ...config import PeftEncoderTrainingBackendConfig
from ...training.delta_extraction import (
    load_peft_encoder_base_parameters_into_model,
)
from ...training.modeling import build_peft_text_encoder_with_linear_head_from_config
from ...update.materialization import PeftEncoderMaterializedState
from . import trainable_model as ptm


class PeftEncoderPartitionRuntimeConfig(Protocol):
    """Partitioned PEFT encoder model build에 필요한 runtime config surface."""

    device: str
    classifier_dropout: float
    cache_dir: str | None
    local_files_only: bool
    trust_remote_code: bool


class PeftTextEncoderWithLinearHeadFactory(Protocol):
    """PEFT encoder module factory seam for tests and optional dependencies."""

    def __call__(
        self,
        *,
        labels: list[str],
        peft_config: PeftEncoderTrainingBackendConfig,
        runtime_config: PeftEncoderPartitionRuntimeConfig,
        runtime_resource_cache: RuntimeResourceCache | None = None,
    ) -> tuple[nn.Module, Any]:
        """Build one PEFT text encoder/head module and tokenizer."""


@dataclass(frozen=True, slots=True)
class PartitionedPeftTextEncoderWithLinearHeadBuildResult:
    """Partitioned text encoder/head model과 tokenizer build 결과."""

    model: ptm.PartitionedTrainableTextEncoderHeadModules
    tokenizer: Any
    partition_names: tuple[str, ...]


def build_partitioned_peft_text_encoder_with_linear_head_from_config(
    *,
    partition_names: Sequence[str],
    labels: Sequence[str],
    base_parameters: PeftEncoderMaterializedState,
    base_partition_parameters: Mapping[str, PeftEncoderMaterializedState] | None = None,
    peft_config: PeftEncoderTrainingBackendConfig,
    runtime_config: PeftEncoderPartitionRuntimeConfig,
    runtime_resource_cache: RuntimeResourceCache | None = None,
    module_factory: PeftTextEncoderWithLinearHeadFactory = (
        build_peft_text_encoder_with_linear_head_from_config
    ),
) -> PartitionedPeftTextEncoderWithLinearHeadBuildResult:
    """Build one full PEFT text encoder/head module per physical trainable partition.

    이 builder는 partition 이름의 method 의미를 해석하지 않는다. caller가 넘긴
    partition 이름마다 같은 global base state를 로드한 full text encoder/head
    module을 만들고, partition-local parameter key는 각 module의 canonical named
    parameter를 그대로 보존한다.
    """

    plan = ptm.TrainableAdapterPartitionPlan.from_names(partition_names)
    normalized_labels = _normalize_labels(labels)
    tokenizer: Any | None = None
    partition_specs: list[ptm.TextEncoderHeadPartitionSpec] = []

    for partition_name in plan.partition_names:
        partition_model, partition_tokenizer = module_factory(
            labels=list(normalized_labels),
            peft_config=peft_config,
            runtime_config=runtime_config,
            runtime_resource_cache=runtime_resource_cache,
        )
        load_peft_encoder_base_parameters_into_model(
            model=partition_model,  # type: ignore[arg-type]
            labels=normalized_labels,
            base_parameters=(
                base_partition_parameters.get(partition_name, base_parameters)
                if base_partition_parameters is not None
                else base_parameters
            ),
            device=runtime_config.device,
        )
        if tokenizer is None:
            tokenizer = partition_tokenizer
        partition_specs.append(
            ptm.TextEncoderHeadPartitionSpec(
                partition_name=partition_name,
                module=partition_model,
            )
        )

    if tokenizer is None:  # pragma: no cover - plan rejects empty names first.
        raise ValueError("partition_names must not be empty.")
    return PartitionedPeftTextEncoderWithLinearHeadBuildResult(
        model=ptm.PartitionedTrainableTextEncoderHeadModules(
            partitions=tuple(partition_specs),
            composition_policy=plan.composition_policy,
        ),
        tokenizer=tokenizer,
        partition_names=plan.partition_names,
    )


def _normalize_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(label).strip() for label in labels if str(label).strip())
    if not normalized:
        raise ValueError("labels must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("labels must not contain duplicates.")
    return normalized
