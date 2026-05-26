"""Physical trainable-adapter partition model primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

PARTITION_COMPOSITION_SUM_LOGITS = "sum_logits"


class TextFeatureExtractor(Protocol):
    """Frozen text feature extractor surface used by partitioned adapters."""

    def __call__(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return pooled features for adapter/head partitions."""


class PartitionedTrainableTextClassifier(Protocol):
    """Partition별 trainable classifier forward/update surface."""

    def forward_partition(
        self,
        partition_name: str,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return logits from one trainable partition."""

    def partition_parameters(self, partition_name: str) -> tuple[nn.Parameter, ...]:
        """Return optimizer parameters for one partition."""

    def partition_parameter_tensors(
        self,
        partition_name: str,
    ) -> dict[str, nn.Parameter]:
        """Return trainable tensors with keys local to one partition."""


@dataclass(frozen=True, slots=True)
class TrainableAdapterPartitionPlan:
    """Method-owned partition names interpreted by adapter-family execution."""

    partition_names: tuple[str, ...]
    composition_policy: str = PARTITION_COMPOSITION_SUM_LOGITS

    @classmethod
    def from_names(
        cls,
        partition_names: Sequence[str],
        *,
        composition_policy: str = PARTITION_COMPOSITION_SUM_LOGITS,
    ) -> TrainableAdapterPartitionPlan:
        normalized = tuple(_normalize_partition_name(name) for name in partition_names)
        if len(set(normalized)) != len(normalized):
            raise ValueError("partition_names must not contain duplicates.")
        return cls(
            partition_names=normalized,
            composition_policy=_normalize_composition_policy(composition_policy),
        )


@dataclass(frozen=True, slots=True)
class AdapterClassifierPartitionSpec:
    """One physical adapter/head partition module pair."""

    partition_name: str
    adapter: nn.Module
    classifier: nn.Module

    def __post_init__(self) -> None:
        _normalize_partition_name(self.partition_name)


@dataclass(frozen=True, slots=True)
class TextClassifierPartitionSpec:
    """One full text-classifier module owned by a physical partition."""

    partition_name: str
    module: nn.Module

    def __post_init__(self) -> None:
        _normalize_partition_name(self.partition_name)


class AdapterClassifierPartition(nn.Module):
    """Trainable adapter and classifier head for a single partition."""

    def __init__(self, *, adapter: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.classifier = classifier

    def forward(self, features: Tensor) -> Tensor:
        adapted = self.adapter(features)
        return self.classifier(adapted)


class PartitionedTrainableTextClassifierModules(nn.Module):
    """Physical partitions backed by full text-classifier modules.

    이 wrapper는 PEFT adapter가 transformer 내부에 붙는 `LoraTextClassifier` 같은
    모델을 partition 단위로 감싼다. feature-space adapter를 새로 가정하지 않고,
    각 partition module이 자기 forward와 trainable parameter 이름을 소유한다.
    """

    def __init__(
        self,
        *,
        partitions: Sequence[TextClassifierPartitionSpec],
        composition_policy: str = PARTITION_COMPOSITION_SUM_LOGITS,
    ) -> None:
        super().__init__()
        self.composition_policy = _normalize_composition_policy(composition_policy)
        self.partitions = nn.ModuleDict(
            {
                _normalize_partition_name(spec.partition_name): spec.module
                for spec in partitions
            }
        )
        if len(self.partitions) != len(partitions):
            raise ValueError("partitions must not contain duplicate names.")
        if not self.partitions:
            raise ValueError("at least one text-classifier partition is required.")

    def forward_partition(
        self,
        partition_name: str,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        partition = self.require_partition(partition_name)
        return partition(input_ids=input_ids, attention_mask=attention_mask)

    def forward_composed(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        partition_names: Sequence[str] | None = None,
    ) -> Tensor:
        names = (
            tuple(self.partitions.keys())
            if partition_names is None
            else tuple(_normalize_partition_name(name) for name in partition_names)
        )
        if not names:
            raise ValueError("partition_names must not be empty.")
        logits = [
            self.forward_partition(
                name,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            for name in names
        ]
        if self.composition_policy == PARTITION_COMPOSITION_SUM_LOGITS:
            result = logits[0]
            for value in logits[1:]:
                result = result + value
            return result
        raise ValueError(f"Unsupported composition policy: {self.composition_policy}")

    def require_partition(self, partition_name: str) -> nn.Module:
        normalized = _normalize_partition_name(partition_name)
        try:
            return self.partitions[normalized]
        except KeyError as error:
            raise ValueError(
                f"Unknown text-classifier partition: {normalized}"
            ) from error

    def partition_parameters(self, partition_name: str) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for parameter in self.require_partition(partition_name).parameters()
            if parameter.requires_grad
        )

    def partition_parameter_tensors(
        self,
        partition_name: str,
    ) -> dict[str, nn.Parameter]:
        partition = self.require_partition(partition_name)
        return {
            name: parameter
            for name, parameter in partition.named_parameters()
            if parameter.requires_grad
        }

    def partition_names(self) -> tuple[str, ...]:
        return tuple(self.partitions.keys())


class PartitionedTrainableAdapterClassifier(nn.Module):
    """Frozen backbone plus physical adapter/head partitions.

    이 primitive는 FedMatch의 `sigma/psi` 의미를 알지 않는다. caller가 넘긴
    partition 이름에 대해 별도 trainable adapter/head set을 보관하고, composition
    policy에 따라 evaluation용 logits를 만든다.
    """

    def __init__(
        self,
        *,
        feature_extractor: TextFeatureExtractor,
        partitions: Sequence[AdapterClassifierPartitionSpec],
        composition_policy: str = PARTITION_COMPOSITION_SUM_LOGITS,
    ) -> None:
        super().__init__()
        if not isinstance(feature_extractor, nn.Module):
            raise TypeError("feature_extractor must be a torch nn.Module.")
        self.feature_extractor = feature_extractor
        self.composition_policy = _normalize_composition_policy(composition_policy)
        self.partitions = nn.ModuleDict(
            {
                _normalize_partition_name(spec.partition_name): (
                    AdapterClassifierPartition(
                        adapter=spec.adapter,
                        classifier=spec.classifier,
                    )
                )
                for spec in partitions
            }
        )
        if len(self.partitions) != len(partitions):
            raise ValueError("partitions must not contain duplicate names.")
        if not self.partitions:
            raise ValueError("at least one trainable adapter partition is required.")
        _freeze_module_parameters(self.feature_extractor)

    def forward_partition(
        self,
        partition_name: str,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        partition = self.require_partition(partition_name)
        features = self.extract_frozen_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return partition(features)

    def forward_composed(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        partition_names: Sequence[str] | None = None,
    ) -> Tensor:
        names = (
            tuple(self.partitions.keys())
            if partition_names is None
            else tuple(_normalize_partition_name(name) for name in partition_names)
        )
        if not names:
            raise ValueError("partition_names must not be empty.")
        logits = [
            self.forward_partition(
                name,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            for name in names
        ]
        if self.composition_policy == PARTITION_COMPOSITION_SUM_LOGITS:
            result = logits[0]
            for value in logits[1:]:
                result = result + value
            return result
        raise ValueError(f"Unsupported composition policy: {self.composition_policy}")

    def extract_frozen_features(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            features = self.feature_extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return features.detach()

    def require_partition(self, partition_name: str) -> AdapterClassifierPartition:
        normalized = _normalize_partition_name(partition_name)
        try:
            return self.partitions[normalized]
        except KeyError as error:
            raise ValueError(
                f"Unknown trainable adapter partition: {normalized}"
            ) from error

    def partition_parameters(self, partition_name: str) -> tuple[nn.Parameter, ...]:
        return tuple(self.require_partition(partition_name).parameters())

    def partition_parameter_tensors(
        self,
        partition_name: str,
    ) -> dict[str, nn.Parameter]:
        """Return trainable tensors with keys local to the physical partition."""

        partition = self.require_partition(partition_name)
        return {
            name: parameter
            for name, parameter in partition.named_parameters()
            if parameter.requires_grad
        }

    def partition_named_parameters(
        self,
        partition_name: str,
    ) -> dict[str, nn.Parameter]:
        partition = self.require_partition(partition_name)
        return {
            f"{_normalize_partition_name(partition_name)}.{name}": parameter
            for name, parameter in partition.named_parameters()
        }

    def partition_names(self) -> tuple[str, ...]:
        return tuple(self.partitions.keys())


def snapshot_partition_parameters(
    model: PartitionedTrainableAdapterClassifier,
    partition_name: str,
) -> dict[str, Tensor]:
    """Detached clone snapshot for one physical partition."""

    return {
        name: parameter.detach().clone()
        for name, parameter in model.partition_named_parameters(partition_name).items()
    }


def snapshot_partition_parameter_tensors(
    model: PartitionedTrainableTextClassifier,
    partition_name: str,
) -> dict[str, Tensor]:
    """Detached clone snapshot keyed by names local to one partition."""

    return {
        name: parameter.detach().clone()
        for name, parameter in model.partition_parameter_tensors(partition_name).items()
    }


def parameters_changed(
    *,
    before: Mapping[str, Tensor],
    after: Mapping[str, Tensor],
) -> bool:
    """Return whether any parameter tensor changed."""

    if set(before) != set(after):
        raise ValueError("parameter snapshot keys must match.")
    return any(not torch.equal(after[name], before[name]) for name in before)


def _freeze_module_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _normalize_partition_name(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("partition name must not be empty.")
    return normalized


def _normalize_composition_policy(value: str) -> str:
    normalized = str(value).strip()
    if normalized != PARTITION_COMPOSITION_SUM_LOGITS:
        raise ValueError(f"Unsupported composition policy: {normalized}")
    return normalized
