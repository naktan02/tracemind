"""Method-owned FL SSL local objective port."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Protocol

from torch import Tensor

from methods.federated_ssl.method_module_resolution import (
    import_method_family_module,
)


@dataclass(frozen=True, slots=True)
class FederatedSslLocalObjectiveSpec:
    """client local objective의 method-owned metadata."""

    objective_name: str
    required_batch_views: tuple[str, ...] = ()
    metric_prefix: str = "local_objective"
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective_name.strip():
            raise ValueError("objective_name must not be empty.")
        if not self.metric_prefix.strip():
            raise ValueError("metric_prefix must not be empty.")


class FederatedSslLocalObjective(Protocol):
    """runtime adapter가 method-owned local objective를 호출하는 port."""

    spec: FederatedSslLocalObjectiveSpec


@dataclass(frozen=True, slots=True)
class PartitionedObjectiveParameterTensors:
    """partitioned local objective가 비교/정규화에 쓰는 trainable tensor 묶음."""

    reference: Mapping[str, Tensor]
    trainable: Mapping[str, Tensor]


@dataclass(frozen=True, slots=True)
class TensorLocalObjectiveResult:
    """tensor-level local objective step의 loss와 진단값."""

    total_loss: Tensor
    partition_losses: Mapping[str, Tensor]
    loss_components: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]
    debug_tensors: Mapping[str, Tensor]


class PartitionedTensorLocalObjective(Protocol):
    """partitioned update-family runtime이 호출하는 method-owned objective."""

    def compute_supervised_loss(
        self,
        *,
        labeled_logits: Tensor,
        labels: Tensor,
    ) -> TensorLocalObjectiveResult:
        """labeled batch logits로 supervised partition loss를 계산한다."""

    def build_confidence_mask(
        self,
        *,
        weak_logits: Tensor,
    ) -> Tensor:
        """weak-view logits에서 unsupervised objective가 사용할 row mask를 만든다."""

    def compute_unsupervised_loss(
        self,
        *,
        weak_logits: Tensor,
        selected_strong_logits: Tensor,
        parameter_tensors: PartitionedObjectiveParameterTensors,
        selected_helper_weak_probabilities: Tensor | Sequence[Tensor] | None = None,
        enable_inter_client_consistency: bool = True,
    ) -> TensorLocalObjectiveResult:
        """unlabeled logits와 partition tensor로 unsupervised loss를 계산한다."""


def requires_method_helper_probability_provider(
    *,
    method_name: str,
    local_ssl_policy_name: str,
) -> bool:
    """method-local helper probability provider 요구 여부를 반환한다."""

    requirements_module = _import_method_runtime_requirements_module(method_name)
    if requirements_module is None:
        return False
    resolver = getattr(
        requirements_module,
        "requires_helper_probability_provider",
        None,
    )
    if resolver is None:
        return False
    return bool(resolver(local_ssl_policy_name=local_ssl_policy_name))


def _import_method_runtime_requirements_module(method_name: str) -> ModuleType | None:
    return import_method_family_module(
        method_name=method_name,
        module_leaf="runtime_requirements",
    )
