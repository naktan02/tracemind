"""Query SSL adaptation objective 공통 인터페이스."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from torch import Tensor

from agent.src.services.training.query_adaptation.modeling import LoraTextClassifier


class QuerySslStepOutput(Protocol):
    """Query SSL objective 한 step의 loss와 metric 결과."""

    total_loss: Tensor

    @property
    def loss_components(self) -> Mapping[str, Tensor]:
        """epoch history에 `train_{name}`으로 기록할 loss component."""

    @property
    def metrics(self) -> Mapping[str, Tensor]:
        """epoch history에 `train_{name}`으로 기록할 scalar metric."""


class QuerySslObjective(Protocol):
    """중앙 Query SSL trainer가 호출하는 objective seam."""

    objective_name: str

    @property
    def uses_labeled_batches(self) -> bool:
        """labeled train loader를 step에 함께 공급해야 하는지 여부."""

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        """objective별 loader 전제조건을 검증한다."""

    def compute_step(
        self,
        *,
        model: LoraTextClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepOutput:
        """model과 batch로 한 optimization step의 objective를 계산한다."""


__all__ = [
    "QuerySslObjective",
    "QuerySslStepOutput",
]
