"""Query SSL objective algorithm 공통 인터페이스."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor

QuerySslAlgorithmFactory = Callable[[Mapping[str, Any]], "QuerySslAlgorithm"]


@dataclass(frozen=True, slots=True)
class QuerySslStepResult:
    """Query SSL algorithm 한 step의 loss와 metric 결과."""

    total_loss: Tensor
    loss_components: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]
    debug_tensors: Mapping[str, Tensor] = field(default_factory=dict)


class TextBatchClassifier(Protocol):
    """tokenized text batch를 label logits로 바꾸는 classifier protocol."""

    def __call__(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """input_ids와 attention_mask로 label logits를 계산한다."""


class QuerySslAlgorithm(Protocol):
    """중앙 Query SSL trainer가 호출하는 algorithm seam."""

    algorithm_name: str

    @property
    def uses_labeled_batches(self) -> bool:
        """labeled train loader를 step에 함께 공급해야 하는지 여부."""

    def validate_loaders(
        self,
        *,
        train_loader_length: int,
        unlabeled_loader_length: int,
    ) -> None:
        """algorithm별 loader 전제조건을 검증한다."""

    def compute_step(
        self,
        *,
        model: TextBatchClassifier,
        labeled_batch: dict[str, Tensor] | None,
        unlabeled_batch: dict[str, Any],
    ) -> QuerySslStepResult:
        """model과 batch로 algorithm-specific optimization step을 계산한다."""


class StatefulQuerySslAlgorithm(Protocol):
    """resume checkpoint가 소비하는 선택적 Query SSL algorithm state 계약."""

    def export_state(self) -> Mapping[str, Any]:
        """학습 재개에 필요한 algorithm-local state를 내보낸다."""

    def load_state(self, state: Mapping[str, Any]) -> None:
        """저장된 algorithm-local state를 복원한다."""


def configure_query_ssl_algorithm_training(
    algorithm: QuerySslAlgorithm,
    *,
    num_train_iter: int,
) -> None:
    """algorithm이 필요로 할 때만 전체 train iteration 수를 전달한다."""

    if num_train_iter <= 0:
        raise ValueError("num_train_iter must be positive.")
    configure_training = getattr(algorithm, "configure_training", None)
    if callable(configure_training):
        configure_training(num_train_iter=num_train_iter)


def configure_query_ssl_algorithm_dataset(
    algorithm: QuerySslAlgorithm,
    *,
    num_classes: int,
    unlabeled_row_count: int,
) -> None:
    """algorithm이 필요로 할 때만 dataset-level state 크기를 전달한다."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    if unlabeled_row_count <= 0:
        raise ValueError("unlabeled_row_count must be positive.")
    configure_dataset = getattr(algorithm, "configure_dataset", None)
    if callable(configure_dataset):
        configure_dataset(
            num_classes=num_classes,
            unlabeled_row_count=unlabeled_row_count,
        )


def _require_non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _normalize_view_names(view_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized_names = tuple(
        _require_non_empty(str(view_name), field_name="view_names item")
        for view_name in view_names
    )
    if not normalized_names:
        raise ValueError("required view_names must not be empty.")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("required view_names must be unique.")
    return normalized_names


@dataclass(frozen=True, slots=True)
class QuerySslRequiredViews:
    """Query SSL algorithm이 unlabeled row에 요구하는 view surface."""

    view_names: tuple[str, ...]
    view_builder_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "view_names",
            _normalize_view_names(self.view_names),
        )
        object.__setattr__(
            self,
            "view_builder_name",
            _require_non_empty(
                self.view_builder_name,
                field_name="view_builder_name",
            ),
        )


@dataclass(frozen=True, slots=True)
class QuerySslAlgorithmDescriptor:
    """Query SSL algorithm metadata와 factory를 함께 둔 method-local spec."""

    algorithm_name: str
    display_name: str
    required_views: QuerySslRequiredViews
    algorithm_factory: QuerySslAlgorithmFactory
    default_uses_labeled_batches: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "algorithm_name",
            _require_non_empty(self.algorithm_name, field_name="algorithm_name"),
        )
        object.__setattr__(
            self,
            "display_name",
            _require_non_empty(self.display_name, field_name="display_name"),
        )

    def build_algorithm(self, parameters: Mapping[str, Any]) -> QuerySslAlgorithm:
        """Hydra parameter mapping으로 algorithm instance를 만든다."""

        return self.algorithm_factory(parameters)
