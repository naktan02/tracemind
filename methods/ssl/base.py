"""Query SSL objective algorithm 공통 인터페이스."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor
from torch import device as TorchDevice

QuerySslAlgorithmFactory = Callable[[Mapping[str, Any]], "QuerySslAlgorithm"]

QUERY_SSL_BATCH_SURFACE_WEAK_ONLY = "weak_only"
QUERY_SSL_BATCH_SURFACE_WEAK_STRONG = "weak_strong"

QUERY_SSL_MODEL_OUTPUT_LOGITS = "logits"
QUERY_SSL_MODEL_OUTPUT_POOLED_FEATURES = "pooled_features"

QUERY_SSL_ALGORITHM_STATE_STATELESS = "stateless"
QUERY_SSL_ALGORITHM_STATE_DATASET_STATE = "dataset_state"
QUERY_SSL_ALGORITHM_STATE_STEP_COUNTER = "step_counter"
QUERY_SSL_ALGORITHM_STATE_ADAPTIVE_THRESHOLD = "adaptive_threshold"
QUERY_SSL_ALGORITHM_STATE_DISTRIBUTION_EMA = "distribution_ema"
QUERY_SSL_ALGORITHM_STATE_FEATURE_QUEUE = "feature_queue"
QUERY_SSL_ALGORITHM_STATE_PROBABILITY_QUEUE = "probability_queue"

QUERY_SSL_INPUT_TRANSFORM_NONE = "none"

QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP = "single_loss_step"
QUERY_SSL_OPTIMIZER_LIFECYCLE_AUXILIARY_TRAINABLE_MODULE = "auxiliary_trainable_module"

QUERY_SSL_TEACHER_STATE_NONE = "none"


@dataclass(frozen=True, slots=True)
class QuerySslStepResult:
    """Query SSL algorithm 한 step의 loss와 metric 결과."""

    total_loss: Tensor
    loss_components: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]
    debug_tensors: Mapping[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QuerySslStepContext:
    """학습 루프가 algorithm step에 제공하는 실행 위치/환경 context."""

    epoch_index: int
    step_index: int
    global_step: int
    total_train_steps: int
    num_classes: int
    device: TorchDevice

    def __post_init__(self) -> None:
        if self.epoch_index <= 0:
            raise ValueError("epoch_index must be positive.")
        if self.step_index <= 0:
            raise ValueError("step_index must be positive.")
        if self.global_step <= 0:
            raise ValueError("global_step must be positive.")
        if self.total_train_steps <= 0:
            raise ValueError("total_train_steps must be positive.")
        if self.global_step > self.total_train_steps:
            raise ValueError("global_step must not exceed total_train_steps.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")


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


def compute_query_ssl_algorithm_step(
    algorithm: QuerySslAlgorithm,
    *,
    model: TextBatchClassifier,
    labeled_batch: dict[str, Tensor] | None,
    unlabeled_batch: dict[str, Any],
    step_context: QuerySslStepContext | None = None,
) -> QuerySslStepResult:
    """context-aware algorithm은 context seam으로, 기존 algorithm은 그대로 호출한다."""

    compute_step_with_context = getattr(algorithm, "compute_step_with_context", None)
    if step_context is not None and callable(compute_step_with_context):
        return compute_step_with_context(
            model=model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
            step_context=step_context,
        )
    return algorithm.compute_step(
        model=model,
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
    )


def _require_non_empty(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _normalize_requirement_set(
    values: Iterable[str],
    *,
    field_name: str,
) -> frozenset[str]:
    normalized_values = frozenset(
        _require_non_empty(str(value), field_name=f"{field_name} item")
        for value in values
    )
    if not normalized_values:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized_values


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
class QuerySslRuntimeRequirements:
    """Query SSL algorithm이 runtime adapter에 요구하는 capability 선언.

    각 field 값은 닫힌 enum이 아니다. 공통 built-in 값은 이 파일의 상수를
    재사용하되, 새 capability는 실제 producer/consumer와 test가 생길 때 그
    owner 경계에서 추가한다.
    """

    batch_surface: str = QUERY_SSL_BATCH_SURFACE_WEAK_STRONG
    model_outputs: frozenset[str] = field(
        default_factory=lambda: frozenset({QUERY_SSL_MODEL_OUTPUT_LOGITS})
    )
    algorithm_state_surface: frozenset[str] = field(
        default_factory=lambda: frozenset({QUERY_SSL_ALGORITHM_STATE_STATELESS})
    )
    input_transform_surface: str = QUERY_SSL_INPUT_TRANSFORM_NONE
    optimizer_lifecycle: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {QUERY_SSL_OPTIMIZER_LIFECYCLE_SINGLE_LOSS_STEP}
        )
    )
    teacher_state: str = QUERY_SSL_TEACHER_STATE_NONE
    step_context_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "batch_surface",
            _require_non_empty(self.batch_surface, field_name="batch_surface"),
        )
        object.__setattr__(
            self,
            "model_outputs",
            _normalize_requirement_set(
                self.model_outputs,
                field_name="model_outputs",
            ),
        )
        object.__setattr__(
            self,
            "algorithm_state_surface",
            _normalize_requirement_set(
                self.algorithm_state_surface,
                field_name="algorithm_state_surface",
            ),
        )
        object.__setattr__(
            self,
            "input_transform_surface",
            _require_non_empty(
                self.input_transform_surface,
                field_name="input_transform_surface",
            ),
        )
        object.__setattr__(
            self,
            "optimizer_lifecycle",
            _normalize_requirement_set(
                self.optimizer_lifecycle,
                field_name="optimizer_lifecycle",
            ),
        )
        object.__setattr__(
            self,
            "teacher_state",
            _require_non_empty(self.teacher_state, field_name="teacher_state"),
        )
        object.__setattr__(
            self,
            "step_context_required",
            bool(self.step_context_required),
        )


@dataclass(frozen=True, slots=True)
class QuerySslAlgorithmDescriptor:
    """Query SSL algorithm metadata와 factory를 함께 둔 method-local spec."""

    algorithm_name: str
    display_name: str
    required_views: QuerySslRequiredViews
    algorithm_factory: QuerySslAlgorithmFactory
    default_uses_labeled_batches: bool = True
    runtime_requirements: QuerySslRuntimeRequirements = field(
        default_factory=QuerySslRuntimeRequirements
    )

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
