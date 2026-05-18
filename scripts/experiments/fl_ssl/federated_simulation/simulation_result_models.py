"""FL SSL simulation 결과/평가 모델."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ClientRoundSummary:
    """client 하나의 라운드 참여 요약."""

    client_id: str
    candidate_count: int
    accepted_count: int
    update_generated: bool
    delta_l2_norm: float | None = None
    aggregation_example_count: int | None = None
    client_train_time_seconds: float | None = None
    client_payload_bytes: int | None = None
    pseudo_label_confidence_mean: float | None = None
    pseudo_label_margin_mean: float | None = None
    pseudo_label_correct_count: int = 0
    pseudo_label_evaluated_count: int = 0
    accepted_label_distribution: dict[str, int] = field(default_factory=dict)
    rejected_label_distribution: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationEvaluation:
    """validation 평가 결과."""

    row_count: int
    top1_accuracy: float
    accepted_ratio: float
    loss: float = 0.0
    loss_kind: str = "not_computed"
    accuracy_top_1: float = 0.0
    correct_top_1: int = 0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    weighted_f1: float = 0.0
    balanced_accuracy: float = 0.0
    worst_category_f1: str | None = None
    worst_category_f1_value: float | None = None
    worst_category_recall: float | None = None
    worst_category_precision: float | None = None
    expected_calibration_error: float = 0.0
    max_calibration_error: float = 0.0
    overconfidence_gap: float = 0.0
    mean_true_label_probability: float = 0.0
    mean_top_1_probability: float = 0.0
    mean_margin_top1_top2: float = 0.0
    mean_correct_top_1_probability: float = 0.0
    mean_incorrect_top_1_probability: float = 0.0
    score_distribution_kind: str = "not_computed"
    selection_confidence_kind: str | None = None
    mean_selection_confidence: float = 0.0
    mean_selection_margin: float = 0.0
    per_label: dict[str, dict[str, int | float]] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    classification_report: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.accuracy_top_1 == 0.0 and self.top1_accuracy != 0.0:
            self.accuracy_top_1 = self.top1_accuracy


@dataclass(slots=True)
class ClientEvaluationSummary:
    """client별 heldout validation shard 평가 요약."""

    client_id: str
    validation: SimulationEvaluation


@dataclass(slots=True)
class SimulationRoundSummary:
    """한 라운드 종료 후 요약."""

    round_id: str
    model_revision: str
    prototype_version: str
    update_count: int
    validation: SimulationEvaluation
    clients: tuple[ClientRoundSummary, ...]
    round_time_seconds: float | None = None
    total_payload_bytes: int | None = None
    gpu_memory_peak_mb: float | None = None


@dataclass(slots=True)
class SimulationResult:
    """전체 simulation 요약."""

    initial_model_revision: str
    initial_prototype_version: str
    initial_validation: SimulationEvaluation
    final_validation: SimulationEvaluation
    rounds: tuple[SimulationRoundSummary, ...]
    client_evaluations: tuple[ClientEvaluationSummary, ...] = ()
    report_path: str | None = None
