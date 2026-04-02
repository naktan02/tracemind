"""Federated simulation용 task/config 변환 helper."""

from __future__ import annotations

from main_server.src.services.rounds.models import RoundOpenRequest
from shared.src.domain.entities.artifacts.model_manifest import ModelManifest

from .models import FederatedTrainingTaskConfig


def build_round_open_request(
    *,
    active_manifest: ModelManifest,
    round_id: str,
    training_task_config: FederatedTrainingTaskConfig,
) -> RoundOpenRequest:
    """simulation training task 설정을 round open request로 변환한다."""
    return RoundOpenRequest(
        active_manifest=active_manifest,
        round_id=round_id,
        local_epochs=int(training_task_config.local_epochs),
        batch_size=int(training_task_config.batch_size),
        learning_rate=float(training_task_config.learning_rate),
        max_steps=int(training_task_config.max_steps),
        objective_config=dict(training_task_config.objective_config),
        selection_policy=dict(training_task_config.selection_policy),
        min_required_examples=int(training_task_config.min_required_examples),
        gradient_clip_norm=(
            None
            if training_task_config.gradient_clip_norm is None
            else float(training_task_config.gradient_clip_norm)
        ),
    )


def build_legacy_task_config(
    *,
    confidence_threshold: float | None,
    margin_threshold: float | None,
    max_examples: int | None,
    min_required_examples: int | None,
    gradient_clip_norm: float | None,
) -> FederatedTrainingTaskConfig:
    """legacy leaf 인자들을 canonical training task config로 모은다."""
    return FederatedTrainingTaskConfig(
        min_required_examples=min_required_examples or 1,
        gradient_clip_norm=gradient_clip_norm,
        objective_config={
            "loss": "diagonal_scale_heuristic",
            "confidence_threshold": resolve_threshold(
                confidence_threshold,
                fallback=None,
                default=0.6,
            ),
            "margin_threshold": resolve_threshold(
                margin_threshold,
                fallback=None,
                default=0.02,
            ),
            "score_policy_name": "max_cosine",
            "acceptance_policy_name": "top1_margin_threshold",
            "privacy_guard_name": "diagonal_scale_clip_only",
        },
        selection_policy={"max_examples": max_examples or 128},
    )


def resolve_threshold(
    value: float | None,
    *,
    fallback: object,
    default: float,
) -> float:
    """override/fallback/default 우선순위로 threshold를 결정한다."""
    if value is not None:
        return float(value)
    if fallback is None:
        return default
    if isinstance(fallback, bool):
        raise ValueError("Threshold config must not be bool.")
    return float(fallback)


def resolve_optional_positive_int(value: object) -> int | None:
    """optional positive integer config를 안전하게 해석한다."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("score_top_k must not be bool.")
    parsed = int(value)
    if parsed < 1:
        raise ValueError("score_top_k must be positive.")
    return parsed
