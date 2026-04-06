"""여러 contract가 실제로 공유하는 공통 닫힌 값 타입."""

from __future__ import annotations

from enum import StrEnum


class TrainingScope(StrEnum):
    """허용된 학습 범위 식별자."""

    ADAPTER_ONLY = "adapter_only"
    HEAD_ONLY = "head_only"
    SELECTED_ENCODER_BLOCK = "selected_encoder_block"
    FULL_ENCODER = "full_encoder"


class TrainingTaskType(StrEnum):
    """학습 task 유형 식별자."""

    PSEUDO_LABEL_SELF_TRAINING = "pseudo_label_self_training"
    FEEDBACK_SUPERVISED = "feedback_supervised"


__all__ = ["TrainingScope", "TrainingTaskType"]
