"""여러 contract가 실제로 공유하는 공통 닫힌 값 타입."""

from __future__ import annotations

from enum import StrEnum


class TrainingScope(StrEnum):
    """허용된 학습 범위 식별자.

    `selected_encoder_block`는 future LoRA/PEFT family translation에서
    encoder 내부의 제한된 trainable scope를 표현할 때 사용할 수 있다.
    `full_encoder`는 현재 paper-track 기본값이 아니라 upper-bound 또는
    확장 비교축으로 해석한다.
    """

    ADAPTER_ONLY = "adapter_only"
    HEAD_ONLY = "head_only"
    SELECTED_ENCODER_BLOCK = "selected_encoder_block"
    FULL_ENCODER = "full_encoder"


class TrainingTaskType(StrEnum):
    """학습 task 유형 식별자."""

    PSEUDO_LABEL_SELF_TRAINING = "pseudo_label_self_training"
    FEEDBACK_SUPERVISED = "feedback_supervised"


__all__ = ["TrainingScope", "TrainingTaskType"]
