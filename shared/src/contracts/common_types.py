"""여러 contract가 실제로 공유하는 공통 닫힌 값 타입."""

from __future__ import annotations

from enum import StrEnum


class TrainingScope(StrEnum):
    """허용된 학습 범위 식별자.

    `selected_encoder_block`는 future LoRA/PEFT family translation에서
    encoder 내부의 제한된 trainable scope를 표현할 때 사용할 수 있다.
    `adapter_and_head`는 PEFT adapter와 classifier/decision head가 함께
    학습·동기화되는 shared update surface를 표현한다.
    `full_encoder`는 현재 paper-track 기본값이 아니라 upper-bound 또는
    확장 비교축으로 해석한다.
    """

    ADAPTER_ONLY = "adapter_only"
    ADAPTER_AND_HEAD = "adapter_and_head"
    HEAD_ONLY = "head_only"
    SELECTED_ENCODER_BLOCK = "selected_encoder_block"
    FULL_ENCODER = "full_encoder"


class TrainingTaskType(StrEnum):
    """학습 task 유형 식별자.

    이 값은 서버 round task와 agent/simulation local trainer 사이를 지나는
    canonical discriminator다. method 구현과 method 이름은 `methods/`가
    소유하며, 여기에는 boundary가 이해해야 하는 capability 수준 task kind만 둔다.
    """

    PSEUDO_LABEL_SELF_TRAINING = "pseudo_label_self_training"
    FEDERATED_SSL_METHOD_LOCAL_STEP = "federated_ssl_method_local_step"
    FEEDBACK_SUPERVISED = "feedback_supervised"


LEGACY_TRAINING_TASK_TYPE_ALIASES = {
    "fedmatch_local_step": TrainingTaskType.FEDERATED_SSL_METHOD_LOCAL_STEP,
}


def normalize_training_task_type(value: TrainingTaskType | str) -> TrainingTaskType:
    """구형 method-specific task_type을 canonical capability 값으로 정규화한다."""

    if isinstance(value, TrainingTaskType):
        return value
    text = str(value).strip()
    if text in LEGACY_TRAINING_TASK_TYPE_ALIASES:
        return LEGACY_TRAINING_TASK_TYPE_ALIASES[text]
    return TrainingTaskType(text)
