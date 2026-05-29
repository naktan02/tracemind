"""Secure update codec seam for FL update envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shared.src.contracts.training_contracts import (
    TrainingTask,
    TrainingUpdateEnvelope,
)


class SecureUpdateCodec(Protocol):
    """Agent upload와 server acceptance 사이의 secure update codec."""

    codec_name: str

    def encode_for_submission(
        self,
        *,
        envelope: TrainingUpdateEnvelope,
        training_task: TrainingTask,
    ) -> TrainingUpdateEnvelope:
        """agent가 만든 plaintext update envelope을 제출 형태로 변환한다."""

    def decode_submission(
        self,
        *,
        envelope: TrainingUpdateEnvelope,
        training_task: TrainingTask,
    ) -> TrainingUpdateEnvelope:
        """server가 받은 제출 envelope을 aggregation 가능한 형태로 변환한다."""


@dataclass(frozen=True, slots=True)
class NoOpSecureUpdateCodec:
    """secure aggregation을 쓰지 않는 현재 baseline codec."""

    codec_name: str = "noop"

    def encode_for_submission(
        self,
        *,
        envelope: TrainingUpdateEnvelope,
        training_task: TrainingTask,
    ) -> TrainingUpdateEnvelope:
        if training_task.secure_aggregation.required:
            raise ValueError(
                "NoOpSecureUpdateCodec cannot satisfy required secure aggregation."
            )
        return envelope

    def decode_submission(
        self,
        *,
        envelope: TrainingUpdateEnvelope,
        training_task: TrainingTask,
    ) -> TrainingUpdateEnvelope:
        if envelope.secure_aggregation is not None:
            raise ValueError(
                "NoOpSecureUpdateCodec cannot decode secure aggregation submission."
            )
        if training_task.secure_aggregation.required:
            raise ValueError(
                "NoOpSecureUpdateCodec cannot satisfy required secure aggregation."
            )
        return envelope


SecureUpdateCodecFactory = type[NoOpSecureUpdateCodec]

_SECURE_UPDATE_CODEC_REGISTRY: dict[str, SecureUpdateCodecFactory] = {
    "noop": NoOpSecureUpdateCodec,
}


def build_secure_update_codec(codec_name: str) -> SecureUpdateCodec:
    """codec 이름으로 secure update codec을 생성한다."""

    normalized_name = codec_name.strip().lower()
    factory = _SECURE_UPDATE_CODEC_REGISTRY.get(normalized_name)
    if factory is None:
        raise ValueError(f"Unsupported secure update codec: {codec_name}.")
    return factory()
