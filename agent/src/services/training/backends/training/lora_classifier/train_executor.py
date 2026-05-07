"""LoRA-classifier train executor seam."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import TrainingTask

from .config import LoraClassifierTrainingBackendConfig
from .row_extractor import LoraClassifierTrainingRow


@dataclass(frozen=True, slots=True)
class LoraClassifierTrainArtifacts:
    """LoRA train executor가 payload builder에 돌려주는 artifact snapshot."""

    lora_delta_artifact_ref: str
    classifier_head_delta_artifact_ref: str
    delta_l2_norm: float = 0.0


class LoraClassifierTrainExecutor(Protocol):
    """실제 PEFT/LoRA 학습 실행기가 만족해야 하는 boundary."""

    def train(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        rows: Sequence[LoraClassifierTrainingRow],
        label_schema: tuple[str, ...],
        config: LoraClassifierTrainingBackendConfig,
        created_at: datetime,
    ) -> LoraClassifierTrainArtifacts:
        """agent-local raw text rows로 LoRA/classifier delta artifact를 만든다."""


class NotImplementedLoraClassifierTrainExecutor:
    """실제 LoRA train step이 붙기 전 명시적으로 실패하는 executor."""

    def train(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        rows: Sequence[LoraClassifierTrainingRow],
        label_schema: tuple[str, ...],
        config: LoraClassifierTrainingBackendConfig,
        created_at: datetime,
    ) -> LoraClassifierTrainArtifacts:
        del training_task, model_manifest, rows, label_schema, config, created_at
        raise NotImplementedError(
            "LoRA-classifier train executor is not implemented yet. "
            "Use the payload-only backend path until PEFT artifact materialization "
            "is wired."
        )
