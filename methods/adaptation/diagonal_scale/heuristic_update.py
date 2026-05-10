"""Diagonal-scale heuristic local update 계산 core."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from methods.adaptation.diagonal_scale.config import (
    DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS,
    TRAINING_BACKEND_EXTRA_SCOPE,
    DiagonalScaleHeuristicTrainingBackendConfig,
)
from shared.src.contracts.adapter_contract_families.base import VECTOR_ADAPTER_DELTA_V1
from shared.src.contracts.adapter_contract_families.diagonal_scale import (
    DIAGONAL_SCALE_ADAPTER_KIND,
    VectorAdapterDelta,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    ClientMetricKeys,
    TrainingObjectiveConfig,
    TrainingTask,
)
from shared.src.domain.entities.training.pseudo_label_candidate import (
    PseudoLabelCandidate,
)


class DiagonalScaleAcceptedExample(Protocol):
    """Diagonal-scale heuristic update core가 읽는 accepted example 최소 shape."""

    update_embedding: Sequence[float]
    candidate: PseudoLabelCandidate | None


def build_diagonal_scale_heuristic_config(
    objective_config: TrainingObjectiveConfig | None,
) -> DiagonalScaleHeuristicTrainingBackendConfig:
    """objective config에서 diagonal-scale heuristic 설정을 읽는다."""

    extras = (
        {}
        if objective_config is None
        else objective_config.get_component_extras(
            TRAINING_BACKEND_EXTRA_SCOPE,
            legacy_keys=DIAGONAL_SCALE_HEURISTIC_TRAINING_BACKEND_EXTRA_KEYS,
        )
    )
    return DiagonalScaleHeuristicTrainingBackendConfig.from_mapping(extras)


def build_diagonal_scale_heuristic_update(
    *,
    training_task: TrainingTask,
    model_manifest: ModelManifest,
    accepted_examples: Sequence[DiagonalScaleAcceptedExample],
    created_at: datetime,
    config: DiagonalScaleHeuristicTrainingBackendConfig,
    adapter_kind: str = DIAGONAL_SCALE_ADAPTER_KIND,
) -> VectorAdapterDelta:
    """accepted examples에서 deterministic diagonal-scale delta를 계산한다."""

    examples = tuple(accepted_examples)
    if not examples:
        raise ValueError("accepted_examples must not be empty.")

    embedding_dim = len(examples[0].update_embedding)
    weighted_sums = [0.0] * embedding_dim
    label_counts: dict[str, int] = defaultdict(int)
    total_weight = 0.0
    total_confidence = 0.0
    total_margin = 0.0

    for example in examples:
        if len(example.update_embedding) != embedding_dim:
            raise ValueError("All accepted embeddings must share the same dimension.")

        if example.candidate is None:
            raise ValueError("Accepted example must carry a pseudo-label candidate.")

        weight = max(example.candidate.sample_weight, 1e-6)
        total_weight += weight
        total_confidence += example.candidate.confidence
        total_margin += example.candidate.margin
        label_counts[example.candidate.label] += 1
        for index, value in enumerate(example.update_embedding):
            weighted_sums[index] += weight * float(value)

    effective_scale = max(
        config.minimum_effective_scale,
        training_task.learning_rate
        * max(training_task.local_epochs, 1)
        * max(training_task.max_steps, 1)
        * config.delta_scale_multiplier,
    )
    dimension_deltas = [
        max(
            -config.max_abs_delta,
            min(
                config.max_abs_delta,
                (value / total_weight) * effective_scale,
            ),
        )
        for value in weighted_sums
    ]

    return VectorAdapterDelta(
        schema_version=VECTOR_ADAPTER_DELTA_V1,
        model_id=model_manifest.model_id,
        base_model_revision=model_manifest.model_revision,
        training_scope=training_task.training_scope,
        dimension_deltas=dimension_deltas,
        example_count=len(examples),
        mean_confidence=total_confidence / len(examples),
        mean_margin=total_margin / len(examples),
        label_counts=dict(sorted(label_counts.items())),
        created_at=created_at,
        adapter_kind=adapter_kind,
    )


def build_diagonal_scale_client_metrics(
    update: VectorAdapterDelta,
) -> dict[str, float]:
    """diagonal-scale update에서 envelope용 client metric을 추출한다."""

    return {
        ClientMetricKeys.MEAN_CONFIDENCE: update.mean_confidence,
        ClientMetricKeys.MEAN_MARGIN: update.mean_margin or 0.0,
        ClientMetricKeys.DELTA_L2_NORM: update.l2_norm(),
    }
