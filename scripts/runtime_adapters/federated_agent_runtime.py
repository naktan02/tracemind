"""federated simulation이 쓰는 agent runtime bridge."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def build_federated_scoring_service(
    *,
    objective_config: Any,
    similarity_name: str,
    shared_state: Any | None = None,
) -> Any:
    """agent ScoringService를 simulation 경계 뒤에서 조립한다."""

    from agent.src.services.inference.scoring_service import ScoringService

    return ScoringService.from_objective_config(
        objective_config,
        similarity_name=similarity_name,
        shared_state=shared_state,
    )


def build_federated_training_examples(
    *,
    rows: list[Mapping[str, Any]],
    adapter: Any,
    adapter_state: Any,
    prototype_pack: Any,
    model_id: str,
    scoring_service: Any,
    objective_config: Any | None,
    parse_created_at: Callable[[str], datetime],
) -> tuple[Any, ...]:
    """simulation row를 agent runtime training example로 변환한다."""

    from agent.src.services.training.backends.inputs.base import (
        WEAK_STRONG_PAIR_BACKEND_NAME,
    )
    from agent.src.services.training.backends.inputs.models import (
        TrainingExampleBuildRequest,
        TrainingExampleSource,
    )
    from agent.src.services.training.examples.service import TrainingExampleService

    backend_name = _resolve_example_generation_backend_name(
        objective_config=objective_config,
    )
    if backend_name == WEAK_STRONG_PAIR_BACKEND_NAME:
        _validate_weak_strong_rows(rows)

    service = (
        TrainingExampleService()
        if objective_config is None
        else TrainingExampleService.from_objective_config(objective_config)
    )
    source_rows = tuple(
        TrainingExampleSource(
            query_id=str(row["query_id"]),
            text=str(row["text"]),
            occurred_at=parse_created_at(str(row["created_at"])),
            weak_text=_optional_row_value(row, "weak_text"),
            strong_text=_optional_row_value(row, "strong_text"),
            weak_translated_text=_optional_row_value(row, "weak_translated_text"),
            strong_translated_text=_optional_row_value(row, "strong_translated_text"),
        )
        for row in rows
    )
    return service.build_examples(
        TrainingExampleBuildRequest(
            source_rows=source_rows,
            adapter=adapter,
            adapter_state=adapter_state,
            prototype_pack=prototype_pack,
            model_id=model_id,
            scoring_service=scoring_service,
        )
    )


def select_federated_pseudo_labels(
    *,
    scored_events: tuple[Any, ...],
    training_task: Any,
) -> Any:
    """validation pseudo-label selection을 agent policy로 실행한다."""

    from agent.src.services.training.selection.pseudo_label_service import (
        PseudoLabelSelectionService,
    )

    return PseudoLabelSelectionService().select(
        scored_events=scored_events,
        training_task=training_task,
    )


def build_federated_local_training_service(
    *,
    client_state_root: Path,
) -> Any:
    """simulation client state root 기준 local training service를 만든다."""

    from agent.src.infrastructure.repositories.training_artifact_repository import (
        TrainingArtifactRepository,
    )
    from agent.src.services.training.execution.local_training_service import (
        LocalTrainingService,
    )

    return LocalTrainingService(
        repository=TrainingArtifactRepository(state_root=client_state_root)
    )


def run_federated_local_training(
    *,
    local_training_service: Any,
    training_examples: tuple[Any, ...],
    training_task: Any,
    model_manifest: Any,
) -> Any:
    """LocalTrainingRequest 생성 세부사항을 scripts runtime bridge에 숨긴다."""

    from agent.src.services.training.execution.local_training_service import (
        LocalTrainingRequest,
    )

    return local_training_service.run(
        LocalTrainingRequest(
            training_examples=training_examples,
            training_task=training_task,
            model_manifest=model_manifest,
        )
    )


def resolve_federated_training_backend_adapter_kind(
    *,
    objective_config: Any,
) -> str:
    """simulation config 검증용으로 local training backend의 adapter kind를 읽는다."""

    from agent.src.services.training.backends.training.registry import (
        build_shared_adapter_training_backend,
    )

    backend = build_shared_adapter_training_backend(
        objective_config.training_backend_name,
        objective_config=objective_config,
    )
    return str(backend.adapter_kind)


def _resolve_example_generation_backend_name(
    *,
    objective_config: Any | None,
) -> str:
    from methods.federated_ssl.training_defaults import DEFAULT_TRAINING_PROFILE

    if objective_config is None:
        return DEFAULT_TRAINING_PROFILE.example_generation_backend_name
    return (
        objective_config.example_generation_backend_name
        or DEFAULT_TRAINING_PROFILE.example_generation_backend_name
    )


def _validate_weak_strong_rows(rows: list[Mapping[str, Any]]) -> None:
    for row in rows:
        if row.get("weak_text") and row.get("strong_text"):
            continue
        raise ValueError(
            "weak_strong_pair simulation requires each row to include both "
            "weak_text and strong_text."
        )


def _optional_row_value(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)
