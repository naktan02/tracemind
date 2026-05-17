"""FL simulation rows를 agent training example request로 mapping한다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from scripts.runtime_adapters.federated_agent.backend_resolver import (
    resolve_example_generation_backend_name,
)
from scripts.runtime_adapters.federated_agent.row_validator import (
    require_rows_supported_by_example_backend,
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

    from agent.src.services.training.backends.inputs.models import (
        TrainingExampleBuildRequest,
        TrainingExampleSource,
    )
    from agent.src.services.training.examples.service import TrainingExampleService

    backend_name = resolve_example_generation_backend_name(
        objective_config=objective_config,
    )
    require_rows_supported_by_example_backend(
        rows=rows,
        backend_name=backend_name,
    )

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
            weak_text=_resolve_weak_text(row),
            strong_text=_resolve_strong_text(row),
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


def _optional_row_value(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)


def _resolve_weak_text(row: Mapping[str, Any]) -> str | None:
    legacy_value = _optional_row_value(row, "weak_text")
    if legacy_value is not None:
        return legacy_value
    if row.get("aug_0") or row.get("aug_1"):
        return str(row["text"])
    return None


def _resolve_strong_text(row: Mapping[str, Any]) -> str | None:
    legacy_value = _optional_row_value(row, "strong_text")
    if legacy_value is not None:
        return legacy_value
    return _optional_row_value(row, "aug_0") or _optional_row_value(row, "aug_1")
