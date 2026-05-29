"""FL SSL final projection dispatcher."""

from __future__ import annotations

from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRunRequest,
)
from scripts.support.configured_callable import load_configured_callable


def build_final_projection_artifacts(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    runtime_resource_cache: Any | None = None,
) -> dict[str, Any] | None:
    """update family config가 선언한 final projection builder를 실행한다."""

    config = request.final_projection_config
    if not config.enabled:
        return {"enabled": False, "reason": "disabled_by_config"}
    builder_path = request.round_runtime_config.final_projection_builder
    if not builder_path:
        return {
            "enabled": False,
            "reason": "final_projection_builder_not_configured",
            "update_family_name": request.round_runtime_config.update_family_name,
        }
    try:
        builder = _load_final_projection_builder(builder_path)
        return builder(
            request=request,
            active=active,
            runtime_resource_cache=runtime_resource_cache,
        )
    except Exception as exc:
        if config.fail_on_error:
            raise
        return {
            "enabled": False,
            "reason": "projection_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _load_final_projection_builder(builder_path: str) -> Any:
    return load_configured_callable(
        builder_path,
        field_name="round_runtime.final_projection_builder",
    )
