"""FL generic server runtime bridge for simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    server_step_execution,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRunRequest,
)
from scripts.support.configured_callable import load_configured_callable


def run_supervised_seed_step(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_index: int,
) -> server_step_execution.ServerStepExecution:
    """Supervised seed step이 가능한 update_family면 실행한다."""

    runtime_payload = request.round_runtime_config.runtime_payload_for_update_family()
    if runtime_payload is None:
        raise ValueError(
            f"{request.round_runtime_config.update_family_name} runtime config "
            "is required."
        )

    if not hasattr(active.adapter_state, "label_schema"):
        raise ValueError(
            "supervised_seed_step requires active state with label_schema."
        )
    if not bootstrapped.dataset_split.bootstrap_rows:
        raise ValueError("supervised_seed_step requires server bootstrap_rows.")
    if request.ssl_method_config is None:
        raise ValueError("supervised_seed_step requires ssl_method_config.")

    from methods.federated_ssl.hooks.server_step import (
        resolve_method_supervised_seed_step_parameters,
    )

    seed_parameters = resolve_method_supervised_seed_step_parameters(
        method_name=request.ssl_method_config.name,
        effective_parameters=request.ssl_method_config.effective_parameters,
        default_epochs=request.training_task_config.local_epochs,
        default_batch_size=request.training_task_config.batch_size,
        round_index=round_index,
    )

    seed_artifact_names = _seed_artifact_names(
        _load_server_round_callable(
            request=request,
            callable_name="supervised_seed_artifact_names",
        )()
    )
    build_supervised_seed_revision = _load_server_round_callable(
        request=request,
        callable_name="supervised_seed_revision_builder",
    )
    build_supervised_seed_projection = _load_server_round_callable(
        request=request,
        callable_name="supervised_seed_projection_builder",
    )
    supervised_seed_step_seed_func = _load_server_round_callable(
        request=request,
        callable_name="supervised_seed_seed",
    )

    from scripts.experiments.fl_ssl.federated_simulation.io.run_artifact_writer import (
        RunArtifactWriter,
    )
    from scripts.experiments.fl_ssl.federated_simulation.model_revisions import (
        build_simulation_model_revision,
    )
    from scripts.runtime_adapters.federated_server.aggregation_artifacts import (
        build_server_aggregate_artifact_refs,
        build_simulation_aggregation_context,
    )

    now = datetime.now(timezone.utc)
    next_model_revision = build_supervised_seed_revision(
        base_model_revision=build_simulation_model_revision(round_index)
    )
    artifact_refs = build_server_aggregate_artifact_refs(
        artifact_namespace=request.round_runtime_config.update_family_name,
        next_model_revision=next_model_revision,
        artifact_names=seed_artifact_names,
    )
    projection = build_supervised_seed_projection(
        adapter_state=active.adapter_state,
        bootstrap_rows=bootstrapped.dataset_split.bootstrap_rows,
        aggregation_context=build_simulation_aggregation_context(
            output_dir=request.output_dir,
            next_model_revision=active.adapter_state.model_revision,
            aggregated_at=now,
        ),
        runtime_payload=runtime_payload,
        trainer_runtime_config=request.local_trainer_runtime_config,
        runtime_resource_cache=bootstrapped.runtime_resource_cache,
        seed=supervised_seed_step_seed_func(
            base_seed=request.seed,
            round_index=round_index,
        ),
        epochs=seed_parameters.epochs,
        batch_size=seed_parameters.batch_size,
        learning_rate=float(request.training_task_config.learning_rate),
        gradient_clip_norm=request.training_task_config.gradient_clip_norm,
        next_model_revision=next_model_revision,
        updated_at=now,
        artifact_refs_by_name=artifact_refs.refs_by_name,
        artifact_format=artifact_refs.artifact_format,
    )
    publication = bootstrapped.server_runtime.publish_shared_adapter_projection(
        base_manifest=active.manifest,
        next_state=projection.next_state,
        artifacts=projection.artifacts,
        published_at=now,
        notes=(
            "supervised_seed_step published from server bootstrap_rows before "
            f"round_{round_index:04d}"
        ),
    )
    RunArtifactWriter().save_model_manifest(
        output_dir=request.output_dir,
        manifest=publication.next_manifest,
    )
    return server_step_execution.ServerStepExecution(
        active=ActiveSimulationState(
            manifest=publication.next_manifest,
            adapter_state=publication.next_state,
        ),
        model_revision=next_model_revision,
        metrics=projection.metrics,
    )


def build_final_projection_artifacts(
    *,
    request: SimulationRunRequest,
    active: ActiveSimulationState,
    runtime_resource_cache: Any | None = None,
) -> dict[str, Any] | None:
    """최종 global update-family projection artifact를 만든다."""

    require_state = _load_server_round_callable(
        request=request,
        callable_name="final_projection_state_resolver",
    )
    build_final_projection_artifacts_from_state = _load_server_round_callable(
        request=request,
        callable_name="final_projection_artifacts_builder",
    )

    from scripts.runtime_adapters.federated_server.aggregation_artifacts import (
        build_simulation_aggregation_context,
    )

    adapter_state = require_state(active.adapter_state)
    return build_final_projection_artifacts_from_state(
        rows_by_dataset_name=_projection_rows_by_dataset_name(request),
        adapter_state=adapter_state,
        aggregation_context=build_simulation_aggregation_context(
            output_dir=request.output_dir,
            next_model_revision=adapter_state.model_revision,
            aggregated_at=adapter_state.updated_at,
        ),
        objective_config=request.training_task_config.objective_config,
        runtime_config=request.local_trainer_runtime_config,
        batch_size=int(request.training_task_config.batch_size),
        projection_dir=request.output_dir / "projections",
        seed=request.seed,
        runtime_resource_cache=runtime_resource_cache,
    )


def _projection_rows_by_dataset_name(
    request: SimulationRunRequest,
) -> dict[str, Any]:
    from scripts.experiments.fl_ssl.federated_simulation.models import (
        PROJECTION_DATASET_TEST,
        PROJECTION_DATASET_VALIDATION,
    )

    rows_by_name = {
        PROJECTION_DATASET_VALIDATION: request.validation_rows,
        PROJECTION_DATASET_TEST: request.test_rows,
    }
    return {
        dataset_name: rows
        for dataset_name, rows in rows_by_name.items()
        if dataset_name in request.final_projection_config.dataset_names and rows
    }


def _load_server_round_callable(
    *,
    request: SimulationRunRequest,
    callable_name: str,
) -> Any:
    path = _server_round_callable_path(
        request.round_runtime_config,
        callable_name,
    )
    if path is None:
        raise NotImplementedError(
            "round_runtime.server_round_runtime is missing required callable: "
            f"{callable_name}"
        )
    return load_configured_callable(
        path,
        field_name=f"round_runtime.server_round_runtime.{callable_name}",
    )


def _server_round_callable_path(
    round_runtime_config: object,
    callable_name: str,
) -> str | None:
    resolver = getattr(round_runtime_config, "server_round_callable_path", None)
    if callable(resolver):
        return resolver(callable_name)
    mapping = getattr(round_runtime_config, "server_round_runtime", {})
    if not isinstance(mapping, Mapping):
        return None
    normalized_name = callable_name.strip().lower().replace("-", "_")
    raw_value = mapping.get(normalized_name)
    if raw_value is None:
        return None
    normalized_value = str(raw_value).strip()
    return normalized_value or None


def _seed_artifact_names(source: Sequence[object]) -> tuple[str, ...]:
    artifact_names = tuple(str(item).strip() for item in source if str(item).strip())
    if not artifact_names:
        raise ValueError("supervised_seed_artifact_names must not be empty.")
    return artifact_names
