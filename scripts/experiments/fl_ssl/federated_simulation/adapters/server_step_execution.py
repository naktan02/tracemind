"""FL simulation server step execution adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from methods.adaptation.lora_classifier.federated_ssl.supervised_seed_step import (
    run_lora_classifier_supervised_seed_step_core,
)
from methods.adaptation.text_classifier.aggregation import (
    peft_encoder_fedavg_projection as peft_fedavg_projection,
)
from methods.adaptation.text_classifier.aggregation import (
    peft_encoder_state_projection as peft_state_projection,
)
from methods.adaptation.text_classifier.peft_encoder.update.materialization import (
    materialize_base_lora_classifier_state,
)
from methods.federated_ssl.capability_plan import (
    SERVER_STEP_NONE,
    SERVER_STEP_SUPERVISED_SEED,
    FederatedSslCapabilityPlan,
)
from methods.federated_ssl.server_step import (
    resolve_method_supervised_seed_step_parameters,
)
from scripts.experiments.fl_ssl.federated_simulation.flow.state import (
    ActiveSimulationState,
    BootstrappedSimulation,
)
from scripts.experiments.fl_ssl.federated_simulation.io.run_artifact_writer import (
    RunArtifactWriter,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    SimulationRunRequest,
)
from scripts.runtime_adapters.federated_server.aggregation_artifacts import (
    build_server_aggregate_artifact_refs,
    build_simulation_aggregation_context,
)
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
    LoraClassifierState,
)


def require_supported_server_step(
    capability_plan: FederatedSslCapabilityPlan,
) -> None:
    """v1 simulation에서 지원되는 server step인지 확인한다."""

    if capability_plan.server_step_policy_name == SERVER_STEP_NONE:
        return
    if capability_plan.server_step_policy_name == SERVER_STEP_SUPERVISED_SEED:
        return
    raise NotImplementedError(
        "server_step_policy is declared but not implemented in simulation runtime: "
        f"{capability_plan.server_step_policy_name}"
    )


@dataclass(frozen=True, slots=True)
class ServerStepExecution:
    """round open 전에 실행된 server-side step 결과."""

    active: ActiveSimulationState
    metrics: dict[str, float]
    model_revision: str | None = None


def run_server_step_if_supported(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    capability_plan: FederatedSslCapabilityPlan,
    round_index: int,
) -> ServerStepExecution:
    """capability plan에 선언된 server-side step을 round open 전에 실행한다."""

    if capability_plan.server_step_policy_name == SERVER_STEP_NONE:
        return ServerStepExecution(active=active, metrics={})
    if capability_plan.server_step_policy_name != SERVER_STEP_SUPERVISED_SEED:
        raise NotImplementedError(
            "server_step_policy is declared but not implemented in simulation "
            f"runtime: {capability_plan.server_step_policy_name}"
        )
    return _run_lora_classifier_supervised_seed_step(
        request=request,
        bootstrapped=bootstrapped,
        active=active,
        round_index=round_index,
    )


def _run_lora_classifier_supervised_seed_step(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_index: int,
) -> ServerStepExecution:
    if request.round_runtime_config.adapter_family_name != LORA_CLASSIFIER_ADAPTER_KIND:
        raise NotImplementedError(
            "supervised_seed_step currently supports lora_classifier simulation "
            "runtime."
        )
    if request.round_runtime_config.lora_classifier is None:
        raise ValueError("lora_classifier runtime config is required.")
    if not isinstance(active.adapter_state, LoraClassifierState):
        raise ValueError("supervised_seed_step requires active LoraClassifierState.")
    if not bootstrapped.dataset_split.bootstrap_rows:
        raise ValueError("supervised_seed_step requires server bootstrap_rows.")
    if request.ssl_method_config is None:
        raise ValueError("supervised_seed_step requires ssl_method_config.")

    lora_config = request.round_runtime_config.lora_classifier.training_backend_config
    labels = tuple(str(label) for label in active.adapter_state.label_schema)
    seed_parameters = resolve_method_supervised_seed_step_parameters(
        method_name=request.ssl_method_config.name,
        effective_parameters=request.ssl_method_config.effective_parameters,
        default_epochs=request.training_task_config.local_epochs,
        default_batch_size=request.training_task_config.batch_size,
        round_index=round_index,
    )
    now = datetime.now(timezone.utc)
    base_parameters = materialize_base_lora_classifier_state(
        base_state=active.adapter_state,
        context=build_simulation_aggregation_context(
            output_dir=request.output_dir,
            next_model_revision=active.adapter_state.model_revision,
            aggregated_at=now,
        ),
    )
    seed_result = run_lora_classifier_supervised_seed_step_core(
        labels=labels,
        base_parameters=base_parameters,
        bootstrap_rows=bootstrapped.dataset_split.bootstrap_rows,
        lora_config=lora_config,
        trainer_runtime_config=request.local_trainer_runtime_config,
        runtime_resource_cache=bootstrapped.runtime_resource_cache,
        seed=int(request.seed) + 7919 + int(round_index),
        epochs=seed_parameters.epochs,
        batch_size=seed_parameters.batch_size,
        learning_rate=float(request.training_task_config.learning_rate),
        gradient_clip_norm=request.training_task_config.gradient_clip_norm,
    )
    next_model_revision = f"sim_rev_{round_index:04d}_server_seed"
    artifact_refs = build_server_aggregate_artifact_refs(
        adapter_family_name=LORA_CLASSIFIER_ADAPTER_KIND,
        next_model_revision=next_model_revision,
        artifact_names=(
            peft_fedavg_projection.LORA_ADAPTER_ARTIFACT_SLOT,
            peft_fedavg_projection.CLASSIFIER_HEAD_ARTIFACT_SLOT,
        ),
    )
    projection = peft_state_projection.build_lora_classifier_state_projection(
        base_state=active.adapter_state,
        base_parameters=base_parameters,
        next_model_revision=next_model_revision,
        updated_at=now,
        lora_adapter_artifact_ref=artifact_refs.refs_by_name[
            peft_fedavg_projection.LORA_ADAPTER_ARTIFACT_SLOT
        ],
        classifier_head_artifact_ref=artifact_refs.refs_by_name[
            peft_fedavg_projection.CLASSIFIER_HEAD_ARTIFACT_SLOT
        ],
        artifact_format=artifact_refs.artifact_format,
        lora_parameter_deltas=seed_result.lora_parameter_deltas,
        classifier_head_weight_deltas=seed_result.classifier_head_weight_deltas,
        classifier_head_bias_deltas=seed_result.classifier_head_bias_deltas,
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
    return ServerStepExecution(
        active=ActiveSimulationState(
            manifest=publication.next_manifest,
            adapter_state=publication.next_state,
        ),
        model_revision=next_model_revision,
        metrics=seed_result.metrics,
    )
