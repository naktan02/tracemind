"""PEFT encoder server-step runtime adapter for FL simulation."""

from __future__ import annotations

from datetime import datetime, timezone

from methods.adaptation.peft_text_classifier.aggregation import (
    peft_encoder_fedavg_projection as peft_fedavg_projection,
)
from methods.adaptation.peft_text_classifier.aggregation import (
    peft_encoder_state_projection as peft_state_projection,
)
from methods.adaptation.peft_text_classifier.federated_ssl import (
    supervised_seed_step,
)
from methods.adaptation.peft_text_classifier.runtime_family import (
    is_peft_encoder_adapter_family,
    peft_encoder_runtime_payload,
)
from methods.adaptation.peft_text_classifier.update.materialization import (
    materialize_base_peft_encoder_state,
)
from methods.federated_ssl.server_step import (
    resolve_method_supervised_seed_step_parameters,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters import (
    server_step_execution,
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
    LoraClassifierState,
)
from shared.src.contracts.adapter_contract_families.peft_classifier import (
    PeftClassifierState,
)


def run_peft_encoder_supervised_seed_step(
    *,
    request: SimulationRunRequest,
    bootstrapped: BootstrappedSimulation,
    active: ActiveSimulationState,
    round_index: int,
) -> server_step_execution.ServerStepExecution:
    """PEFT-backed text classifier server supervised seed step을 실행한다."""

    adapter_family_name = request.round_runtime_config.adapter_family_name
    if not is_peft_encoder_adapter_family(adapter_family_name):
        raise NotImplementedError(
            "supervised_seed_step currently supports PEFT-backed classifier "
            "simulation runtime."
        )
    runtime_payload = peft_encoder_runtime_payload(request.round_runtime_config)
    if runtime_payload is None:
        raise ValueError("PEFT-backed classifier runtime config is required.")
    if not isinstance(active.adapter_state, LoraClassifierState | PeftClassifierState):
        raise ValueError("supervised_seed_step requires active classifier state.")
    if not bootstrapped.dataset_split.bootstrap_rows:
        raise ValueError("supervised_seed_step requires server bootstrap_rows.")
    if request.ssl_method_config is None:
        raise ValueError("supervised_seed_step requires ssl_method_config.")

    lora_config = runtime_payload.training_backend_config
    labels = tuple(str(label) for label in active.adapter_state.label_schema)
    seed_parameters = resolve_method_supervised_seed_step_parameters(
        method_name=request.ssl_method_config.name,
        effective_parameters=request.ssl_method_config.effective_parameters,
        default_epochs=request.training_task_config.local_epochs,
        default_batch_size=request.training_task_config.batch_size,
        round_index=round_index,
    )
    now = datetime.now(timezone.utc)
    base_parameters = materialize_base_peft_encoder_state(
        base_state=active.adapter_state,
        context=build_simulation_aggregation_context(
            output_dir=request.output_dir,
            next_model_revision=active.adapter_state.model_revision,
            aggregated_at=now,
        ),
    )
    seed_result = supervised_seed_step.run_peft_encoder_supervised_seed_step_core(
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
        artifact_namespace=request.round_runtime_config.update_family_name,
        next_model_revision=next_model_revision,
        artifact_names=(
            _adapter_artifact_slot(active.adapter_state),
            peft_fedavg_projection.CLASSIFIER_HEAD_ARTIFACT_SLOT,
        ),
    )
    projection = peft_state_projection.build_peft_encoder_state_projection(
        base_state=active.adapter_state,
        base_parameters=base_parameters,
        next_model_revision=next_model_revision,
        updated_at=now,
        lora_adapter_artifact_ref=artifact_refs.refs_by_name[
            _adapter_artifact_slot(active.adapter_state)
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
    return server_step_execution.ServerStepExecution(
        active=ActiveSimulationState(
            manifest=publication.next_manifest,
            adapter_state=publication.next_state,
        ),
        model_revision=next_model_revision,
        metrics=seed_result.metrics,
    )


def _adapter_artifact_slot(
    adapter_state: LoraClassifierState | PeftClassifierState,
) -> str:
    if isinstance(adapter_state, PeftClassifierState):
        return peft_fedavg_projection.PEFT_ADAPTER_ARTIFACT_SLOT
    return peft_fedavg_projection.LORA_ADAPTER_ARTIFACT_SLOT
