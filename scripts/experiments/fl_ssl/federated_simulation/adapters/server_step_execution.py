"""FL simulation server step execution adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from main_server.src.services.federation.rounds.aggregation.artifact_refs import (
    AggregatedArtifactRefBuilder,
    AggregationArtifactStore,
)
from main_server.src.services.federation.rounds.aggregation.executor import (
    DEFAULT_AGGREGATED_ARTIFACT_FORMAT,
)
from methods.adaptation.lora_classifier.aggregation.fedavg import (
    CLASSIFIER_HEAD_ARTIFACT_SLOT,
    LORA_ADAPTER_ARTIFACT_SLOT,
)
from methods.adaptation.lora_classifier.aggregation.materialization import (
    materialize_base_lora_classifier_state,
)
from methods.adaptation.lora_classifier.aggregation.state_projection import (
    build_lora_classifier_state_projection,
)
from methods.adaptation.lora_classifier.training.delta_extraction import (
    extract_lora_classifier_parameter_deltas,
    load_lora_classifier_base_parameters_into_model,
)
from methods.adaptation.lora_classifier.training.loops import (
    set_seed,
    train_classifier,
)
from methods.adaptation.lora_classifier.training.modeling import (
    build_lora_text_classifier_from_config,
)
from methods.adaptation.query_classifier_adaptation.data import build_dataloader
from methods.federated.aggregation.base import FederatedAggregationContext
from methods.federated_ssl.capability_plan import (
    SERVER_STEP_NONE,
    SERVER_STEP_SUPERVISED_SEED,
    FederatedSslCapabilityPlan,
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
from shared.src.contracts.adapter_contract_families.lora_classifier import (
    LORA_CLASSIFIER_ADAPTER_KIND,
    LoraClassifierState,
)
from shared.src.contracts.model_contracts import ModelManifest


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

    lora_config = request.round_runtime_config.lora_classifier.training_backend_config
    device = request.local_trainer_runtime_config.device
    labels = tuple(str(label) for label in active.adapter_state.label_schema)
    label_to_index = {label: index for index, label in enumerate(labels)}
    effective_epochs = _server_seed_epochs(
        request=request,
        round_index=round_index,
    )
    batch_size = _server_seed_batch_size(request=request)
    if effective_epochs <= 0:
        raise ValueError("supervised_seed_step server epochs must be positive.")

    set_seed(int(request.seed) + 7919 + int(round_index))
    model, tokenizer = build_lora_text_classifier_from_config(
        labels=list(labels),
        lora_config=lora_config,
        runtime_config=request.local_trainer_runtime_config,
        runtime_resource_cache=bootstrapped.runtime_resource_cache,
    )
    artifact_store = AggregationArtifactStore(
        state_root=request.output_dir / "main_server" / "aggregation_artifacts"
    )
    now = datetime.now(timezone.utc)
    base_parameters = materialize_base_lora_classifier_state(
        base_state=active.adapter_state,
        context=FederatedAggregationContext(
            next_model_revision=active.adapter_state.model_revision,
            aggregated_at=now,
            artifact_loader=artifact_store,
        ),
    )
    load_lora_classifier_base_parameters_into_model(
        model=model,
        labels=labels,
        base_parameters=base_parameters,
        device=device,
    )
    train_loader = build_dataloader(
        rows=bootstrapped.dataset_split.bootstrap_rows,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=lora_config.max_length,
        task_prefix=lora_config.task_prefix,
        shuffle=True,
    )
    train_classifier(
        model=model,
        train_loader=train_loader,
        selection_loader=train_loader,
        categories=list(labels),
        device=device,
        epochs=effective_epochs,
        learning_rate=float(request.training_task_config.learning_rate),
        classifier_learning_rate=float(request.training_task_config.learning_rate),
        weight_decay=0.0,
        max_grad_norm=(
            0.0
            if request.training_task_config.gradient_clip_norm is None
            else float(request.training_task_config.gradient_clip_norm)
        ),
        log_every_steps=0,
    )
    lora_deltas, head_weight_deltas, head_bias_deltas = (
        extract_lora_classifier_parameter_deltas(
            model=model,
            base_parameters=base_parameters,
            labels=labels,
        )
    )
    next_model_revision = f"sim_rev_{round_index:04d}_server_seed"
    artifact_ref_builder = AggregatedArtifactRefBuilder(
        artifact_ref_prefix=f"server-aggregate://{LORA_CLASSIFIER_ADAPTER_KIND}",
        artifact_format=DEFAULT_AGGREGATED_ARTIFACT_FORMAT,
    )
    lora_adapter_artifact_ref = artifact_ref_builder.build_ref(
        next_model_revision=next_model_revision,
        artifact_name=LORA_ADAPTER_ARTIFACT_SLOT,
    )
    classifier_head_artifact_ref = artifact_ref_builder.build_ref(
        next_model_revision=next_model_revision,
        artifact_name=CLASSIFIER_HEAD_ARTIFACT_SLOT,
    )
    projection = build_lora_classifier_state_projection(
        base_state=active.adapter_state,
        base_parameters=base_parameters,
        next_model_revision=next_model_revision,
        updated_at=now,
        lora_adapter_artifact_ref=lora_adapter_artifact_ref,
        classifier_head_artifact_ref=classifier_head_artifact_ref,
        artifact_format=artifact_ref_builder.artifact_format,
        lora_parameter_deltas=lora_deltas,
        classifier_head_weight_deltas=head_weight_deltas,
        classifier_head_bias_deltas=head_bias_deltas,
    )
    for artifact_ref, payload in projection.artifacts.items():
        artifact_store.save_json_artifact_ref(
            artifact_ref=artifact_ref,
            payload=dict(payload),
        )
    adapter_family = bootstrapped.server_runtime.round_manager.adapter_family
    next_state_payload = adapter_family.state_to_payload(projection.next_state)
    bootstrapped.server_runtime.state_repository.save_shared_adapter_state(
        next_state_payload
    )
    next_manifest = ModelManifest(
        schema_version=active.manifest.schema_version,
        model_id=active.manifest.model_id,
        model_revision=projection.next_state.model_revision,
        published_at=now,
        artifact_kind="shared_adapter_state",
        artifact_ref=bootstrapped.server_runtime.state_repository.ref_for_revision(
            projection.next_state.model_revision
        ),
        auxiliary_artifact_versions=dict(active.manifest.auxiliary_artifact_versions),
        training_scope=active.manifest.training_scope,
        training_enabled=active.manifest.training_enabled,
        compatible_task_types=active.manifest.compatible_task_types,
        base_model_id=active.manifest.base_model_id,
        base_model_revision=active.manifest.base_model_revision,
        notes=(
            "supervised_seed_step published from server bootstrap_rows before "
            f"round_{round_index:04d}"
        ),
    )
    bootstrapped.server_runtime.activate_manifest(next_manifest)
    RunArtifactWriter().save_model_manifest(
        output_dir=request.output_dir,
        manifest=next_manifest,
    )
    return ServerStepExecution(
        active=ActiveSimulationState(
            manifest=next_manifest,
            adapter_state=projection.next_state,
        ),
        model_revision=next_model_revision,
        metrics={
            "server_step_supervised_seed": 1.0,
            "server_step_labeled_count": float(
                len(bootstrapped.dataset_split.bootstrap_rows)
            ),
            "server_step_epochs": float(effective_epochs),
            "server_step_batch_size": float(batch_size),
        },
    )


def _server_seed_epochs(
    *,
    request: SimulationRunRequest,
    round_index: int,
) -> int:
    effective_parameters = (
        {}
        if request.ssl_method_config is None
        else request.ssl_method_config.effective_parameters
    )
    key = "server_pretrain_epochs" if round_index == 1 else "server_epochs"
    value = effective_parameters.get(key, request.training_task_config.local_epochs)
    return int(value)


def _server_seed_batch_size(*, request: SimulationRunRequest) -> int:
    effective_parameters = (
        {}
        if request.ssl_method_config is None
        else request.ssl_method_config.effective_parameters
    )
    value = effective_parameters.get(
        "server_batch_size",
        request.training_task_config.batch_size,
    )
    batch_size = int(value)
    if batch_size <= 0:
        raise ValueError("supervised_seed_step server batch size must be positive.")
    return batch_size
