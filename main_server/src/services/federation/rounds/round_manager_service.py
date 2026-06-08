"""FL round의 task 발행과 pair publication을 조정한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping
from uuid import uuid4

from main_server.src.infrastructure.repositories import (
    shared_adapter_state_repository as shared_adapter_state_repository_module,
)
from main_server.src.infrastructure.repositories import (
    shared_adapter_update_repository as shared_adapter_update_repository_module,
)
from main_server.src.services.federation.rounds.boundary.models import (
    RoundOpenRequest,
)
from main_server.src.services.federation.rounds.payload_adapters.models import (
    SharedAdapterRoundPayloadAdapter,
)
from methods.federated_ssl.capabilities.plan import (
    build_default_method_capability_plan,
)
from methods.federated_ssl.compatibility import (
    validate_federated_ssl_capability_compatibility,
)
from methods.federated_ssl.execution_plan import build_federated_ssl_execution_plan
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from methods.federated_ssl.runtime_fallbacks import (
    RUNTIME_FALLBACK_SERVER_ROUND_PROFILE,
    build_runtime_fallback_secure_aggregation_config,
    build_runtime_fallback_training_objective_config,
    build_runtime_fallback_training_selection_policy,
    build_runtime_strategy_training_objective_config,
)
from shared.src.contracts.model_contracts import ModelManifest
from shared.src.contracts.training_contracts import (
    SecureAggregationConfig,
    TrainingConfigScalar,
    TrainingObjectiveConfig,
    TrainingSelectionPolicy,
    TrainingTask,
    TrainingUpdateEnvelope,
)
from shared.src.domain.entities.training.shared_adapter_state import SharedAdapterState
from shared.src.domain.services.clock import Clock, SystemUtcClock

SharedAdapterStateRepository = (
    shared_adapter_state_repository_module.SharedAdapterStateRepository
)
SharedAdapterUpdateRepository = (
    shared_adapter_update_repository_module.SharedAdapterUpdateRepository
)


@dataclass(slots=True)
class RoundPublication:
    """한 라운드 집계 후 발행되는 새 active manifest 메타데이터."""

    next_manifest: ModelManifest
    next_state: SharedAdapterState
    aggregated_metrics: dict[str, float]
    update_count: int


@dataclass(slots=True)
class RoundPublicationRequest:
    """집계 후 active manifest 발행 요청."""

    base_manifest: ModelManifest
    updates: tuple[TrainingUpdateEnvelope, ...] | list[TrainingUpdateEnvelope]
    next_model_revision: str | None = None
    next_auxiliary_artifact_versions: Mapping[str, str] = field(default_factory=dict)
    published_at: datetime | None = None


@dataclass(slots=True)
class RoundManagerService:
    """라운드용 task를 만들고 새 active manifest를 발행한다."""

    payload_adapter: SharedAdapterRoundPayloadAdapter
    artifact_repository: SharedAdapterStateRepository = field(
        default_factory=SharedAdapterStateRepository
    )
    update_payload_repository: SharedAdapterUpdateRepository = field(
        default_factory=SharedAdapterUpdateRepository
    )
    clock: Clock = field(default_factory=SystemUtcClock)

    def create_training_task(
        self,
        request: RoundOpenRequest,
        *,
        runtime_surface: Mapping[str, object] | None = None,
    ) -> TrainingTask:
        if request.round_id is None:
            raise ValueError("RoundOpenRequest.round_id must be set.")
        resolved_runtime_surface = self._resolve_runtime_surface_payload(
            runtime_surface
        )
        return TrainingTask(
            schema_version="training_task.v1",
            task_id=request.task_id or f"task_{request.round_id}_{uuid4().hex[:8]}",
            round_id=request.round_id,
            model_id=request.active_manifest.model_id,
            model_revision=request.active_manifest.model_revision,
            task_type=request.task_type,
            training_scope=request.active_manifest.training_scope,
            local_epochs=request.local_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            max_steps=request.max_steps,
            objective_config=self._resolve_objective_config(
                request.objective_config,
                strategy=request.strategy,
                batch_size=request.batch_size,
            ),
            selection_policy=self._resolve_selection_policy(request.selection_policy),
            secure_aggregation=self._resolve_secure_aggregation(
                request.secure_aggregation
            ),
            fssl_method=getattr(request.strategy, "fssl_method", None),
            fssl_execution=self._resolve_fssl_execution_payload(
                request.strategy,
                runtime_surface=resolved_runtime_surface,
            ),
            fssl_capability_plan=self._resolve_fssl_capability_plan_payload(
                request.strategy
            ),
            fssl_context=(
                dict(request.fssl_context) if request.fssl_context is not None else None
            ),
            deadline_at=request.deadline_at,
            gradient_clip_norm=request.gradient_clip_norm,
            min_required_examples=request.min_required_examples,
            notes=request.notes,
        )

    def publish_next_pair(self, request: RoundPublicationRequest) -> RoundPublication:
        if not request.updates:
            raise ValueError(
                "At least one update is required to publish the next pair."
            )

        effective_published_at = request.published_at or self.clock.now()
        base_state = self._load_base_state(request.base_manifest)
        next_revision = (
            request.next_model_revision
            or f"{request.base_manifest.model_revision}_next"
        )
        update_payloads = [
            self._load_update_payload(update) for update in request.updates
        ]
        aggregation = self.payload_adapter.aggregation_backend.aggregate(
            base_state=base_state,
            update_payloads=update_payloads,
            next_model_revision=next_revision,
            aggregated_at=effective_published_at,
        )
        next_state_payload = self.payload_adapter.state_to_payload(
            aggregation.next_state
        )
        self.artifact_repository.save_shared_adapter_state(next_state_payload)
        next_auxiliary_versions = _build_next_auxiliary_artifact_versions(
            base_manifest=request.base_manifest,
            next_auxiliary_artifact_versions=(request.next_auxiliary_artifact_versions),
        )
        next_manifest = ModelManifest(
            schema_version=request.base_manifest.schema_version,
            model_id=request.base_manifest.model_id,
            model_revision=aggregation.next_state.model_revision,
            published_at=effective_published_at,
            artifact_kind="shared_adapter_state",
            artifact_ref=self.artifact_repository.ref_for_revision(
                aggregation.next_state.model_revision
            ),
            auxiliary_artifact_versions=next_auxiliary_versions,
            training_scope=request.base_manifest.training_scope,
            training_enabled=request.base_manifest.training_enabled,
            compatible_task_types=request.base_manifest.compatible_task_types,
            base_model_id=request.base_manifest.base_model_id,
            base_model_revision=request.base_manifest.base_model_revision,
            notes=(
                f"round_active_pair_only published from {request.updates[0].round_id}"
            ),
        )
        return RoundPublication(
            next_manifest=next_manifest,
            next_state=aggregation.next_state,
            aggregated_metrics=aggregation.aggregated_metrics,
            update_count=aggregation.update_count,
        )

    def _load_base_state(self, base_manifest: ModelManifest) -> SharedAdapterState:
        payload = self.artifact_repository.load_shared_adapter_state_from_ref(
            base_manifest.artifact_ref
        )
        state = self.payload_adapter.state_from_payload(payload)
        if state.adapter_kind != self.payload_adapter.aggregation_backend.adapter_kind:
            raise ValueError(
                "Base state adapter_kind does not match the configured "
                f"aggregation backend: {state.adapter_kind}"
            )
        return state

    def _load_update_payload(self, update: TrainingUpdateEnvelope):
        payload = self.update_payload_repository.load_shared_adapter_update_from_ref(
            update.payload_ref
        )
        if update.payload_format not in self.payload_adapter.accepted_update_formats:
            raise ValueError(
                "Unsupported payload_format for adapter_kind "
                f"{payload.adapter_kind}: {update.payload_format}"
            )
        return self.payload_adapter.update_from_payload(payload)

    @staticmethod
    def _resolve_objective_config(
        source: TrainingObjectiveConfig | Mapping[str, TrainingConfigScalar] | None,
        *,
        strategy: object | None = None,
        batch_size: int,
    ) -> TrainingObjectiveConfig:
        if source is not None and strategy is not None:
            raise ValueError(
                "Round open request must not provide both strategy and "
                "objective_config."
            )
        if strategy is not None:
            return build_runtime_strategy_training_objective_config(
                local_update_profile_name=getattr(
                    strategy,
                    "local_update_profile",
                    None,
                ),
                strategy_mode=getattr(strategy, "mode", "composed"),
                ssl_method_name=getattr(strategy, "ssl_method", None),
                fssl_method_name=getattr(strategy, "fssl_method", None),
                server_update_policy_name=getattr(
                    strategy,
                    "server_update_policy",
                    None,
                ),
                aggregation_backend_name=getattr(
                    strategy,
                    "aggregation_backend",
                    None,
                ),
                unlabeled_batch_size=batch_size,
                parameter_overrides=getattr(strategy, "parameter_overrides", None),
            )
        if isinstance(source, TrainingObjectiveConfig):
            return source
        if source is None:
            return build_runtime_fallback_training_objective_config()
        return TrainingObjectiveConfig.from_mapping(source)

    @staticmethod
    def _resolve_selection_policy(
        source: TrainingSelectionPolicy | Mapping[str, TrainingConfigScalar] | None,
    ) -> TrainingSelectionPolicy:
        if isinstance(source, TrainingSelectionPolicy):
            return source
        if source is None:
            return build_runtime_fallback_training_selection_policy()
        return TrainingSelectionPolicy.from_mapping(source)

    @staticmethod
    def _resolve_secure_aggregation(
        source: (
            SecureAggregationConfig | Mapping[str, TrainingConfigScalar] | bool | None
        ),
    ) -> SecureAggregationConfig:
        if isinstance(source, SecureAggregationConfig):
            return source
        if isinstance(source, bool):
            return build_runtime_fallback_secure_aggregation_config(
                overrides={"required": source}
            )
        if source is None:
            return build_runtime_fallback_secure_aggregation_config()
        return SecureAggregationConfig.from_mapping(source)

    @staticmethod
    def _resolve_fssl_execution_payload(
        strategy: object | None,
        *,
        runtime_surface: Mapping[str, object],
    ) -> dict[str, object] | None:
        method_name = _strategy_text(strategy, "fssl_method")
        if method_name is None:
            return None
        descriptor = resolve_federated_ssl_method_descriptor(method_name)
        plan = build_federated_ssl_execution_plan(
            fl_method={
                "name": descriptor.name,
                "descriptor_name": descriptor.name,
                "composition_mode": "method_owned",
            },
            security_policy=None,
            method_descriptor=descriptor,
        )
        payload = plan.to_mapping()
        payload["runtime_surface"] = dict(runtime_surface)
        return payload

    def _resolve_runtime_surface_payload(
        self,
        source: Mapping[str, object] | None,
    ) -> dict[str, object]:
        return {
            "payload_adapter_kind": (
                _mapping_text(source, "payload_adapter_kind")
                or self.payload_adapter.adapter_kind
            ),
            "update_family_name": (
                _mapping_text(source, "update_family_name")
                or RUNTIME_FALLBACK_SERVER_ROUND_PROFILE.update_family_name
            ),
            "aggregation_backend_name": (
                _mapping_text(source, "aggregation_backend_name")
                or RUNTIME_FALLBACK_SERVER_ROUND_PROFILE.aggregation_backend_name
            ),
        }

    @staticmethod
    def _resolve_fssl_capability_plan_payload(
        strategy: object | None,
    ) -> dict[str, object] | None:
        method_name = _strategy_text(strategy, "fssl_method")
        if method_name is None:
            return None
        descriptor = resolve_federated_ssl_method_descriptor(method_name)
        if not descriptor.runtime_capabilities.live_server_supported:
            raise ValueError(
                f"fssl_method={method_name!r}는 live server runtime을 "
                "지원하지 않습니다."
            )
        capability_plan = build_default_method_capability_plan(
            method_descriptor=descriptor,
            method_config=_strategy_method_config(strategy, method_name=method_name),
            server_update_policy_name=_strategy_text(
                strategy,
                "server_update_policy",
            ),
        )
        validate_federated_ssl_capability_compatibility(
            method_descriptor=descriptor,
            capability_plan=capability_plan,
        )
        return capability_plan.to_payload()


def _strategy_text(strategy: object | None, field_name: str) -> str | None:
    if strategy is None:
        return None
    raw_value = getattr(strategy, field_name, None)
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None


def _mapping_text(
    source: Mapping[str, object] | None,
    field_name: str,
) -> str | None:
    if source is None:
        return None
    raw_value = source.get(field_name)
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None


def _strategy_method_config(
    strategy: object | None,
    *,
    method_name: str,
) -> dict[str, object]:
    method_config: dict[str, object] = {
        "name": method_name,
        "use_original_parameters": True,
        "parameter_overrides": {},
    }
    scenario = _strategy_text(strategy, "scenario")
    if scenario is not None:
        method_config["scenario"] = scenario
    parameter_overrides = getattr(strategy, "parameter_overrides", None)
    if isinstance(parameter_overrides, Mapping):
        method_config["parameter_overrides"] = dict(parameter_overrides)
    return method_config


def _build_next_auxiliary_artifact_versions(
    *,
    base_manifest: ModelManifest,
    next_auxiliary_artifact_versions: Mapping[str, str],
) -> dict[str, str]:
    """부속 artifact version을 중립 map으로 누적한다."""

    result = dict(base_manifest.auxiliary_artifact_versions)
    result.update(
        {
            str(key): str(value)
            for key, value in next_auxiliary_artifact_versions.items()
            if str(key).strip() and str(value).strip()
        }
    )
    return result
