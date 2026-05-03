"""Entrypoint별 experiment compile warning/validation policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from main_server.src.services.experiment_workspace.catalog_service import (
    ExperimentCatalogService,
)
from main_server.src.services.experiment_workspace.payloads import CatalogItemPayload
from shared.src.contracts.workspace_manifest_contracts import (
    WorkspaceManifestPayload,
)


@dataclass(frozen=True, slots=True)
class ExperimentCompileContext:
    """Entrypoint-specific compile policy가 읽는 canonical context."""

    manifest: WorkspaceManifestPayload
    entrypoint_item: CatalogItemPayload
    effective_groups: dict[str, str]
    hydra_override_map: dict[str, str]
    catalog_service: ExperimentCatalogService


class ExperimentCompilePolicy(Protocol):
    """Entrypoint-specific compile policy."""

    def collect_warnings(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> tuple[str, ...]:
        """compile preview에 추가할 warning을 반환한다."""

    def validate_requirements(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> None:
        """compile 전제조건을 검사하고 실패 시 ValueError를 올린다."""


@dataclass(frozen=True, slots=True)
class NoOpExperimentCompilePolicy:
    """추가 warning/validation이 없는 기본 policy."""

    def collect_warnings(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> tuple[str, ...]:
        del context
        return ()

    def validate_requirements(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> None:
        del context


@dataclass(frozen=True, slots=True)
class FederatedSimulationCompilePolicy:
    """federated simulation entrypoint의 preview warning policy."""

    def collect_warnings(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> tuple[str, ...]:
        warnings = [
            "run_federated_simulation의 client_count는 live agent roster가 아니라 "
            "synthetic simulation participant 수를 뜻한다."
        ]
        if _resolve_effective_string_override(
            context.hydra_override_map,
            "shard_policy.name",
        ) not in (None, "label_dominant"):
            return tuple(warnings)

        dataset_name = context.effective_groups.get("dataset")
        federated_run_preset = context.effective_groups.get("federated_run_preset")
        if dataset_name is None or federated_run_preset is None:
            return tuple(warnings)

        dataset_raw = context.catalog_service.load_config_group_item(
            relative_dir="scripts/conf/dataset",
            item_name=dataset_name,
        )
        label_count = len(
            [
                category
                for category in dataset_raw.get("prototype_expected_categories", [])
                if isinstance(category, str) and category.strip()
            ]
        )
        if label_count <= 0:
            return tuple(warnings)

        preset_raw = context.catalog_service.load_config_group_item(
            relative_dir="scripts/conf/fl_ssl/run_preset",
            item_name=federated_run_preset,
        )
        client_count = _resolve_effective_int_override(
            context.hydra_override_map,
            key="federated_run_preset.client_count",
            fallback=preset_raw.get("client_count"),
        )
        if client_count is None:
            return tuple(warnings)
        if client_count > label_count + 1:
            warnings.append(
                "현재 label_dominant shard policy에서는 "
                f"client_count={client_count}, label_count={label_count} 조합에서 "
                "빈 shard 또는 거의 비어 있는 shard가 생길 수 있다."
            )
        return tuple(warnings)

    def validate_requirements(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> None:
        del context


@dataclass(frozen=True, slots=True)
class FixMatchCompilePolicy:
    """FixMatch entrypoint의 dataset readiness validation policy."""

    def collect_warnings(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> tuple[str, ...]:
        del context
        return ()

    def validate_requirements(
        self,
        *,
        context: ExperimentCompileContext,
    ) -> None:
        if _has_non_null_override(
            context.hydra_override_map,
            "unlabeled_jsonl",
            "query_ssl_train_source.unlabeled_jsonl",
        ):
            return

        train_source_name = context.effective_groups.get("query_ssl_train_source")
        if train_source_name is None:
            raise ValueError(
                "FixMatch compile readiness check failed: "
                "query_ssl_train_source preset을 결정할 수 없습니다."
            )
        train_source_raw = context.catalog_service.load_config_group_item(
            relative_dir="scripts/conf/central_ssl/train_source",
            item_name=train_source_name,
        )
        source_unlabeled = _string_or_none(train_source_raw.get("unlabeled_jsonl"))
        if source_unlabeled is None or source_unlabeled == "null":
            raise ValueError(
                "FixMatch compile readiness check failed: "
                f"query_ssl_train_source={train_source_name} 에 "
                "unlabeled_jsonl이 없습니다."
            )
        if source_unlabeled != "${dataset.unlabeled_query_pool_jsonl}":
            return

        dataset_name = context.effective_groups.get("dataset")
        if dataset_name is None:
            raise ValueError(
                "FixMatch compile readiness check failed: "
                "dataset preset을 결정할 수 없습니다."
            )
        dataset_raw = context.catalog_service.load_config_group_item(
            relative_dir="scripts/conf/dataset",
            item_name=dataset_name,
        )
        dataset_unlabeled = _string_or_none(
            dataset_raw.get("unlabeled_query_pool_jsonl")
        )
        if dataset_unlabeled is None or dataset_unlabeled == "null":
            raise ValueError(
                "FixMatch compile readiness check failed: "
                f"dataset={dataset_name} 는 "
                "unlabeled_query_pool_jsonl이 아직 비어 있습니다. "
                "다른 query_ssl_train_source preset을 고르거나 "
                "unlabeled_jsonl override를 직접 제공하세요."
            )


@dataclass(slots=True)
class ExperimentCompilePolicyRegistry:
    """Entrypoint 이름별 compile policy registry."""

    _policies: dict[str, ExperimentCompilePolicy] = field(default_factory=dict)
    _default_policy: ExperimentCompilePolicy = field(
        default_factory=NoOpExperimentCompilePolicy
    )

    def register(
        self,
        entrypoint_name: str,
        policy: ExperimentCompilePolicy,
    ) -> None:
        self._policies[entrypoint_name] = policy

    def resolve(self, entrypoint_name: str) -> ExperimentCompilePolicy:
        return self._policies.get(entrypoint_name, self._default_policy)


DEFAULT_EXPERIMENT_COMPILE_POLICY_REGISTRY = ExperimentCompilePolicyRegistry()
DEFAULT_EXPERIMENT_COMPILE_POLICY_REGISTRY.register(
    "run_federated_simulation",
    FederatedSimulationCompilePolicy(),
)
DEFAULT_EXPERIMENT_COMPILE_POLICY_REGISTRY.register(
    "train_lora_fixmatch",
    FixMatchCompilePolicy(),
)


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _has_non_null_override(
    hydra_override_map: dict[str, str],
    *candidate_keys: str,
) -> bool:
    for candidate_key in candidate_keys:
        value = hydra_override_map.get(candidate_key)
        if value is None:
            continue
        if value.lower() != "null":
            return True
    return False


def _resolve_effective_string_override(
    hydra_override_map: dict[str, str],
    key: str,
) -> str | None:
    value = hydra_override_map.get(key)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() == "null":
        return None
    return stripped


def _resolve_effective_int_override(
    hydra_override_map: dict[str, str],
    *,
    key: str,
    fallback: object,
) -> int | None:
    value = hydra_override_map.get(key)
    if value is None:
        if isinstance(fallback, int):
            return fallback
        if isinstance(fallback, float):
            return int(fallback)
        if isinstance(fallback, str):
            stripped = fallback.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                return None
        return None
    try:
        return int(value)
    except ValueError:
        return None


__all__ = [
    "DEFAULT_EXPERIMENT_COMPILE_POLICY_REGISTRY",
    "ExperimentCompileContext",
    "ExperimentCompilePolicy",
    "ExperimentCompilePolicyRegistry",
    "FederatedSimulationCompilePolicy",
    "FixMatchCompilePolicy",
    "NoOpExperimentCompilePolicy",
]
