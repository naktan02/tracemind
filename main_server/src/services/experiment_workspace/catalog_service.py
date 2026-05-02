"""개발자용 read-only experiment catalog service."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from main_server.src.services.experiment_workspace.catalog_build_context import (
    ExperimentCatalogBuildContext,
)
from main_server.src.services.experiment_workspace.catalog_constants import (
    AGENT_LIVE_STORED_EVENT_RUNTIME_PATH,
    FEDERATED_SIMULATION_RUNTIME_PATH,
)
from main_server.src.services.experiment_workspace.catalog_tracks import (
    build_catalog_tracks,
)
from main_server.src.services.experiment_workspace.payloads import (
    ExperimentCatalogPayload,
)
from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.config.training_defaults import (
    DEFAULT_TRAINING_PROFILE,
    build_default_secure_aggregation_config,
    build_default_training_selection_policy,
)
from shared.src.contracts.common_types import TrainingTaskType
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingTaskPayload,
)


class ExperimentCatalogService:
    """현재 코드/설정에서 읽어오는 read-only 전략 catalog."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self._repo_root = repo_root or Path(__file__).resolve().parents[4]

    def build_catalog(self) -> ExperimentCatalogPayload:
        """현재 저장소 상태를 반영한 catalog payload를 조립한다."""

        return ExperimentCatalogPayload(
            generated_at=datetime.now(tz=timezone.utc),
            source_root=str(self._repo_root),
            tracks=build_catalog_tracks(self._build_context()),
        )

    def load_config_group_item(
        self,
        *,
        relative_dir: str,
        item_name: str,
    ) -> dict[str, object]:
        """Hydra config group item 하나를 직접 읽는다."""

        path = self._repo_root / relative_dir / f"{item_name}.yaml"
        if not path.exists():
            raise ValueError(
                f"Unknown config group item: dir={relative_dir}, item={item_name}."
            )
        return self._load_yaml_mapping(path)

    def load_relative_yaml_mapping(self, relative_path: str) -> dict[str, object]:
        """repo root 기준 상대 경로 YAML mapping을 읽는다."""

        path = self._repo_root / relative_path
        if not path.exists():
            raise ValueError(f"Unknown YAML path: {relative_path}.")
        return self._load_yaml_mapping(path)

    def _build_context(self) -> ExperimentCatalogBuildContext:
        return ExperimentCatalogBuildContext(
            repo_root=self._repo_root,
            load_yaml_mapping=self._load_yaml_mapping,
            iter_yaml_files=self._iter_yaml_files,
            relative_repo_path=self._relative_repo_path,
            resolve_script_path=self._resolve_script_path,
            source_of_truth_for_module=self._source_of_truth_for_module,
            build_catalog_training_task=self._build_catalog_training_task,
            resolve_example_generation_runtime_paths=(
                self._resolve_example_generation_runtime_paths
            ),
            resolve_scoring_backend_runtime_paths=(
                self._resolve_scoring_backend_runtime_paths
            ),
        )

    def _build_catalog_training_task(
        self,
        training_scope: str,
        objective_config: TrainingObjectiveConfig,
    ) -> TrainingTaskPayload:
        return TrainingTaskPayload(
            task_id="catalog_preview_task",
            round_id="catalog_preview_round",
            model_id="catalog_preview_model",
            model_revision="catalog_preview_revision",
            task_type=TrainingTaskType.PSEUDO_LABEL_SELF_TRAINING,
            training_scope=training_scope,
            local_epochs=DEFAULT_TRAINING_PROFILE.local_epochs,
            batch_size=DEFAULT_TRAINING_PROFILE.batch_size,
            learning_rate=DEFAULT_TRAINING_PROFILE.learning_rate,
            max_steps=DEFAULT_TRAINING_PROFILE.max_steps,
            objective_config=objective_config,
            selection_policy=build_default_training_selection_policy(),
            min_required_examples=DEFAULT_TRAINING_PROFILE.min_required_examples,
            gradient_clip_norm=DEFAULT_TRAINING_PROFILE.gradient_clip_norm,
            secure_aggregation=build_default_secure_aggregation_config(),
        )

    def _resolve_example_generation_runtime_paths(
        self,
        entry: RegistryCatalogEntry,
    ) -> tuple[str, ...]:
        runtime_paths = [FEDERATED_SIMULATION_RUNTIME_PATH]
        if bool(entry.metadata.get("supports_stored_event_rebuild")):
            runtime_paths.append(AGENT_LIVE_STORED_EVENT_RUNTIME_PATH)
        return tuple(runtime_paths)

    def _resolve_scoring_backend_runtime_paths(
        self,
        entry: RegistryCatalogEntry,
    ) -> tuple[str, ...]:
        runtime_paths = [FEDERATED_SIMULATION_RUNTIME_PATH]
        if not bool(entry.metadata.get("requires_shared_state")):
            runtime_paths.append(AGENT_LIVE_STORED_EVENT_RUNTIME_PATH)
        return tuple(runtime_paths)

    def _iter_yaml_files(self, relative_dir: str) -> tuple[Path, ...]:
        root = self._repo_root / relative_dir
        return tuple(sorted(root.glob("*.yaml")))

    def _load_yaml_mapping(self, path: Path) -> dict[str, object]:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        if raw is None:
            return {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"Expected mapping config at {path}.")
        return dict(raw)

    def _resolve_script_path(self, job_config_path: str) -> str:
        if job_config_path.startswith("scripts/conf/experiments/"):
            return job_config_path.replace(
                "scripts/conf/experiments/",
                "scripts/experiments/",
            ).replace(".yaml", ".py")
        if job_config_path.startswith("scripts/conf/prototypes/"):
            return job_config_path.replace(
                "scripts/conf/prototypes/",
                "scripts/prototypes/",
            ).replace(".yaml", ".py")
        if job_config_path.startswith("scripts/conf/datasets/"):
            return job_config_path.replace(
                "scripts/conf/datasets/",
                "scripts/datasets/",
            ).replace(".yaml", ".py")
        raise ValueError(
            "Unsupported job config path for catalog script resolution: "
            f"{job_config_path}."
        )

    def _source_of_truth_for_module(self, module_name: str) -> str:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return module_name
        return self._relative_repo_path(Path(spec.origin))

    def _relative_repo_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self._repo_root))
        except ValueError:
            return str(resolved)
