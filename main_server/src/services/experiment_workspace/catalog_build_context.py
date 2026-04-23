"""Experiment catalog builder가 공유하는 context/callback 계약."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from shared.src.config.registry_catalog_metadata import RegistryCatalogEntry
from shared.src.contracts.training_contracts import (
    TrainingObjectiveConfig,
    TrainingTaskPayload,
)

LoadYamlMapping = Callable[[Path], dict[str, object]]
IterYamlFiles = Callable[[str], tuple[Path, ...]]
RelativeRepoPath = Callable[[Path], str]
ResolveScriptPath = Callable[[str], str]
SourceOfTruthForModule = Callable[[str], str]
BuildCatalogTrainingTask = Callable[
    [str, TrainingObjectiveConfig],
    TrainingTaskPayload,
]
RuntimePathResolver = Callable[[RegistryCatalogEntry], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ExperimentCatalogBuildContext:
    """Track/section builder가 공통으로 쓰는 repo-local dependency 묶음."""

    repo_root: Path
    load_yaml_mapping: LoadYamlMapping
    iter_yaml_files: IterYamlFiles
    relative_repo_path: RelativeRepoPath
    resolve_script_path: ResolveScriptPath
    source_of_truth_for_module: SourceOfTruthForModule
    build_catalog_training_task: BuildCatalogTrainingTask
    resolve_example_generation_runtime_paths: RuntimePathResolver
    resolve_scoring_backend_runtime_paths: RuntimePathResolver
