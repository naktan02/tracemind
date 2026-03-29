"""Prototype strategy 실험 설정 로딩."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "scripts/experiments/configs/prototype_strategy/default.yaml"
)


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    train_jsonl: Path
    validation_jsonl: Path
    test_jsonl: Path


@dataclass(slots=True, frozen=True)
class EmbeddingConfig:
    backend: str
    model_id: str
    revision: str
    batch_size: int
    cache_dir: Path
    device: str
    task_prefix: str
    local_files_only: bool
    hash_dim: int


@dataclass(slots=True, frozen=True)
class ThresholdConfig:
    confidence_threshold: float
    margin_threshold: float


@dataclass(slots=True, frozen=True)
class ProjectionConfig:
    sample_size: int
    reducers: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class KMeansConfig:
    candidate_ks: tuple[int, ...]
    silhouette_sample_size: int


@dataclass(slots=True, frozen=True)
class DbscanConfig:
    eps_values: tuple[float, ...]
    min_samples_values: tuple[int, ...]
    search_sample_size: int
    min_cluster_coverage: float


@dataclass(slots=True, frozen=True)
class StrategyConfig:
    kmeans: KMeansConfig
    dbscan: DbscanConfig


@dataclass(slots=True, frozen=True)
class OutputConfig:
    output_dir: Path


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    seed: int


@dataclass(slots=True, frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    thresholds: ThresholdConfig
    projection: ProjectionConfig
    strategies: StrategyConfig
    output: OutputConfig
    runtime: RuntimeConfig


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _require_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _require_list(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list.")
    return value


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {key}")
    return value


def _build_config(payload: dict[str, Any]) -> ExperimentConfig:
    dataset_payload = _require_dict(payload.get("dataset", {}), field_name="dataset")
    embedding_payload = _require_dict(
        payload.get("embedding", {}),
        field_name="embedding",
    )
    thresholds_payload = _require_dict(
        payload.get("thresholds", {}),
        field_name="thresholds",
    )
    projection_payload = _require_dict(
        payload.get("projection", {}),
        field_name="projection",
    )
    strategies_payload = _require_dict(
        payload.get("strategies", {}),
        field_name="strategies",
    )
    kmeans_payload = _require_dict(
        strategies_payload.get("kmeans", {}),
        field_name="strategies.kmeans",
    )
    dbscan_payload = _require_dict(
        strategies_payload.get("dbscan", {}),
        field_name="strategies.dbscan",
    )
    output_payload = _require_dict(payload.get("output", {}), field_name="output")
    runtime_payload = _require_dict(payload.get("runtime", {}), field_name="runtime")

    reducers = tuple(
        str(value)
        for value in _require_list(
            projection_payload.get("reducers", []),
            field_name="projection.reducers",
        )
    )

    return ExperimentConfig(
        dataset=DatasetConfig(
            train_jsonl=_resolve_path(_require_string(dataset_payload, "train_jsonl")),
            validation_jsonl=_resolve_path(
                _require_string(dataset_payload, "validation_jsonl")
            ),
            test_jsonl=_resolve_path(_require_string(dataset_payload, "test_jsonl")),
        ),
        embedding=EmbeddingConfig(
            backend=_require_string(embedding_payload, "backend"),
            model_id=_require_string(embedding_payload, "model_id"),
            revision=str(embedding_payload.get("revision", "main")),
            batch_size=int(embedding_payload.get("batch_size", 16)),
            cache_dir=_resolve_path(_require_string(embedding_payload, "cache_dir")),
            device=str(embedding_payload.get("device", "auto")),
            task_prefix=str(embedding_payload.get("task_prefix", "")),
            local_files_only=bool(embedding_payload.get("local_files_only", False)),
            hash_dim=int(embedding_payload.get("hash_dim", 256)),
        ),
        thresholds=ThresholdConfig(
            confidence_threshold=float(
                thresholds_payload.get("confidence_threshold", 0.8)
            ),
            margin_threshold=float(thresholds_payload.get("margin_threshold", 0.15)),
        ),
        projection=ProjectionConfig(
            sample_size=int(projection_payload.get("sample_size", 5000)),
            reducers=reducers,
        ),
        strategies=StrategyConfig(
            kmeans=KMeansConfig(
                candidate_ks=tuple(
                    int(value)
                    for value in _require_list(
                        kmeans_payload.get("candidate_ks", []),
                        field_name="strategies.kmeans.candidate_ks",
                    )
                ),
                silhouette_sample_size=int(
                    kmeans_payload.get("silhouette_sample_size", 2000)
                ),
            ),
            dbscan=DbscanConfig(
                eps_values=tuple(
                    float(value)
                    for value in _require_list(
                        dbscan_payload.get("eps_values", []),
                        field_name="strategies.dbscan.eps_values",
                    )
                ),
                min_samples_values=tuple(
                    int(value)
                    for value in _require_list(
                        dbscan_payload.get("min_samples_values", []),
                        field_name="strategies.dbscan.min_samples_values",
                    )
                ),
                search_sample_size=int(dbscan_payload.get("search_sample_size", 3000)),
                min_cluster_coverage=float(
                    dbscan_payload.get("min_cluster_coverage", 0.6)
                ),
            ),
        ),
        output=OutputConfig(
            output_dir=_resolve_path(_require_string(output_payload, "output_dir"))
        ),
        runtime=RuntimeConfig(
            seed=int(runtime_payload.get("seed", 42)),
        ),
    )


def load_experiment_config(
    *,
    config_path: Path | None = None,
    overrides: tuple[str, ...] = (),
) -> ExperimentConfig:
    """YAML 설정과 dotlist override를 병합해 typed config를 만든다."""
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    yaml_config = OmegaConf.load(resolved_path)
    override_config = OmegaConf.from_dotlist(list(overrides))
    merged = OmegaConf.merge(yaml_config, override_config)
    payload = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Experiment config root must be a mapping.")
    return _build_config(payload)
