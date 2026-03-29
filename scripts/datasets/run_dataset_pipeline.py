"""YAML 레지스트리 기반 데이터셋 파이프라인 러너."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

# 프로젝트 루트를 PYTHONPATH에 추가 (스크립트 직접 실행 지원)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.datasets.build_labeled_query_set import (
    build_labeled_query_set,
    load_mapping_config,
)
from scripts.datasets.download_dataset import (
    build_default_output_path,
    download_huggingface_dataset_to_csv,
)
from scripts.datasets.split_labeled_query_set import build_split_artifacts
from scripts.prototypes.seed_prototypes import seed_prototype_pack

PIPELINE_STAGE_ORDER = ("download", "map", "split", "prototype")


@dataclass(slots=True, frozen=True)
class PipelineDefaults:
    raw_dir: Path
    labeled_query_set_dir: Path
    split_dir: Path
    prototype_pack_dir: Path
    pipeline_run_dir: Path
    cache_dir: Path


@dataclass(slots=True, frozen=True)
class SourceConfig:
    name: str
    kind: str
    dataset_id: str
    split: str
    data_file: str | None = None
    output_filename: str | None = None
    mapping_config: Path | None = None
    reference_urls: tuple[str, ...] = ()
    revision: str | None = None


@dataclass(slots=True, frozen=True)
class SplitConfig:
    source: str
    split_name: str
    validation_ratio: float = 0.1
    seed: int = 42


@dataclass(slots=True, frozen=True)
class PrototypeConfig:
    source: str
    backend: str
    embedding_model_id: str
    embedding_model_revision: str = "main"
    translation_model_id: str | None = None
    translation_model_revision: str | None = None
    translation_direction: str | None = None
    batch_size: int = 16
    device: str = "auto"
    task_prefix: str = ""
    local_files_only: bool = False
    expected_categories: tuple[str, ...] = ()
    prototype_version_prefix: str = "proto"
    hash_dim: int = 256


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    name: str
    description: str
    stages: tuple[str, ...]
    sources: dict[str, SourceConfig]
    split: SplitConfig | None = None
    prototype: PrototypeConfig | None = None


@dataclass(slots=True, frozen=True)
class DatasetRegistry:
    schema_version: str
    defaults: PipelineDefaults
    datasets: dict[str, DatasetConfig]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset preparation stages from a YAML registry."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/datasets/registry.yaml"),
        help="Path to the dataset registry YAML.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=[],
        help="Dataset alias to run. Repeat to run multiple datasets. Defaults to all.",
    )
    parser.add_argument(
        "--only-stage",
        action="append",
        dest="only_stages",
        choices=PIPELINE_STAGE_ORDER,
        default=[],
        help="Limit execution to specific stage(s). Repeat as needed.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List dataset aliases defined in the registry and exit.",
    )
    return parser.parse_args()


def _as_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _resolve_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string field: {key}")
    return value


def load_registry(path: Path) -> DatasetRegistry:
    resolved_path = path if path.is_absolute() else PROJECT_ROOT / path
    payload = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Registry root must be a mapping.")

    defaults_raw = _as_dict(payload.get("defaults", {}), field_name="defaults")
    defaults = PipelineDefaults(
        raw_dir=_resolve_path(_require_string(defaults_raw, "raw_dir")) or PROJECT_ROOT,
        labeled_query_set_dir=_resolve_path(
            _require_string(defaults_raw, "labeled_query_set_dir")
        )
        or PROJECT_ROOT,
        split_dir=_resolve_path(_require_string(defaults_raw, "split_dir")) or PROJECT_ROOT,
        prototype_pack_dir=_resolve_path(
            _require_string(defaults_raw, "prototype_pack_dir")
        )
        or PROJECT_ROOT,
        pipeline_run_dir=_resolve_path(
            _require_string(defaults_raw, "pipeline_run_dir")
        )
        or PROJECT_ROOT,
        cache_dir=_resolve_path(_require_string(defaults_raw, "cache_dir")) or PROJECT_ROOT,
    )

    datasets_raw = _as_dict(payload.get("datasets", {}), field_name="datasets")
    datasets: dict[str, DatasetConfig] = {}
    for dataset_name, dataset_value in sorted(datasets_raw.items()):
        dataset_payload = _as_dict(dataset_value, field_name=f"datasets.{dataset_name}")
        stages = dataset_payload.get("stages", [])
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"datasets.{dataset_name}.stages must be a non-empty list.")
        invalid_stages = [stage for stage in stages if stage not in PIPELINE_STAGE_ORDER]
        if invalid_stages:
            raise ValueError(
                f"datasets.{dataset_name}.stages contains unsupported stages: {invalid_stages}"
            )

        sources_raw = _as_dict(dataset_payload.get("sources", {}), field_name="sources")
        sources: dict[str, SourceConfig] = {}
        for source_name, source_value in sorted(sources_raw.items()):
            source_payload = _as_dict(
                source_value,
                field_name=f"datasets.{dataset_name}.sources.{source_name}",
            )
            reference_urls = source_payload.get("reference_urls", [])
            if not isinstance(reference_urls, list):
                raise ValueError(
                    f"datasets.{dataset_name}.sources.{source_name}.reference_urls must be a list."
                )
            sources[source_name] = SourceConfig(
                name=source_name,
                kind=_require_string(source_payload, "kind"),
                dataset_id=_require_string(source_payload, "dataset_id"),
                split=_require_string(source_payload, "split"),
                data_file=source_payload.get("data_file"),
                output_filename=source_payload.get("output_filename"),
                mapping_config=_resolve_path(source_payload.get("mapping_config")),
                reference_urls=tuple(str(url) for url in reference_urls),
                revision=source_payload.get("revision"),
            )

        split_config = None
        split_raw = dataset_payload.get("split")
        if split_raw is not None:
            split_payload = _as_dict(split_raw, field_name=f"datasets.{dataset_name}.split")
            split_config = SplitConfig(
                source=_require_string(split_payload, "source"),
                split_name=_require_string(split_payload, "split_name"),
                validation_ratio=float(split_payload.get("validation_ratio", 0.1)),
                seed=int(split_payload.get("seed", 42)),
            )

        prototype_config = None
        prototype_raw = dataset_payload.get("prototype")
        if prototype_raw is not None:
            prototype_payload = _as_dict(
                prototype_raw,
                field_name=f"datasets.{dataset_name}.prototype",
            )
            expected_categories = prototype_payload.get("expected_categories", [])
            if not isinstance(expected_categories, list):
                raise ValueError(
                    f"datasets.{dataset_name}.prototype.expected_categories must be a list."
                )
            prototype_config = PrototypeConfig(
                source=_require_string(prototype_payload, "source"),
                backend=_require_string(prototype_payload, "backend"),
                embedding_model_id=_require_string(prototype_payload, "embedding_model_id"),
                embedding_model_revision=str(
                    prototype_payload.get("embedding_model_revision", "main")
                ),
                translation_model_id=prototype_payload.get("translation_model_id"),
                translation_model_revision=prototype_payload.get(
                    "translation_model_revision"
                ),
                translation_direction=prototype_payload.get("translation_direction"),
                batch_size=int(prototype_payload.get("batch_size", 16)),
                device=str(prototype_payload.get("device", "auto")),
                task_prefix=str(prototype_payload.get("task_prefix", "")),
                local_files_only=bool(prototype_payload.get("local_files_only", False)),
                expected_categories=tuple(str(value) for value in expected_categories),
                prototype_version_prefix=str(
                    prototype_payload.get("prototype_version_prefix", "proto")
                ),
                hash_dim=int(prototype_payload.get("hash_dim", 256)),
            )

        datasets[dataset_name] = DatasetConfig(
            name=dataset_name,
            description=str(dataset_payload.get("description", "")),
            stages=tuple(stage for stage in PIPELINE_STAGE_ORDER if stage in stages),
            sources=sources,
            split=split_config,
            prototype=prototype_config,
        )

    return DatasetRegistry(
        schema_version=str(payload.get("schema_version", "")),
        defaults=defaults,
        datasets=datasets,
    )


def resolve_dataset_output_path(
    *,
    source: SourceConfig,
    defaults: PipelineDefaults,
) -> Path:
    if source.output_filename:
        return defaults.raw_dir / source.output_filename
    return build_default_output_path(
        dataset_id=source.dataset_id,
        split=source.split,
        output_dir=defaults.raw_dir,
    )


def resolve_mapped_output_paths(
    *,
    source: SourceConfig,
    defaults: PipelineDefaults,
) -> dict[str, str]:
    if source.mapping_config is None:
        raise ValueError(f"Source '{source.name}' does not define mapping_config.")
    mapping_config = load_mapping_config(source.mapping_config)
    dataset_id = mapping_config["dataset_id"]
    jsonl_path = defaults.labeled_query_set_dir / f"{dataset_id}.jsonl"
    manifest_path = defaults.labeled_query_set_dir / f"{dataset_id}.manifest.json"
    return {
        "jsonl": str(jsonl_path),
        "manifest": str(manifest_path),
    }


def resolve_split_output_paths(
    *,
    split: SplitConfig,
    defaults: PipelineDefaults,
) -> dict[str, str]:
    return {
        "train_jsonl": str(defaults.split_dir / f"{split.split_name}.train.jsonl"),
        "validation_jsonl": str(defaults.split_dir / f"{split.split_name}.validation.jsonl"),
        "manifest": str(defaults.split_dir / f"{split.split_name}.manifest.json"),
    }


def resolve_prototype_input_jsonl(
    *,
    dataset: DatasetConfig,
    mapped_outputs: dict[str, dict[str, str]],
    split_output: dict[str, str] | None,
    defaults: PipelineDefaults,
) -> Path:
    if dataset.prototype is None:
        raise ValueError(f"Dataset '{dataset.name}' has no prototype config.")

    source = dataset.prototype.source
    if source == "split_train":
        if split_output is not None:
            return Path(split_output["train_jsonl"])
        if dataset.split is not None:
            return defaults.split_dir / f"{dataset.split.split_name}.train.jsonl"
        raise ValueError(
            f"Dataset '{dataset.name}' prototype.source=split_train requires split config."
        )

    if source.startswith("mapped:"):
        mapped_name = source.removeprefix("mapped:")
        mapped_output = mapped_outputs.get(mapped_name)
        if mapped_output is not None:
            return Path(mapped_output["jsonl"])
        source_config = dataset.sources.get(mapped_name)
        if source_config is None:
            raise ValueError(
                f"Dataset '{dataset.name}' prototype source '{source}' is unknown."
            )
        if source_config.mapping_config is None:
            raise ValueError(
                f"Dataset '{dataset.name}' source '{mapped_name}' has no mapping_config."
            )
        resolved_outputs = resolve_mapped_output_paths(
            source=source_config,
            defaults=defaults,
        )
        return Path(resolved_outputs["jsonl"])

    raise ValueError(
        f"Dataset '{dataset.name}' has unsupported prototype.source='{source}'."
    )


def run_dataset(
    *,
    dataset: DatasetConfig,
    defaults: PipelineDefaults,
    registry_path: Path,
    only_stages: set[str],
) -> Path:
    selected_stages = [stage for stage in dataset.stages if not only_stages or stage in only_stages]
    if only_stages:
        unsupported_requested = sorted(only_stages.difference(dataset.stages))
        if unsupported_requested:
            raise ValueError(
                f"Dataset '{dataset.name}' does not configure stage(s): {unsupported_requested}"
            )

    raw_outputs: dict[str, str] = {}
    mapped_outputs: dict[str, dict[str, str]] = {}
    split_output: dict[str, str] | None = None
    prototype_output: dict[str, str] | None = None

    if "download" in selected_stages:
        for source_name, source in dataset.sources.items():
            if source.kind != "huggingface":
                raise ValueError(
                    f"Dataset '{dataset.name}' source '{source_name}' has unsupported kind '{source.kind}'."
                )
            output_path = resolve_dataset_output_path(source=source, defaults=defaults)
            download_huggingface_dataset_to_csv(
                dataset_id=source.dataset_id,
                split=source.split,
                output_dir=defaults.raw_dir,
                cache_dir=defaults.cache_dir,
                data_file=source.data_file,
                output_path=output_path,
                revision=source.revision,
            )
            raw_outputs[source_name] = str(output_path)

    if "map" in selected_stages:
        for source_name, source in dataset.sources.items():
            if source.mapping_config is None:
                continue
            raw_csv_path = Path(
                raw_outputs.get(
                    source_name,
                    resolve_dataset_output_path(
                        source=source,
                        defaults=defaults,
                    ),
                )
            )
            jsonl_path, manifest_path = build_labeled_query_set(
                raw_csv_path=raw_csv_path,
                mapping_config_path=source.mapping_config,
                output_dir=defaults.labeled_query_set_dir,
            )
            mapped_outputs[source_name] = {
                "jsonl": str(jsonl_path),
                "manifest": str(manifest_path),
            }

    if "split" in selected_stages:
        if dataset.split is None:
            raise ValueError(f"Dataset '{dataset.name}' does not define split config.")
        mapped_output = mapped_outputs.get(dataset.split.source)
        if mapped_output is None:
            source = dataset.sources.get(dataset.split.source)
            if source is None:
                raise ValueError(
                    f"Dataset '{dataset.name}' split source '{dataset.split.source}' is unknown."
                )
            mapped_output = resolve_mapped_output_paths(
                source=source,
                defaults=defaults,
            )
        if mapped_output is None:
            raise ValueError(
                f"Dataset '{dataset.name}' split source '{dataset.split.source}' was not generated."
            )
        train_path, validation_path, manifest_path = build_split_artifacts(
            input_jsonl=Path(mapped_output["jsonl"]),
            split_name=dataset.split.split_name,
            validation_ratio=dataset.split.validation_ratio,
            seed=dataset.split.seed,
            output_dir=defaults.split_dir,
        )
        split_output = {
            "train_jsonl": str(train_path),
            "validation_jsonl": str(validation_path),
            "manifest": str(manifest_path),
        }
    elif dataset.split is not None:
        split_output = resolve_split_output_paths(
            split=dataset.split,
            defaults=defaults,
        )

    if "prototype" in selected_stages:
        if dataset.prototype is None:
            raise ValueError(f"Dataset '{dataset.name}' does not define prototype config.")
        prototype_input_jsonl = resolve_prototype_input_jsonl(
            dataset=dataset,
            mapped_outputs=mapped_outputs,
            split_output=split_output,
            defaults=defaults,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prototype_version = f"{dataset.prototype.prototype_version_prefix}_{timestamp}"
        (
            pack_path,
            build_state_path,
            manifest_path,
            main_server_pack_path,
            main_server_build_state_path,
        ) = seed_prototype_pack(
            input_jsonl=prototype_input_jsonl,
            output_dir=defaults.prototype_pack_dir,
            build_state_output_dir=defaults.prototype_pack_dir.parent
            / "prototype_build_states",
            prototype_version=prototype_version,
            backend=dataset.prototype.backend,
            embedding_model_id=dataset.prototype.embedding_model_id,
            embedding_model_revision=dataset.prototype.embedding_model_revision,
            translation_model_id=dataset.prototype.translation_model_id,
            translation_model_revision=dataset.prototype.translation_model_revision,
            translation_direction=dataset.prototype.translation_direction,
            batch_size=dataset.prototype.batch_size,
            cache_dir=defaults.cache_dir,
            device=dataset.prototype.device,
            task_prefix=dataset.prototype.task_prefix,
            local_files_only=dataset.prototype.local_files_only,
            expected_categories=list(dataset.prototype.expected_categories),
            hash_dim=dataset.prototype.hash_dim,
        )
        prototype_output = {
            "prototype_build_state": str(build_state_path),
            "prototype_pack": str(pack_path),
            "manifest": str(manifest_path),
            "main_server_build_state": str(main_server_build_state_path),
            "main_server_pack": str(main_server_pack_path),
            "input_jsonl": str(prototype_input_jsonl),
        }

    defaults.pipeline_run_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_manifest_path = defaults.pipeline_run_dir / f"{dataset.name}__{run_timestamp}.manifest.json"
    run_manifest = {
        "schema_version": "dataset_pipeline_run.v1",
        "dataset_name": dataset.name,
        "description": dataset.description,
        "registry_path": str(registry_path),
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "executed_stages": selected_stages,
        "raw_outputs": raw_outputs,
        "mapped_outputs": mapped_outputs,
        "split_output": split_output,
        "prototype_output": prototype_output,
        "sources": {
            source_name: {
                "kind": source.kind,
                "dataset_id": source.dataset_id,
                "split": source.split,
                "data_file": source.data_file,
                "reference_urls": list(source.reference_urls),
                "mapping_config": None
                if source.mapping_config is None
                else str(source.mapping_config),
            }
            for source_name, source in dataset.sources.items()
        },
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return run_manifest_path


def main() -> None:
    args = parse_args()
    registry = load_registry(args.registry)

    if args.list_datasets:
        for dataset_name, dataset in registry.datasets.items():
            print(f"{dataset_name}\t{','.join(dataset.stages)}")
        return

    selected_dataset_names = args.datasets or list(registry.datasets.keys())
    for dataset_name in selected_dataset_names:
        dataset = registry.datasets.get(dataset_name)
        if dataset is None:
            raise ValueError(f"Unknown dataset alias: {dataset_name}")
        run_manifest_path = run_dataset(
            dataset=dataset,
            defaults=registry.defaults,
            registry_path=(args.registry if args.registry.is_absolute() else PROJECT_ROOT / args.registry),
            only_stages=set(args.only_stages),
        )
        print(f"dataset={dataset_name} run_manifest={run_manifest_path}")


if __name__ == "__main__":
    main()
