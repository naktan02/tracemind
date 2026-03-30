"""labeled query set으로 prototype pack을 생성한다."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# 프로젝트 루트를 PYTHONPATH에 추가 (스크립트 직접 실행 지원)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_SERVER_ROOT = PROJECT_ROOT / "main-server"
if str(MAIN_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_SERVER_ROOT))

from src.services.prototypes.prototype_build_state_service import (  # noqa: E402
    PrototypeBuildStateService,
)
from src.services.prototypes.prototype_pack_service import (  # noqa: E402
    PrototypePackService,
)

from agent.src.infrastructure.model_adapters.embedding.factory import (  # noqa: E402
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from scripts.prototypes.build_strategies import (  # noqa: E402
    PrototypeBuildRequest,
    PrototypeBuildStrategy,
    SinglePrototypeBuildStrategy,
    describe_prototype_build_strategy,
)
from shared.src.contracts.prototype_build_state_contracts import (  # noqa: E402
    PrototypeBuildStatePayload,
)
from shared.src.domain.entities.artifacts.prototype_pack import (  # noqa: E402
    PrototypePack,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_metadata_from_manifests(input_jsonl: Path) -> tuple[str, str | None]:
    direct_manifest = load_json(input_jsonl.with_suffix(".manifest.json"))
    if direct_manifest is not None and direct_manifest.get("mapping_version"):
        return direct_manifest["mapping_version"], direct_manifest.get("dataset_id")

    split_manifest_candidates: list[Path] = []
    for suffix in (".train.jsonl", ".validation.jsonl"):
        if input_jsonl.name.endswith(suffix):
            split_manifest_candidates.append(
                input_jsonl.parent
                / input_jsonl.name.replace(suffix, ".manifest.json")
            )
    for manifest_path in split_manifest_candidates:
        split_manifest = load_json(manifest_path)
        if split_manifest is not None and split_manifest.get("source_mapping_version"):
            return (
                split_manifest["source_mapping_version"],
                split_manifest.get("source_dataset_id"),
            )

    raise ValueError(
        "Could not resolve mapping_version from manifests. "
        "Generate the labeled_query_set manifest or split manifest first."
    )


def group_rows_by_label(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row["mapped_label_4"]].append(row)
    return dict(sorted(rows_by_label.items()))


def build_pack_dict(pack: PrototypePack) -> dict[str, Any]:
    return {
        "schema_version": pack.schema_version,
        "prototype_version": pack.prototype_version,
        "embedding_model_id": pack.embedding_model_id,
        "embedding_model_revision": pack.embedding_model_revision,
        "translation_model_id": pack.translation_model_id,
        "translation_model_revision": pack.translation_model_revision,
        "translation_direction": pack.translation_direction,
        "mapping_version": pack.mapping_version,
        "build_method": pack.build_method,
        "distance_metric": pack.distance_metric,
        "built_at": pack.built_at.isoformat(),
        "categories": {
            category: {
                "centroid": prototype.centroid,
                "sample_count": prototype.sample_count,
            }
            for category, prototype in pack.categories.items()
        },
    }


def build_state_dict(build_state: PrototypeBuildStatePayload) -> dict[str, Any]:
    return {
        "schema_version": build_state.schema_version,
        "prototype_version": build_state.prototype_version,
        "embedding_backend": build_state.embedding_backend,
        "embedding_model_id": build_state.embedding_model_id,
        "embedding_model_revision": build_state.embedding_model_revision,
        "normalize_embeddings": build_state.normalize_embeddings,
        "task_prefix": build_state.task_prefix,
        "translation_model_id": build_state.translation_model_id,
        "translation_model_revision": build_state.translation_model_revision,
        "translation_direction": build_state.translation_direction,
        "mapping_version": build_state.mapping_version,
        "build_method": build_state.build_method,
        "distance_metric": build_state.distance_metric,
        "built_at": build_state.built_at.isoformat(),
        "categories": {
            category: {
                "embedding_sum": category_state.embedding_sum,
                "sample_count": category_state.sample_count,
            }
            for category, category_state in build_state.categories.items()
        },
    }


def seed_prototype_pack(
    *,
    input_jsonl: Path,
    output_dir: Path,
    build_state_output_dir: Path,
    prototype_version: str,
    backend: str,
    embedding_model_id: str,
    embedding_model_revision: str,
    translation_model_id: str | None,
    translation_model_revision: str | None,
    translation_direction: str | None,
    batch_size: int,
    cache_dir: Path,
    device: str,
    task_prefix: str,
    local_files_only: bool,
    expected_categories: list[str],
    hash_dim: int,
    build_strategy: PrototypeBuildStrategy | None = None,
) -> tuple[Path, Path | None, Path, Path, Path | None]:
    rows = load_jsonl(input_jsonl)
    rows_by_label = group_rows_by_label(rows)
    mapping_version, source_dataset_id = resolve_metadata_from_manifests(input_jsonl)
    if build_strategy is None:
        build_strategy = SinglePrototypeBuildStrategy()
    adapter = EmbeddingAdapterFactory.create(
        EmbeddingAdapterSpec(
            backend=backend,
            model_id=embedding_model_id,
            revision=embedding_model_revision,
            device=device,
            batch_size=batch_size,
            cache_dir=str(cache_dir),
            task_prefix=task_prefix,
            hash_dim=hash_dim,
            local_files_only=local_files_only,
        )
    )

    embeddings_by_category: dict[str, list[list[float]]] = {}
    label_counts: Counter[str] = Counter()
    for category, label_rows in rows_by_label.items():
        texts = [row["text"] for row in label_rows]
        print(f"embedding_category={category} rows={len(texts)}", flush=True)
        embeddings = adapter.embed_texts(texts)
        embeddings_by_category[category] = embeddings
        label_counts[category] = len(label_rows)
        print(f"embedded_category={category} rows={len(texts)}", flush=True)

    built_at = datetime.now(timezone.utc)
    build_result = build_strategy.build(
        PrototypeBuildRequest(
            embeddings_by_category=embeddings_by_category,
            prototype_version=prototype_version,
            embedding_backend=backend,
            embedding_model_id=embedding_model_id,
            embedding_model_revision=embedding_model_revision,
            normalize_embeddings=True,
            task_prefix=task_prefix,
            translation_model_id=translation_model_id,
            translation_model_revision=translation_model_revision,
            translation_direction=translation_direction,
            mapping_version=mapping_version,
            built_at=built_at,
            required_categories=expected_categories or None,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    build_state_output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / f"{prototype_version}.json"
    build_state_path: Path | None = None
    manifest_path = output_dir / f"{prototype_version}.manifest.json"
    build_state_payload = build_result.build_state_payload
    if build_state_payload is not None:
        build_state_path = build_state_output_dir / f"{prototype_version}.json"
        build_state_payload = PrototypeBuildStatePayload.model_validate(
            build_state_payload.model_dump(mode="python")
        )
        build_state_path.write_text(
            json.dumps(
                build_state_payload.model_dump(mode="json"),
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
    pack_payload = build_result.pack_payload
    pack_path.write_text(
        json.dumps(
            pack_payload.model_dump(mode="json"),
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    main_server_build_state_path: Path | None = None
    if build_state_payload is not None:
        main_server_build_state_path = PrototypeBuildStateService().publish_state(
            build_state_payload
        )
    main_server_pack_path = PrototypePackService().publish_pack(pack_payload)
    manifest = {
        "prototype_version": prototype_version,
        "input_jsonl": str(input_jsonl),
        "source_dataset_id": source_dataset_id,
        "backend": backend,
        "embedding_model_id": embedding_model_id,
        "embedding_model_revision": embedding_model_revision,
        "mapping_version": mapping_version,
        "built_at": built_at.isoformat(),
        "category_counts": dict(sorted(label_counts.items())),
        "expected_categories": expected_categories,
        "prototype_builder": describe_prototype_build_strategy(build_strategy),
        "output_build_state": (
            None if build_state_path is None else str(build_state_path)
        ),
        "output_pack": str(pack_path),
        "main_server_build_state": (
            None
            if main_server_build_state_path is None
            else str(main_server_build_state_path)
        ),
        "main_server_pack": str(main_server_pack_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return (
        pack_path,
        build_state_path,
        manifest_path,
        main_server_pack_path,
        main_server_build_state_path,
    )


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="prototypes/seed_prototypes",
)
def main(cfg: DictConfig) -> None:
    built_at = datetime.now(timezone.utc)
    prototype_version = (
        cfg.prototype_version
        or built_at.strftime("proto_%Y_%m_%d_%H%M%S")
    )
    embedding_spec = instantiate(cfg.embedding.spec)
    build_strategy = instantiate(cfg.prototype_builder)
    (
        pack_path,
        build_state_path,
        manifest_path,
        main_server_pack_path,
        main_server_build_state_path,
    ) = seed_prototype_pack(
        input_jsonl=Path(cfg.input_jsonl),
        output_dir=Path(cfg.output_dir),
        build_state_output_dir=Path(cfg.build_state_output_dir),
        prototype_version=prototype_version,
        backend=embedding_spec.backend,
        embedding_model_id=embedding_spec.model_id,
        embedding_model_revision=embedding_spec.revision,
        translation_model_id=cfg.translation_model_id,
        translation_model_revision=cfg.translation_model_revision,
        translation_direction=cfg.translation_direction,
        batch_size=embedding_spec.batch_size,
        cache_dir=Path(embedding_spec.cache_dir or "hf_cache"),
        device=embedding_spec.device,
        task_prefix=embedding_spec.task_prefix,
        local_files_only=embedding_spec.local_files_only,
        expected_categories=list(cfg.expected_categories),
        hash_dim=embedding_spec.hash_dim,
        build_strategy=build_strategy,
    )
    print(f"prototype_build_state={build_state_path}")
    print(f"prototype_pack={pack_path}")
    print(f"manifest={manifest_path}")
    print(f"main_server_build_state={main_server_build_state_path}")
    print(f"main_server_pack={main_server_pack_path}")


if __name__ == "__main__":
    main()
