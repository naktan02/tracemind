"""labeled query set으로 prototype pack을 생성한다."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 프로젝트 루트를 PYTHONPATH에 추가 (스크립트 직접 실행 지원)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAIN_SERVER_ROOT = PROJECT_ROOT / "main-server"
if str(MAIN_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_SERVER_ROOT))

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from shared.src.contracts.prototype_build_state_contracts import (
    PrototypeBuildStatePayload,
)
from shared.src.contracts.prototype_contracts import PrototypePackPayload
from shared.src.domain.entities.artifacts.prototype_pack import PrototypePack
from scripts.prototypes.prototype_pack_builder import PrototypePackBuilder
from src.services.prototypes.prototype_build_state_service import (
    PrototypeBuildStateService,
)
from src.services.prototypes.prototype_pack_service import PrototypePackService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a PrototypePack from a labeled query set JSONL."
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
        help="Path to the labeled query set train JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/prototype_packs"),
        help="Directory where prototype pack outputs are written.",
    )
    parser.add_argument(
        "--build-state-output-dir",
        type=Path,
        default=Path("data/processed/prototype_build_states"),
        help="Directory where prototype build-state outputs are written.",
    )
    parser.add_argument(
        "--prototype-version",
        default="",
        help="Prototype pack version. Defaults to a UTC timestamp-based version.",
    )
    parser.add_argument(
        "--backend",
        choices=EmbeddingAdapterFactory.supported_backends(),
        default="hash_debug",
        help="Embedding backend used to build centroids.",
    )
    parser.add_argument(
        "--embedding-model-id",
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="Embedding model identifier recorded in the output pack.",
    )
    parser.add_argument(
        "--embedding-model-revision",
        default="main",
        help="Embedding model revision recorded in the output pack.",
    )
    parser.add_argument(
        "--translation-model-id",
        default=None,
        help="Optional translation model identifier.",
    )
    parser.add_argument(
        "--translation-model-revision",
        default=None,
        help="Optional translation model revision.",
    )
    parser.add_argument(
        "--translation-direction",
        default=None,
        help="Optional translation direction metadata.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the embedding backend.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("hf_cache"),
        help="Cache directory for model downloads.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device for the embedding backend (auto/cpu/cuda/cuda:0/mps).",
    )
    parser.add_argument(
        "--task-prefix",
        default="",
        help="Optional prefix added before each text for embedding models.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model files only from the local cache.",
    )
    parser.add_argument(
        "--expected-category",
        action="append",
        dest="expected_categories",
        default=[],
        help="Expected mapped_label_4 category. Repeat for multiple categories.",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=256,
        help="Vector dimension for the hash_debug backend.",
    )
    return parser.parse_args()


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
) -> tuple[Path, Path, Path, Path, Path]:
    rows = load_jsonl(input_jsonl)
    rows_by_label = group_rows_by_label(rows)
    mapping_version, source_dataset_id = resolve_metadata_from_manifests(input_jsonl)
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
    builder = PrototypePackBuilder()
    build_state = builder.build_state(
        embeddings_by_category,
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
    pack = builder.build_pack_from_state(build_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    build_state_output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / f"{prototype_version}.json"
    build_state_path = build_state_output_dir / f"{prototype_version}.json"
    manifest_path = output_dir / f"{prototype_version}.manifest.json"
    build_state_payload = PrototypeBuildStatePayload.model_validate(
        build_state_dict(build_state)
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
    pack_payload = PrototypePackPayload.model_validate(build_pack_dict(pack))
    pack_path.write_text(
        json.dumps(pack_payload.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
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
        "output_build_state": str(build_state_path),
        "output_pack": str(pack_path),
        "main_server_build_state": str(main_server_build_state_path),
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


def main() -> None:
    args = parse_args()
    built_at = datetime.now(timezone.utc)
    prototype_version = args.prototype_version or built_at.strftime("proto_%Y_%m_%d_%H%M%S")
    (
        pack_path,
        build_state_path,
        manifest_path,
        main_server_pack_path,
        main_server_build_state_path,
    ) = seed_prototype_pack(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        build_state_output_dir=args.build_state_output_dir,
        prototype_version=prototype_version,
        backend=args.backend,
        embedding_model_id=args.embedding_model_id,
        embedding_model_revision=args.embedding_model_revision,
        translation_model_id=args.translation_model_id,
        translation_model_revision=args.translation_model_revision,
        translation_direction=args.translation_direction,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        device=args.device,
        task_prefix=args.task_prefix,
        local_files_only=args.local_files_only,
        expected_categories=args.expected_categories,
        hash_dim=args.hash_dim,
    )
    print(f"prototype_build_state={build_state_path}")
    print(f"prototype_pack={pack_path}")
    print(f"manifest={manifest_path}")
    print(f"main_server_build_state={main_server_build_state_path}")
    print(f"main_server_pack={main_server_pack_path}")


if __name__ == "__main__":
    main()
