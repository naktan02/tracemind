"""신규 labeled query set으로 single-centroid build state를 갱신한다."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from agent.src.infrastructure.model_adapters.embedding.factory import (  # noqa: E402
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from main_server.src.services.prototypes.prototype_build_state_service import (  # noqa: E402
    PrototypeBuildStateService,
)
from main_server.src.services.prototypes.prototype_pack_service import (  # noqa: E402
    PrototypePackService,
)
from scripts.prototypes.io import (  # noqa: E402
    group_rows_by_label,
    load_jsonl,
    resolve_metadata_from_manifests,
)
from shared.src.contracts.prototype_build_state_contracts import (  # noqa: E402
    PrototypeBuildStatePayload,
    dump_prototype_build_state_payload,
    load_prototype_build_state_payload,
)
from shared.src.contracts.prototype_contracts import (  # noqa: E402
    dump_prototype_pack_payload,
)
from shared.src.services.prototypes.payload_serialization import (  # noqa: E402
    build_single_prototype_pack_payload,
)
from shared.src.services.prototypes.prototype_pack_builder import (  # noqa: E402
    PrototypePackBuilder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally update a single-centroid prototype build state with new "
            "labeled JSONL data."
        )
    )
    parser.add_argument(
        "--base-build-state",
        required=True,
        type=Path,
        help="Path to the existing prototype build state JSON file.",
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        type=Path,
        help="Path to the new labeled query set JSONL.",
    )
    parser.add_argument(
        "--prototype-version",
        default="",
        help="New prototype version. Defaults to a UTC timestamp-based version.",
    )
    parser.add_argument(
        "--prototype-pack-output-dir",
        type=Path,
        default=Path("data/processed/prototype_packs"),
        help="Directory where updated prototype packs are written.",
    )
    parser.add_argument(
        "--build-state-output-dir",
        type=Path,
        default=Path("data/processed/prototype_build_states"),
        help="Directory where updated build states are written.",
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
        help="Cache directory for model files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device for the embedding backend (auto/cpu/cuda/cuda:0/mps).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model files only from the local cache.",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=256,
        help="Vector dimension for the hash_debug backend.",
    )
    return parser.parse_args()


def update_prototype_build_state(
    *,
    base_build_state_path: Path,
    input_jsonl: Path,
    prototype_pack_output_dir: Path,
    build_state_output_dir: Path,
    prototype_version: str,
    batch_size: int,
    cache_dir: Path,
    device: str,
    local_files_only: bool,
    hash_dim: int,
) -> tuple[Path, Path, Path, Path, Path]:
    base_state = load_prototype_build_state_payload(base_build_state_path)
    rows = load_jsonl(input_jsonl)
    if not rows:
        raise ValueError("input_jsonl must contain at least one labeled row.")

    mapping_version, source_dataset_id = resolve_metadata_from_manifests(input_jsonl)
    if mapping_version != base_state.mapping_version:
        raise ValueError(
            "mapping_version mismatch between base build state and new labeled data."
        )

    rows_by_label = group_rows_by_label(rows)
    base_categories = sorted(base_state.categories)
    unexpected_categories = sorted(
        category for category in rows_by_label if category not in base_state.categories
    )
    if unexpected_categories:
        raise ValueError(
            "New labeled data contains categories not present in the base "
            f"build state: {unexpected_categories}"
        )

    adapter = EmbeddingAdapterFactory.create(
        EmbeddingAdapterSpec(
            backend=base_state.embedding_backend,
            model_id=base_state.embedding_model_id,
            revision=base_state.embedding_model_revision,
            device=device,
            batch_size=batch_size,
            cache_dir=str(cache_dir),
            task_prefix=base_state.task_prefix,
            normalize_embeddings=base_state.normalize_embeddings,
            hash_dim=hash_dim,
            local_files_only=local_files_only,
        )
    )

    embeddings_by_category: dict[str, list[list[float]]] = {}
    added_label_counts: Counter[str] = Counter()
    for category, label_rows in rows_by_label.items():
        texts = [row["text"] for row in label_rows]
        print(f"embedding_category={category} rows={len(texts)}", flush=True)
        embeddings = adapter.embed_texts(texts)
        embeddings_by_category[category] = embeddings
        added_label_counts[category] = len(label_rows)
        print(f"embedded_category={category} rows={len(texts)}", flush=True)

    built_at = datetime.now(timezone.utc)
    builder = PrototypePackBuilder()
    merged_state = builder.merge_build_state(
        base_state,
        embeddings_by_category,
        prototype_version=prototype_version,
        built_at=built_at,
        required_categories=base_categories,
    )
    pack = builder.build_pack_from_state(merged_state)

    build_state_output_dir.mkdir(parents=True, exist_ok=True)
    prototype_pack_output_dir.mkdir(parents=True, exist_ok=True)
    build_state_path = build_state_output_dir / f"{prototype_version}.json"
    pack_path = prototype_pack_output_dir / f"{prototype_version}.json"
    manifest_path = prototype_pack_output_dir / f"{prototype_version}.manifest.json"

    build_state_payload = PrototypeBuildStatePayload.model_validate(merged_state)
    dump_prototype_build_state_payload(build_state_path, build_state_payload)
    pack_payload = build_single_prototype_pack_payload(pack)
    dump_prototype_pack_payload(pack_path, pack_payload)

    main_server_build_state_path = PrototypeBuildStateService().publish_state(
        build_state_payload
    )
    main_server_pack_path = PrototypePackService().publish_pack(pack_payload)
    manifest = {
        "prototype_version": prototype_version,
        "base_build_state": str(base_build_state_path),
        "input_jsonl": str(input_jsonl),
        "source_dataset_id": source_dataset_id,
        "mapping_version": mapping_version,
        "built_at": built_at.isoformat(),
        "embedding_backend": base_state.embedding_backend,
        "embedding_model_id": base_state.embedding_model_id,
        "embedding_model_revision": base_state.embedding_model_revision,
        "added_category_counts": dict(sorted(added_label_counts.items())),
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
        build_state_path,
        pack_path,
        manifest_path,
        main_server_build_state_path,
        main_server_pack_path,
    )


def main() -> None:
    args = parse_args()
    built_at = datetime.now(timezone.utc)
    prototype_version = args.prototype_version or built_at.strftime(
        "proto_%Y_%m_%d_%H%M%S"
    )
    (
        build_state_path,
        pack_path,
        manifest_path,
        main_server_build_state_path,
        main_server_pack_path,
    ) = update_prototype_build_state(
        base_build_state_path=args.base_build_state,
        input_jsonl=args.input_jsonl,
        prototype_pack_output_dir=args.prototype_pack_output_dir,
        build_state_output_dir=args.build_state_output_dir,
        prototype_version=prototype_version,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        device=args.device,
        local_files_only=args.local_files_only,
        hash_dim=args.hash_dim,
    )
    print(f"prototype_build_state={build_state_path}")
    print(f"prototype_pack={pack_path}")
    print(f"manifest={manifest_path}")
    print(f"main_server_build_state={main_server_build_state_path}")
    print(f"main_server_pack={main_server_pack_path}")


if __name__ == "__main__":
    main()
