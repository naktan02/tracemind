"""prototype pack 생성 재사용 함수."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from main_server.src.services.prototypes.prototype_rebuild_service import (
    PrototypeRebuildRequest,
    PrototypeRebuildService,
    ReferenceRebuildPrototypePublicationStrategy,
)
from scripts.prototypes.io import (
    group_rows_by_label,
    load_jsonl,
    resolve_metadata_from_manifests,
)
from shared.src.services.prototypes.build_strategies import (
    PrototypeBuildRequest,
    PrototypeBuildStrategy,
    SinglePrototypeBuildStrategy,
    describe_prototype_build_strategy,
)


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

    rebuild_service = PrototypeRebuildService(
        build_strategy=build_strategy,
        publication_strategy=ReferenceRebuildPrototypePublicationStrategy(
            reference_pack_output_dir=output_dir,
            reference_build_state_output_dir=build_state_output_dir,
        ),
    )
    rebuild_result = rebuild_service.rebuild(
        PrototypeRebuildRequest(
            build_request=PrototypeBuildRequest(
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
                built_at=None,
                required_categories=expected_categories or None,
            )
        )
    )
    pack_path = rebuild_result.reference_pack_path
    if pack_path is None:
        raise ValueError(
            "Reference rebuild publication must produce a local pack path."
        )

    manifest_path = output_dir / f"{prototype_version}.manifest.json"
    manifest = {
        "prototype_version": prototype_version,
        "input_jsonl": str(input_jsonl),
        "source_dataset_id": source_dataset_id,
        "backend": backend,
        "embedding_model_id": embedding_model_id,
        "embedding_model_revision": embedding_model_revision,
        "mapping_version": mapping_version,
        "built_at": rebuild_result.pack_payload.built_at.isoformat(),
        "category_counts": dict(sorted(label_counts.items())),
        "expected_categories": expected_categories,
        "prototype_builder": describe_prototype_build_strategy(build_strategy),
        "output_build_state": (
            None
            if rebuild_result.reference_build_state_path is None
            else str(rebuild_result.reference_build_state_path)
        ),
        "output_pack": str(pack_path),
        "main_server_build_state": (
            None
            if rebuild_result.published_build_state_path is None
            else str(rebuild_result.published_build_state_path)
        ),
        "main_server_pack": str(rebuild_result.published_pack_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return (
        pack_path,
        rebuild_result.reference_build_state_path,
        manifest_path,
        rebuild_result.published_pack_path,
        rebuild_result.published_build_state_path,
    )
