"""실험 결과 아티팩트를 catalog preset처럼 재사용 가능하게 노출한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from main_server.src.services.experiment_workspace.payloads import (
    CatalogItemPayload,
)


def build_generated_lora_train_source_items(
    *,
    repo_root: Path,
    relative_repo_path,
) -> tuple[CatalogItemPayload, ...]:
    items: list[CatalogItemPayload] = []

    for manifest_path in _iter_teacher_split_manifests(repo_root):
        payload = _load_json_object(manifest_path)
        bootstrap_version = str(
            payload.get("bootstrap_version", manifest_path.parent.name)
        )
        train_jsonl = _string_or_none(payload.get("teacher_seed_jsonl"))
        if train_jsonl is None:
            continue
        items.append(
            _build_generated_preset_item(
                item_name=f"generated_lora_train_source__{bootstrap_version}",
                display_name=f"saved: {bootstrap_version}",
                family_name="train_source",
                core_method_name="artifact_train_source",
                preset_group="lora_train_source",
                compiled_selector_name="dataset_train",
                source_of_truth=relative_repo_path(manifest_path),
                description="기존 bootstrap split에서 생성된 seed train JSONL입니다.",
                default_override_patch={"train_jsonl": train_jsonl},
                metadata={
                    "generated_from": "teacher_split_manifest",
                    "train_jsonl": train_jsonl,
                    "unlabeled_ratio": payload.get("unlabeled_ratio"),
                    "teacher_seed_row_count": payload.get("teacher_seed_row_count"),
                },
            )
        )

    for subset_dir in _iter_query_ssl_subset_dirs(repo_root):
        seed_path = _find_first(subset_dir, "seed_train*.jsonl")
        if seed_path is None:
            continue
        subset_name = subset_dir.name
        items.append(
            _build_generated_preset_item(
                item_name=f"generated_lora_train_source__{subset_name}",
                display_name=f"saved subset: {subset_name}",
                family_name="train_source",
                core_method_name="artifact_train_source",
                preset_group="lora_train_source",
                compiled_selector_name="dataset_train",
                source_of_truth=relative_repo_path(seed_path),
                description=(
                    "기존 query SSL subset에서 재사용하는 labeled seed train입니다."
                ),
                default_override_patch={
                    "train_jsonl": relative_repo_path(seed_path),
                },
                metadata={
                    "generated_from": "query_ssl_subset",
                    "train_jsonl": relative_repo_path(seed_path),
                },
            )
        )

    return tuple(sorted(items, key=lambda item: item.display_name, reverse=True))


def build_generated_query_ssl_train_source_items(
    *,
    repo_root: Path,
    relative_repo_path,
) -> tuple[CatalogItemPayload, ...]:
    items: list[CatalogItemPayload] = []

    for manifest_path in _iter_teacher_split_manifests(repo_root):
        payload = _load_json_object(manifest_path)
        bootstrap_version = str(
            payload.get("bootstrap_version", manifest_path.parent.name)
        )
        train_jsonl = _string_or_none(payload.get("teacher_seed_jsonl"))
        unlabeled_jsonl = _string_or_none(payload.get("teacher_unlabeled_jsonl"))
        if train_jsonl is None or unlabeled_jsonl is None:
            continue
        items.append(
            _build_generated_preset_item(
                item_name=f"generated_query_ssl_train_source__{bootstrap_version}",
                display_name=f"saved: {bootstrap_version}",
                family_name="train_source",
                core_method_name="artifact_query_ssl_source",
                preset_group="query_ssl_train_source",
                compiled_selector_name="dataset_default",
                source_of_truth=relative_repo_path(manifest_path),
                description=(
                    "기존 bootstrap split에서 생성된 labeled seed + unlabeled "
                    "pool입니다."
                ),
                default_override_patch={
                    "train_jsonl": train_jsonl,
                    "unlabeled_jsonl": unlabeled_jsonl,
                },
                metadata={
                    "generated_from": "teacher_split_manifest",
                    "train_jsonl": train_jsonl,
                    "unlabeled_jsonl": unlabeled_jsonl,
                    "teacher_seed_row_count": payload.get("teacher_seed_row_count"),
                    "teacher_unlabeled_row_count": payload.get(
                        "teacher_unlabeled_row_count"
                    ),
                },
            )
        )

    for subset_dir in _iter_query_ssl_subset_dirs(repo_root):
        seed_path = _find_first(subset_dir, "seed_train*.jsonl")
        unlabeled_path = _find_first(subset_dir, "unlabeled_pool*.jsonl")
        if seed_path is None or unlabeled_path is None:
            continue
        subset_name = subset_dir.name
        items.append(
            _build_generated_preset_item(
                item_name=f"generated_query_ssl_train_source__{subset_name}",
                display_name=f"saved subset: {subset_name}",
                family_name="train_source",
                core_method_name="artifact_query_ssl_source",
                preset_group="query_ssl_train_source",
                compiled_selector_name="dataset_default",
                source_of_truth=relative_repo_path(subset_dir),
                description=(
                    "기존 query SSL subset에서 재사용하는 labeled seed + "
                    "unlabeled pool입니다."
                ),
                default_override_patch={
                    "train_jsonl": relative_repo_path(seed_path),
                    "unlabeled_jsonl": relative_repo_path(unlabeled_path),
                },
                metadata={
                    "generated_from": "query_ssl_subset",
                    "train_jsonl": relative_repo_path(seed_path),
                    "unlabeled_jsonl": relative_repo_path(unlabeled_path),
                },
            )
        )

    return tuple(sorted(items, key=lambda item: item.display_name, reverse=True))


def build_generated_bootstrap_teacher_source_items(
    *,
    repo_root: Path,
    relative_repo_path,
) -> tuple[CatalogItemPayload, ...]:
    items: list[CatalogItemPayload] = []
    for manifest_path in _iter_teacher_split_manifests(repo_root):
        payload = _load_json_object(manifest_path)
        bootstrap_version = str(
            payload.get("bootstrap_version", manifest_path.parent.name)
        )
        teacher_train_jsonl = _string_or_none(payload.get("teacher_seed_jsonl"))
        teacher_unlabeled_jsonl = _string_or_none(
            payload.get("teacher_unlabeled_jsonl")
        )
        if teacher_train_jsonl is None or teacher_unlabeled_jsonl is None:
            continue
        items.append(
            _build_generated_preset_item(
                item_name=f"generated_bootstrap_teacher_source__{bootstrap_version}",
                display_name=f"saved: {bootstrap_version}",
                family_name="bootstrap_teacher_source",
                core_method_name="artifact_bootstrap_teacher_source",
                preset_group="bootstrap_teacher_source",
                compiled_selector_name="dataset_default",
                source_of_truth=relative_repo_path(manifest_path),
                description="기존 teacher bootstrap split 결과를 다시 사용합니다.",
                default_override_patch={
                    "teacher_train_jsonl": teacher_train_jsonl,
                    "teacher_unlabeled_jsonl": teacher_unlabeled_jsonl,
                },
                metadata={
                    "generated_from": "teacher_split_manifest",
                    "teacher_train_jsonl": teacher_train_jsonl,
                    "teacher_unlabeled_jsonl": teacher_unlabeled_jsonl,
                    "teacher_seed_row_count": payload.get("teacher_seed_row_count"),
                    "teacher_unlabeled_row_count": payload.get(
                        "teacher_unlabeled_row_count"
                    ),
                },
            )
        )
    return tuple(sorted(items, key=lambda item: item.display_name, reverse=True))


def build_generated_initial_checkpoint_items(
    *,
    repo_root: Path,
    relative_repo_path,
) -> tuple[CatalogItemPayload, ...]:
    items: list[CatalogItemPayload] = []

    fixed_classifier_dir = repo_root / "data/processed/classifier_heads"
    for manifest_path in sorted(fixed_classifier_dir.glob("*.manifest.json")):
        payload = _load_json_object(manifest_path)
        classifier_version = str(
            payload.get(
                "classifier_version",
                manifest_path.stem.removesuffix(".manifest"),
            )
        )
        items.append(
            _build_generated_preset_item(
                item_name=f"generated_initial_checkpoint__fixed__{classifier_version}",
                display_name=f"fixed seed: {classifier_version}",
                family_name="initial_checkpoint",
                core_method_name="fixed_classifier_manifest",
                preset_group="query_adaptation_initial_checkpoint",
                compiled_selector_name="required",
                source_of_truth=relative_repo_path(manifest_path),
                description=(
                    "기존 고정 분류기 seed manifest를 초기 체크포인트로 사용합니다."
                ),
                default_override_patch={
                    "manifest_path": relative_repo_path(manifest_path),
                },
                metadata={
                    "generated_from": "fixed_classifier_manifest",
                    "manifest_path": relative_repo_path(manifest_path),
                    "train_jsonl": payload.get("train_jsonl"),
                    "selection_accuracy_top_1": _nested_accuracy(payload),
                },
            )
        )

    lora_classifier_dir = repo_root / "data/processed/lora_classifier_heads"
    for manifest_path in sorted(lora_classifier_dir.glob("*.manifest.json")):
        payload = _load_json_object(manifest_path)
        trainer_version = str(
            payload.get("trainer_version", manifest_path.stem.removesuffix(".manifest"))
        )
        items.append(
            _build_generated_preset_item(
                item_name=f"generated_initial_checkpoint__lora__{trainer_version}",
                display_name=f"saved LoRA: {trainer_version}",
                family_name="initial_checkpoint",
                core_method_name="lora_classifier_manifest",
                preset_group="query_adaptation_initial_checkpoint",
                compiled_selector_name="required",
                source_of_truth=relative_repo_path(manifest_path),
                description=(
                    "기존 LoRA + classifier manifest를 warm-start 체크포인트로 "
                    "사용합니다."
                ),
                default_override_patch={
                    "manifest_path": relative_repo_path(manifest_path),
                },
                metadata={
                    "generated_from": "lora_classifier_manifest",
                    "manifest_path": relative_repo_path(manifest_path),
                    "train_jsonl": payload.get("train_jsonl"),
                    "selection_accuracy_top_1": _nested_accuracy(payload),
                },
            )
        )

    return tuple(sorted(items, key=lambda item: item.display_name, reverse=True))


def _build_generated_preset_item(
    *,
    item_name: str,
    display_name: str,
    family_name: str,
    core_method_name: str,
    preset_group: str,
    compiled_selector_name: str,
    source_of_truth: str,
    description: str,
    default_override_patch: dict[str, object],
    metadata: dict[str, object],
) -> CatalogItemPayload:
    return CatalogItemPayload(
        item_name=item_name,
        display_name=display_name,
        item_kind="generated_artifact_preset",
        family_name=family_name,
        core_method_name=core_method_name,
        variant_profile_name=item_name,
        compiled_selector_name=compiled_selector_name,
        preset_group=preset_group,
        description=description,
        source_of_truth=source_of_truth,
        source_kind="generated_artifact",
        compile_support="preset_selector",
        default_override_patch=default_override_patch,
        tags=("generated_artifact",),
        metadata=metadata,
    )


def _iter_teacher_split_manifests(repo_root: Path) -> tuple[Path, ...]:
    base_dir = repo_root / "data/processed/lora_bootstrap_classifier_teacher"
    return tuple(sorted(base_dir.glob("*/teacher_split.manifest.json")))


def _iter_query_ssl_subset_dirs(repo_root: Path) -> tuple[Path, ...]:
    base_dir = repo_root / "data/processed/query_ssl_smoke_subsets"
    if not base_dir.exists():
        return ()
    return tuple(sorted(path for path in base_dir.iterdir() if path.is_dir()))


def _find_first(directory: Path, pattern: str) -> Path | None:
    for path in sorted(directory.glob(pattern)):
        return path
    return None


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object manifest: {path}")
    return payload


def _nested_accuracy(payload: dict[str, Any]) -> float | None:
    best_selection_report = payload.get("best_selection_report")
    if not isinstance(best_selection_report, dict):
        return None
    value = best_selection_report.get("accuracy_top_1")
    return float(value) if isinstance(value, (float, int)) else None


def _string_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None
