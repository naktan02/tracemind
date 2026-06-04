"""중앙 supervised/SSL method 비교용 공통 UMAP figure 생성."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from methods.adaptation.peft_text_encoder.training.modeling import (
    PeftTextEncoderWithLinearHead,
    require_transformer_stack,
)
from methods.adaptation.query_text_views.data import (
    build_dataloader as build_query_text_dataloader,
)
from methods.adaptation.text_encoder_classifier.modeling import (
    TextEncoderWithLinearHead,
    load_classifier_head_state_if_configured,
    load_transformer_backbone,
    load_transformer_tokenizer,
)
from methods.adaptation.text_encoder_classifier.projection import (
    collect_pooled_classifier_features,
    reduce_features_2d,
)
from scripts.runtime_adapters.embedding_runtime import resolve_runtime_device_name
from shared.src.contracts.labeled_query_row_contracts import load_labeled_query_rows


@dataclass(frozen=True, slots=True)
class RunSpec:
    """논문용 projection에 올릴 run report 지정."""

    label: str
    report_path: Path


@dataclass(frozen=True, slots=True)
class RunManifest:
    """report.json에서 projection 후처리에 필요한 정보만 정규화한 값."""

    label: str
    report_path: Path
    trainer_version: str
    categories: tuple[str, ...]
    eval_sets: dict[str, Path]
    classifier_path: Path
    adapter_dir: Path | None
    model_dir: Path | None
    backbone: dict[str, Any]
    schema_version: str


@dataclass(frozen=True, slots=True)
class MethodFeatureSet:
    """한 method/run의 한 split feature 묶음."""

    method_label: str
    trainer_version: str
    split: str
    features: np.ndarray
    row_ids: tuple[str, ...]
    true_labels: tuple[str, ...]
    predicted_labels: tuple[str, ...]
    top_1_probabilities: tuple[float, ...]


def main() -> None:
    args = _parse_args()
    run_specs = [_parse_run_spec(value) for value in args.run]
    splits = tuple(args.split or ["test"])
    output_dir = resolve_projection_output_dir(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        output_root=Path(args.output_root),
        figure_version=args.figure_version,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    manifests = [_load_run_manifest(spec) for spec in run_specs]
    _validate_category_schema(manifests)

    outputs: dict[str, Any] = {
        "schema_version": "central_method_projection_figure.v1",
        "reducer": args.reducer,
        "device": device,
        "seed": int(args.seed),
        "max_rows_per_run": args.max_rows_per_run,
        "output_dir": str(output_dir),
        "figure_version": output_dir.name,
        "runs": [
            {
                "label": manifest.label,
                "trainer_version": manifest.trainer_version,
                "report_path": str(manifest.report_path),
                "schema_version": manifest.schema_version,
            }
            for manifest in manifests
        ],
        "splits": {},
    }

    for split in splits:
        feature_sets = [
            extract_method_features(
                manifest=manifest,
                split=split,
                device=device,
                max_rows_per_run=args.max_rows_per_run,
                seed=int(args.seed),
                batch_size=args.batch_size,
            )
            for manifest in manifests
        ]
        split_outputs = write_split_projection_artifacts(
            split=split,
            feature_sets=feature_sets,
            categories=list(manifests[0].categories),
            output_dir=output_dir,
            reducer_name=str(args.reducer),
            seed=int(args.seed),
            n_neighbors=int(args.n_neighbors),
        )
        outputs["splits"][split] = split_outputs

    manifest_path = output_dir / "method_projection_manifest.json"
    _write_json(manifest_path, outputs)
    print(f"output_dir={output_dir}")
    print(f"projection_manifest={manifest_path}")


def extract_method_features(
    *,
    manifest: RunManifest,
    split: str,
    device: str,
    max_rows_per_run: int | None,
    seed: int,
    batch_size: int | None,
) -> MethodFeatureSet:
    """저장된 best artifact를 복원해 split feature를 추출한다."""

    if split not in manifest.eval_sets:
        raise ValueError(
            f"run {manifest.label!r} does not include eval split {split!r}."
        )
    model, tokenizer = _load_model_from_manifest(manifest=manifest, device=device)
    rows = load_labeled_query_rows(manifest.eval_sets[split])
    selected_indices = select_row_indices(
        row_count=len(rows),
        max_rows=max_rows_per_run,
        seed=seed,
        salt=f"{manifest.label}:{split}",
    )
    selected_rows = [rows[index] for index in selected_indices]
    label_to_index = {label: index for index, label in enumerate(manifest.categories)}
    effective_batch_size = int(batch_size or _default_eval_batch_size(manifest))
    dataloader = build_query_text_dataloader(
        rows=selected_rows,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        batch_size=effective_batch_size,
        max_length=int(manifest.backbone["max_length"]),
        task_prefix=str(manifest.backbone.get("task_prefix", "")),
        shuffle=False,
    )
    collected = collect_pooled_classifier_features(
        model=model,
        dataloader=dataloader,
        categories=list(manifest.categories),
        device=device,
        collect_labels=False,
    )
    return MethodFeatureSet(
        method_label=manifest.label,
        trainer_version=manifest.trainer_version,
        split=split,
        features=collected.features,
        row_ids=tuple(str(row["query_id"]) for row in selected_rows),
        true_labels=tuple(str(row["mapped_label_4"]) for row in selected_rows),
        predicted_labels=collected.predicted_labels,
        top_1_probabilities=collected.top_1_probabilities,
    )


def write_split_projection_artifacts(
    *,
    split: str,
    feature_sets: Sequence[MethodFeatureSet],
    categories: list[str],
    output_dir: Path,
    reducer_name: str,
    seed: int,
    n_neighbors: int,
) -> dict[str, Any]:
    """한 split의 method feature를 합쳐 공통 2D projection 산출물을 쓴다."""

    _validate_feature_sets(feature_sets)
    all_features = np.concatenate(
        [feature_set.features for feature_set in feature_sets]
    )
    projection = reduce_features_2d(
        features=all_features,
        reducer_name=reducer_name,
        seed=seed,
        n_neighbors=n_neighbors,
    )
    rows = _build_projection_rows(
        feature_sets=feature_sets,
        coordinates=projection.coordinates,
    )
    axis_limits = _axis_limits(rows)
    split_dir = output_dir / split
    method_outputs = _write_method_projection_artifacts(
        split_dir=split_dir,
        feature_sets=feature_sets,
        coordinates=projection.coordinates,
        rows=rows,
        categories=categories,
        axis_limits=axis_limits,
        reducer_name=projection.reducer,
    )
    return {
        "reducer": projection.reducer,
        "fallback_reason": projection.fallback_reason,
        "row_count": len(rows),
        "method_count": len(method_outputs),
        "methods": method_outputs,
    }


def select_row_indices(
    *,
    row_count: int,
    max_rows: int | None,
    seed: int,
    salt: str,
) -> list[int]:
    """row sampling을 재현 가능하게 수행하고 원래 순서를 유지한다."""

    if max_rows is None or max_rows <= 0 or row_count <= max_rows:
        return list(range(row_count))
    salt_digest = hashlib.sha256(salt.encode("utf-8")).hexdigest()
    salt_seed = int(salt_digest[:8], 16)
    rng = np.random.default_rng(seed + salt_seed)
    selected = rng.choice(row_count, size=int(max_rows), replace=False)
    return sorted(int(index) for index in selected.tolist())


def parse_run_specs(values: Sequence[str]) -> list[RunSpec]:
    """CLI run spec 목록을 파싱한다."""

    return [_parse_run_spec(value) for value in values]


def resolve_projection_output_dir(
    *,
    output_dir: Path | None,
    output_root: Path,
    figure_version: str | None,
    created_at: datetime | None = None,
) -> Path:
    """projection 산출물 저장 디렉터리를 결정한다."""

    if output_dir is not None:
        return output_dir
    if figure_version is not None and figure_version.strip():
        return output_root / figure_version.strip()
    timestamp = created_at or datetime.now()
    return output_root / timestamp.strftime("%Y_%m_%d_%H%M%S")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build common reducer method projection figures from central run reports."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Method label and report path in label=path/to/report.json format.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["validation", "test"],
        help="Eval split to project. Defaults to validation and test.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Exact directory for method projection artifacts. Overrides "
            "--output-root and --figure-version."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="runs/figures/central_ssl/method_projection",
        help="Root directory where a dated projection directory is created.",
    )
    parser.add_argument(
        "--figure-version",
        default=None,
        help="Optional output directory name under --output-root.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Runtime device name. Defaults to auto.",
    )
    parser.add_argument(
        "--reducer",
        choices=["umap", "pca"],
        default="umap",
        help="2D reducer to fit jointly per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reducer and optional sampling seed.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors upper bound.",
    )
    parser.add_argument(
        "--max-rows-per-run",
        type=int,
        default=None,
        help="Optional deterministic row subsample per run/split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional feature extraction batch size override.",
    )
    return parser.parse_args()


def _parse_run_spec(value: str) -> RunSpec:
    if "=" not in value:
        raise ValueError(
            f"--run must use label=path/to/report.json format (got {value!r})."
        )
    label, raw_path = value.split("=", 1)
    normalized_label = label.strip()
    if not normalized_label:
        raise ValueError("--run label must not be empty.")
    report_path = Path(raw_path.strip())
    if not report_path.exists():
        raise FileNotFoundError(f"run report does not exist: {report_path}")
    return RunSpec(label=normalized_label, report_path=report_path)


def _load_run_manifest(spec: RunSpec) -> RunManifest:
    payload = json.loads(spec.report_path.read_text(encoding="utf-8"))
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"report manifest must be an object: {spec.report_path}")
    backbone = manifest.get("backbone")
    if not isinstance(backbone, dict):
        raise ValueError(
            f"report manifest.backbone must be an object: {spec.report_path}"
        )
    eval_sets_payload = manifest.get("eval_sets")
    if not isinstance(eval_sets_payload, dict):
        raise ValueError(
            f"report manifest.eval_sets must be an object: {spec.report_path}"
        )
    categories_payload = manifest.get("categories")
    if not isinstance(categories_payload, list) or not categories_payload:
        raise ValueError(
            f"report manifest.categories must be a non-empty list: {spec.report_path}"
        )
    classifier_path = _required_existing_path(
        manifest.get("classifier_path"),
        field_name="manifest.classifier_path",
        report_path=spec.report_path,
    )
    adapter_dir = _optional_existing_path(manifest.get("adapter_dir"))
    model_dir = _optional_existing_path(manifest.get("model_dir"))
    if adapter_dir is None and model_dir is None:
        raise ValueError(
            "report manifest must include adapter_dir or model_dir for projection "
            f"model loading: {spec.report_path}"
        )
    if adapter_dir is not None and model_dir is not None:
        raise ValueError(
            f"report has both adapter_dir and model_dir: {spec.report_path}"
        )
    trainer_version = str(
        manifest.get("trainer_version") or spec.report_path.parents[1].name
    )
    return RunManifest(
        label=spec.label,
        report_path=spec.report_path,
        trainer_version=trainer_version,
        categories=tuple(str(category) for category in categories_payload),
        eval_sets={
            str(name): Path(str(path)) for name, path in eval_sets_payload.items()
        },
        classifier_path=classifier_path,
        adapter_dir=adapter_dir,
        model_dir=model_dir,
        backbone=dict(backbone),
        schema_version=str(payload.get("schema_version") or ""),
    )


def _load_model_from_manifest(
    *,
    manifest: RunManifest,
    device: str,
) -> tuple[TextEncoderWithLinearHead, Any]:
    if manifest.model_dir is not None:
        return _load_full_text_encoder_model(manifest=manifest, device=device)
    if manifest.adapter_dir is not None:
        return _load_peft_text_encoder_model(manifest=manifest, device=device)
    raise ValueError(f"run {manifest.label!r} has no loadable model artifact.")


def _load_peft_text_encoder_model(
    *,
    manifest: RunManifest,
    device: str,
) -> tuple[PeftTextEncoderWithLinearHead, Any]:
    assert manifest.adapter_dir is not None
    AutoModel, AutoTokenizer, _LoraConfig, _TaskType, _get_peft_model, PeftModel = (
        require_transformer_stack()
    )
    backbone_base = load_transformer_backbone(
        model_cls=AutoModel,
        model_id=str(manifest.backbone["backbone_model_id"]),
        revision=str(manifest.backbone["backbone_revision"]),
        cache_dir=None,
        local_files_only=True,
        trust_remote_code=False,
    )
    backbone = PeftModel.from_pretrained(
        backbone_base,
        manifest.adapter_dir,
        is_trainable=False,
    )
    tokenizer = _load_tokenizer_for_artifact(
        tokenizer_cls=AutoTokenizer,
        artifact_dir=manifest.adapter_dir,
        manifest=manifest,
    )
    model = PeftTextEncoderWithLinearHead(
        backbone=backbone,
        hidden_size=int(backbone.config.hidden_size),
        num_labels=len(manifest.categories),
        classifier_dropout=0.0,
    ).to(device)
    load_classifier_head_state_if_configured(
        model=model,
        categories=list(manifest.categories),
        classifier_path=manifest.classifier_path,
    )
    model.eval()
    return model, tokenizer


def _load_full_text_encoder_model(
    *,
    manifest: RunManifest,
    device: str,
) -> tuple[TextEncoderWithLinearHead, Any]:
    assert manifest.model_dir is not None
    AutoModel, AutoTokenizer, _LoraConfig, _TaskType, _get_peft_model, _PeftModel = (
        require_transformer_stack()
    )
    tokenizer = _load_tokenizer_for_artifact(
        tokenizer_cls=AutoTokenizer,
        artifact_dir=manifest.model_dir,
        manifest=manifest,
    )
    backbone = AutoModel.from_pretrained(
        str(manifest.model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = TextEncoderWithLinearHead(
        backbone=backbone,
        hidden_size=int(backbone.config.hidden_size),
        num_labels=len(manifest.categories),
        classifier_dropout=0.0,
    ).to(device)
    load_classifier_head_state_if_configured(
        model=model,
        categories=list(manifest.categories),
        classifier_path=manifest.classifier_path,
    )
    model.eval()
    return model, tokenizer


def _load_tokenizer_for_artifact(
    *,
    tokenizer_cls: Any,
    artifact_dir: Path,
    manifest: RunManifest,
) -> Any:
    try:
        tokenizer = tokenizer_cls.from_pretrained(
            str(artifact_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        return tokenizer
    except Exception:
        return load_transformer_tokenizer(
            tokenizer_cls=tokenizer_cls,
            model_id=str(manifest.backbone["tokenizer_model_id"]),
            revision=str(manifest.backbone["tokenizer_revision"]),
            cache_dir=None,
            local_files_only=True,
            trust_remote_code=False,
        )


def _build_projection_rows(
    *,
    feature_sets: Sequence[MethodFeatureSet],
    coordinates: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for feature_set in feature_sets:
        count = int(feature_set.features.shape[0])
        method_coordinates = coordinates[offset : offset + count]
        for row_index, (x, y) in enumerate(method_coordinates):
            true_label = feature_set.true_labels[row_index]
            predicted_label = feature_set.predicted_labels[row_index]
            rows.append(
                {
                    "method_label": feature_set.method_label,
                    "trainer_version": feature_set.trainer_version,
                    "split": feature_set.split,
                    "row_index": row_index,
                    "query_id": feature_set.row_ids[row_index],
                    "x": round(float(x), 6),
                    "y": round(float(y), 6),
                    "label": true_label,
                    "predicted_label": predicted_label,
                    "is_correct": true_label == predicted_label,
                    "top_1_probability": round(
                        float(feature_set.top_1_probabilities[row_index]),
                        6,
                    ),
                }
            )
        offset += count
    return rows


def draw_method_figure(
    *,
    figure_path: Path,
    rows: Sequence[dict[str, Any]],
    categories: list[str],
    title: str,
    axis_limits: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    if not rows:
        raise ValueError("projection rows must not be empty.")
    color_map = _category_color_map(categories)
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    for category in categories:
        category_rows = [row for row in rows if str(row["label"]) == category]
        if not category_rows:
            continue
        axis.scatter(
            [float(row["x"]) for row in category_rows],
            [float(row["y"]) for row in category_rows],
            s=12,
            alpha=0.74,
            color=color_map[category],
            label=category,
        )
    incorrect_rows = [row for row in rows if not bool(row["is_correct"])]
    if incorrect_rows:
        axis.scatter(
            [float(row["x"]) for row in incorrect_rows],
            [float(row["y"]) for row in incorrect_rows],
            s=34,
            facecolors="none",
            edgecolors="black",
            linewidths=0.7,
            label="incorrect",
        )
    axis.set_title(title)
    axis.set_xlim(*axis_limits[0])
    axis.set_ylim(*axis_limits[1])
    axis.set_xticks([])
    axis.set_yticks([])
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)))
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)


def _write_method_projection_artifacts(
    *,
    split_dir: Path,
    feature_sets: Sequence[MethodFeatureSet],
    coordinates: np.ndarray,
    rows: Sequence[dict[str, Any]],
    categories: list[str],
    axis_limits: tuple[tuple[float, float], tuple[float, float]],
    reducer_name: str,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_method[str(row["method_label"])].append(dict(row))

    offset = 0
    for feature_set in feature_sets:
        count = int(feature_set.features.shape[0])
        method_label = feature_set.method_label
        method_name = _safe_filename(method_label)
        method_rows = rows_by_method[method_label]
        method_coordinates = coordinates[offset : offset + count]
        points_path = split_dir / f"{method_name}.method_projection.jsonl"
        figure_path = split_dir / f"{method_name}.method_projection.png"
        features_path = split_dir / f"{method_name}.method_features.npz"
        _write_jsonl(points_path, method_rows)
        _write_feature_npz(
            path=features_path,
            feature_sets=[feature_set],
            coordinates=method_coordinates,
        )
        draw_method_figure(
            figure_path=figure_path,
            rows=method_rows,
            categories=categories,
            title=f"{split_dir.name} {method_label} {reducer_name.upper()} projection",
            axis_limits=axis_limits,
        )
        outputs[method_label] = {
            "row_count": len(method_rows),
            "points_jsonl": str(points_path),
            "features_npz": str(features_path),
            "figure_png": str(figure_path),
        }
        offset += count
    return outputs


def _axis_limits(
    rows: Sequence[dict[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not rows:
        raise ValueError("projection rows must not be empty.")
    all_x = [float(row["x"]) for row in rows]
    all_y = [float(row["y"]) for row in rows]
    x_margin = max(1e-6, (max(all_x) - min(all_x)) * 0.05)
    y_margin = max(1e-6, (max(all_y) - min(all_y)) * 0.05)
    return (
        (min(all_x) - x_margin, max(all_x) + x_margin),
        (min(all_y) - y_margin, max(all_y) + y_margin),
    )


def _validate_feature_sets(feature_sets: Sequence[MethodFeatureSet]) -> None:
    if not feature_sets:
        raise ValueError("At least one feature set is required.")
    feature_dims = {
        int(feature_set.features.shape[1])
        for feature_set in feature_sets
        if feature_set.features.ndim == 2
    }
    if len(feature_dims) != 1:
        raise ValueError(f"All feature sets must share one dimension: {feature_dims}")
    empty_methods = [
        feature_set.method_label
        for feature_set in feature_sets
        if int(feature_set.features.shape[0]) == 0
    ]
    if empty_methods:
        raise ValueError(f"Feature sets must not be empty: {empty_methods}")


def _validate_category_schema(manifests: Sequence[RunManifest]) -> None:
    if not manifests:
        raise ValueError("At least one run is required.")
    expected = manifests[0].categories
    mismatches = [
        manifest.label
        for manifest in manifests
        if tuple(manifest.categories) != tuple(expected)
    ]
    if mismatches:
        raise ValueError(
            f"All runs must use the same category order. Mismatched runs: {mismatches}"
        )


def _category_color_map(categories: Sequence[str]) -> dict[str, Any]:
    color_cycle = plt.get_cmap("tab10")
    return {
        category: color_cycle(index % 10) for index, category in enumerate(categories)
    }


def _safe_filename(value: str) -> str:
    characters = [
        character.lower() if character.isalnum() else "_" for character in value.strip()
    ]
    return "".join(characters).strip("_") or "method"


def _write_feature_npz(
    *,
    path: Path,
    feature_sets: Sequence[MethodFeatureSet],
    coordinates: np.ndarray,
) -> None:
    features = np.concatenate([feature_set.features for feature_set in feature_sets])
    method_labels: list[str] = []
    trainer_versions: list[str] = []
    splits: list[str] = []
    row_ids: list[str] = []
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    top_1_probabilities: list[float] = []
    for feature_set in feature_sets:
        count = int(feature_set.features.shape[0])
        method_labels.extend([feature_set.method_label] * count)
        trainer_versions.extend([feature_set.trainer_version] * count)
        splits.extend([feature_set.split] * count)
        row_ids.extend(feature_set.row_ids)
        true_labels.extend(feature_set.true_labels)
        predicted_labels.extend(feature_set.predicted_labels)
        top_1_probabilities.extend(feature_set.top_1_probabilities)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        coordinates=coordinates.astype(np.float32),
        method_labels=np.asarray(method_labels),
        trainer_versions=np.asarray(trainer_versions),
        splits=np.asarray(splits),
        row_ids=np.asarray(row_ids),
        true_labels=np.asarray(true_labels),
        predicted_labels=np.asarray(predicted_labels),
        top_1_probabilities=np.asarray(top_1_probabilities, dtype=np.float32),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def _required_existing_path(
    value: object,
    *,
    field_name: str,
    report_path: Path,
) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required in {report_path}")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _optional_existing_path(value: object) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"artifact path does not exist: {path}")
    return path


def _default_eval_batch_size(manifest: RunManifest) -> int:
    payload = json.loads(manifest.report_path.read_text(encoding="utf-8"))
    raw_value = payload.get("manifest", {}).get("eval_batch_size", 32)
    return int(raw_value)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return resolve_runtime_device_name(device)


if __name__ == "__main__":
    main()
