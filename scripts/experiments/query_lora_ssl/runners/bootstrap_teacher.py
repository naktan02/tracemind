"""Fixed classifier teacher -> LoRA student bootstrap runner."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

from scripts.datasets.lib.split import split_rows
from scripts.experiments.fixed_classifier.runner import (
    load_fixed_classifier_artifacts,
    predict_fixed_classifier_rows,
    train_fixed_embedding_classifier,
    write_fixed_classifier_artifacts,
)
from scripts.experiments.query_lora_ssl.config.pseudo_label_algorithm import (
    resolve_pseudo_label_algorithm,
)
from scripts.experiments.query_lora_ssl.io.teacher_pseudo_label_artifact_writer import (
    TeacherPseudoLabelArtifactWriter,
)
from scripts.experiments.query_lora_ssl.io.teacher_pseudo_label_builder import (
    TeacherPseudoLabelBuilder,
)
from scripts.experiments.query_lora_ssl.runners.pseudo_label import (
    run_pseudo_label_self_training,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)

BOOTSTRAP_SPLIT_SCHEMA_VERSION = "fixed_classifier_teacher_split.v1"
BOOTSTRAP_SUMMARY_SCHEMA_VERSION = "fixed_classifier_lora_bootstrap.v1"


@dataclass(slots=True)
class BootstrapSplitArtifacts:
    """teacher seed / unlabeled split 산출물."""

    seed_train_jsonl: Path
    unlabeled_jsonl: Path
    manifest_path: Path


def run_fixed_classifier_teacher_lora_student_bootstrap(
    *,
    cfg: DictConfig,
    teacher_seed_rows: Sequence[LabeledQueryRow] | None = None,
    teacher_unlabeled_rows: Sequence[LabeledQueryRow] | None = None,
    export_root: str | Path | None = None,
    generated_at: datetime | None = None,
) -> dict[str, str]:
    """초기 teacher는 fixed classifier, student는 LoRA classifier로 연결한다."""

    effective_generated_at = generated_at or datetime.now(tz=timezone.utc)
    run_id = _resolve_run_id(cfg=cfg, generated_at=effective_generated_at)
    resolved_export_root = (
        Path(str(cfg.bootstrap_export_root))
        if export_root is None
        else Path(str(export_root))
    )
    export_dir = resolved_export_root / run_id
    export_dir.mkdir(parents=True, exist_ok=True)

    (
        effective_teacher_seed_rows,
        effective_teacher_unlabeled_rows,
        teacher_seed_jsonl_ref,
        teacher_unlabeled_jsonl_ref,
        split_artifacts,
    ) = _resolve_teacher_and_unlabeled_rows(
        cfg=cfg,
        run_id=run_id,
        export_dir=export_dir,
        teacher_seed_rows=teacher_seed_rows,
        teacher_unlabeled_rows=teacher_unlabeled_rows,
    )

    eval_set_map = {name: Path(str(path)) for name, path in cfg.eval_sets.items()}
    eval_rows_by_name = {
        name: load_labeled_query_rows(path) for name, path in eval_set_map.items()
    }
    teacher_reuse_manifest_path = str(
        getattr(cfg, "teacher_reuse_manifest_path", "") or ""
    ).strip()
    if teacher_reuse_manifest_path:
        trained_teacher, teacher_outputs = load_fixed_classifier_artifacts(
            manifest_path=teacher_reuse_manifest_path,
            device=str(cfg.runtime.device),
            batch_size=int(cfg.teacher_eval_batch_size),
            cache_dir=str(getattr(cfg.embedding, "cache_dir", "") or "") or None,
            local_files_only=bool(getattr(cfg.runtime, "local_files_only", False)),
        )
        teacher_outputs = {
            **teacher_outputs,
            "reused_teacher_manifest": teacher_reuse_manifest_path,
        }
    else:
        embedding_spec = instantiate(cfg.embedding.spec)
        teacher_classifier_version = (
            str(getattr(cfg, "teacher_classifier_version", "") or "").strip()
            or f"{run_id}_teacher"
        )

        trained_teacher = train_fixed_embedding_classifier(
            train_rows=effective_teacher_seed_rows,
            eval_rows_by_name=eval_rows_by_name,
            selection_set_name=str(cfg.selection_set),
            embedding_spec=embedding_spec,
            embed_chunk_size=int(cfg.teacher_embed_chunk_size),
            train_batch_size=int(cfg.teacher_train_batch_size),
            eval_batch_size=int(cfg.teacher_eval_batch_size),
            epochs=int(cfg.teacher_epochs),
            learning_rate=float(cfg.teacher_learning_rate),
            weight_decay=float(cfg.teacher_weight_decay),
        )
        teacher_outputs = write_fixed_classifier_artifacts(
            classifier_version=teacher_classifier_version,
            created_at=effective_generated_at,
            train_jsonl_ref=teacher_seed_jsonl_ref,
            eval_set_map={name: str(path) for name, path in eval_set_map.items()},
            selection_set_name=str(cfg.selection_set),
            output_dir_root=str(cfg.teacher_output_dir),
            model_output_dir=str(cfg.teacher_model_output_dir),
            epochs=int(cfg.teacher_epochs),
            train_batch_size=int(cfg.teacher_train_batch_size),
            learning_rate=float(cfg.teacher_learning_rate),
            weight_decay=float(cfg.teacher_weight_decay),
            trained=trained_teacher,
        )

    predictions = predict_fixed_classifier_rows(
        trained=trained_teacher,
        rows=effective_teacher_unlabeled_rows,
        embed_chunk_size=int(cfg.teacher_embed_chunk_size),
        eval_batch_size=int(cfg.teacher_eval_batch_size),
    )
    teacher_export = TeacherPseudoLabelBuilder().build_export(
        rows=effective_teacher_unlabeled_rows,
        predictions=predictions,
        pseudo_label_algorithm=resolve_pseudo_label_algorithm(cfg),
        generated_at=effective_generated_at,
        run_id=run_id,
    )
    pseudo_label_rows = teacher_export.pseudo_label_rows
    if not pseudo_label_rows:
        raise ValueError(
            "Teacher bootstrap did not accept any pseudo-label rows. "
            "Lower the threshold or provide a different unlabeled pool."
        )

    teacher_artifact_writer = TeacherPseudoLabelArtifactWriter()
    prediction_artifacts = teacher_artifact_writer.write_prediction_artifacts(
        export_dir=export_dir,
        prediction_trace_rows=teacher_export.prediction_trace_rows,
        prediction_summary=teacher_export.prediction_summary,
    )

    student_outputs = _run_student_lora_bootstrap(
        cfg=cfg,
        run_id=run_id,
        teacher_seed_jsonl_ref=teacher_seed_jsonl_ref,
        seed_train_rows=effective_teacher_seed_rows,
        pseudo_label_rows=pseudo_label_rows,
        generated_at=effective_generated_at,
        categories_override=tuple(trained_teacher.categories),
    )

    bootstrap_summary = {
        "schema_version": BOOTSTRAP_SUMMARY_SCHEMA_VERSION,
        "bootstrap_version": run_id,
        "generated_at": effective_generated_at.isoformat(),
        "teacher_train_jsonl": teacher_seed_jsonl_ref,
        "teacher_unlabeled_jsonl": teacher_unlabeled_jsonl_ref,
        "teacher_seed_row_count": len(effective_teacher_seed_rows),
        "teacher_unlabeled_row_count": len(effective_teacher_unlabeled_rows),
        "teacher_outputs": teacher_outputs,
        "prediction_trace_jsonl": str(prediction_artifacts.prediction_trace_jsonl),
        "prediction_summary_json": str(prediction_artifacts.prediction_summary_json),
        "student_outputs": student_outputs,
        "split_manifest": (
            None if split_artifacts is None else str(split_artifacts.manifest_path)
        ),
    }
    bootstrap_summary_path = teacher_artifact_writer.write_bootstrap_summary(
        export_dir=export_dir,
        bootstrap_summary=bootstrap_summary,
    )

    outputs = {
        "bootstrap_summary": str(bootstrap_summary_path),
        "prediction_trace_jsonl": str(prediction_artifacts.prediction_trace_jsonl),
        "prediction_summary_json": str(prediction_artifacts.prediction_summary_json),
        **teacher_outputs,
        **student_outputs,
    }
    if split_artifacts is not None:
        outputs.update(
            {
                "teacher_seed_jsonl": str(split_artifacts.seed_train_jsonl),
                "teacher_unlabeled_jsonl": str(split_artifacts.unlabeled_jsonl),
                "teacher_split_manifest": str(split_artifacts.manifest_path),
            }
        )
    return outputs


def _resolve_teacher_and_unlabeled_rows(
    *,
    cfg: DictConfig,
    run_id: str,
    export_dir: Path,
    teacher_seed_rows: Sequence[LabeledQueryRow] | None,
    teacher_unlabeled_rows: Sequence[LabeledQueryRow] | None,
) -> tuple[
    list[LabeledQueryRow],
    list[LabeledQueryRow],
    str,
    str,
    BootstrapSplitArtifacts | None,
]:
    if teacher_seed_rows is not None and teacher_unlabeled_rows is not None:
        return (
            list(teacher_seed_rows),
            list(teacher_unlabeled_rows),
            "in_memory/teacher_seed_rows.jsonl",
            "in_memory/teacher_unlabeled_rows.jsonl",
            None,
        )

    teacher_train_path = Path(str(cfg.teacher_train_jsonl))
    explicit_unlabeled = getattr(cfg, "teacher_unlabeled_jsonl", None)
    if teacher_seed_rows is None:
        base_rows = load_labeled_query_rows(teacher_train_path)
    else:
        base_rows = list(teacher_seed_rows)

    if teacher_unlabeled_rows is not None:
        return (
            base_rows,
            list(teacher_unlabeled_rows),
            str(teacher_train_path),
            "in_memory/teacher_unlabeled_rows.jsonl",
            None,
        )

    if explicit_unlabeled:
        unlabeled_path = Path(str(explicit_unlabeled))
        return (
            base_rows,
            load_labeled_query_rows(unlabeled_path),
            str(teacher_train_path),
            str(unlabeled_path),
            None,
        )

    if not bool(cfg.bootstrap_split.enabled):
        raise ValueError(
            "Provide teacher_unlabeled_jsonl or enable bootstrap_split.enabled."
        )

    seed_rows, unlabeled_rows = split_rows(
        base_rows,
        validation_ratio=float(cfg.bootstrap_split.unlabeled_ratio),
        seed=int(cfg.bootstrap_split.seed),
    )
    seed_path = export_dir / "teacher_seed_train.jsonl"
    unlabeled_path = export_dir / "teacher_unlabeled_pool.jsonl"
    manifest_path = export_dir / "teacher_split.manifest.json"
    dump_labeled_query_rows(seed_path, seed_rows)
    dump_labeled_query_rows(unlabeled_path, unlabeled_rows)
    _write_json(
        manifest_path,
        {
            "schema_version": BOOTSTRAP_SPLIT_SCHEMA_VERSION,
            "bootstrap_version": run_id,
            "source_train_jsonl": str(teacher_train_path),
            "seed": int(cfg.bootstrap_split.seed),
            "unlabeled_ratio": float(cfg.bootstrap_split.unlabeled_ratio),
            "teacher_seed_row_count": len(seed_rows),
            "teacher_unlabeled_row_count": len(unlabeled_rows),
            "teacher_seed_jsonl": str(seed_path),
            "teacher_unlabeled_jsonl": str(unlabeled_path),
        },
    )
    return (
        seed_rows,
        unlabeled_rows,
        str(seed_path),
        str(unlabeled_path),
        BootstrapSplitArtifacts(
            seed_train_jsonl=seed_path,
            unlabeled_jsonl=unlabeled_path,
            manifest_path=manifest_path,
        ),
    )


def _resolve_run_id(
    *,
    cfg: DictConfig,
    generated_at: datetime,
) -> str:
    bootstrap_version = str(getattr(cfg, "bootstrap_version", "") or "").strip()
    if bootstrap_version:
        return bootstrap_version
    return generated_at.strftime("lora_bootstrap_%Y_%m_%d_%H%M%S")


def _run_student_lora_bootstrap(
    *,
    cfg: DictConfig,
    run_id: str,
    teacher_seed_jsonl_ref: str,
    seed_train_rows: Sequence[LabeledQueryRow],
    pseudo_label_rows: Sequence[LabeledQueryRow],
    generated_at: datetime,
    categories_override: Sequence[str],
) -> dict[str, str]:
    return run_pseudo_label_self_training(
        cfg=cfg,
        seed_train_rows=seed_train_rows,
        pseudo_label_rows=pseudo_label_rows,
        include_seed_train_rows=bool(
            getattr(cfg, "student_include_seed_train_rows", False)
        ),
        train_jsonl_ref=teacher_seed_jsonl_ref,
        trainer_version_override=_resolve_student_trainer_version(cfg, run_id),
        export_root=str(cfg.pseudo_label_export_root),
        generated_at=generated_at,
        categories_override=categories_override,
    )


def _resolve_student_trainer_version(cfg: DictConfig, run_id: str) -> str:
    trainer_version = str(getattr(cfg, "trainer_version", "") or "").strip()
    if trainer_version:
        return trainer_version
    return f"{run_id}_student"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
