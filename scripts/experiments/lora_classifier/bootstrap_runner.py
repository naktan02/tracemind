"""Fixed classifier teacher -> LoRA student bootstrap runner."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from agent.src.services.training.query_adaptation.ssl.registry import (
    build_query_ssl_algorithm,
)
from scripts.datasets.lib.split import split_rows
from scripts.experiments.fixed_classifier.runner import (
    FixedClassifierPrediction,
    load_fixed_classifier_artifacts,
    predict_fixed_classifier_rows,
    train_fixed_embedding_classifier,
    write_fixed_classifier_artifacts,
)
from scripts.labeled_query_rows import (
    LabeledQueryRow,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)

from .pseudo_label_algorithm_config import resolve_pseudo_label_algorithm
from .pseudo_label_runner import run_pseudo_label_self_training

TEACHER_PREDICTION_TRACE_SCHEMA_VERSION = "fixed_classifier_teacher_trace.v1"
TEACHER_PREDICTION_SUMMARY_SCHEMA_VERSION = "fixed_classifier_teacher_summary.v1"
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
    resolved_pseudo_label_algorithm = resolve_pseudo_label_algorithm(cfg)
    pseudo_label_rows, prediction_trace_rows, prediction_summary = (
        _build_teacher_pseudo_label_rows(
            rows=effective_teacher_unlabeled_rows,
            predictions=predictions,
            pseudo_label_algorithm=resolved_pseudo_label_algorithm,
            generated_at=effective_generated_at,
            run_id=run_id,
        )
    )
    if not pseudo_label_rows:
        raise ValueError(
            "Teacher bootstrap did not accept any pseudo-label rows. "
            "Lower the threshold or provide a different unlabeled pool."
        )

    prediction_trace_path = export_dir / "teacher_unlabeled_predictions.jsonl"
    prediction_summary_path = export_dir / "teacher_unlabeled_predictions.summary.json"
    _write_jsonl(prediction_trace_path, prediction_trace_rows)
    _write_json(prediction_summary_path, prediction_summary)

    student_cfg = _clone_cfg(
        cfg=cfg,
        overrides={
            "train_jsonl": teacher_seed_jsonl_ref,
            "trainer_version": (
                str(getattr(cfg, "trainer_version", "") or "").strip()
                or f"{run_id}_student"
            ),
        },
    )
    student_outputs = run_pseudo_label_self_training(
        cfg=student_cfg,
        seed_train_rows=effective_teacher_seed_rows,
        pseudo_label_rows=pseudo_label_rows,
        include_seed_train_rows=bool(
            getattr(cfg, "student_include_seed_train_rows", False)
        ),
        export_root=str(cfg.pseudo_label_export_root),
        generated_at=effective_generated_at,
        categories_override=tuple(trained_teacher.categories),
    )

    bootstrap_summary_path = export_dir / "bootstrap.summary.json"
    bootstrap_summary = {
        "schema_version": BOOTSTRAP_SUMMARY_SCHEMA_VERSION,
        "bootstrap_version": run_id,
        "generated_at": effective_generated_at.isoformat(),
        "teacher_train_jsonl": teacher_seed_jsonl_ref,
        "teacher_unlabeled_jsonl": teacher_unlabeled_jsonl_ref,
        "teacher_seed_row_count": len(effective_teacher_seed_rows),
        "teacher_unlabeled_row_count": len(effective_teacher_unlabeled_rows),
        "teacher_outputs": teacher_outputs,
        "prediction_trace_jsonl": str(prediction_trace_path),
        "prediction_summary_json": str(prediction_summary_path),
        "student_outputs": student_outputs,
        "split_manifest": (
            None if split_artifacts is None else str(split_artifacts.manifest_path)
        ),
    }
    _write_json(bootstrap_summary_path, bootstrap_summary)

    outputs = {
        "bootstrap_summary": str(bootstrap_summary_path),
        "prediction_trace_jsonl": str(prediction_trace_path),
        "prediction_summary_json": str(prediction_summary_path),
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


def _build_teacher_pseudo_label_rows(
    *,
    rows: Sequence[LabeledQueryRow],
    predictions: Sequence[FixedClassifierPrediction],
    pseudo_label_algorithm,
    generated_at: datetime,
    run_id: str,
) -> tuple[list[LabeledQueryRow], list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length.")

    algorithm = build_query_ssl_algorithm(
        pseudo_label_algorithm.acceptance_policy_name
    )
    accepted_rows: list[LabeledQueryRow] = []
    trace_rows: list[dict[str, Any]] = []
    accepted_label_counts: Counter[str] = Counter()
    hidden_label_counts: Counter[str] = Counter()
    accepted_hidden_label_counts: Counter[str] = Counter()
    accepted_correct = 0

    for row, prediction in zip(rows, predictions, strict=True):
        hidden_label = str(row["mapped_label_4"])
        decision = algorithm.evaluate(
            evidence=_build_teacher_evidence(
                row=row,
                prediction=prediction,
                generated_at=generated_at,
                run_id=run_id,
            ),
            config=pseudo_label_algorithm.config,
        )
        hidden_label_counts[hidden_label] += 1
        if decision.accepted:
            accepted_label_counts[decision.label] += 1
            accepted_hidden_label_counts[hidden_label] += 1
            if decision.label == hidden_label:
                accepted_correct += 1
            accepted_rows.append(
                LabeledQueryRow(
                    query_id=str(row["query_id"]),
                    text=str(row["text"]),
                    raw_label_scheme="pseudo_label",
                    raw_label=decision.label,
                    mapped_label_4=decision.label,
                    locale=str(row["locale"]),
                    annotation_source="fixed_classifier_teacher_bootstrap",
                    approved_by=None,
                    created_at=generated_at.isoformat(),
                )
            )
        trace_rows.append(
            {
                "schema_version": TEACHER_PREDICTION_TRACE_SCHEMA_VERSION,
                "query_id": str(row["query_id"]),
                "hidden_true_label": hidden_label,
                "predicted_label": prediction.predicted_label,
                "confidence": round(prediction.confidence, 6),
                "margin": round(prediction.margin, 6),
                "runner_up_label": prediction.runner_up_label,
                "runner_up_score": round(
                    0.0
                    if prediction.runner_up_score is None
                    else float(prediction.runner_up_score),
                    6,
                ),
                "threshold_accepted": decision.accepted,
                "final_accepted": decision.accepted,
                "is_prediction_correct": prediction.predicted_label == hidden_label,
                "category_scores": {
                    label: round(score, 6)
                    for label, score in sorted(prediction.raw_scores.items())
                },
            }
        )

    summary = {
        "schema_version": TEACHER_PREDICTION_SUMMARY_SCHEMA_VERSION,
        "bootstrap_version": run_id,
        "total_rows": len(rows),
        "accepted_count": len(accepted_rows),
        "accepted_ratio": round(
            len(accepted_rows) / len(rows) if rows else 0.0,
            6,
        ),
        "pseudo_label_algorithm": (
            pseudo_label_algorithm.to_manifest_entry()
        ),
        "accepted_label_counts": dict(sorted(accepted_label_counts.items())),
        "hidden_label_counts": dict(sorted(hidden_label_counts.items())),
        "accepted_hidden_label_counts": dict(
            sorted(accepted_hidden_label_counts.items())
        ),
        "accepted_hidden_label_accuracy": round(
            accepted_correct / len(accepted_rows) if accepted_rows else 0.0,
            6,
        ),
    }
    return accepted_rows, trace_rows, summary


def _build_teacher_evidence(
    *,
    row: LabeledQueryRow,
    prediction: FixedClassifierPrediction,
    generated_at: datetime,
    run_id: str,
) -> Any:
    from shared.src.domain.entities.training.pseudo_label_evidence import (
        PSEUDO_LABEL_EVIDENCE_V1,
        PseudoLabelEvidence,
    )

    return PseudoLabelEvidence(
        schema_version=PSEUDO_LABEL_EVIDENCE_V1,
        evidence_id=f"{run_id}:{row['query_id']}",
        source_event_ref=str(row["query_id"]),
        occurred_at=_parse_row_timestamp(str(row["created_at"]), generated_at),
        label=prediction.predicted_label,
        confidence=prediction.confidence,
        confidence_kind="classifier_posterior_top1",
        margin=prediction.margin,
        top1_label=prediction.predicted_label,
        top1_score=prediction.confidence,
        top2_label=prediction.runner_up_label,
        top2_score=(
            0.0
            if prediction.runner_up_score is None
            else float(prediction.runner_up_score)
        ),
        raw_scores=dict(prediction.raw_scores),
    )


def _parse_row_timestamp(value: str, fallback: datetime) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return fallback


def _resolve_run_id(
    *,
    cfg: DictConfig,
    generated_at: datetime,
) -> str:
    bootstrap_version = str(getattr(cfg, "bootstrap_version", "") or "").strip()
    if bootstrap_version:
        return bootstrap_version
    return generated_at.strftime("lora_bootstrap_%Y_%m_%d_%H%M%S")


def _clone_cfg(
    *,
    cfg: DictConfig,
    overrides: Mapping[str, object],
) -> DictConfig:
    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in overrides.items():
        setattr(cloned, key, value)
    return cloned


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


__all__ = ["run_fixed_classifier_teacher_lora_student_bootstrap"]
