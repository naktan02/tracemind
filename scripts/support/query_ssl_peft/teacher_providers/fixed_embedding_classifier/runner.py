"""고정 임베딩 + classifier 실험 helper."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from scripts.runtime_adapters.embedding_runtime import (
    create_embedding_adapter,
    resolve_runtime_device_name,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow

from .artifacts import write_fixed_classifier_artifacts
from .common import prepare_fixed_classifier_run_context
from .evaluation import evaluate_classifier, print_evaluation_report
from .models import TrainedFixedClassifier
from .row_embeddings import embed_rows
from .training import build_label_index, labels_to_tensor, train_classifier_head


def train_fixed_embedding_classifier(
    *,
    train_rows: list[LabeledQueryRow],
    eval_rows_by_name: dict[str, list[LabeledQueryRow]],
    selection_set_name: str,
    embedding_spec: Any,
    embed_chunk_size: int,
    train_batch_size: int,
    eval_batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainedFixedClassifier:
    """메모리 row 기준으로 fixed embedding classifier를 학습/평가한다."""

    categories, label_to_index = build_label_index(train_rows)
    training_device = resolve_runtime_device_name(embedding_spec.device)
    adapter = create_embedding_adapter(embedding_spec)

    print(f"embedding_train_rows={len(train_rows)}", flush=True)
    train_features = embed_rows(
        rows=train_rows,
        adapter=adapter,
        chunk_size=embed_chunk_size,
    )
    train_targets = labels_to_tensor(train_rows, label_to_index)

    eval_features_by_name: dict[str, torch.Tensor] = {}
    eval_targets_by_name: dict[str, torch.Tensor] = {}
    for dataset_name, rows in eval_rows_by_name.items():
        print(f"embedding_eval_set={dataset_name} rows={len(rows)}", flush=True)
        eval_features_by_name[dataset_name] = embed_rows(
            rows=rows,
            adapter=adapter,
            chunk_size=embed_chunk_size,
        )
        eval_targets_by_name[dataset_name] = labels_to_tensor(rows, label_to_index)

    if selection_set_name not in eval_features_by_name:
        raise ValueError(
            f"selection_set '{selection_set_name}' is not included in eval rows."
        )

    model, history, best_selection_report = train_classifier_head(
        train_features=train_features,
        train_targets=train_targets,
        selection_features=eval_features_by_name[selection_set_name],
        selection_targets=eval_targets_by_name[selection_set_name],
        categories=categories,
        training_device=training_device,
        epochs=epochs,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    results: dict[str, Any] = {}
    for dataset_name, rows in eval_rows_by_name.items():
        del rows
        report = evaluate_classifier(
            model=model,
            features=eval_features_by_name[dataset_name],
            targets=eval_targets_by_name[dataset_name],
            categories=categories,
            eval_batch_size=eval_batch_size,
            device=training_device,
        )
        results[dataset_name] = report
        print_evaluation_report(dataset_name=dataset_name, report=report)

    return TrainedFixedClassifier(
        model=model,
        adapter=adapter,
        embedding_spec=embedding_spec,
        categories=categories,
        label_to_index=label_to_index,
        training_device=training_device,
        history=history,
        best_selection_report=best_selection_report,
        eval_results=results,
    )


def run_fixed_embedding_classifier(
    *,
    cfg: DictConfig,
    train_rows: list[LabeledQueryRow] | None = None,
    eval_rows_by_name: dict[str, list[LabeledQueryRow]] | None = None,
    train_jsonl_ref: str | None = None,
    output_dir_root: str | None = None,
    model_output_dir: str | None = None,
    classifier_version: str | None = None,
) -> dict[str, str]:
    """Hydra config 기준 fixed embedding classifier를 실행한다."""

    context = prepare_fixed_classifier_run_context(
        cfg=cfg,
        train_rows=train_rows,
        eval_rows_by_name=eval_rows_by_name,
        train_jsonl_ref=train_jsonl_ref,
        output_dir_root=output_dir_root,
        model_output_dir=model_output_dir,
        classifier_version=classifier_version,
    )
    trained = train_fixed_embedding_classifier(
        train_rows=context.train_rows,
        eval_rows_by_name=context.eval_rows_by_name,
        selection_set_name=context.effective_selection_set,
        embedding_spec=context.embedding_spec,
        embed_chunk_size=int(context.cfg.embed_chunk_size),
        train_batch_size=int(context.cfg.train_batch_size),
        eval_batch_size=int(context.cfg.train_batch_size),
        epochs=int(context.cfg.epochs),
        learning_rate=float(context.cfg.learning_rate),
        weight_decay=float(context.cfg.weight_decay),
    )
    outputs = write_fixed_classifier_artifacts(
        classifier_version=context.classifier_version,
        created_at=context.created_at,
        train_jsonl_ref=context.effective_train_jsonl_ref,
        eval_set_map={name: str(path) for name, path in context.eval_set_map.items()},
        selection_set_name=context.effective_selection_set,
        output_dir_root=context.output_dir_root,
        model_output_dir=context.model_output_dir,
        epochs=int(context.cfg.epochs),
        train_batch_size=int(context.cfg.train_batch_size),
        learning_rate=float(context.cfg.learning_rate),
        weight_decay=float(context.cfg.weight_decay),
        trained=trained,
    )
    print(f"output_dir={outputs['output_dir']}")
    print(f"model_path={outputs['model_path']}")
    print(f"manifest={outputs['manifest']}")
    print(f"report_json={outputs['report_json']}")
    return outputs
