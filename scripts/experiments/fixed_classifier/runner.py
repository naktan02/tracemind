"""고정 임베딩 + classifier 실험 helper."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from methods.adaptation.common.classification_evaluation import (
    build_classification_evaluation_report,
)
from methods.adaptation.common.selection_training_loop import (
    SelectionTrackedEpochResult,
    run_selection_tracked_training_loop,
)
from scripts.runtime_adapters.embedding_runtime import (
    create_embedding_adapter,
    resolve_runtime_device_name,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow
from shared.src.domain.services.classification_report import (
    render_confusion_table,
    render_per_category_table,
)
from shared.src.domain.services.embedding_adapter import EmbeddingAdapter

from .artifacts import write_fixed_classifier_artifacts
from .common import prepare_fixed_classifier_run_context
from .models import FixedClassifierPrediction, TrainedFixedClassifier


def batched_rows(
    rows: list[LabeledQueryRow],
    chunk_size: int,
) -> list[list[LabeledQueryRow]]:
    return [
        rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)
    ]


def embed_rows(
    *,
    rows: list[LabeledQueryRow],
    adapter: EmbeddingAdapter,
    chunk_size: int,
) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for chunk in batched_rows(rows, chunk_size):
        texts = [row["text"] for row in chunk]
        embeddings = adapter.embed_texts(texts)
        tensors.append(torch.tensor(embeddings, dtype=torch.float32))
    return torch.cat(tensors, dim=0)


def build_label_index(
    rows: list[LabeledQueryRow],
) -> tuple[list[str], dict[str, int]]:
    categories = sorted({row["mapped_label_4"] for row in rows})
    return categories, {category: index for index, category in enumerate(categories)}


def labels_to_tensor(
    rows: list[LabeledQueryRow],
    label_to_index: dict[str, int],
) -> torch.Tensor:
    indices = [label_to_index[row["mapped_label_4"]] for row in rows]
    return torch.tensor(indices, dtype=torch.long)


def evaluate_classifier(
    *,
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    categories: list[str],
    eval_batch_size: int,
    device: str,
) -> dict[str, Any]:
    """분류 결과 지표를 계산한다."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    actual_labels: list[str] = []
    predicted_labels: list[str] = []
    true_probs: list[float] = []
    top_1_probs: list[float] = []
    margins: list[float] = []
    total_loss = 0.0
    total_rows = len(targets)

    with torch.no_grad():
        for start in range(0, total_rows, eval_batch_size):
            end = min(start + eval_batch_size, total_rows)
            batch_features = features[start:end].to(device)
            batch_targets = targets[start:end].to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
            predicted = top_indices[:, 0]
            true_probability = probabilities.gather(
                1,
                batch_targets.unsqueeze(1),
            ).squeeze(1)

            total_loss += float(loss.item()) * len(batch_targets)
            actual_labels.extend(
                categories[index] for index in batch_targets.cpu().tolist()
            )
            predicted_labels.extend(
                categories[index] for index in predicted.cpu().tolist()
            )
            true_probs.extend(true_probability.cpu().tolist())
            top_1_probs.extend(top_values[:, 0].cpu().tolist())
            margins.extend((top_values[:, 0] - top_values[:, 1]).cpu().tolist())

    return build_classification_evaluation_report(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        true_probs=true_probs,
        top_1_values=top_1_probs,
        margins=margins,
        total_loss=total_loss,
        total_rows=total_rows,
    )


def train_classifier_head(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    selection_features: torch.Tensor,
    selection_targets: torch.Tensor,
    categories: list[str],
    training_device: str,
    epochs: int,
    train_batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    """고정 임베딩 위 linear classifier head를 학습한다."""

    model = nn.Linear(train_features.shape[1], len(categories)).to(training_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    def train_epoch(_epoch: int) -> SelectionTrackedEpochResult:
        model.train()
        permutation = torch.randperm(len(train_targets))
        epoch_loss_total = 0.0

        for start in range(0, len(train_targets), train_batch_size):
            end = min(start + train_batch_size, len(train_targets))
            indices = permutation[start:end]
            batch_features = train_features[indices].to(training_device)
            batch_targets = train_targets[indices].to(training_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss_total += float(loss.item()) * len(indices)

        return SelectionTrackedEpochResult(
            train_loss_total=epoch_loss_total,
            train_loss_denominator=len(train_targets),
        )

    def evaluate_selection() -> dict[str, Any]:
        return evaluate_classifier(
            model=model,
            features=selection_features,
            targets=selection_targets,
            categories=categories,
            eval_batch_size=train_batch_size,
            device=training_device,
        )

    history, best_selection_report = run_selection_tracked_training_loop(
        model=model,
        epochs=epochs,
        train_epoch=train_epoch,
        evaluate_selection=evaluate_selection,
        best_checkpoint_error_message=(
            "Classifier training did not produce a best checkpoint."
        ),
        log_epoch_summary=lambda message: print(message, flush=True),
    )
    return model, history, best_selection_report


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
        print(
            f"[{dataset_name}] "
            f"accuracy_top_1={report['accuracy_top_1']:.4f} "
            f"rows={report['rows_total']} "
            f"mean_true_prob={report['mean_true_label_probability']:.4f} "
            f"mean_margin={report['mean_margin_top1_top2']:.4f}",
            flush=True,
        )
        print(render_confusion_table(report["confusion_matrix"]))
        print()
        print(
            render_per_category_table(
                report["per_category"],
                primary_metric_key="mean_true_label_probability",
                top_1_metric_key="mean_top_1_probability",
                primary_header="mean_true_prob",
                top_1_header="mean_top1_prob",
            )
        )
        print()

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


def predict_fixed_classifier_rows(
    *,
    trained: TrainedFixedClassifier,
    rows: list[LabeledQueryRow],
    embed_chunk_size: int,
    eval_batch_size: int,
) -> list[FixedClassifierPrediction]:
    """학습된 fixed classifier로 unlabeled row를 추론한다."""

    features = embed_rows(
        rows=rows,
        adapter=trained.adapter,
        chunk_size=embed_chunk_size,
    )
    predictions: list[FixedClassifierPrediction] = []
    trained.model.eval()

    with torch.no_grad():
        for start in range(0, len(rows), eval_batch_size):
            end = min(start + eval_batch_size, len(rows))
            batch_rows = rows[start:end]
            batch_features = features[start:end].to(trained.training_device)
            logits = trained.model(batch_features)
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
            for row, probability_row, top_value_row, top_index_row in zip(
                batch_rows,
                probabilities.cpu(),
                top_values.cpu(),
                top_indices.cpu(),
                strict=True,
            ):
                top1_index = int(top_index_row[0].item())
                top2_index = int(top_index_row[1].item())
                raw_scores = {
                    category: float(probability_row[index].item())
                    for index, category in enumerate(trained.categories)
                }
                predictions.append(
                    FixedClassifierPrediction(
                        query_id=str(row["query_id"]),
                        predicted_label=trained.categories[top1_index],
                        confidence=float(top_value_row[0].item()),
                        margin=float((top_value_row[0] - top_value_row[1]).item()),
                        runner_up_label=trained.categories[top2_index],
                        runner_up_score=float(top_value_row[1].item()),
                        raw_scores=raw_scores,
                    )
                )
    return predictions


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
