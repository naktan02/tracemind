"""고정 임베딩 + classifier 실험 helper."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
)
from agent.src.infrastructure.runtime import resolve_runtime_device
from scripts.classification_report import (
    build_confusion_matrix,
    render_confusion_table,
    render_per_category_table,
    safe_divide,
    summarize_per_category,
)
from scripts.labeled_query_rows import LabeledQueryRow, load_labeled_query_rows
from scripts.run_artifacts import build_run_dir
from shared.src.domain.services import EmbeddingAdapter
from shared.src.domain.value_objects.embedding_adapter_spec import EmbeddingAdapterSpec


@dataclass(slots=True)
class TrainedFixedClassifier:
    """학습된 fixed encoder + classifier bundle."""

    model: nn.Module
    adapter: EmbeddingAdapter
    embedding_spec: Any
    categories: list[str]
    label_to_index: dict[str, int]
    training_device: str
    history: list[dict[str, Any]]
    best_selection_report: dict[str, Any]
    eval_results: dict[str, Any]


@dataclass(slots=True)
class FixedClassifierPrediction:
    """teacher classifier의 unlabeled row 추론 결과."""

    query_id: str
    predicted_label: str
    confidence: float
    margin: float
    runner_up_label: str | None
    runner_up_score: float | None
    raw_scores: dict[str, float]


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

    correct = sum(
        1
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
        if actual == predicted
    )
    confusion_matrix = build_confusion_matrix(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )
    per_category = summarize_per_category(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        primary_values=true_probs,
        top_1_values=top_1_probs,
        margins=margins,
        primary_metric_key="mean_true_label_probability",
        top_1_metric_key="mean_top_1_probability",
    )
    return {
        "rows_total": total_rows,
        "loss": round(safe_divide(total_loss, total_rows), 6),
        "accuracy_top_1": round(safe_divide(correct, total_rows), 6),
        "correct_top_1": correct,
        "mean_true_label_probability": round(
            safe_divide(sum(true_probs), len(true_probs)),
            6,
        ),
        "mean_top_1_probability": round(
            safe_divide(sum(top_1_probs), len(top_1_probs)),
            6,
        ),
        "mean_margin_top1_top2": round(
            safe_divide(sum(margins), len(margins)),
            6,
        ),
        "confusion_matrix": confusion_matrix,
        "per_category": per_category,
    }


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
    history: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_selection_report: dict[str, Any] | None = None
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
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

        selection_report = evaluate_classifier(
            model=model,
            features=selection_features,
            targets=selection_targets,
            categories=categories,
            eval_batch_size=train_batch_size,
            device=training_device,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(
                safe_divide(epoch_loss_total, len(train_targets)),
                6,
            ),
            "selection_loss": selection_report["loss"],
            "selection_accuracy_top_1": selection_report["accuracy_top_1"],
        }
        history.append(epoch_record)
        print(
            f"[epoch={epoch}] "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"selection_loss={epoch_record['selection_loss']:.4f} "
            f"selection_accuracy={epoch_record['selection_accuracy_top_1']:.4f}",
            flush=True,
        )

        accuracy = float(selection_report["accuracy_top_1"])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
            best_selection_report = selection_report

    if best_state_dict is None or best_selection_report is None:
        raise RuntimeError("Classifier training did not produce a best checkpoint.")

    model.load_state_dict(best_state_dict)
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
    training_device = resolve_runtime_device(embedding_spec.device)
    adapter = EmbeddingAdapterFactory.create(embedding_spec)

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


def write_fixed_classifier_artifacts(
    *,
    classifier_version: str,
    created_at: datetime,
    train_jsonl_ref: str,
    eval_set_map: dict[str, str],
    selection_set_name: str,
    output_dir_root: str,
    model_output_dir: str,
    epochs: int,
    train_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    trained: TrainedFixedClassifier,
) -> dict[str, str]:
    """fixed classifier 산출물을 저장한다."""

    output_dir = build_run_dir(
        output_dir_root,
        run_id=classifier_version,
        created_at=created_at,
    )
    resolved_model_output_dir = Path(model_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    model_path = resolved_model_output_dir / f"{classifier_version}.pt"
    manifest_path = resolved_model_output_dir / f"{classifier_version}.manifest.json"
    report_path = report_dir / "report.json"

    torch.save(
        {
            "classifier_state_dict": trained.model.state_dict(),
            "categories": trained.categories,
            "embedding_model_id": trained.embedding_spec.model_id,
            "embedding_model_revision": trained.embedding_spec.revision,
            "backend": trained.embedding_spec.backend,
        },
        model_path,
    )
    manifest = {
        "classifier_version": classifier_version,
        "train_jsonl": train_jsonl_ref,
        "eval_sets": dict(eval_set_map),
        "selection_set": selection_set_name,
        "embedding_backend": trained.embedding_spec.backend,
        "embedding_model_id": trained.embedding_spec.model_id,
        "embedding_model_revision": trained.embedding_spec.revision,
        "task_prefix": trained.embedding_spec.task_prefix,
        "device": trained.training_device,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "categories": trained.categories,
        "model_path": str(model_path),
        "best_selection_report": trained.best_selection_report,
        "history": trained.history,
    }
    report = {
        "schema_version": "classifier_head_eval.v1",
        "classifier_version": classifier_version,
        "manifest": manifest,
        "results": trained.eval_results,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "manifest": str(manifest_path),
        "report_json": str(report_path),
    }


def load_fixed_classifier_artifacts(
    *,
    manifest_path: str | Path,
    device: str,
    batch_size: int,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> tuple[TrainedFixedClassifier, dict[str, str]]:
    """저장된 fixed classifier artifact를 teacher/reference로 다시 연다."""

    resolved_manifest_path = Path(str(manifest_path))
    manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    model_path = Path(str(manifest["model_path"]))
    serialized = cast(
        dict[str, Any],
        torch.load(model_path, map_location="cpu"),
    )
    categories = [str(category) for category in serialized["categories"]]
    state_dict = cast(dict[str, torch.Tensor], serialized["classifier_state_dict"])
    hidden_size = int(serialized.get("hidden_size", state_dict["weight"].shape[1]))
    training_device = resolve_runtime_device(device)
    model = nn.Linear(hidden_size, len(categories)).to(training_device)
    model.load_state_dict(state_dict)
    model.eval()

    embedding_spec = EmbeddingAdapterSpec(
        backend=str(manifest["embedding_backend"]),
        model_id=str(manifest["embedding_model_id"]),
        revision=str(manifest["embedding_model_revision"]),
        device=training_device,
        batch_size=batch_size,
        cache_dir=cache_dir,
        task_prefix=str(manifest.get("task_prefix", "")),
        local_files_only=local_files_only,
    )
    adapter = EmbeddingAdapterFactory.create(embedding_spec)
    classifier_version = str(manifest["classifier_version"])
    report_path = (
        Path("runs/train_classifier") / classifier_version / "reports" / "report.json"
    )
    outputs = {
        "model_path": str(model_path),
        "manifest": str(resolved_manifest_path),
        "report_json": str(report_path),
    }
    if report_path.exists():
        outputs["output_dir"] = str(report_path.parent.parent)

    return (
        TrainedFixedClassifier(
            model=model,
            adapter=adapter,
            embedding_spec=embedding_spec,
            categories=categories,
            label_to_index={label: index for index, label in enumerate(categories)},
            training_device=training_device,
            history=[],
            best_selection_report=dict(manifest.get("best_selection_report", {})),
            eval_results={},
        ),
        outputs,
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

    eval_set_map = {name: Path(str(path)) for name, path in cfg.eval_sets.items()}
    if cfg.selection_set not in eval_set_map:
        raise ValueError(
            f"selection_set '{cfg.selection_set}' is not included in eval_sets."
        )

    effective_train_rows = (
        load_labeled_query_rows(Path(str(cfg.train_jsonl)))
        if train_rows is None
        else list(train_rows)
    )
    effective_eval_rows = (
        {name: load_labeled_query_rows(path) for name, path in eval_set_map.items()}
        if eval_rows_by_name is None
        else {name: list(rows) for name, rows in eval_rows_by_name.items()}
    )
    created_at = datetime.now(timezone.utc)
    effective_classifier_version = classifier_version or (
        cfg.classifier_version or created_at.strftime("clf_%Y_%m_%d_%H%M%S")
    )
    embedding_spec = instantiate(cfg.embedding.spec)
    trained = train_fixed_embedding_classifier(
        train_rows=effective_train_rows,
        eval_rows_by_name=effective_eval_rows,
        selection_set_name=str(cfg.selection_set),
        embedding_spec=embedding_spec,
        embed_chunk_size=int(cfg.embed_chunk_size),
        train_batch_size=int(cfg.train_batch_size),
        eval_batch_size=int(cfg.train_batch_size),
        epochs=int(cfg.epochs),
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    outputs = write_fixed_classifier_artifacts(
        classifier_version=effective_classifier_version,
        created_at=created_at,
        train_jsonl_ref=str(train_jsonl_ref or cfg.train_jsonl),
        eval_set_map={name: str(path) for name, path in eval_set_map.items()},
        selection_set_name=str(cfg.selection_set),
        output_dir_root=str(output_dir_root or cfg.output_dir),
        model_output_dir=str(model_output_dir or cfg.model_output_dir),
        epochs=int(cfg.epochs),
        train_batch_size=int(cfg.train_batch_size),
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        trained=trained,
    )
    print(f"output_dir={outputs['output_dir']}")
    print(f"model_path={outputs['model_path']}")
    print(f"manifest={outputs['manifest']}")
    print(f"report_json={outputs['report_json']}")
    return outputs


__all__ = [
    "FixedClassifierPrediction",
    "TrainedFixedClassifier",
    "batched_rows",
    "build_label_index",
    "embed_rows",
    "evaluate_classifier",
    "labels_to_tensor",
    "load_fixed_classifier_artifacts",
    "predict_fixed_classifier_rows",
    "run_fixed_embedding_classifier",
    "train_classifier_head",
    "train_fixed_embedding_classifier",
    "write_fixed_classifier_artifacts",
]
