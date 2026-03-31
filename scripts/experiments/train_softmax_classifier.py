"""고정 임베딩 위에 linear classifier head + softmax를 학습하고 평가한다."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from agent.src.infrastructure.model_adapters.embedding.factory import (  # noqa: E402
    EmbeddingAdapterFactory,
)
from agent.src.infrastructure.runtime import resolve_runtime_device  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def batched_rows(
    rows: list[dict[str, Any]],
    chunk_size: int,
) -> list[list[dict[str, Any]]]:
    return [
        rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)
    ]


def embed_rows(
    *,
    rows: list[dict[str, Any]],
    adapter: Any,
    chunk_size: int,
) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for chunk in batched_rows(rows, chunk_size):
        texts = [row["text"] for row in chunk]
        embeddings = adapter.embed_texts(texts)
        tensors.append(torch.tensor(embeddings, dtype=torch.float32))
    return torch.cat(tensors, dim=0)


def build_label_index(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    categories = sorted({row["mapped_label_4"] for row in rows})
    return categories, {category: index for index, category in enumerate(categories)}


def labels_to_tensor(
    rows: list[dict[str, Any]],
    label_to_index: dict[str, int],
) -> torch.Tensor:
    indices = [label_to_index[row["mapped_label_4"]] for row in rows]
    return torch.tensor(indices, dtype=torch.long)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_confusion_matrix(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = {
        actual: {predicted: 0 for predicted in categories} for actual in categories
    }
    for actual, predicted in zip(actual_labels, predicted_labels, strict=True):
        matrix[actual][predicted] += 1
    return matrix


def summarize_per_category(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    true_probs: list[float],
    top_1_probs: list[float],
    margins: list[float],
) -> dict[str, dict[str, float | int]]:
    support_counter = Counter(actual_labels)
    predicted_counter = Counter(predicted_labels)
    correct_counter = Counter(
        actual
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
        if actual == predicted
    )
    true_prob_buckets: dict[str, list[float]] = defaultdict(list)
    top_1_prob_buckets: dict[str, list[float]] = defaultdict(list)
    margin_buckets: dict[str, list[float]] = defaultdict(list)
    for actual, true_prob, top_1_prob, margin in zip(
        actual_labels,
        true_probs,
        top_1_probs,
        margins,
        strict=True,
    ):
        true_prob_buckets[actual].append(true_prob)
        top_1_prob_buckets[actual].append(top_1_prob)
        margin_buckets[actual].append(margin)

    per_category: dict[str, dict[str, float | int]] = {}
    for category in categories:
        support = support_counter[category]
        correct = correct_counter[category]
        predicted = predicted_counter[category]
        precision = safe_divide(correct, predicted)
        recall = safe_divide(correct, support)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_category[category] = {
            "support": support,
            "predicted": predicted,
            "correct": correct,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "mean_true_label_probability": round(
                safe_divide(
                    sum(true_prob_buckets[category]),
                    len(true_prob_buckets[category]),
                ),
                6,
            ),
            "mean_top_1_probability": round(
                safe_divide(
                    sum(top_1_prob_buckets[category]),
                    len(top_1_prob_buckets[category]),
                ),
                6,
            ),
            "mean_margin_top1_top2": round(
                safe_divide(
                    sum(margin_buckets[category]),
                    len(margin_buckets[category]),
                ),
                6,
            ),
        }
    return per_category


def render_confusion_table(confusion_matrix: dict[str, dict[str, int]]) -> str:
    categories = list(confusion_matrix)
    header = ["actual \\ predicted"] + categories
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for actual in categories:
        row = [actual]
        for predicted in categories:
            row.append(str(confusion_matrix[actual][predicted]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_per_category_table(per_category: dict[str, dict[str, float | int]]) -> str:
    header = [
        "category",
        "support",
        "precision",
        "recall",
        "f1",
        "mean_true_prob",
        "mean_top1_prob",
        "mean_margin",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for category in sorted(per_category):
        metrics = per_category[category]
        row = [
            category,
            str(metrics["support"]),
            f"{float(metrics['precision']):.4f}",
            f"{float(metrics['recall']):.4f}",
            f"{float(metrics['f1']):.4f}",
            f"{float(metrics['mean_true_label_probability']):.4f}",
            f"{float(metrics['mean_top_1_probability']):.4f}",
            f"{float(metrics['mean_margin_top1_top2']):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def evaluate_classifier(
    *,
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    categories: list[str],
    eval_batch_size: int,
    device: str,
) -> dict[str, Any]:
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
        true_probs=true_probs,
        top_1_probs=top_1_probs,
        margins=margins,
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


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiments/train_softmax_classifier",
)
def main(cfg: DictConfig) -> None:
    eval_set_map = {name: Path(str(path)) for name, path in cfg.eval_sets.items()}
    if cfg.selection_set not in eval_set_map:
        raise ValueError(
            f"selection_set '{cfg.selection_set}' is not included in eval_sets."
        )

    train_rows = load_jsonl(Path(cfg.train_jsonl))
    categories, label_to_index = build_label_index(train_rows)
    embedding_spec = instantiate(cfg.embedding.spec)
    training_device = resolve_runtime_device(embedding_spec.device)
    classifier_version = cfg.classifier_version or datetime.now(timezone.utc).strftime(
        "clf_%Y_%m_%d_%H%M%S"
    )

    adapter = EmbeddingAdapterFactory.create(embedding_spec)

    print(f"embedding_train_rows={len(train_rows)}", flush=True)
    train_features = embed_rows(
        rows=train_rows,
        adapter=adapter,
        chunk_size=int(cfg.embed_chunk_size),
    )
    train_targets = labels_to_tensor(train_rows, label_to_index)

    eval_features_by_name: dict[str, torch.Tensor] = {}
    eval_targets_by_name: dict[str, torch.Tensor] = {}
    eval_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for dataset_name, path in eval_set_map.items():
        rows = load_jsonl(path)
        eval_rows_by_name[dataset_name] = rows
        print(f"embedding_eval_set={dataset_name} rows={len(rows)}", flush=True)
        eval_features_by_name[dataset_name] = embed_rows(
            rows=rows,
            adapter=adapter,
            chunk_size=int(cfg.embed_chunk_size),
        )
        eval_targets_by_name[dataset_name] = labels_to_tensor(rows, label_to_index)

    selection_features = eval_features_by_name[str(cfg.selection_set)]
    selection_targets = eval_targets_by_name[str(cfg.selection_set)]
    model, history, best_selection_report = train_classifier_head(
        train_features=train_features,
        train_targets=train_targets,
        selection_features=selection_features,
        selection_targets=selection_targets,
        categories=categories,
        training_device=training_device,
        epochs=int(cfg.epochs),
        train_batch_size=int(cfg.train_batch_size),
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    results: dict[str, Any] = {}
    for dataset_name in eval_set_map:
        report = evaluate_classifier(
            model=model,
            features=eval_features_by_name[dataset_name],
            targets=eval_targets_by_name[dataset_name],
            categories=categories,
            eval_batch_size=int(cfg.train_batch_size),
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
        print(render_per_category_table(report["per_category"]))
        print()

    output_dir = Path(str(cfg.output_dir))
    model_output_dir = Path(str(cfg.model_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / classifier_version
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_output_dir / f"{classifier_version}.pt"
    manifest_path = model_output_dir / f"{classifier_version}.manifest.json"
    report_path = report_dir / "report.json"

    torch.save(
        {
            "classifier_state_dict": model.state_dict(),
            "categories": categories,
            "embedding_model_id": embedding_spec.model_id,
            "embedding_model_revision": embedding_spec.revision,
            "backend": embedding_spec.backend,
        },
        model_path,
    )
    manifest = {
        "classifier_version": classifier_version,
        "train_jsonl": str(cfg.train_jsonl),
        "eval_sets": {name: str(path) for name, path in eval_set_map.items()},
        "selection_set": cfg.selection_set,
        "embedding_backend": embedding_spec.backend,
        "embedding_model_id": embedding_spec.model_id,
        "embedding_model_revision": embedding_spec.revision,
        "task_prefix": embedding_spec.task_prefix,
        "device": training_device,
        "epochs": int(cfg.epochs),
        "train_batch_size": int(cfg.train_batch_size),
        "learning_rate": float(cfg.learning_rate),
        "weight_decay": float(cfg.weight_decay),
        "categories": categories,
        "model_path": str(model_path),
        "best_selection_report": best_selection_report,
        "history": history,
    }
    report = {
        "schema_version": "classifier_head_eval.v1",
        "classifier_version": classifier_version,
        "manifest": manifest,
        "results": results,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"model_path={model_path}")
    print(f"manifest={manifest_path}")
    print(f"report_json={report_path}")


if __name__ == "__main__":
    main()
