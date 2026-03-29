"""PrototypePack을 labeled query set에 대해 평가한다."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.src.infrastructure.model_adapters.embedding.factory import (  # noqa: E402
    EmbeddingAdapterFactory,
)
from agent.src.services.inference.scoring_service import ScoringService  # noqa: E402
from shared.src.contracts.prototype_contracts import (  # noqa: E402
    extract_category_centroids,
    load_prototype_pack_payload,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def predict_label(scores: dict[str, float]) -> tuple[str, float, float]:
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    predicted_label, top_1_score = ranked[0]
    top_2_score = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
    return predicted_label, top_1_score, top_1_score - top_2_score


def build_confusion_matrix(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = {
        actual: {predicted: 0 for predicted in categories}
        for actual in categories
    }
    for actual, predicted in zip(actual_labels, predicted_labels, strict=True):
        matrix[actual][predicted] += 1
    return matrix


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_per_category(
    *,
    categories: list[str],
    actual_labels: list[str],
    predicted_labels: list[str],
    true_scores: list[float],
    top_1_scores: list[float],
    margins: list[float],
) -> dict[str, dict[str, float | int]]:
    support_counter = Counter(actual_labels)
    predicted_counter = Counter(predicted_labels)
    correct_counter = Counter(
        actual
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
        if actual == predicted
    )
    true_score_buckets: dict[str, list[float]] = defaultdict(list)
    top_1_score_buckets: dict[str, list[float]] = defaultdict(list)
    margin_buckets: dict[str, list[float]] = defaultdict(list)
    for actual, true_score, top_1_score, margin in zip(
        actual_labels,
        true_scores,
        top_1_scores,
        margins,
        strict=True,
    ):
        true_score_buckets[actual].append(true_score)
        top_1_score_buckets[actual].append(top_1_score)
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
            "mean_true_label_score": round(
                safe_divide(sum(true_score_buckets[category]), len(true_score_buckets[category])),
                6,
            ),
            "mean_top_1_score": round(
                safe_divide(sum(top_1_score_buckets[category]), len(top_1_score_buckets[category])),
                6,
            ),
            "mean_margin_top1_top2": round(
                safe_divide(sum(margin_buckets[category]), len(margin_buckets[category])),
                6,
            ),
        }
    return per_category


def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    prototypes: dict[str, list[float]],
    embeddings: list[list[float]],
) -> dict[str, Any]:
    scoring_service = ScoringService()
    categories = sorted(prototypes)
    actual_labels: list[str] = []
    predicted_labels: list[str] = []
    top_1_scores: list[float] = []
    true_scores: list[float] = []
    margins: list[float] = []

    for row, embedding in zip(rows, embeddings, strict=True):
        actual_label = row["mapped_label_4"]
        scores = scoring_service.score(embedding=embedding, prototypes=prototypes)
        predicted_label, top_1_score, margin = predict_label(scores)

        actual_labels.append(actual_label)
        predicted_labels.append(predicted_label)
        top_1_scores.append(top_1_score)
        true_scores.append(scores[actual_label])
        margins.append(margin)

    total = len(rows)
    correct = sum(
        1
        for actual, predicted in zip(actual_labels, predicted_labels, strict=True)
        if actual == predicted
    )
    accuracy = safe_divide(correct, total)
    confusion_matrix = build_confusion_matrix(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )
    per_category = summarize_per_category(
        categories=categories,
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
        true_scores=true_scores,
        top_1_scores=top_1_scores,
        margins=margins,
    )

    return {
        "rows_total": total,
        "accuracy_top_1": round(accuracy, 6),
        "correct_top_1": correct,
        "mean_true_label_score": round(safe_divide(sum(true_scores), len(true_scores)), 6),
        "mean_top_1_score": round(safe_divide(sum(top_1_scores), len(top_1_scores)), 6),
        "mean_margin_top1_top2": round(safe_divide(sum(margins), len(margins)), 6),
        "confusion_matrix": confusion_matrix,
        "per_category": per_category,
    }


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
        "mean_true_score",
        "mean_top1_score",
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
            f"{float(metrics['mean_true_label_score']):.4f}",
            f"{float(metrics['mean_top_1_score']):.4f}",
            f"{float(metrics['mean_margin_top1_top2']):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="prototypes/evaluate_prototype_pack",
)
def main(cfg: DictConfig) -> None:
    if not cfg.prototype_pack:
        raise ValueError("prototype_pack must be set.")

    payload = load_prototype_pack_payload(Path(cfg.prototype_pack))
    prototypes = extract_category_centroids(payload)
    spec_cfg = OmegaConf.create(
        {
            "_target_": "agent.src.infrastructure.model_adapters.embedding.factory.EmbeddingAdapterSpec",
            "backend": cfg.embedding.backend,
            "model_id": (
                payload.embedding_model_id
                if cfg.respect_pack_embedding_identity
                else cfg.embedding.model_id
            ),
            "revision": (
                payload.embedding_model_revision
                if cfg.respect_pack_embedding_identity
                else cfg.embedding.revision
            ),
            "device": cfg.runtime.device,
            "batch_size": cfg.embedding.batch_size,
            "cache_dir": cfg.embedding.cache_dir,
            "task_prefix": cfg.embedding.task_prefix,
            "hash_dim": cfg.embedding.hash_dim,
            "local_files_only": cfg.runtime.local_files_only,
        }
    )
    embedding_spec = instantiate(spec_cfg)
    adapter = EmbeddingAdapterFactory.create(
        embedding_spec
    )

    output_dir = Path(cfg.output_dir) / payload.prototype_version
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, raw_input_path in cfg.eval_sets.items():
        input_jsonl = Path(str(raw_input_path))
        rows = load_jsonl(input_jsonl)
        texts = [row["text"] for row in rows]
        embeddings = adapter.embed_texts(texts)
        evaluation = evaluate_rows(
            rows=rows,
            prototypes=prototypes,
            embeddings=embeddings,
        )
        report = {
            "prototype_version": payload.prototype_version,
            "prototype_pack": str(cfg.prototype_pack),
            "dataset_name": dataset_name,
            "input_jsonl": str(input_jsonl),
            "embedding_model_id": payload.embedding_model_id,
            "embedding_model_revision": payload.embedding_model_revision,
            "build_method": payload.build_method,
            "distance_metric": payload.distance_metric,
            "results": evaluation,
        }
        output_path = output_dir / f"{dataset_name}.json"
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        print(
            f"[{dataset_name}] "
            f"accuracy_top_1={evaluation['accuracy_top_1']:.4f} "
            f"rows={evaluation['rows_total']} "
            f"mean_true_score={evaluation['mean_true_label_score']:.4f} "
            f"mean_margin={evaluation['mean_margin_top1_top2']:.4f}"
        )
        print(render_confusion_table(evaluation["confusion_matrix"]))
        print()
        print(render_per_category_table(evaluation["per_category"]))
        print()
        print(f"report_json={output_path}")
        print()


if __name__ == "__main__":
    main()
