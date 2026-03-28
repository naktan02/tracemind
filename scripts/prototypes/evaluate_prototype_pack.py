"""PrototypePack을 labeled query set에 대해 평가한다."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.src.infrastructure.model_adapters.embedding.factory import (  # noqa: E402
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from agent.src.services.scoring_service import ScoringService  # noqa: E402
from shared.src.contracts.prototype_contracts import (  # noqa: E402
    extract_category_centroids,
    load_prototype_pack_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a prototype pack on one or more labeled query set JSONL files."
    )
    parser.add_argument(
        "--prototype-pack",
        required=True,
        type=Path,
        help="Path to the prototype pack JSON file.",
    )
    parser.add_argument(
        "--eval-set",
        action="append",
        dest="eval_sets",
        default=[],
        help="Evaluation set in the form name=path/to/file.jsonl. Repeat for multiple sets.",
    )
    parser.add_argument(
        "--backend",
        choices=EmbeddingAdapterFactory.supported_backends(),
        default="transformers_mxbai",
        help="Embedding backend used for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the embedding backend.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("hf_cache"),
        help="Cache directory for model files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device for the embedding backend (auto/cpu/cuda/cuda:0/mps).",
    )
    parser.add_argument(
        "--task-prefix",
        default="",
        help="Optional prefix added before each text for embedding models.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model files only from the local cache.",
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=256,
        help="Vector dimension for the hash_debug backend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/evaluations/prototype_packs"),
        help="Directory where evaluation reports are written.",
    )
    return parser.parse_args()


def parse_eval_set(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            "eval-set must be in the form name=path/to/file.jsonl."
        )
    name, raw_path = value.split("=", 1)
    dataset_name = name.strip()
    path = Path(raw_path.strip())
    if not dataset_name:
        raise ValueError("eval-set name must not be empty.")
    if not raw_path.strip():
        raise ValueError("eval-set path must not be empty.")
    return dataset_name, path


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


def main() -> None:
    args = parse_args()
    if not args.eval_sets:
        raise ValueError("At least one --eval-set must be provided.")

    payload = load_prototype_pack_payload(args.prototype_pack)
    prototypes = extract_category_centroids(payload)
    adapter = EmbeddingAdapterFactory.create(
        EmbeddingAdapterSpec(
            backend=args.backend,
            model_id=payload.embedding_model_id,
            revision=payload.embedding_model_revision,
            device=args.device,
            normalize_embeddings=True,
            batch_size=args.batch_size,
            cache_dir=str(args.cache_dir),
            task_prefix=args.task_prefix,
            hash_dim=args.hash_dim,
            local_files_only=args.local_files_only,
        )
    )

    output_dir = args.output_dir / payload.prototype_version
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_eval_set in args.eval_sets:
        dataset_name, input_jsonl = parse_eval_set(raw_eval_set)
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
            "prototype_pack": str(args.prototype_pack),
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
