"""PrototypePack 평가 Hydra entrypoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from agent.src.infrastructure.model_adapters.embedding.factory import (  # noqa: E402
    EmbeddingAdapterFactory,
)
from scripts.classification_report import (  # noqa: E402
    render_confusion_table,
    render_per_category_table,
)
from scripts.prototypes.evaluation import evaluate_rows  # noqa: E402
from scripts.prototypes.io import load_jsonl  # noqa: E402
from scripts.run_artifacts import build_run_dir  # noqa: E402
from shared.src.contracts.prototype_contracts import (  # noqa: E402
    extract_category_prototypes,
    load_prototype_pack_payload,
)
from shared.src.domain.value_objects import EmbeddingAdapterSpec  # noqa: E402


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="jobs/prototypes/evaluate_prototype_pack",
)
def main(cfg: DictConfig) -> None:
    if not cfg.prototype_pack:
        raise ValueError("prototype_pack must be set.")

    payload = load_prototype_pack_payload(Path(cfg.prototype_pack))
    prototypes = extract_category_prototypes(payload)
    spec_cfg = OmegaConf.create(
        {
            "_target_": "shared.src.domain.value_objects.EmbeddingAdapterSpec",
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
    embedding_spec = instantiate(spec_cfg, _convert_="object")
    if not isinstance(embedding_spec, EmbeddingAdapterSpec):
        raise TypeError(
            "embedding.spec must instantiate to EmbeddingAdapterSpec, "
            f"got {type(embedding_spec)!r}."
        )
    adapter = EmbeddingAdapterFactory.create(embedding_spec)

    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = build_run_dir(
        cfg.output_dir,
        run_id=run_id,
        created_at=created_at,
    )
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

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
        output_path = reports_dir / f"{dataset_name}.json"
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
        print(
            render_per_category_table(
                evaluation["per_category"],
                primary_metric_key="mean_true_label_score",
                top_1_metric_key="mean_top_1_score",
                primary_header="mean_true_score",
                top_1_header="mean_top1_score",
            )
        )
        print()
        print(f"report_json={output_path}")
        print()

    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
