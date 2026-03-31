"""Prototype strategy experiment CLI entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
)
from scripts.experiments.prototype_strategy.io_utils import (
    load_jsonl_rows,
    resolve_output_dir,
)
from scripts.experiments.prototype_strategy.runner import (
    PrototypeExperimentRequest,
    build_strategies,
    render_validation_summary,
)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="experiments/prototype_strategy",
)
def main(config: DictConfig) -> None:
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_output_dir(
        config.output.base_dir,
        run_id,
        created_at=created_at,
    )
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    adapter = EmbeddingAdapterFactory.create(instantiate(config.embedding.spec))
    runner = instantiate(config.runner)
    summary = runner.run(
        PrototypeExperimentRequest(
            strategies=build_strategies(
                strategy_name=str(config.strategy.name),
                seed=int(config.strategy.seed),
                kmeans_candidate_ks=tuple(config.strategy.kmeans_candidate_ks),
                kmeans_silhouette_sample_size=int(
                    config.strategy.kmeans_silhouette_sample_size
                ),
                dbscan_eps_values=tuple(config.strategy.dbscan_eps_values),
                dbscan_min_samples_values=tuple(
                    config.strategy.dbscan_min_samples_values
                ),
                dbscan_search_sample_size=int(config.strategy.dbscan_search_sample_size),
                dbscan_min_cluster_coverage=float(
                    config.strategy.dbscan_min_cluster_coverage
                ),
            ),
            train_rows=load_jsonl_rows(config.dataset.train_jsonl),
            validation_rows=load_jsonl_rows(config.dataset.validation_jsonl),
            test_rows=load_jsonl_rows(config.dataset.test_jsonl),
            adapter=adapter,
            output_dir=output_dir,
            run_id=run_id,
            projection_reducers=tuple(config.projection_reducers),
        )
    )
    print(f"output_dir={output_dir}")
    print(render_validation_summary(summary))


if __name__ == "__main__":
    main()
