"""Prototype threshold sweep CLI entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from scripts.experiments.prototype_analysis.prototype_strategy.io_utils import (
    load_jsonl_rows,
    resolve_output_dir,
)
from scripts.experiments.prototype_analysis.prototype_strategy.sweep import (
    ThresholdPolicyExperimentRequest,
    render_sweep_summary,
)
from scripts.runtime_adapters.embedding_runtime import create_embedding_adapter


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/prototype_analysis/prototype_threshold_sweep",
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

    adapter = create_embedding_adapter(instantiate(config.embedding.spec))
    threshold_policies = tuple(
        instantiate(policy_config) for policy_config in config.threshold_policies
    )
    summary = instantiate(config.runner).run(
        ThresholdPolicyExperimentRequest(
            train_rows=load_jsonl_rows(config.dataset.train_jsonl),
            validation_rows=load_jsonl_rows(config.dataset.validation_jsonl),
            test_rows=load_jsonl_rows(config.dataset.test_jsonl),
            adapter=adapter,
            strategy_name=config.strategy.name,
            seed=int(config.strategy.seed),
            kmeans_candidate_ks=tuple(config.strategy.kmeans_candidate_ks),
            kmeans_silhouette_sample_size=int(
                config.strategy.kmeans_silhouette_sample_size
            ),
            dbscan_eps_values=tuple(config.strategy.dbscan_eps_values),
            dbscan_min_samples_values=tuple(config.strategy.dbscan_min_samples_values),
            dbscan_search_sample_size=int(config.strategy.dbscan_search_sample_size),
            dbscan_min_cluster_coverage=float(
                config.strategy.dbscan_min_cluster_coverage
            ),
            threshold_policies=threshold_policies,
            output_dir=output_dir,
            run_id=run_id,
        )
    )
    print(f"output_dir={output_dir}")
    print(render_sweep_summary(summary))


if __name__ == "__main__":
    main()
