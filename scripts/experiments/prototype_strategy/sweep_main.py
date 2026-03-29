"""Threshold sweep entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from scripts.experiments.prototype_strategy.cli import parse_cli_args
from scripts.experiments.prototype_strategy.io_utils import load_jsonl_rows
from scripts.experiments.prototype_strategy.sweep import (
    ThresholdSweepRunner,
    render_sweep_summary,
)
from scripts.experiments.prototype_strategy.sweep_config import (
    DEFAULT_SWEEP_CONFIG_PATH,
    load_threshold_sweep_config,
)


def main() -> None:
    cli_args = parse_cli_args(default_config_path=DEFAULT_SWEEP_CONFIG_PATH)
    config = load_threshold_sweep_config(
        config_path=cli_args.config_path,
        overrides=cli_args.overrides,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.output.output_dir / run_id

    adapter = EmbeddingAdapterFactory.create(
        EmbeddingAdapterSpec(
            backend=config.embedding.backend,
            model_id=config.embedding.model_id,
            revision=config.embedding.revision,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            cache_dir=str(config.embedding.cache_dir),
            task_prefix=config.embedding.task_prefix,
            hash_dim=config.embedding.hash_dim,
            local_files_only=config.embedding.local_files_only,
        )
    )
    summary = ThresholdSweepRunner().run(
        train_rows=load_jsonl_rows(config.dataset.train_jsonl),
        validation_rows=load_jsonl_rows(config.dataset.validation_jsonl),
        test_rows=load_jsonl_rows(config.dataset.test_jsonl),
        adapter=adapter,
        strategy_name=config.strategy.name,
        seed=config.runtime.seed,
        kmeans_candidate_ks=config.strategy.kmeans.candidate_ks,
        kmeans_silhouette_sample_size=config.strategy.kmeans.silhouette_sample_size,
        dbscan_eps_values=config.strategy.dbscan.eps_values,
        dbscan_min_samples_values=config.strategy.dbscan.min_samples_values,
        dbscan_search_sample_size=config.strategy.dbscan.search_sample_size,
        dbscan_min_cluster_coverage=config.strategy.dbscan.min_cluster_coverage,
        confidence_thresholds=config.threshold_grid.confidence_thresholds,
        margin_thresholds=config.threshold_grid.margin_thresholds,
        output_dir=output_dir,
        run_id=run_id,
    )
    print(f"output_dir={output_dir}")
    print(render_sweep_summary(summary))
