"""Prototype strategy experiment entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
    EmbeddingAdapterSpec,
)
from scripts.experiments.prototype_strategy.cli import parse_cli_args
from scripts.experiments.prototype_strategy.config import (
    DEFAULT_CONFIG_PATH,
    load_experiment_config,
)
from scripts.experiments.prototype_strategy.io_utils import load_jsonl_rows
from scripts.experiments.prototype_strategy.projection import ProjectionService
from scripts.experiments.prototype_strategy.runner import (
    PrototypeExperimentRunner,
    render_validation_summary,
)
from scripts.experiments.prototype_strategy.strategies import (
    DbscanPrototypeStrategy,
    KMeansPrototypeStrategy,
    SinglePrototypeStrategy,
)


def main() -> None:
    cli_args = parse_cli_args(default_config_path=DEFAULT_CONFIG_PATH)
    config = load_experiment_config(
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
    runner = PrototypeExperimentRunner(
        strategies=(
            SinglePrototypeStrategy(),
            KMeansPrototypeStrategy(
                candidate_ks=config.strategies.kmeans.candidate_ks,
                silhouette_sample_size=config.strategies.kmeans.silhouette_sample_size,
                random_state=config.runtime.seed,
            ),
            DbscanPrototypeStrategy(
                eps_values=config.strategies.dbscan.eps_values,
                min_samples_values=config.strategies.dbscan.min_samples_values,
                search_sample_size=config.strategies.dbscan.search_sample_size,
                min_cluster_coverage=config.strategies.dbscan.min_cluster_coverage,
                random_state=config.runtime.seed,
            ),
        ),
        projection_service=ProjectionService(
            seed=config.runtime.seed,
            sample_size=config.projection.sample_size,
        ),
        confidence_threshold=config.thresholds.confidence_threshold,
        margin_threshold=config.thresholds.margin_threshold,
    )
    summary = runner.run(
        train_rows=load_jsonl_rows(config.dataset.train_jsonl),
        validation_rows=load_jsonl_rows(config.dataset.validation_jsonl),
        test_rows=load_jsonl_rows(config.dataset.test_jsonl),
        adapter=adapter,
        output_dir=output_dir,
        run_id=run_id,
        projection_reducers=config.projection.reducers,
    )
    print(f"output_dir={output_dir}")
    print(render_validation_summary(summary))
