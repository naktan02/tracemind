"""Prototype strategy experiment entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone

import hydra
from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterFactory,
)
from hydra.utils import instantiate
from omegaconf import DictConfig

from scripts.experiments.prototype_strategy.io_utils import load_jsonl_rows
from scripts.experiments.prototype_strategy.runner import (
    PrototypeExperimentRequest,
    render_validation_summary,
)

@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="experiments/prototype_strategy",
)
def main(config: DictConfig) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.output.base_dir / run_id

    adapter = EmbeddingAdapterFactory.create(
        instantiate(config.embedding.spec)
    )
    runner = instantiate(config.runner)
    summary = runner.run(
        PrototypeExperimentRequest(
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
