"""LoRA FixMatch baseline entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.experiments.lora_classifier.query_ssl.consistency_runner import (
    run_fixmatch_lora_baseline,
)


@hydra.main(
    version_base=None,
    config_path="..conf",
    config_name="experiments/train_lora_fixmatch",
)
def main(cfg: DictConfig) -> None:
    run_fixmatch_lora_baseline(cfg=cfg)


if __name__ == "__main__":
    main()
