"""중앙집중형 frozen-backbone + LoRA + classifier supervised baseline entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.experiments.lora_classifier.runner import run_supervised_lora_baseline


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="jobs/experiments/train_lora_classifier",
)
def main(cfg: DictConfig) -> None:
    run_supervised_lora_baseline(cfg)


if __name__ == "__main__":
    main()
