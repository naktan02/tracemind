"""중앙집중형 LoRA + classifier pseudo-label self-training entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.experiments.query_lora_ssl.runners.pseudo_label import (
    run_pseudo_label_self_training,
)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/central_ssl_control/train_lora_pseudo_label_classifier",
)
def main(cfg: DictConfig) -> None:
    run_pseudo_label_self_training(cfg=cfg)


if __name__ == "__main__":
    main()
