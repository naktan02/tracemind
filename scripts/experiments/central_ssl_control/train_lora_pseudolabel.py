"""중앙집중형 LoRA + classifier USB PseudoLabel entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.experiments.query_lora_ssl.runners.consistency import (
    run_pseudolabel_lora_baseline,
)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/central_ssl_control/train_lora_pseudolabel",
)
def main(cfg: DictConfig) -> None:
    run_pseudolabel_lora_baseline(cfg=cfg)


if __name__ == "__main__":
    main()
