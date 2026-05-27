"""중앙집중형 PEFT encoder classifier SSL 공통 entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.experiments.central_ssl_control.ssl_mode_router import (
    run_central_ssl_mode,
)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/central_ssl_control/train_peft_ssl_classifier",
)
def main(cfg: DictConfig) -> None:
    run_central_ssl_mode(cfg)


if __name__ == "__main__":
    main()
