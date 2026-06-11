"""중앙집중형 full text encoder supervised control entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.support.query_ssl_text_encoder.runners import (
    full_text_encoder_supervised as full_text_encoder_supervised_runner,
)


@hydra.main(
    version_base=None,
    config_path="../../../../conf",
    config_name="entrypoints/central/ssl_control/run_full_text_encoder_supervised_control",
)
def main(cfg: DictConfig) -> None:
    full_text_encoder_supervised_runner.run_full_text_encoder_supervised_baseline(cfg)


if __name__ == "__main__":
    main()
