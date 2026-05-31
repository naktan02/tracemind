"""중앙집중형 full text encoder supervised control entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.support.query_ssl_peft.runners.full_text_encoder_supervised import (
    run_full_text_encoder_supervised_baseline,
)


@hydra.main(
    version_base=None,
    config_path="../../../../conf",
    config_name="entrypoints/central/ssl_control/run_full_text_encoder_supervised_control",
)
def main(cfg: DictConfig) -> None:
    run_full_text_encoder_supervised_baseline(cfg)


if __name__ == "__main__":
    main()
