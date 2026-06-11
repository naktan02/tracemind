"""중앙집중형 PEFT text encoder consistency SSL entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.support.query_ssl_text_encoder.runners.consistency import (
    run_query_ssl_peft_baseline,
)


@hydra.main(
    version_base=None,
    config_path="../../../../conf",
    config_name="entrypoints/central/ssl_control/run_peft_ssl_control",
)
def main(cfg: DictConfig) -> None:
    run_query_ssl_peft_baseline(cfg)


if __name__ == "__main__":
    main()
