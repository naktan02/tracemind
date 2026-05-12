"""중앙집중형 LoRA + classifier SSL 공통 entrypoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from scripts.experiments.query_lora_ssl.runners.consistency import (
    run_query_ssl_lora_baseline,
)
from scripts.experiments.query_lora_ssl.runners.pseudo_label import (
    run_pseudo_label_self_training,
)

SSL_INPUT_MODE_CONSISTENCY = "consistency"
SSL_INPUT_MODE_PSEUDO_LABEL_REPLAY = "pseudo_label_replay"


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/central_ssl_control/train_lora_ssl_classifier",
)
def main(cfg: DictConfig) -> None:
    mode = str(getattr(cfg, "ssl_input_mode", SSL_INPUT_MODE_CONSISTENCY)).strip()
    if mode == SSL_INPUT_MODE_CONSISTENCY:
        run_query_ssl_lora_baseline(cfg=cfg)
        return
    if mode == SSL_INPUT_MODE_PSEUDO_LABEL_REPLAY:
        run_pseudo_label_self_training(cfg=cfg)
        return
    raise ValueError(
        "Unsupported ssl_input_mode. "
        f"Expected one of: {SSL_INPUT_MODE_CONSISTENCY}, "
        f"{SSL_INPUT_MODE_PSEUDO_LABEL_REPLAY}. Got: {mode!r}."
    )


if __name__ == "__main__":
    main()
