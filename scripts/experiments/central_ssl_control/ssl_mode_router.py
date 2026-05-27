"""중앙 SSL control mode router."""

from __future__ import annotations

from omegaconf import DictConfig

from scripts.experiments.query_peft_ssl.runners.consistency import (
    run_query_ssl_lora_baseline,
)
from scripts.experiments.query_peft_ssl.runners.pseudo_label import (
    run_pseudo_label_self_training,
)

SSL_INPUT_MODE_CONSISTENCY = "consistency"
SSL_INPUT_MODE_PSEUDO_LABEL_REPLAY = "pseudo_label_replay"


def run_central_ssl_mode(cfg: DictConfig) -> None:
    """ssl_input_mode에 맞는 중앙 SSL runner를 실행한다."""

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
