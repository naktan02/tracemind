"""중앙 SSL control runner router."""

from __future__ import annotations

from omegaconf import DictConfig

from scripts.support.configured_callable import load_configured_callable


def run_central_ssl_mode(cfg: DictConfig) -> None:
    """Hydra input_mode leaf가 선언한 중앙 SSL runner를 실행한다."""

    runner_cfg = getattr(cfg, "central_ssl_runner", None)
    if runner_cfg is None:
        raise ValueError(
            "central_ssl_runner is required. Select "
            "strategy_axes/ssl_objective/input_mode=<mode> instead of overriding only "
            "ssl_input_mode."
        )
    mode = str(getattr(cfg, "ssl_input_mode", "") or "").strip()
    runner_mode = str(getattr(runner_cfg, "mode", "") or "").strip()
    if mode != runner_mode:
        raise ValueError(
            "ssl_input_mode must match central_ssl_runner.mode. Select "
            "strategy_axes/ssl_objective/input_mode=<mode> so the mode and runner stay "
            f"coupled by config. Got ssl_input_mode={mode!r}, "
            f"central_ssl_runner.mode={runner_mode!r}."
        )

    runner = load_configured_callable(
        str(runner_cfg.callable_path),
        field_name="central_ssl_runner.callable_path",
    )
    runner(cfg=cfg)
