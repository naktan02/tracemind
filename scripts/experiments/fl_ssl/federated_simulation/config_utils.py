"""FL SSL Hydra config 해석 helper."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


def to_plain_dict(cfg: DictConfig) -> dict[str, object]:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Expected DictConfig section to resolve to a dict.")
    return raw


def optional_plain_dict(cfg: DictConfig, section_name: str) -> dict[str, object]:
    if section_name not in cfg:
        return {}
    return to_plain_dict(cfg[section_name])
