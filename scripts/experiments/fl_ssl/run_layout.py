"""FL SSL 실행 산출물 경로 규칙."""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT = "materialized_client_split"
FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN = "runtime_split_from_train"


def build_fl_ssl_run_dir(
    base_dir: str | Path,
    *,
    cfg: DictConfig,
    run_id: str,
) -> Path:
    """FL SSL run을 split/method-composition 아래에 배치한다."""

    return (
        Path(str(base_dir))
        / resolve_fl_ssl_split_slug(cfg)
        / resolve_fl_ssl_method_composition_slug(cfg)
        / _slugify(run_id)
    )


def resolve_fl_ssl_split_slug(cfg: DictConfig) -> str:
    """output path에서 data split 축을 표현하는 slug를 만든다."""

    source_mode = str(
        _select(
            cfg,
            "fl_data.source_mode",
            default=FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN,
        )
    )
    if source_mode == FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT:
        split_manifest = _select(cfg, "fl_data.split_manifest", default=None)
        if split_manifest:
            return _slugify(_manifest_path_slug(Path(str(split_manifest))))
        return "materialized_client_split"

    seed = _select(cfg, "seed", default=None)
    client_count = _select(cfg, "federated_run_budget.client_count", default=None)
    shard_policy = _select(cfg, "shard_policy.name", default=None)
    shard_alpha = _select(cfg, "shard_policy.alpha", default=None)
    parts = ["runtime_split"]
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    if client_count is not None:
        parts.append(f"clients{int(client_count):02d}")
    if shard_policy:
        parts.append(str(shard_policy))
    if shard_alpha is not None:
        parts.append(f"alpha{_float_slug(float(shard_alpha))}")
    return _slugify("__".join(parts))


def resolve_fl_ssl_method_composition_slug(cfg: DictConfig) -> str:
    """output path에서 method/runtime composition 축을 표현하는 slug를 만든다."""

    query_ssl_method = (
        _select(cfg, "query_ssl_method.name", default=None)
        or _select(cfg, "ssl_method.name", default=None)
        or "unknown_ssl"
    )
    adapter_family = (
        _select(cfg, "round_runtime.adapter_family_name", default=None)
        or "unknown_family"
    )
    aggregation_backend = (
        _select(cfg, "round_runtime.aggregation_backend_name", default=None)
        or _select(cfg, "ssl_method.server_step.aggregation_backend_name", default=None)
        or "unknown_aggregation"
    )
    composition_mode = _select(cfg, "fl_method.composition_mode", default=None)
    parts = [str(query_ssl_method), str(adapter_family), str(aggregation_backend)]
    if composition_mode:
        parts.append(str(composition_mode))
    return _slugify("__".join(parts))


def _select(cfg: DictConfig, key: str, *, default: object) -> object:
    return OmegaConf.select(cfg, key, default=default)


def _manifest_path_slug(path: Path) -> str:
    if path.name == "manifest.json":
        return path.parent.name or "materialized_client_split"
    return path.stem or "materialized_client_split"


def _float_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _slugify(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    if not slug or slug in {".", ".."}:
        return "unknown"
    return slug
