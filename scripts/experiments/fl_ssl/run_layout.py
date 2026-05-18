"""FL SSL 실행 산출물 경로 규칙."""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from methods.federated.client_split import (
    LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT,
    LABELED_EXPOSURE_POLICY_NAMES,
)


def build_fl_ssl_run_dir(
    base_dir: str | Path,
    *,
    cfg: DictConfig,
    run_id: str,
    run_kind: str = "single",
) -> Path:
    """FL SSL run을 method composition과 실험 변수 아래에 배치한다."""

    base_path = (
        Path(str(base_dir))
        / resolve_fl_ssl_method_family_slug(cfg)
        / resolve_fl_ssl_method_composition_slug(cfg)
        / resolve_fl_ssl_split_slug(cfg)
    )
    if run_kind == "single":
        return base_path / resolve_fl_ssl_run_condition_slug(cfg) / _slugify(run_id)
    if run_kind == "client_count_sweep":
        round_budget = int(_select(cfg, "federated_run_budget.rounds", default=0))
        return (
            base_path
            / "sweeps"
            / f"client_count_rounds{round_budget}"
            / _slugify(run_id)
        )
    if run_kind == "seed_sweep":
        client_count = int(_select(cfg, "federated_run_budget.client_count", default=0))
        round_budget = int(_select(cfg, "federated_run_budget.rounds", default=0))
        return (
            base_path
            / "sweeps"
            / f"seed_clients{client_count}_rounds{round_budget}"
            / _slugify(run_id)
        )
    raise ValueError(f"Unsupported FL SSL run_kind: {run_kind!r}.")


def build_fl_ssl_client_count_sweep_member_dir(
    sweep_output_dir: str | Path,
    *,
    client_count: int,
) -> Path:
    """client-count sweep 내부의 client_count별 sub-run 위치를 만든다."""

    return Path(str(sweep_output_dir)) / f"clients_{int(client_count):02d}"


def build_fl_ssl_seed_sweep_member_dir(
    sweep_output_dir: str | Path,
    *,
    seed: int,
) -> Path:
    """seed sweep 내부의 seed별 sub-run 위치를 만든다."""

    return Path(str(sweep_output_dir)) / f"seed_{int(seed)}"


def resolve_fl_ssl_method_family_slug(cfg: DictConfig) -> str:
    """상위 method family archive 축을 만든다."""

    composition_mode = _select(cfg, "fl_method.composition_mode", default=None)
    if str(composition_mode or "").strip().lower() == "manual":
        return "manual_baselines"
    method_name = _select(cfg, "ssl_method.name", default=None)
    return _slugify(method_name or "method_owned")


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
    return "__".join(
        _slugify(part)
        for part in (query_ssl_method, adapter_family, aggregation_backend)
    )


def resolve_fl_ssl_split_slug(cfg: DictConfig) -> str:
    """output path에서 data split 축을 표현하는 slug를 만든다."""

    seed = _select(cfg, "seed", default=None)
    shard_policy = _select(cfg, "shard_policy.name", default=None)
    shard_alpha = _select(cfg, "shard_policy.alpha", default=None)
    parts: list[str] = []
    if shard_alpha is not None:
        parts.append(_alpha_slug(float(shard_alpha)))
    elif shard_policy:
        parts.append(str(shard_policy))
    else:
        parts.append("split")
    labeled_exposure = _resolve_labeled_exposure_slug(cfg)
    if labeled_exposure is not None:
        parts.append(labeled_exposure)
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    return "_".join(_slugify(part) for part in parts if str(part).strip())


def resolve_fl_ssl_run_condition_slug(cfg: DictConfig) -> str:
    """client 수와 round budget을 leaf 조건 slug로 표현한다."""

    client_count = int(_select(cfg, "federated_run_budget.client_count", default=0))
    round_budget = int(_select(cfg, "federated_run_budget.rounds", default=0))
    return f"clients{client_count}_rounds{round_budget}"


def _select(cfg: DictConfig, key: str, *, default: object) -> object:
    return OmegaConf.select(cfg, key, default=default)


def _resolve_labeled_exposure_slug(cfg: DictConfig) -> str | None:
    configured = _select(cfg, "labeled_exposure_policy.name", default=None)
    if configured is not None:
        normalized = str(configured).strip()
        if normalized and normalized != LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT:
            return normalized
    manifest = str(_select(cfg, "fl_data.split_manifest", default="") or "")
    for policy_name in sorted(LABELED_EXPOSURE_POLICY_NAMES):
        if (
            policy_name != LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT
            and policy_name in manifest
        ):
            return policy_name
    return None


def _alpha_slug(value: float) -> str:
    if 0 < value < 1:
        return "alpha0" + f"{value:g}".split(".", 1)[1]
    return "alpha" + _float_slug(value)


def _float_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _slugify(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    if not slug or slug in {".", ".."}:
        return "unknown"
    return slug
