"""FL SSL 실행 산출물 경로 규칙."""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from methods.federated.client_split import (
    LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT,
    LABELED_EXPOSURE_POLICY_NAMES,
    LABELED_EXPOSURE_SERVER_ONLY_SEED,
    LABELED_EXPOSURE_SHARED_CLIENT_SEED,
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

    if _is_manual_fl_composition(cfg):
        return "manual_baselines"
    method_name = _select(cfg, "ssl_method.name", default=None)
    return _slugify(method_name or "method_owned")


def resolve_fl_ssl_method_composition_slug(cfg: DictConfig) -> str:
    """output path에서 method/runtime composition 축을 표현하는 slug를 만든다."""

    if not _is_manual_fl_composition(cfg):
        method_name = _select(cfg, "ssl_method.name", default=None) or "method_owned"
        adapter_family = _resolve_adapter_runtime_slug(
            cfg,
            _select(cfg, "round_runtime.adapter_family_name", default=None)
        )
        server_update_policy = (
            _select(cfg, "server_update_policy.name", default=None)
            or _select(cfg, "round_runtime.aggregation_backend_name", default=None)
            or "unknown_server_update"
        )
        return "__".join(
            _slugify(part)
            for part in (method_name, adapter_family, server_update_policy)
        )

    query_ssl_method = (
        _select(cfg, "query_ssl_method.name", default=None)
        or _select(cfg, "ssl_method.name", default=None)
        or "unknown_ssl"
    )
    adapter_family = _resolve_adapter_runtime_slug(
        cfg,
        _select(cfg, "round_runtime.adapter_family_name", default=None)
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
    parts: list[str] = []
    data_source = _resolve_data_source_slug(cfg)
    if data_source is not None:
        parts.append(data_source)
    else:
        parts.append("split")
    label_budget = _resolve_label_budget_slug(cfg)
    if label_budget is not None:
        parts.append(label_budget)
    labeled_exposure = _resolve_labeled_exposure_slug(cfg)
    if labeled_exposure is not None:
        parts.append(labeled_exposure)
    if seed is not None:
        parts.append(f"seed{int(seed)}")
    return "_".join(_slugify(part) for part in parts if str(part).strip())


def resolve_fl_ssl_run_condition_slug(cfg: DictConfig) -> str:
    """client 수, round budget, local objective 조건을 leaf slug로 표현한다."""

    client_count = int(_select(cfg, "federated_run_budget.client_count", default=0))
    round_budget = int(_select(cfg, "federated_run_budget.rounds", default=0))
    parts = [f"clients{client_count}", f"rounds{round_budget}"]
    local_regularizer = _resolve_local_regularizer_slug(cfg)
    if local_regularizer is not None:
        parts.append(local_regularizer)
    return "_".join(_slugify(part) for part in parts)


def _select(cfg: DictConfig, key: str, *, default: object) -> object:
    return OmegaConf.select(cfg, key, default=default)


def _is_manual_fl_composition(cfg: DictConfig) -> bool:
    composition_mode = _select(cfg, "fl_method.composition_mode", default=None)
    return str(composition_mode or "").strip().lower() == "manual"


def _resolve_adapter_runtime_slug(cfg: DictConfig, adapter_family: object) -> str:
    family = str(adapter_family or "unknown_family").strip() or "unknown_family"
    adapter_kind = _select(
        cfg,
        f"round_runtime.{family}.peft_adapter_name",
        default=None,
    )
    if adapter_kind is None:
        return family
    normalized_kind = str(adapter_kind).strip()
    if not normalized_kind:
        return family
    normalized_family = family.lower().replace("-", "_")
    normalized_kind_key = normalized_kind.lower().replace("-", "_")
    if normalized_family.startswith(f"{normalized_kind_key}_"):
        return family
    if normalized_family.endswith(f"_{normalized_kind_key}"):
        return family
    return f"{family}_{normalized_kind}"


def _resolve_labeled_exposure_slug(cfg: DictConfig) -> str | None:
    configured = _select(cfg, "labeled_exposure_policy.name", default=None)
    if configured is not None:
        normalized = str(configured).strip()
        if normalized:
            return _compact_labeled_exposure_slug(normalized)
    manifest = str(_select(cfg, "fl_data.split_manifest", default="") or "")
    for policy_name in sorted(LABELED_EXPOSURE_POLICY_NAMES):
        if policy_name in manifest:
            return _compact_labeled_exposure_slug(policy_name)
    return None


def _resolve_data_source_slug(cfg: DictConfig) -> str | None:
    labeled = _select(cfg, "query_data_selection.labeled", default=None)
    unlabeled = _select(cfg, "query_data_selection.unlabeled", default=None)
    if labeled is None or unlabeled is None:
        manifest = str(_select(cfg, "fl_data.split_manifest", default="") or "")
        labeled = labeled or _extract_manifest_component(manifest, prefix="labeled")
        unlabeled = unlabeled or _extract_manifest_component(
            manifest,
            prefix="unlabeled",
        )
    if labeled is None and unlabeled is None:
        return None
    return "_".join(
        _slugify(part)
        for part in (
            f"labeled-{labeled or 'unknown'}",
            f"unlabeled-{unlabeled or 'unknown'}",
        )
    )


def _resolve_label_budget_slug(cfg: DictConfig) -> str | None:
    count_per_class = _select(
        cfg,
        "fl_client_split_materialization.labeled_policy.count_per_class",
        default=None,
    )
    if count_per_class is not None:
        return f"labels_pc{int(count_per_class)}"
    materialized_split_name = _select(cfg, "materialized_split.name", default=None)
    manifest = str(_select(cfg, "fl_data.split_manifest", default="") or "")
    for source in (materialized_split_name, manifest):
        match = re.search(r"(?:^|_)labels_pc(\d+)(?:_|$)", str(source or ""))
        if match:
            return f"labels_pc{match.group(1)}"
    return None


def _resolve_local_regularizer_slug(cfg: DictConfig) -> str | None:
    proximal_mu = _select(
        cfg,
        "training_task.objective.peft_classifier.proximal_mu",
        default=None,
    )
    if proximal_mu is None:
        proximal_mu = _select(
            cfg,
            "training_task.objective.lora_classifier.proximal_mu",
            default=None,
        )
    if proximal_mu is None:
        return None
    normalized_mu = float(proximal_mu)
    if normalized_mu <= 0.0:
        return None
    return f"fedprox_mu{normalized_mu:g}"


def _extract_manifest_component(manifest: str, *, prefix: str) -> str | None:
    if not manifest:
        return None
    next_component = {
        "labeled": "unlabeled",
        "unlabeled": "validation",
        "validation": "test",
    }.get(prefix)
    if next_component is not None:
        match = re.search(
            rf"(?:^|[/_]){re.escape(prefix)}-(.+?)_"
            rf"{re.escape(next_component)}-",
            manifest,
        )
        if match:
            return match.group(1)
    match = re.search(rf"(?:^|[/_]){re.escape(prefix)}-([^/]+)", manifest)
    if match:
        return match.group(1).split("_", 1)[0]
    return None


def _compact_labeled_exposure_slug(policy_name: str) -> str:
    if policy_name == LABELED_EXPOSURE_SHARED_CLIENT_SEED:
        return "shared_client"
    if policy_name == LABELED_EXPOSURE_SERVER_ONLY_SEED:
        return "server_only"
    if policy_name == LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT:
        return "client_local"
    return policy_name


def _slugify(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    if not slug or slug in {".", ".."}:
        return "unknown"
    return slug
