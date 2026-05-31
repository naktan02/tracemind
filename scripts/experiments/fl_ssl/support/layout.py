"""FL SSL 실행 산출물 경로 규칙."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from methods.federated.client_split import (
    LABELED_EXPOSURE_POLICY_NAMES,
    compact_labeled_exposure_policy_slug,
)
from methods.federated_ssl.method_config_surface import (
    default_method_server_update_policy_name,
)
from methods.federated_ssl.registry import resolve_federated_ssl_method_descriptor
from scripts.support.configured_callable import load_configured_callable


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
        update_family = _resolve_update_runtime_slug(cfg)
        server_update_policy = (
            _resolve_method_owned_server_update_policy(cfg, method_name=method_name)
            or _select(cfg, "round_runtime.aggregation_backend_name", default=None)
            or "unknown_server_update"
        )
        return "__".join(
            _slugify(part)
            for part in (method_name, update_family, server_update_policy)
        )

    query_ssl_method = (
        _select(cfg, "query_ssl_method.name", default=None)
        or _select(cfg, "ssl_method.name", default=None)
        or "unknown_ssl"
    )
    update_family = _resolve_update_runtime_slug(cfg)
    aggregation_backend = (
        _select(cfg, "round_runtime.aggregation_backend_name", default=None)
        or _select(cfg, "ssl_method.server_step.aggregation_backend_name", default=None)
        or "unknown_aggregation"
    )
    return "__".join(
        _slugify(part)
        for part in (query_ssl_method, update_family, aggregation_backend)
    )


def _resolve_method_owned_server_update_policy(
    cfg: DictConfig,
    *,
    method_name: str,
) -> str | None:
    try:
        descriptor = resolve_federated_ssl_method_descriptor(str(method_name))
    except ValueError:
        return _select(cfg, "server_update_policy.name", default=None)
    return default_method_server_update_policy_name(descriptor) or _select(
        cfg,
        "server_update_policy.name",
        default=None,
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


def _resolve_update_runtime_slug(cfg: DictConfig) -> str:
    family = (
        _select(cfg, "round_runtime.update_family_name", default=None)
        or "unknown_family"
    )
    family = str(family).strip() or "unknown_family"
    configured_slug = _build_configured_update_family_slug(cfg, family)
    if configured_slug is not None:
        return configured_slug
    return family


def _build_configured_update_family_slug(
    cfg: DictConfig,
    update_family_name: str,
) -> str | None:
    builder_path = _select(
        cfg,
        "round_runtime.composition_slug_builder",
        default=None,
    )
    if builder_path is None:
        return None
    round_runtime = _select(cfg, "round_runtime", default=None)
    if round_runtime is None:
        return None
    builder = load_configured_callable(
        str(builder_path),
        field_name="round_runtime.composition_slug_builder",
    )
    if OmegaConf.is_config(round_runtime):
        runtime_mapping = OmegaConf.to_container(round_runtime, resolve=True)
    else:
        runtime_mapping = round_runtime
    if not isinstance(runtime_mapping, Mapping):
        return None
    slug = builder(
        round_runtime_mapping=runtime_mapping,
        update_family_name=update_family_name,
    )
    normalized_slug = str(slug).strip()
    if not normalized_slug:
        raise ValueError("round_runtime.composition_slug_builder returned empty slug.")
    return normalized_slug


def _resolve_labeled_exposure_slug(cfg: DictConfig) -> str | None:
    configured = _select(cfg, "labeled_exposure_policy.name", default=None)
    if configured is not None:
        normalized = str(configured).strip()
        if normalized:
            return compact_labeled_exposure_policy_slug(normalized)
    manifest = str(_select(cfg, "fl_data.split_manifest", default="") or "")
    for policy_name in sorted(LABELED_EXPOSURE_POLICY_NAMES):
        if policy_name in manifest:
            return compact_labeled_exposure_policy_slug(policy_name)
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
    fl_client_split_name = _select(cfg, "fl_client_split.name", default=None)
    manifest = str(_select(cfg, "fl_data.split_manifest", default="") or "")
    for source in (fl_client_split_name, manifest):
        match = re.search(r"(?:^|_)labels_pc(\d+)(?:_|$)", str(source or ""))
        if match:
            return f"labels_pc{match.group(1)}"
    return None


def _resolve_local_regularizer_slug(cfg: DictConfig) -> str | None:
    proximal_mu = _select_objective_parameter(cfg, "proximal_mu")
    if proximal_mu is None:
        return None
    normalized_mu = float(proximal_mu)
    if normalized_mu <= 0.0:
        return None
    return f"fedprox_mu{normalized_mu:g}"


def _select_objective_parameter(cfg: DictConfig, key: str) -> object | None:
    objective = _select(cfg, "training_task.objective", default=None)
    if objective is None:
        return None
    if OmegaConf.is_config(objective):
        objective_mapping = OmegaConf.to_container(objective, resolve=True)
    else:
        objective_mapping = objective
    if not isinstance(objective_mapping, Mapping):
        return None

    values = _collect_objective_parameter_values(objective_mapping, key)
    normalized_values = {str(value) for value in values if value is not None}
    if not normalized_values:
        return None
    if len(normalized_values) > 1:
        raise ValueError(
            f"training_task.objective has conflicting {key!r} values: "
            f"{sorted(normalized_values)}"
        )
    return values[0]


def _collect_objective_parameter_values(
    objective: Mapping[object, object],
    key: str,
) -> list[object]:
    values: list[object] = []
    dotted_suffix = f".{key}"
    for raw_key, value in objective.items():
        normalized_key = str(raw_key)
        if normalized_key == key or normalized_key.endswith(dotted_suffix):
            values.append(value)
        if isinstance(value, Mapping):
            values.extend(_collect_objective_parameter_values(value, key))
    return values


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


def _slugify(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    if not slug or slug in {".", ".."}:
        return "unknown"
    return slug
