"""FL SSL simulation data source config를 실행 입력으로 해석한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from methods.federated.client_split import (
    LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT,
    LABELED_EXPOSURE_SERVER_ONLY_SEED,
    FederatedLabeledExposurePolicy,
)
from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.fl_ssl.federated_simulation.adapters.sharding import (
    split_rows_for_federation,
    split_rows_into_client_shards,
)
from scripts.experiments.fl_ssl.federated_simulation.config_utils import (
    optional_plain_dict,
)
from scripts.experiments.fl_ssl.federated_simulation.io.client_split_manifest import (
    LoadedFlClientSplit,
    load_materialized_client_split,
)
from scripts.experiments.fl_ssl.federated_simulation.io.rows import load_jsonl_rows
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT,
    FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN,
    FederatedClientShard,
    FederatedDatasetSplit,
    FederatedDataSourceConfig,
)
from shared.src.contracts.labeled_query_row_contracts import LabeledQueryRow


@dataclass(slots=True)
class ResolvedFlDataSource:
    train_rows: list[LabeledQueryRow]
    validation_rows: list[LabeledQueryRow]
    test_rows: list[LabeledQueryRow]
    materialized_dataset_split: FederatedDatasetSplit | None
    data_source_config: FederatedDataSourceConfig


def resolve_fl_data_source(
    *,
    cfg: DictConfig,
    client_count: int,
    bootstrap_ratio: float,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
) -> ResolvedFlDataSource:
    fl_data_cfg = cfg.get("fl_data", {})
    source_mode = str(
        fl_data_cfg.get("source_mode", FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN)
    )
    if source_mode == FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN:
        labeled_rows = load_jsonl_rows(Path(str(cfg.train_jsonl)))
        unlabeled_jsonl = (
            str(cfg.unlabeled_jsonl)
            if "unlabeled_jsonl" in cfg and cfg.unlabeled_jsonl is not None
            else str(cfg.train_jsonl)
        )
        unlabeled_rows = load_jsonl_rows(Path(unlabeled_jsonl))
        labeled_exposure_policy = FederatedLabeledExposurePolicy.from_mapping(
            optional_plain_dict(cfg, "labeled_exposure_policy")
        )
        dataset_split = _build_runtime_dataset_split(
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
            client_count=client_count,
            bootstrap_ratio=bootstrap_ratio,
            seed=seed,
            shard_policy=shard_policy,
            labeled_exposure_policy=labeled_exposure_policy,
        )
        return ResolvedFlDataSource(
            train_rows=[
                *dataset_split.bootstrap_rows,
                *[row for shard in dataset_split.client_shards for row in shard.rows],
            ],
            validation_rows=load_jsonl_rows(Path(str(cfg.validation_jsonl))),
            test_rows=_load_optional_test_rows(cfg),
            materialized_dataset_split=dataset_split,
            data_source_config=FederatedDataSourceConfig(
                source_mode=source_mode,
                source_selection=optional_plain_dict(cfg, "query_data_selection"),
                source_jsonl={
                    "labeled": str(cfg.train_jsonl),
                    "unlabeled": unlabeled_jsonl,
                    "validation": str(cfg.validation_jsonl),
                },
                labeled_policy={"mode": "all"},
                labeled_exposure_policy=labeled_exposure_policy.to_payload(),
            ),
        )
    if source_mode != FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT:
        return _unsupported_fl_data_source(source_mode)

    raw_manifest_path = fl_data_cfg.get("split_manifest")
    if raw_manifest_path is None:
        raise ValueError(
            "fl_data.split_manifest is required when fl_data.source_mode is "
            "materialized_client_split."
        )
    loaded_split = load_materialized_client_split(Path(str(raw_manifest_path)))
    require_manifest_matches_config(
        loaded_split=loaded_split,
        cfg=cfg,
        client_count=client_count,
        bootstrap_ratio=bootstrap_ratio,
        shard_policy=shard_policy,
    )
    manifest = loaded_split.manifest
    return ResolvedFlDataSource(
        train_rows=loaded_split.train_rows,
        validation_rows=loaded_split.validation_rows,
        test_rows=loaded_split.test_rows,
        materialized_dataset_split=loaded_split.dataset_split,
        data_source_config=FederatedDataSourceConfig(
            source_mode=source_mode,
            split_manifest_path=str(manifest.manifest_path or raw_manifest_path),
            split_manifest_sha256=manifest.manifest_sha256,
            split_id=manifest.split_id,
            source_selection=dict(manifest.source_selection),
            source_jsonl=dict(manifest.source_jsonl),
            labeled_policy=dict(manifest.labeled_policy),
            labeled_exposure_policy=dict(manifest.labeled_exposure_policy),
            view_schema=manifest.view_schema.to_payload(),
            test_jsonl=manifest.test_jsonl,
        ),
    )


def _load_optional_test_rows(cfg: DictConfig) -> list[LabeledQueryRow]:
    raw_test_jsonl = cfg.get("test_jsonl")
    if raw_test_jsonl is None:
        return []
    return load_jsonl_rows(Path(str(raw_test_jsonl)))


def require_manifest_matches_config(
    *,
    loaded_split: LoadedFlClientSplit,
    cfg: DictConfig,
    client_count: int,
    bootstrap_ratio: float,
    shard_policy: FederatedShardPolicyConfig,
) -> None:
    manifest = loaded_split.manifest
    if manifest.client_count != client_count:
        raise ValueError(
            "fl_data materialized manifest client_count must match "
            f"federated_run_budget.client_count: {manifest.client_count} != "
            f"{client_count}."
        )
    if abs(manifest.bootstrap_ratio - bootstrap_ratio) > 1e-9:
        raise ValueError(
            "fl_data materialized manifest bootstrap_ratio must match "
            f"federated_run_budget.bootstrap_ratio: {manifest.bootstrap_ratio} != "
            f"{bootstrap_ratio}."
        )
    _require_manifest_shard_policy_matches_config(manifest.shard_policy, shard_policy)
    configured_source_selection = optional_plain_dict(cfg, "query_data_selection")
    if (
        configured_source_selection
        and manifest.source_selection
        and manifest.source_selection != configured_source_selection
    ):
        raise ValueError(
            "fl_data materialized manifest source_selection must match "
            f"query_data_selection: {manifest.source_selection} != "
            f"{configured_source_selection}."
        )


def _unsupported_fl_data_source(source_mode: str) -> ResolvedFlDataSource:
    supported_modes = [
        FL_DATA_SOURCE_RUNTIME_SPLIT_FROM_TRAIN,
        FL_DATA_SOURCE_MATERIALIZED_CLIENT_SPLIT,
    ]
    raise ValueError(
        "Unsupported fl_data.source_mode. "
        f"Expected one of {supported_modes}, got {source_mode!r}."
    )


def _require_manifest_shard_policy_matches_config(
    manifest_policy: dict[str, object],
    shard_policy: FederatedShardPolicyConfig,
) -> None:
    expected = {
        "name": shard_policy.name,
        "client_id_prefix": shard_policy.client_id_prefix,
        "dominant_ratio": shard_policy.dominant_ratio,
        "alpha": shard_policy.alpha,
    }
    for key, expected_value in expected.items():
        actual_value = manifest_policy.get(key)
        if actual_value != expected_value:
            raise ValueError(
                "fl_data materialized manifest shard_policy must match config: "
                f"{key} {actual_value!r} != {expected_value!r}."
            )


def _build_runtime_dataset_split(
    *,
    labeled_rows: list[LabeledQueryRow],
    unlabeled_rows: list[LabeledQueryRow],
    client_count: int,
    bootstrap_ratio: float,
    seed: int,
    shard_policy: FederatedShardPolicyConfig,
    labeled_exposure_policy: FederatedLabeledExposurePolicy,
) -> FederatedDatasetSplit:
    """runtime_split fallback에서도 labeled exposure policy를 명시적으로 적용한다."""

    labeled_split = split_rows_for_federation(
        labeled_rows,
        bootstrap_ratio=bootstrap_ratio,
        client_count=client_count,
        seed=seed,
        shard_policy=shard_policy,
    )
    unlabeled_shards = split_rows_into_client_shards(
        unlabeled_rows,
        client_count=client_count,
        seed=seed + 1,
        shard_policy=shard_policy,
    )
    unlabeled_by_client = {
        shard.client_id: list(shard.rows) for shard in unlabeled_shards
    }
    client_shards: list[FederatedClientShard] = []
    for labeled_shard in labeled_split.client_shards:
        unlabeled = unlabeled_by_client[labeled_shard.client_id]
        if labeled_exposure_policy.name == LABELED_EXPOSURE_SERVER_ONLY_SEED:
            labeled = []
        elif labeled_exposure_policy.name == LABELED_EXPOSURE_CLIENT_LOCAL_SPLIT:
            labeled = list(labeled_shard.rows)
        else:
            labeled = list(labeled_rows)
        client_shards.append(
            FederatedClientShard(
                client_id=labeled_shard.client_id,
                rows=[*labeled, *unlabeled],
                labeled_rows=labeled,
                unlabeled_rows=unlabeled,
                client_pool_split_enforced=True,
            )
        )
    bootstrap_rows = (
        list(labeled_rows)
        if labeled_exposure_policy.name == LABELED_EXPOSURE_SERVER_ONLY_SEED
        else list(labeled_split.bootstrap_rows)
    )
    return FederatedDatasetSplit(
        bootstrap_rows=bootstrap_rows,
        client_shards=tuple(client_shards),
    )
