"""Hydra config 기반 FL client labeled/unlabeled split materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig

from methods.federated.client_split import (
    LABELED_EXPOSURE_SERVER_ONLY_SEED,
    FederatedLabeledExposurePolicy,
    FederatedLabeledPoolPolicy,
    select_labeled_pool_items,
)
from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.fl_ssl.federated_simulation.adapters.sharding import (
    split_rows_for_federation,
    split_rows_into_client_shards,
)
from scripts.experiments.fl_ssl.federated_simulation.config_utils import to_plain_dict
from scripts.experiments.fl_ssl.federated_simulation.io.client_split_manifest import (
    FL_CLIENT_SPLIT_MANIFEST_SCHEMA_VERSION,
    FlClientSplitManifest,
    FlClientSplitViewSchema,
    build_client_entry,
    dump_split_rows,
    validate_rows_have_view_schema,
    write_fl_client_split_manifest,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    load_labeled_query_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class FlClientSplitArtifacts:
    """FL client split materializer가 쓰는 산출물 경로."""

    output_dir: Path
    manifest_json: Path
    bootstrap_labeled_jsonl: Path
    shared_client_labeled_jsonl: Path | None
    validation_jsonl: Path
    test_jsonl: Path


def materialize_fl_client_split(
    *,
    source_labeled_jsonl: Path,
    source_unlabeled_jsonl: Path,
    source_validation_jsonl: Path,
    source_test_jsonl: Path,
    split_id: str,
    output_root: Path,
    seed: int,
    client_count: int,
    bootstrap_ratio: float,
    shard_policy: FederatedShardPolicyConfig,
    source_selection: Mapping[str, object],
    source_jsonl: Mapping[str, str],
    view_schema: FlClientSplitViewSchema,
    labeled_policy: FederatedLabeledPoolPolicy | Mapping[str, object] | None = None,
    labeled_exposure_policy: (
        FederatedLabeledExposurePolicy | Mapping[str, object] | None
    ) = None,
) -> FlClientSplitArtifacts:
    """중앙 Query SSL split을 FL client별 고정 split으로 materialize한다."""

    normalized_split_id = split_id.strip()
    if not normalized_split_id:
        raise ValueError("split_id must not be empty.")
    if client_count <= 0:
        raise ValueError("client_count must be positive.")

    labeled_rows = load_labeled_query_rows(source_labeled_jsonl)
    unlabeled_rows = load_labeled_query_rows(source_unlabeled_jsonl)
    validation_rows = load_labeled_query_rows(source_validation_jsonl)
    test_rows = load_labeled_query_rows(source_test_jsonl)
    resolved_labeled_policy = _resolve_labeled_policy(labeled_policy)
    resolved_labeled_exposure_policy = _resolve_labeled_exposure_policy(
        labeled_exposure_policy
    )
    if resolved_labeled_exposure_policy.name == LABELED_EXPOSURE_SERVER_ONLY_SEED:
        raise ValueError(
            "server_only_seed materialization is planned but not implemented yet."
        )
    validate_rows_have_view_schema(
        unlabeled_rows,
        view_schema=view_schema,
        context="source_unlabeled_jsonl",
    )
    _require_disjoint_query_ids(
        labeled_rows,
        unlabeled_rows,
        left_name="source_labeled_jsonl",
        right_name="source_unlabeled_jsonl",
    )
    selected_labeled_rows = select_labeled_pool_items(
        labeled_rows,
        policy=resolved_labeled_policy,
        seed=seed,
        label_getter=_row_label,
    )

    labeled_split = split_rows_for_federation(
        selected_labeled_rows,
        bootstrap_ratio=bootstrap_ratio,
        client_count=client_count,
        seed=seed,
        shard_policy=shard_policy,
    )
    unlabeled_client_shards = split_rows_into_client_shards(
        list(unlabeled_rows),
        client_count=client_count,
        seed=seed + 1,
        shard_policy=shard_policy,
    )
    unlabeled_rows_by_client = {
        shard.client_id: list(shard.rows) for shard in unlabeled_client_shards
    }
    _require_same_client_ids(
        [shard.client_id for shard in labeled_split.client_shards],
        list(unlabeled_rows_by_client),
    )

    output_dir = (
        output_root
        / resolved_labeled_exposure_policy.storage_group_name
        / normalized_split_id
    )
    clients_dir = output_dir / "clients"
    artifacts = FlClientSplitArtifacts(
        output_dir=output_dir,
        manifest_json=output_dir / "manifest.json",
        bootstrap_labeled_jsonl=output_dir / "bootstrap_labeled.jsonl",
        shared_client_labeled_jsonl=(
            output_dir / "shared_client_labeled.jsonl"
            if resolved_labeled_exposure_policy.shares_same_labeled_rows_across_clients
            else None
        ),
        validation_jsonl=output_dir / "validation.jsonl",
        test_jsonl=output_dir / "test.jsonl",
    )
    dump_split_rows(artifacts.bootstrap_labeled_jsonl, labeled_split.bootstrap_rows)
    if artifacts.shared_client_labeled_jsonl is not None:
        dump_split_rows(artifacts.shared_client_labeled_jsonl, selected_labeled_rows)
    dump_split_rows(artifacts.validation_jsonl, validation_rows)
    dump_split_rows(artifacts.test_jsonl, test_rows)

    client_entries = []
    for labeled_client_shard in labeled_split.client_shards:
        client_id = labeled_client_shard.client_id
        client_dir = clients_dir / client_id
        labeled_path = client_dir / "labeled.jsonl"
        unlabeled_path = client_dir / "unlabeled.jsonl"
        if artifacts.shared_client_labeled_jsonl is None:
            client_labeled_rows = list(labeled_client_shard.rows)
            client_labeled_ref = _relative_ref(output_dir, labeled_path)
            dump_split_rows(labeled_path, client_labeled_rows)
        else:
            client_labeled_rows = list(selected_labeled_rows)
            client_labeled_ref = _relative_ref(
                output_dir,
                artifacts.shared_client_labeled_jsonl,
            )
        client_unlabeled_rows = unlabeled_rows_by_client[client_id]

        dump_split_rows(unlabeled_path, client_unlabeled_rows)
        client_entries.append(
            build_client_entry(
                client_id=client_id,
                labeled_jsonl=client_labeled_ref,
                unlabeled_jsonl=_relative_ref(output_dir, unlabeled_path),
                labeled_rows=client_labeled_rows,
                unlabeled_rows=client_unlabeled_rows,
            )
        )

    manifest = FlClientSplitManifest(
        schema_version=FL_CLIENT_SPLIT_MANIFEST_SCHEMA_VERSION,
        split_id=normalized_split_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        client_count=client_count,
        bootstrap_ratio=bootstrap_ratio,
        shard_policy=_shard_policy_payload(shard_policy),
        client_pool_split={},
        labeled_policy=resolved_labeled_policy.to_payload(),
        labeled_exposure_policy=resolved_labeled_exposure_policy.to_payload(),
        source_selection=dict(source_selection),
        source_jsonl=dict(source_jsonl),
        view_schema=view_schema,
        bootstrap_labeled_jsonl=_relative_ref(
            output_dir,
            artifacts.bootstrap_labeled_jsonl,
        ),
        shared_client_labeled_jsonl=(
            None
            if artifacts.shared_client_labeled_jsonl is None
            else _relative_ref(output_dir, artifacts.shared_client_labeled_jsonl)
        ),
        validation_jsonl=_relative_ref(output_dir, artifacts.validation_jsonl),
        test_jsonl=_relative_ref(output_dir, artifacts.test_jsonl),
        clients=tuple(client_entries),
    )
    write_fl_client_split_manifest(artifacts.manifest_json, manifest)
    return artifacts


def run_fl_client_split_materialization_from_hydra(*, cfg: DictConfig) -> None:
    """Hydra 설정을 FL client split materializer 실행 파라미터로 변환한다."""

    materialization_cfg = cfg.fl_client_split_materialization
    source_jsonl = {
        "labeled": str(materialization_cfg.source_labeled_jsonl),
        "unlabeled": str(materialization_cfg.source_unlabeled_jsonl),
        "validation": str(materialization_cfg.source_validation_jsonl),
        "test": str(materialization_cfg.source_test_jsonl),
    }
    artifacts = materialize_fl_client_split(
        source_labeled_jsonl=_resolve_project_path(source_jsonl["labeled"]),
        source_unlabeled_jsonl=_resolve_project_path(source_jsonl["unlabeled"]),
        source_validation_jsonl=_resolve_project_path(source_jsonl["validation"]),
        source_test_jsonl=_resolve_project_path(source_jsonl["test"]),
        split_id=str(materialization_cfg.split_id),
        output_root=_resolve_project_path(str(materialization_cfg.output_root)),
        seed=int(materialization_cfg.seed),
        client_count=int(materialization_cfg.client_count),
        bootstrap_ratio=float(materialization_cfg.bootstrap_ratio),
        shard_policy=FederatedShardPolicyConfig(**to_plain_dict(cfg.shard_policy)),
        labeled_policy=FederatedLabeledPoolPolicy.from_mapping(
            to_plain_dict(materialization_cfg.labeled_policy)
        ),
        labeled_exposure_policy=FederatedLabeledExposurePolicy.from_mapping(
            to_plain_dict(cfg.labeled_exposure_policy)
        ),
        source_selection=to_plain_dict(cfg.query_data_selection),
        source_jsonl=source_jsonl,
        view_schema=FlClientSplitViewSchema(
            weak_text_field=str(materialization_cfg.view_schema.weak_text_field),
            strong_text_fields=tuple(
                str(field)
                for field in materialization_cfg.view_schema.strong_text_fields
            ),
            require_strong_views=bool(
                materialization_cfg.view_schema.require_strong_views
            ),
        ),
    )

    print(f"output_dir={artifacts.output_dir}")
    print(f"manifest_json={artifacts.manifest_json}")
    print(f"bootstrap_labeled_jsonl={artifacts.bootstrap_labeled_jsonl}")
    print(f"validation_jsonl={artifacts.validation_jsonl}")
    print(f"test_jsonl={artifacts.test_jsonl}")


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _relative_ref(base_dir: Path, path: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def _shard_policy_payload(
    shard_policy: FederatedShardPolicyConfig,
) -> dict[str, object]:
    return {
        "name": shard_policy.name,
        "client_id_prefix": shard_policy.client_id_prefix,
        "dominant_ratio": shard_policy.dominant_ratio,
        "alpha": shard_policy.alpha,
    }


def _row_label(row: LabeledQueryRow) -> str:
    return str(row["mapped_label_4"])


def _resolve_labeled_policy(
    policy: FederatedLabeledPoolPolicy | Mapping[str, object] | None,
) -> FederatedLabeledPoolPolicy:
    if policy is None:
        return FederatedLabeledPoolPolicy()
    if isinstance(policy, FederatedLabeledPoolPolicy):
        return policy
    return FederatedLabeledPoolPolicy.from_mapping(policy)


def _resolve_labeled_exposure_policy(
    policy: FederatedLabeledExposurePolicy | Mapping[str, object] | None,
) -> FederatedLabeledExposurePolicy:
    if policy is None:
        return FederatedLabeledExposurePolicy()
    if isinstance(policy, FederatedLabeledExposurePolicy):
        return policy
    return FederatedLabeledExposurePolicy.from_mapping(policy)


def _require_disjoint_query_ids(
    left_rows: Sequence[LabeledQueryRow],
    right_rows: Sequence[LabeledQueryRow],
    *,
    left_name: str,
    right_name: str,
) -> None:
    left_ids = {str(row["query_id"]) for row in left_rows}
    right_ids = {str(row["query_id"]) for row in right_rows}
    overlap = left_ids & right_ids
    if overlap:
        raise ValueError(
            f"{left_name} and {right_name} must be disjoint by query_id: "
            f"{sorted(overlap)[:5]}."
        )


def _require_same_client_ids(
    labeled_client_ids: Sequence[str],
    unlabeled_client_ids: Sequence[str],
) -> None:
    if set(labeled_client_ids) == set(unlabeled_client_ids):
        return
    raise ValueError(
        "Labeled and unlabeled client shard ids must match: "
        f"{sorted(labeled_client_ids)} != {sorted(unlabeled_client_ids)}."
    )


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="entrypoints/fl_ssl/materialize_fl_client_split",
)
def main(cfg: DictConfig) -> None:
    run_fl_client_split_materialization_from_hydra(cfg=cfg)


if __name__ == "__main__":
    main()
