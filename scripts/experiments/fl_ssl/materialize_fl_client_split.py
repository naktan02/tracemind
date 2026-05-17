"""Hydra config 기반 FL client labeled/unlabeled split materialization."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.fl_ssl.federated_simulation.adapters.sharding import (
    split_rows_for_federation,
    split_rows_into_client_shards,
)
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
    group_labeled_query_rows_by_label,
    load_labeled_query_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class FlClientSplitArtifacts:
    """FL client split materializer가 쓰는 산출물 경로."""

    output_dir: Path
    manifest_json: Path
    bootstrap_labeled_jsonl: Path
    validation_jsonl: Path
    test_jsonl: Path


@dataclass(frozen=True, slots=True)
class FlLabeledPoolPolicy:
    """FL materialized split에 포함할 labeled source pool 선택 정책."""

    mode: str = "all"
    count_per_class: int | None = None
    fraction: float | None = None

    @classmethod
    def from_mapping(cls, source: Mapping[str, object]) -> "FlLabeledPoolPolicy":
        raw_count = source.get("count_per_class")
        raw_fraction = source.get("fraction")
        return cls(
            mode=str(source.get("mode", "all")),
            count_per_class=None if raw_count is None else int(raw_count),
            fraction=None if raw_fraction is None else float(raw_fraction),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "count_per_class": self.count_per_class,
            "fraction": self.fraction,
        }


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
    labeled_policy: FlLabeledPoolPolicy | Mapping[str, object] | None = None,
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
    selected_labeled_rows = _select_labeled_rows(
        labeled_rows,
        policy=resolved_labeled_policy,
        seed=seed,
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

    output_dir = output_root / normalized_split_id
    clients_dir = output_dir / "clients"
    artifacts = FlClientSplitArtifacts(
        output_dir=output_dir,
        manifest_json=output_dir / "manifest.json",
        bootstrap_labeled_jsonl=output_dir / "bootstrap_labeled.jsonl",
        validation_jsonl=output_dir / "validation.jsonl",
        test_jsonl=output_dir / "test.jsonl",
    )
    dump_split_rows(artifacts.bootstrap_labeled_jsonl, labeled_split.bootstrap_rows)
    dump_split_rows(artifacts.validation_jsonl, validation_rows)
    dump_split_rows(artifacts.test_jsonl, test_rows)

    client_entries = []
    for labeled_client_shard in labeled_split.client_shards:
        client_id = labeled_client_shard.client_id
        client_dir = clients_dir / client_id
        labeled_path = client_dir / "labeled.jsonl"
        unlabeled_path = client_dir / "unlabeled.jsonl"
        client_labeled_rows = list(labeled_client_shard.rows)
        client_unlabeled_rows = unlabeled_rows_by_client[client_id]

        dump_split_rows(labeled_path, client_labeled_rows)
        dump_split_rows(unlabeled_path, client_unlabeled_rows)
        client_entries.append(
            build_client_entry(
                client_id=client_id,
                labeled_jsonl=_relative_ref(output_dir, labeled_path),
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
        source_selection=dict(source_selection),
        source_jsonl=dict(source_jsonl),
        view_schema=view_schema,
        bootstrap_labeled_jsonl=_relative_ref(
            output_dir,
            artifacts.bootstrap_labeled_jsonl,
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
        shard_policy=FederatedShardPolicyConfig(**_to_plain_dict(cfg.shard_policy)),
        labeled_policy=FlLabeledPoolPolicy.from_mapping(
            _to_plain_dict(materialization_cfg.labeled_policy)
        ),
        source_selection=_to_plain_dict(cfg.query_data_selection),
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


def _to_plain_dict(cfg: DictConfig) -> dict[str, object]:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Expected DictConfig section to resolve to a dict.")
    return raw


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


def _resolve_labeled_policy(
    policy: FlLabeledPoolPolicy | Mapping[str, object] | None,
) -> FlLabeledPoolPolicy:
    if policy is None:
        return FlLabeledPoolPolicy()
    if isinstance(policy, FlLabeledPoolPolicy):
        return policy
    return FlLabeledPoolPolicy.from_mapping(policy)


def _select_labeled_rows(
    rows: Sequence[LabeledQueryRow],
    *,
    policy: FlLabeledPoolPolicy,
    seed: int,
) -> list[LabeledQueryRow]:
    if policy.mode == "all":
        if policy.count_per_class is not None or policy.fraction is not None:
            raise ValueError(
                "labeled_policy.count_per_class and labeled_policy.fraction must be "
                "null when mode is all."
            )
        return list(rows)
    if policy.mode == "count_per_class":
        if policy.fraction is not None:
            raise ValueError(
                "labeled_policy.fraction must be null when mode is count_per_class."
            )
        return _select_labeled_rows_by_count_per_class(
            rows,
            count_per_class=policy.count_per_class,
            seed=seed,
        )
    if policy.mode == "fraction":
        if policy.count_per_class is not None:
            raise ValueError(
                "labeled_policy.count_per_class must be null when mode is fraction."
            )
        return _select_labeled_rows_by_fraction(
            rows,
            fraction=policy.fraction,
            seed=seed,
        )
    raise ValueError(
        "fl_client_split_materialization.labeled_policy.mode must be one of "
        "'all', 'count_per_class', or 'fraction'."
    )


def _select_labeled_rows_by_count_per_class(
    rows: Sequence[LabeledQueryRow],
    *,
    count_per_class: int | None,
    seed: int,
) -> list[LabeledQueryRow]:
    if count_per_class is None or count_per_class <= 0:
        raise ValueError(
            "labeled_policy.count_per_class must be positive when mode is "
            "count_per_class."
        )
    rng = random.Random(seed)
    selected_rows: list[LabeledQueryRow] = []
    for label, bucket in group_labeled_query_rows_by_label(rows).items():
        if len(bucket) < count_per_class:
            raise ValueError(
                "labeled_policy.count_per_class exceeds source labeled rows for "
                f"{label}: {count_per_class} > {len(bucket)}."
            )
        shuffled_bucket = list(bucket)
        rng.shuffle(shuffled_bucket)
        selected_rows.extend(shuffled_bucket[:count_per_class])
    rng.shuffle(selected_rows)
    return selected_rows


def _select_labeled_rows_by_fraction(
    rows: Sequence[LabeledQueryRow],
    *,
    fraction: float | None,
    seed: int,
) -> list[LabeledQueryRow]:
    if fraction is None or not 0.0 < fraction <= 1.0:
        raise ValueError(
            "labeled_policy.fraction must be between 0 and 1 when mode is fraction."
        )
    rng = random.Random(seed)
    selected_rows: list[LabeledQueryRow] = []
    for bucket in group_labeled_query_rows_by_label(rows).values():
        selected_count = int(round(len(bucket) * fraction))
        if selected_count <= 0 and bucket:
            selected_count = 1
        selected_count = min(selected_count, len(bucket))
        shuffled_bucket = list(bucket)
        rng.shuffle(shuffled_bucket)
        selected_rows.extend(shuffled_bucket[:selected_count])
    rng.shuffle(selected_rows)
    return selected_rows


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
