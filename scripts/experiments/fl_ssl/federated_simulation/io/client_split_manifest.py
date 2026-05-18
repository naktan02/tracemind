"""FL client split manifest IO helper."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from methods.federated.client_split import (
    FederatedLabeledExposurePolicy,
)
from scripts.experiments.fl_ssl.federated_simulation.io import (
    client_split_manifest_models as manifest_models,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedClientShard,
    FederatedDatasetSplit,
)
from shared.src.contracts.labeled_query_row_contracts import (
    LabeledQueryRow,
    count_labeled_query_rows_by_label,
    dump_labeled_query_rows,
    load_labeled_query_rows,
)

FL_CLIENT_SPLIT_MANIFEST_SCHEMA_VERSION = (
    manifest_models.FL_CLIENT_SPLIT_MANIFEST_SCHEMA_VERSION
)
FlClientSplitClientEntry = manifest_models.FlClientSplitClientEntry
FlClientSplitManifest = manifest_models.FlClientSplitManifest
FlClientSplitViewSchema = manifest_models.FlClientSplitViewSchema


@dataclass(frozen=True, slots=True)
class LoadedFlClientSplit:
    """manifest와 materialized row artifact를 함께 로드한 결과."""

    manifest: FlClientSplitManifest
    dataset_split: FederatedDatasetSplit
    validation_rows: list[LabeledQueryRow]
    test_rows: list[LabeledQueryRow]

    @property
    def train_rows(self) -> list[LabeledQueryRow]:
        return [
            *self.dataset_split.bootstrap_rows,
            *[row for shard in self.dataset_split.client_shards for row in shard.rows],
        ]


def load_fl_client_split_manifest(path: str | Path) -> FlClientSplitManifest:
    """manifest JSON을 읽고 sha256을 함께 계산한다."""

    manifest_path = Path(str(path))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("FL client split manifest root must be a mapping.")
    return FlClientSplitManifest.from_payload(
        payload,
        manifest_path=manifest_path,
        manifest_sha256=compute_file_sha256(manifest_path),
    )


def write_fl_client_split_manifest(
    path: str | Path,
    manifest: FlClientSplitManifest,
) -> FlClientSplitManifest:
    """manifest JSON을 deterministic formatting으로 저장하고 sha256을 반환한다."""

    manifest_path = Path(str(path))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_payload(), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return FlClientSplitManifest.from_payload(
        manifest.to_payload(),
        manifest_path=manifest_path,
        manifest_sha256=compute_file_sha256(manifest_path),
    )


def load_materialized_client_split(path: str | Path) -> LoadedFlClientSplit:
    """manifest가 가리키는 bootstrap/client/eval row를 typed split으로 복원한다."""

    manifest = load_fl_client_split_manifest(path)
    if manifest.manifest_path is None:
        raise ValueError("manifest_path must be set after loading a manifest.")
    bootstrap_rows = load_labeled_query_rows(
        resolve_manifest_ref(manifest.manifest_path, manifest.bootstrap_labeled_jsonl)
    )
    labeled_exposure_policy = FederatedLabeledExposurePolicy.from_mapping(
        manifest.labeled_exposure_policy
    )
    client_shards: list[FederatedClientShard] = []
    seen_client_ids: set[str] = set()
    all_train_rows = list(bootstrap_rows)
    for client in manifest.clients:
        if client.client_id in seen_client_ids:
            raise ValueError(
                f"Duplicate client_id in FL split manifest: {client.client_id}"
            )
        seen_client_ids.add(client.client_id)
        labeled_rows = load_labeled_query_rows(
            resolve_manifest_ref(manifest.manifest_path, client.labeled_jsonl)
        )
        unlabeled_rows = load_labeled_query_rows(
            resolve_manifest_ref(manifest.manifest_path, client.unlabeled_jsonl)
        )
        validate_rows_have_view_schema(
            unlabeled_rows,
            view_schema=manifest.view_schema,
            context=f"{client.client_id}.unlabeled_jsonl",
        )
        _require_client_entry_matches_rows(
            client=client,
            labeled_rows=labeled_rows,
            unlabeled_rows=unlabeled_rows,
        )
        _require_disjoint_query_ids(
            labeled_rows,
            unlabeled_rows,
            left_name=f"{client.client_id}.labeled_jsonl",
            right_name=f"{client.client_id}.unlabeled_jsonl",
        )
        all_train_rows.extend(labeled_rows)
        all_train_rows.extend(unlabeled_rows)
        client_shards.append(
            FederatedClientShard(
                client_id=client.client_id,
                rows=[*labeled_rows, *unlabeled_rows],
                labeled_rows=list(labeled_rows),
                unlabeled_rows=list(unlabeled_rows),
                client_pool_split_enforced=True,
            )
        )
    if len(client_shards) != manifest.client_count:
        raise ValueError(
            "manifest.client_count must match clients length: "
            f"{manifest.client_count} != {len(client_shards)}."
        )
    if labeled_exposure_policy.shares_same_labeled_rows_across_clients:
        _require_shared_labeled_rows_are_consistent(
            bootstrap_rows=bootstrap_rows,
            client_shards=client_shards,
        )
    else:
        _require_unique_query_ids(all_train_rows, context="manifest train split")
    validation_rows = load_labeled_query_rows(
        resolve_manifest_ref(manifest.manifest_path, manifest.validation_jsonl)
    )
    test_rows = (
        []
        if manifest.test_jsonl is None
        else load_labeled_query_rows(
            resolve_manifest_ref(manifest.manifest_path, manifest.test_jsonl)
        )
    )
    return LoadedFlClientSplit(
        manifest=manifest,
        dataset_split=FederatedDatasetSplit(
            bootstrap_rows=bootstrap_rows,
            client_shards=tuple(client_shards),
        ),
        validation_rows=validation_rows,
        test_rows=test_rows,
    )


def dump_split_rows(path: str | Path, rows: Sequence[LabeledQueryRow]) -> None:
    """manifest materialization 전용 JSONL writer wrapper."""

    dump_labeled_query_rows(path, rows)


def validate_rows_have_view_schema(
    rows: Sequence[LabeledQueryRow],
    *,
    view_schema: FlClientSplitViewSchema,
    context: str,
) -> None:
    """weak/strong view가 materialized row에 실제 존재하는지 검증한다."""

    required_fields = [view_schema.weak_text_field]
    if view_schema.require_strong_views:
        required_fields.extend(view_schema.strong_text_fields)

    for row in rows:
        missing_fields = [
            field_name
            for field_name in required_fields
            if not str(row.get(field_name, "")).strip()
        ]
        if missing_fields:
            query_id = str(row.get("query_id", "<missing_query_id>"))
            raise ValueError(
                f"{context} row {query_id} is missing required view fields: "
                f"{missing_fields}."
            )


def build_client_entry(
    *,
    client_id: str,
    labeled_jsonl: str,
    unlabeled_jsonl: str,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
) -> FlClientSplitClientEntry:
    return FlClientSplitClientEntry(
        client_id=client_id,
        labeled_jsonl=labeled_jsonl,
        unlabeled_jsonl=unlabeled_jsonl,
        labeled_count=len(labeled_rows),
        unlabeled_count=len(unlabeled_rows),
        labeled_label_distribution=count_labeled_query_rows_by_label(labeled_rows),
        unlabeled_label_distribution=count_labeled_query_rows_by_label(unlabeled_rows),
    )


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(str(path)).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_ref(manifest_path: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _require_client_entry_matches_rows(
    *,
    client: FlClientSplitClientEntry,
    labeled_rows: Sequence[LabeledQueryRow],
    unlabeled_rows: Sequence[LabeledQueryRow],
) -> None:
    if client.labeled_count != len(labeled_rows):
        raise ValueError(
            f"{client.client_id} labeled_count does not match artifact rows: "
            f"{client.labeled_count} != {len(labeled_rows)}."
        )
    if client.unlabeled_count != len(unlabeled_rows):
        raise ValueError(
            f"{client.client_id} unlabeled_count does not match artifact rows: "
            f"{client.unlabeled_count} != {len(unlabeled_rows)}."
        )
    if client.labeled_label_distribution != count_labeled_query_rows_by_label(
        labeled_rows
    ):
        raise ValueError(f"{client.client_id} labeled_label_distribution drift.")
    if client.unlabeled_label_distribution != count_labeled_query_rows_by_label(
        unlabeled_rows
    ):
        raise ValueError(f"{client.client_id} unlabeled_label_distribution drift.")


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


def _require_shared_labeled_rows_are_consistent(
    *,
    bootstrap_rows: Sequence[LabeledQueryRow],
    client_shards: Sequence[FederatedClientShard],
) -> None:
    if not client_shards:
        return
    expected_labeled_ids = {
        str(row["query_id"]) for row in client_shards[0].labeled_rows
    }
    if not expected_labeled_ids:
        raise ValueError("shared_client_seed manifest must expose client labeled rows.")
    for shard in client_shards[1:]:
        labeled_ids = {str(row["query_id"]) for row in shard.labeled_rows}
        if labeled_ids != expected_labeled_ids:
            raise ValueError(
                "shared_client_seed manifest requires identical labeled rows "
                "for every client."
            )
    all_unlabeled_rows = [
        row for shard in client_shards for row in shard.unlabeled_rows
    ]
    _require_unique_query_ids(
        all_unlabeled_rows,
        context="manifest shared_client_seed unlabeled split",
    )
    _require_disjoint_query_ids(
        bootstrap_rows,
        all_unlabeled_rows,
        left_name="bootstrap_labeled_jsonl",
        right_name="client unlabeled rows",
    )
    _require_disjoint_query_ids(
        client_shards[0].labeled_rows,
        all_unlabeled_rows,
        left_name="shared client labeled rows",
        right_name="client unlabeled rows",
    )


def _require_unique_query_ids(
    rows: Sequence[LabeledQueryRow],
    *,
    context: str,
) -> None:
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in seen_ids:
            duplicate_ids.add(query_id)
        seen_ids.add(query_id)
    if duplicate_ids:
        raise ValueError(
            f"{context} must not contain duplicate query_id values: "
            f"{sorted(duplicate_ids)[:5]}."
        )
