"""FL client split manifest IO helper."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

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

FL_CLIENT_SPLIT_MANIFEST_SCHEMA_VERSION = "fl_client_split_manifest.v1"


@dataclass(frozen=True, slots=True)
class FlClientSplitViewSchema:
    """FL unlabeled pool이 노출하는 weak/strong text view 규약."""

    weak_text_field: str = "text"
    strong_text_fields: tuple[str, ...] = ("aug_0", "aug_1")
    require_strong_views: bool = True

    @classmethod
    def from_mapping(cls, source: Mapping[str, object]) -> "FlClientSplitViewSchema":
        strong_fields = source.get("strong_text_fields", ("aug_0", "aug_1"))
        if not isinstance(strong_fields, Sequence) or isinstance(strong_fields, str):
            raise ValueError("view_schema.strong_text_fields must be a sequence.")
        return cls(
            weak_text_field=str(source.get("weak_text_field", "text")),
            strong_text_fields=tuple(str(field) for field in strong_fields),
            require_strong_views=bool(source.get("require_strong_views", True)),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "weak_text_field": self.weak_text_field,
            "strong_text_fields": list(self.strong_text_fields),
            "require_strong_views": self.require_strong_views,
        }


@dataclass(frozen=True, slots=True)
class FlClientSplitClientEntry:
    """manifest에 기록되는 client별 materialized artifact."""

    client_id: str
    labeled_jsonl: str
    unlabeled_jsonl: str
    labeled_count: int
    unlabeled_count: int
    labeled_label_distribution: dict[str, int] = field(default_factory=dict)
    unlabeled_label_distribution: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        source: Mapping[str, object],
    ) -> "FlClientSplitClientEntry":
        return cls(
            client_id=_required_str(source, "client_id"),
            labeled_jsonl=_required_str(source, "labeled_jsonl"),
            unlabeled_jsonl=_required_str(source, "unlabeled_jsonl"),
            labeled_count=int(source.get("labeled_count", 0)),
            unlabeled_count=int(source.get("unlabeled_count", 0)),
            labeled_label_distribution=_int_dict(
                source.get("labeled_label_distribution", {})
            ),
            unlabeled_label_distribution=_int_dict(
                source.get("unlabeled_label_distribution", {})
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "labeled_jsonl": self.labeled_jsonl,
            "unlabeled_jsonl": self.unlabeled_jsonl,
            "labeled_count": self.labeled_count,
            "unlabeled_count": self.unlabeled_count,
            "labeled_label_distribution": dict(self.labeled_label_distribution),
            "unlabeled_label_distribution": dict(self.unlabeled_label_distribution),
        }


@dataclass(frozen=True, slots=True)
class FlClientSplitManifest:
    """고정 client split을 재사용하기 위한 manifest."""

    schema_version: str
    split_id: str
    created_at: str
    seed: int
    client_count: int
    bootstrap_ratio: float
    shard_policy: dict[str, object]
    client_pool_split: dict[str, object]
    source_selection: dict[str, object]
    source_jsonl: dict[str, str]
    view_schema: FlClientSplitViewSchema
    bootstrap_labeled_jsonl: str
    validation_jsonl: str
    test_jsonl: str | None
    clients: tuple[FlClientSplitClientEntry, ...]
    manifest_path: Path | None = None
    manifest_sha256: str | None = None

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        manifest_path: Path | None = None,
        manifest_sha256: str | None = None,
    ) -> "FlClientSplitManifest":
        schema_version = _required_str(payload, "schema_version")
        if schema_version != FL_CLIENT_SPLIT_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported FL client split manifest schema_version: "
                f"{schema_version!r}."
            )
        clients = payload.get("clients", [])
        if not isinstance(clients, Sequence) or isinstance(clients, str):
            raise ValueError("manifest.clients must be a sequence.")
        return cls(
            schema_version=schema_version,
            split_id=_required_str(payload, "split_id"),
            created_at=_required_str(payload, "created_at"),
            seed=int(payload["seed"]),
            client_count=int(payload["client_count"]),
            bootstrap_ratio=float(payload["bootstrap_ratio"]),
            shard_policy=dict(_mapping(payload.get("shard_policy", {}))),
            client_pool_split=dict(_mapping(payload.get("client_pool_split", {}))),
            source_selection=dict(_mapping(payload.get("source_selection", {}))),
            source_jsonl=_str_dict(payload.get("source_jsonl", {})),
            view_schema=FlClientSplitViewSchema.from_mapping(
                _mapping(payload.get("view_schema", {}))
            ),
            bootstrap_labeled_jsonl=_required_str(
                payload,
                "bootstrap_labeled_jsonl",
            ),
            validation_jsonl=_required_str(payload, "validation_jsonl"),
            test_jsonl=_optional_str(payload.get("test_jsonl")),
            clients=tuple(
                FlClientSplitClientEntry.from_mapping(_mapping(client))
                for client in clients
            ),
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_id": self.split_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "client_count": self.client_count,
            "bootstrap_ratio": self.bootstrap_ratio,
            "shard_policy": dict(self.shard_policy),
            "client_pool_split": dict(self.client_pool_split),
            "source_selection": dict(self.source_selection),
            "source_jsonl": dict(self.source_jsonl),
            "view_schema": self.view_schema.to_payload(),
            "bootstrap_labeled_jsonl": self.bootstrap_labeled_jsonl,
            "validation_jsonl": self.validation_jsonl,
            "test_jsonl": self.test_jsonl,
            "clients": [client.to_payload() for client in self.clients],
        }


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


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping, got {type(value).__name__}.")
    return value


def _required_str(source: Mapping[str, object], key: str) -> str:
    value = _optional_str(source.get(key))
    if value is None:
        raise ValueError(f"manifest.{key} must not be empty.")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _str_dict(value: object) -> dict[str, str]:
    return {str(key): str(item) for key, item in _mapping(value).items()}


def _int_dict(value: object) -> dict[str, int]:
    return {str(key): int(item) for key, item in _mapping(value).items()}
