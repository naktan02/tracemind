"""FL client split manifest payload 모델."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
    labeled_policy: dict[str, object]
    labeled_exposure_policy: dict[str, object]
    source_selection: dict[str, object]
    source_jsonl: dict[str, str]
    view_schema: FlClientSplitViewSchema
    bootstrap_labeled_jsonl: str
    shared_client_labeled_jsonl: str | None
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
            labeled_policy=dict(
                _mapping(payload.get("labeled_policy", {"mode": "all"}))
            ),
            labeled_exposure_policy=dict(
                _mapping(
                    payload.get(
                        "labeled_exposure_policy",
                        {"name": "client_local_split"},
                    )
                )
            ),
            source_selection=dict(_mapping(payload.get("source_selection", {}))),
            source_jsonl=_str_dict(payload.get("source_jsonl", {})),
            view_schema=FlClientSplitViewSchema.from_mapping(
                _mapping(payload.get("view_schema", {}))
            ),
            bootstrap_labeled_jsonl=_required_str(
                payload,
                "bootstrap_labeled_jsonl",
            ),
            shared_client_labeled_jsonl=_optional_str(
                payload.get("shared_client_labeled_jsonl")
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
            "labeled_policy": dict(self.labeled_policy),
            "labeled_exposure_policy": dict(self.labeled_exposure_policy),
            "source_selection": dict(self.source_selection),
            "source_jsonl": dict(self.source_jsonl),
            "view_schema": self.view_schema.to_payload(),
            "bootstrap_labeled_jsonl": self.bootstrap_labeled_jsonl,
            "shared_client_labeled_jsonl": self.shared_client_labeled_jsonl,
            "validation_jsonl": self.validation_jsonl,
            "test_jsonl": self.test_jsonl,
            "clients": [client.to_payload() for client in self.clients],
        }


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
