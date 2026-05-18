"""FL client split manifest materialization tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

from methods.federated.shard_policy.base import FederatedShardPolicyConfig
from scripts.experiments.fl_ssl.federated_simulation.config_request import (
    build_simulation_request_from_config,
)
from scripts.experiments.fl_ssl.federated_simulation.io.client_split_manifest import (
    FL_CLIENT_SPLIT_MANIFEST_SCHEMA_VERSION,
    FlClientSplitViewSchema,
    load_materialized_client_split,
)
from scripts.experiments.fl_ssl.materialize_fl_client_split import (
    materialize_fl_client_split,
)
from shared.src.contracts.labeled_query_row_contracts import (
    dump_labeled_query_rows,
)


def _row(
    query_id: str,
    label: str,
    *,
    with_views: bool = True,
) -> dict[str, str]:
    row = {
        "query_id": query_id,
        "text": f"{label} weak {query_id}",
        "raw_label_scheme": "test",
        "raw_label": label,
        "mapped_label_4": label,
        "locale": "eng_Latn",
        "annotation_source": "test",
        "approved_by": "test",
        "created_at": "2026-03-29T00:00:00+00:00",
    }
    if with_views:
        row["aug_0"] = f"{label} strong0 {query_id}"
        row["aug_1"] = f"{label} strong1 {query_id}"
    return row


def _write_source_rows(tmp_path: Path) -> dict[str, Path]:
    labeled_rows = [_row(f"l_a_{index}", "anxiety") for index in range(6)] + [
        _row(f"l_n_{index}", "normal") for index in range(6)
    ]
    unlabeled_rows = [_row(f"u_a_{index}", "anxiety") for index in range(8)] + [
        _row(f"u_n_{index}", "normal") for index in range(8)
    ]
    validation_rows = [_row("v_a", "anxiety"), _row("v_n", "normal")]
    test_rows = [_row("t_a", "anxiety"), _row("t_n", "normal")]

    paths = {
        "labeled": tmp_path / "labeled.jsonl",
        "unlabeled": tmp_path / "unlabeled.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "test": tmp_path / "test.jsonl",
    }
    dump_labeled_query_rows(paths["labeled"], labeled_rows)
    dump_labeled_query_rows(paths["unlabeled"], unlabeled_rows)
    dump_labeled_query_rows(paths["validation"], validation_rows)
    dump_labeled_query_rows(paths["test"], test_rows)
    return paths


def _materialize_test_split(
    tmp_path: Path,
    *,
    client_count: int = 2,
    labeled_policy: dict[str, object] | None = None,
) -> Path:
    paths = _write_source_rows(tmp_path)
    artifacts = materialize_fl_client_split(
        source_labeled_jsonl=paths["labeled"],
        source_unlabeled_jsonl=paths["unlabeled"],
        source_validation_jsonl=paths["validation"],
        source_test_jsonl=paths["test"],
        split_id="test_split",
        output_root=tmp_path / "splits",
        seed=42,
        client_count=client_count,
        bootstrap_ratio=0.25,
        shard_policy=FederatedShardPolicyConfig(
            name="label_dominant",
            dominant_ratio=0.5,
            client_id_prefix="agent",
        ),
        source_selection={
            "labeled": "ourafla_reddit",
            "unlabeled": "ourafla_reddit",
            "validation": "ourafla_reddit",
            "test": "ourafla_reddit",
        },
        source_jsonl={name: str(path) for name, path in paths.items()},
        view_schema=FlClientSplitViewSchema(
            weak_text_field="text",
            strong_text_fields=("aug_0", "aug_1"),
            require_strong_views=True,
        ),
        labeled_policy=labeled_policy,
    )
    return artifacts.manifest_json


def _loaded_query_ids(manifest_path: Path) -> set[str]:
    loaded = load_materialized_client_split(manifest_path)
    return {
        row["query_id"]
        for shard in loaded.dataset_split.client_shards
        for row in shard.rows
    } | {row["query_id"] for row in loaded.dataset_split.bootstrap_rows}


def test_materialize_fl_client_split_writes_manifest_and_client_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path = _materialize_test_split(tmp_path)

    loaded = load_materialized_client_split(manifest_path)

    assert loaded.manifest.schema_version == FL_CLIENT_SPLIT_MANIFEST_SCHEMA_VERSION
    assert loaded.manifest.split_id == "test_split"
    assert loaded.manifest.manifest_sha256
    assert loaded.manifest.view_schema.weak_text_field == "text"
    assert loaded.manifest.view_schema.strong_text_fields == ("aug_0", "aug_1")
    assert loaded.manifest.client_pool_split == {}
    assert loaded.manifest.labeled_policy["mode"] == "all"
    assert len(loaded.dataset_split.bootstrap_rows) > 0
    assert len(loaded.dataset_split.client_shards) == 2
    assert {shard.client_id for shard in loaded.dataset_split.client_shards} == {
        "agent_01",
        "agent_02",
    }
    for shard in loaded.dataset_split.client_shards:
        assert shard.client_pool_split_enforced is True
        assert shard.labeled_rows
        assert shard.unlabeled_rows
        assert {row["query_id"] for row in shard.rows} == {
            row["query_id"] for row in [*shard.labeled_rows, *shard.unlabeled_rows]
        }
        assert all(row["aug_0"] and row["aug_1"] for row in shard.unlabeled_rows)
    assert [row["query_id"] for row in loaded.validation_rows] == ["v_a", "v_n"]
    assert [row["query_id"] for row in loaded.test_rows] == ["t_a", "t_n"]
    assert _loaded_query_ids(manifest_path) == {
        f"l_a_{index}" for index in range(6)
    } | {f"l_n_{index}" for index in range(6)} | {
        f"u_a_{index}" for index in range(8)
    } | {f"u_n_{index}" for index in range(8)}


def test_materialize_fl_client_split_supports_labeled_count_per_class_policy(
    tmp_path: Path,
) -> None:
    manifest_path = _materialize_test_split(
        tmp_path,
        labeled_policy={
            "mode": "count_per_class",
            "count_per_class": 2,
            "fraction": None,
        },
    )

    loaded = load_materialized_client_split(manifest_path)
    train_rows = [
        *loaded.dataset_split.bootstrap_rows,
        *[row for shard in loaded.dataset_split.client_shards for row in shard.rows],
    ]
    labeled_rows = [row for row in train_rows if str(row["query_id"]).startswith("l_")]
    unlabeled_rows = [
        row for row in train_rows if str(row["query_id"]).startswith("u_")
    ]

    assert loaded.manifest.labeled_policy == {
        "mode": "count_per_class",
        "count_per_class": 2,
        "fraction": None,
    }
    assert len(labeled_rows) == 4
    assert len(unlabeled_rows) == 16
    assert {
        label: sum(1 for row in labeled_rows if row["mapped_label_4"] == label)
        for label in {"anxiety", "normal"}
    } == {"anxiety": 2, "normal": 2}


def test_materialize_fl_client_split_requires_unlabeled_views(
    tmp_path: Path,
) -> None:
    paths = _write_source_rows(tmp_path)
    dump_labeled_query_rows(
        paths["unlabeled"],
        [_row("u_missing", "anxiety", with_views=False)],
    )

    with pytest.raises(ValueError, match="required view fields"):
        materialize_fl_client_split(
            source_labeled_jsonl=paths["labeled"],
            source_unlabeled_jsonl=paths["unlabeled"],
            source_validation_jsonl=paths["validation"],
            source_test_jsonl=paths["test"],
            split_id="missing_views",
            output_root=tmp_path / "splits",
            seed=42,
            client_count=2,
            bootstrap_ratio=0.25,
            shard_policy=FederatedShardPolicyConfig(
                name="label_dominant",
                dominant_ratio=0.5,
                client_id_prefix="agent",
            ),
            source_selection={},
            source_jsonl={name: str(path) for name, path in paths.items()},
            view_schema=FlClientSplitViewSchema(
                weak_text_field="text",
                strong_text_fields=("aug_0", "aug_1"),
                require_strong_views=True,
            ),
            labeled_policy=None,
        )


def test_run_federated_simulation_config_loads_materialized_split(
    tmp_path: Path,
) -> None:
    manifest_path = _materialize_test_split(tmp_path)

    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "execution_context/embedding_adapter=hash_debug",
                "execution_context/runtime_env=cpu_local",
                "federated_run_budget.client_count=2",
                "federated_run_budget.bootstrap_ratio=0.25",
                "fl_data.source_mode=materialized_client_split",
                f"fl_data.split_manifest={manifest_path}",
                "shard_policy.dominant_ratio=0.5",
            ],
        )

    request = build_simulation_request_from_config(
        cfg,
        output_dir=tmp_path / "run",
        seed=999,
    )

    assert request.seed == 999
    assert request.materialized_dataset_split is not None
    assert request.data_source_config.source_mode == "materialized_client_split"
    assert request.data_source_config.split_id == "test_split"
    assert request.data_source_config.split_manifest_sha256
    assert request.data_source_config.view_schema == {
        "weak_text_field": "text",
        "strong_text_fields": ["aug_0", "aug_1"],
        "require_strong_views": True,
    }
    assert request.data_source_config.labeled_policy["mode"] == "all"
    assert [
        shard.client_id for shard in request.materialized_dataset_split.client_shards
    ] == ["agent_01", "agent_02"]
    assert request.train_rows == (
        load_materialized_client_split(manifest_path).train_rows
    )
    assert [row["query_id"] for row in request.validation_rows] == ["v_a", "v_n"]


def test_run_federated_simulation_rejects_manifest_client_count_drift(
    tmp_path: Path,
) -> None:
    manifest_path = _materialize_test_split(tmp_path, client_count=2)

    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "execution_context/embedding_adapter=hash_debug",
                "execution_context/runtime_env=cpu_local",
                "federated_run_budget.client_count=3",
                "federated_run_budget.bootstrap_ratio=0.25",
                "fl_data.source_mode=materialized_client_split",
                f"fl_data.split_manifest={manifest_path}",
                "shard_policy.dominant_ratio=0.5",
            ],
        )

    with pytest.raises(ValueError, match="client_count"):
        build_simulation_request_from_config(cfg, output_dir=tmp_path / "run")


def test_run_federated_simulation_allows_materialized_split_ratio_metadata_drift(
    tmp_path: Path,
) -> None:
    manifest_path = _materialize_test_split(tmp_path, client_count=2)

    with initialize_config_module(version_base=None, config_module="conf"):
        cfg = compose(
            config_name="entrypoints/fl_ssl/run_federated_simulation",
            overrides=[
                "execution_context/embedding_adapter=hash_debug",
                "execution_context/runtime_env=cpu_local",
                "federated_run_budget.client_count=2",
                "federated_run_budget.bootstrap_ratio=0.25",
                "fl_data.source_mode=materialized_client_split",
                f"fl_data.split_manifest={manifest_path}",
                "shard_policy.dominant_ratio=0.5",
                "client_pool_split.labeled_ratio=0.2",
                "client_pool_split.unlabeled_ratio=0.8",
            ],
        )

    request = build_simulation_request_from_config(cfg, output_dir=tmp_path / "run")

    assert request.materialized_dataset_split is not None
    assert _loaded_query_ids(manifest_path) == {
        row["query_id"] for row in request.train_rows
    }
