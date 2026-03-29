"""run_federated_simulation 스크립트 unit tests."""

from __future__ import annotations

from agent.src.infrastructure.model_adapters.embedding.factory import (
    EmbeddingAdapterSpec,
)
from scripts.experiments.run_federated_simulation import (
    run_simulation,
    split_rows_for_federation,
)


def _row(query_id: str, text: str, label: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "text": text,
        "raw_label_scheme": "ourafla_4class.v1",
        "raw_label": label.title(),
        "mapped_label_4": label,
        "locale": "eng_Latn",
        "annotation_source": "test",
        "approved_by": "test",
        "created_at": "2026-03-29T00:00:00+00:00",
    }


def test_split_rows_for_federation_keeps_bootstrap_and_client_data_separate() -> None:
    rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("d1", "sad sad", "depression"),
        _row("d2", "sad sad", "depression"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("s1", "die die", "suicidal"),
        _row("s2", "die die", "suicidal"),
    ]

    split = split_rows_for_federation(
        rows,
        bootstrap_ratio=0.5,
        client_count=2,
        seed=42,
    )

    bootstrap_ids = {row["query_id"] for row in split.bootstrap_rows}
    client_ids = {
        row["query_id"]
        for shard in split.client_shards
        for row in shard.rows
    }

    assert bootstrap_ids
    assert client_ids
    assert bootstrap_ids.isdisjoint(client_ids)


def test_run_simulation_completes_one_round_with_small_fixture(tmp_path) -> None:
    train_rows = [
        _row("a1", "panic panic", "anxiety"),
        _row("a2", "panic panic", "anxiety"),
        _row("a3", "panic panic", "anxiety"),
        _row("d1", "sad sad", "depression"),
        _row("d2", "sad sad", "depression"),
        _row("d3", "sad sad", "depression"),
        _row("n1", "calm calm", "normal"),
        _row("n2", "calm calm", "normal"),
        _row("n3", "calm calm", "normal"),
        _row("s1", "die die", "suicidal"),
        _row("s2", "die die", "suicidal"),
        _row("s3", "die die", "suicidal"),
    ]
    validation_rows = [
        _row("va", "panic panic", "anxiety"),
        _row("vd", "sad sad", "depression"),
        _row("vn", "calm calm", "normal"),
        _row("vs", "die die", "suicidal"),
    ]

    result = run_simulation(
        train_rows=train_rows,
        validation_rows=validation_rows,
        output_dir=tmp_path / "simulation",
        client_count=4,
        rounds=1,
        bootstrap_ratio=1 / 3,
        seed=7,
        embedding_spec=EmbeddingAdapterSpec(
            backend="hash_debug",
            model_id="hash_debug",
            revision="sim",
            hash_dim=32,
        ),
        model_id="tracemind-embed-sim",
        training_scope="adapter_only",
        confidence_threshold=0.0,
        margin_threshold=0.0,
        max_examples=4,
        min_required_examples=1,
        gradient_clip_norm=1.0,
    )

    assert result.rounds
    assert result.rounds[0].update_count > 0
    assert result.rounds[0].model_revision == "sim_rev_0001"
    assert result.rounds[0].prototype_version == "proto_sim_0001"
