"""FL SSL fixed peer-probe subset selection tests."""

from __future__ import annotations

from methods.federated_ssl.diagnostics.sampling import (
    hash_query_ids,
    select_deterministic_diagnostic_rows,
    select_label_balanced_probe_rows,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.diagnostic_view import (
    build_client_diagnostic_unlabeled_view,
)
from scripts.experiments.fl_ssl.federated_simulation.adapters.peer_probe import (
    build_fixed_peer_probe,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedDiagnosticViewConfig,
    FederatedPeerProbeConfig,
)


def _row(query_id: str, label: str) -> dict[str, str]:
    return {
        "query_id": query_id,
        "text": f"{label} text",
        "raw_label_scheme": "mapped_label_4",
        "raw_label": label,
        "mapped_label_4": label,
        "locale": "en",
        "annotation_source": "test",
        "approved_by": None,
        "created_at": "2026-01-01T00:00:00Z",
    }


def test_fixed_peer_probe_selects_label_balanced_deterministic_subset() -> None:
    rows = [
        _row("a1", "anxiety"),
        _row("a2", "anxiety"),
        _row("d1", "depression"),
        _row("d2", "depression"),
        _row("n1", "normal"),
        _row("n2", "normal"),
        _row("s1", "suicidal"),
        _row("s2", "suicidal"),
    ]
    config = FederatedPeerProbeConfig(max_rows=4, seed_offset=10)

    selected_a, manifest_a = build_fixed_peer_probe(
        rows=rows,
        config=config,
        run_seed=42,
    )
    selected_b, manifest_b = build_fixed_peer_probe(
        rows=list(reversed(rows)),
        config=config,
        run_seed=42,
    )

    assert [row["query_id"] for row in selected_a] == [
        row["query_id"] for row in selected_b
    ]
    assert manifest_a is not None
    assert manifest_b is not None
    assert manifest_a.query_ids_sha256 == manifest_b.query_ids_sha256
    assert manifest_a.row_count == 4
    assert manifest_a.requested_max_rows == 4
    assert manifest_a.label_distribution == {
        "anxiety": 1,
        "depression": 1,
        "normal": 1,
        "suicidal": 1,
    }


def test_label_balanced_probe_core_is_order_independent() -> None:
    rows = [
        _row("a1", "anxiety"),
        _row("a2", "anxiety"),
        _row("d1", "depression"),
        _row("d2", "depression"),
        _row("n1", "normal"),
        _row("n2", "normal"),
        _row("s1", "suicidal"),
        _row("s2", "suicidal"),
    ]

    selected_a = select_label_balanced_probe_rows(rows=rows, max_rows=4, seed=52)
    selected_b = select_label_balanced_probe_rows(
        rows=list(reversed(rows)),
        max_rows=4,
        seed=52,
    )

    assert [row["query_id"] for row in selected_a] == [
        row["query_id"] for row in selected_b
    ]
    assert {row["mapped_label_4"] for row in selected_a} == {
        "anxiety",
        "depression",
        "normal",
        "suicidal",
    }


def test_probe_query_id_hash_uses_sequence_order() -> None:
    assert hash_query_ids(("a", "b", "c")) == hash_query_ids(("a", "b", "c"))
    assert hash_query_ids(("a", "b", "c")) != hash_query_ids(("c", "b", "a"))


def test_fixed_peer_probe_disabled_uses_original_rows_without_manifest() -> None:
    rows = [_row("a1", "anxiety"), _row("d1", "depression")]

    selected, manifest = build_fixed_peer_probe(
        rows=rows,
        config=FederatedPeerProbeConfig(enabled=False),
        run_seed=42,
    )

    assert selected == tuple(rows)
    assert manifest is None


def test_client_diagnostic_view_selects_deterministic_subset() -> None:
    rows = [_row(f"q{index:03d}", "normal") for index in range(10)]
    config = FederatedDiagnosticViewConfig(max_rows=4, seed_offset=3)

    selected_a = build_client_diagnostic_unlabeled_view(
        rows=rows,
        config=config,
        run_seed=42,
        round_index=2,
        client_id="agent_01",
    )
    selected_b = build_client_diagnostic_unlabeled_view(
        rows=list(reversed(rows)),
        config=config,
        run_seed=42,
        round_index=2,
        client_id="agent_01",
    )
    selected_c = build_client_diagnostic_unlabeled_view(
        rows=rows,
        config=config,
        run_seed=42,
        round_index=3,
        client_id="agent_01",
    )

    assert len(selected_a) == 4
    assert [row["query_id"] for row in selected_a] == [
        row["query_id"] for row in selected_b
    ]
    assert [row["query_id"] for row in selected_a] != [
        row["query_id"] for row in selected_c
    ]


def test_diagnostic_view_sampling_core_is_order_independent() -> None:
    rows = [_row(f"q{index:03d}", "normal") for index in range(10)]

    selected_a = select_deterministic_diagnostic_rows(
        rows=rows,
        max_rows=4,
        run_seed=42,
        seed_offset=3,
        round_index=2,
        client_id="agent_01",
    )
    selected_b = select_deterministic_diagnostic_rows(
        rows=list(reversed(rows)),
        max_rows=4,
        run_seed=42,
        seed_offset=3,
        round_index=2,
        client_id="agent_01",
    )

    assert [row["query_id"] for row in selected_a] == [
        row["query_id"] for row in selected_b
    ]
    assert len(selected_a) == 4


def test_client_diagnostic_view_disabled_uses_full_pool() -> None:
    rows = [_row("a1", "anxiety"), _row("d1", "depression")]

    selected = build_client_diagnostic_unlabeled_view(
        rows=rows,
        config=FederatedDiagnosticViewConfig(enabled=False, max_rows=1),
        run_seed=42,
        round_index=1,
        client_id="agent_01",
    )

    assert selected == tuple(rows)
