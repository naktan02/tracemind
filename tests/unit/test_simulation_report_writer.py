"""FL simulation report writer 검증."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fl_ssl.federated_simulation.io import (
    simulation_report_writer,
)
from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedReportConfig,
)


def test_simulation_report_writer_writes_track_report_json(tmp_path: Path) -> None:
    report_config = FederatedReportConfig(
        schema_version="fl_ssl_report.v1",
        track="fl_ssl_main_comparison",
        table_role="main_comparison",
        labeled_ratio=0.1,
        unlabeled_ratio=0.9,
        seed_count=3,
        primary_metrics=["macro_f1"],
        secondary_metrics=["communication_cost"],
    )
    payload: dict[str, object] = {
        "schema_version": "fl_ssl_report.v1",
        "track": "fl_ssl_main_comparison",
        "metrics": {"primary": {"macro_f1": 0.5}},
    }

    path = simulation_report_writer.SimulationReportWriter().write(
        output_dir=tmp_path,
        report_config=report_config,
        payload=payload,
    )

    assert path == tmp_path / "reports" / "fl_ssl_main_comparison.report.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert path.read_text(encoding="utf-8").endswith("\n")
