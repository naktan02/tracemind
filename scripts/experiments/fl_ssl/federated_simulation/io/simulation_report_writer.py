"""FL simulation report JSON writer."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.fl_ssl.federated_simulation.models import (
    FederatedReportConfig,
)


class SimulationReportWriter:
    """report 경로 계산과 JSON serialization만 담당한다."""

    def write(
        self,
        *,
        output_dir: Path,
        report_config: FederatedReportConfig,
        payload: dict[str, object],
    ) -> Path:
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"{report_config.track}.report.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return path
