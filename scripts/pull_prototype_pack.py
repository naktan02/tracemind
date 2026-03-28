"""중앙 서버에서 현재 활성 PrototypePack을 agent 로컬 캐시에 내려받는다."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.src.services.prototype_sync_service import PrototypeSyncService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull the current active prototype pack from the main server."
    )
    parser.add_argument(
        "--server-base-url",
        required=True,
        help="Base URL of the main server, for example http://127.0.0.1:8000/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pointer = PrototypeSyncService().pull_current(server_base_url=args.server_base_url)
    print(f"prototype_version={pointer.prototype_version}")
    print(f"activated_at={pointer.activated_at.isoformat()}")


if __name__ == "__main__":
    main()
