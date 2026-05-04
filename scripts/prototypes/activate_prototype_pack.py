"""메인서버 state 저장소의 활성 PrototypePack 버전을 변경한다."""

from __future__ import annotations

import argparse

from scripts.runtime_adapters.prototype_publication_runtime import (
    activate_prototype_pack_version,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Activate a published prototype pack in the main-server state store."
        )
    )
    parser.add_argument(
        "--prototype-version",
        required=True,
        help="Prototype pack version to mark as active.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pointer = activate_prototype_pack_version(args.prototype_version)
    print(f"prototype_version={pointer.prototype_version}")
    print(f"activated_at={pointer.activated_at.isoformat()}")


if __name__ == "__main__":
    main()
