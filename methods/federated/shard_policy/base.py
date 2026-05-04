"""Federated shard policy method contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class FederatedShardPolicyConfig:
    """client shard 분할 정책 설정."""

    name: str
    client_id_prefix: str
    dominant_ratio: float | None = None
    alpha: float | None = None


@dataclass(frozen=True, slots=True)
class FederatedClientShardAssignment(Generic[ItemT]):
    """한 client에 할당된 item 묶음."""

    client_id: str
    items: tuple[ItemT, ...]


@dataclass(frozen=True, slots=True)
class FederatedShardSplit(Generic[ItemT]):
    """bootstrap item과 client assignment로 나눈 split."""

    bootstrap_items: tuple[ItemT, ...]
    client_shards: tuple[FederatedClientShardAssignment[ItemT], ...]
