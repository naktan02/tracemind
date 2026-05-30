"""FL SSL partitioned update hook specs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FederatedSslPartitionedUpdateHook:
    """method가 update-family runtime에 요구하는 logical partition surface."""

    hook_name: str
    partition_names: tuple[str, ...]
    upload_partitions: tuple[str, ...] = ()
    aggregate_partitions: tuple[str, ...] = ()
    l1_sparse_partitions: tuple[str, ...] = ()
    published_state_expression: str | None = None
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hook_name.strip():
            raise ValueError("hook_name must not be empty.")
        partitions = _normalize_unique_names(
            self.partition_names,
            field_name="partition_names",
        )
        object.__setattr__(self, "partition_names", partitions)
        object.__setattr__(
            self,
            "upload_partitions",
            self._normalize_known_partition_subset(
                self.upload_partitions,
                field_name="upload_partitions",
            ),
        )
        object.__setattr__(
            self,
            "aggregate_partitions",
            self._normalize_known_partition_subset(
                self.aggregate_partitions,
                field_name="aggregate_partitions",
            ),
        )
        object.__setattr__(
            self,
            "l1_sparse_partitions",
            self._normalize_known_partition_subset(
                self.l1_sparse_partitions,
                field_name="l1_sparse_partitions",
            ),
        )
        if self.published_state_expression is not None:
            expression = self.published_state_expression.strip()
            if not expression:
                raise ValueError("published_state_expression must not be empty.")
            object.__setattr__(self, "published_state_expression", expression)

    def _normalize_known_partition_subset(
        self,
        values: tuple[str, ...],
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        normalized = _normalize_unique_names(
            values,
            field_name=field_name,
            allow_empty=True,
        )
        unknown = sorted(set(normalized) - set(self.partition_names))
        if unknown:
            raise ValueError(f"{field_name} contains unknown partitions: {unknown}.")
        return normalized


def _normalize_unique_names(
    values: tuple[str, ...],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(str(name).strip() for name in values if str(name).strip())
    if not allow_empty and not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must be unique.")
    return normalized
