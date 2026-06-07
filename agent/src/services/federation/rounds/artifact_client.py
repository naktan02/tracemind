"""서버 소유 round artifact materialization client."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field

import httpx
import torch
from torch import Tensor

RoundArtifactHttpClientFactory = Callable[[], AbstractContextManager[httpx.Client]]


@dataclass(slots=True)
class RoundArtifactClient:
    """server-owned aggregation artifact ref를 HTTP로 materialize한다."""

    server_base_url: str
    timeout: float = 30.0
    _transport: httpx.BaseTransport | None = field(default=None, repr=False)
    _client_factory: RoundArtifactHttpClientFactory | None = field(
        default=None,
        repr=False,
    )

    def _client(self) -> AbstractContextManager[httpx.Client]:
        if self._client_factory is not None:
            return self._client_factory()
        kwargs: dict = {"base_url": self.server_base_url, "timeout": self.timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def load_json_artifact(
        self,
        *,
        artifact_ref: str,
    ) -> Mapping[str, object]:
        """server-owned JSON artifact ref를 materialize한다."""

        with self._client() as client:
            response = client.get(
                "/api/v1/fl/rounds/aggregation-artifacts/json",
                params={"artifact_ref": artifact_ref},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Aggregation JSON artifact response must be an object.")
        return payload

    def load_safetensors_artifact(
        self,
        *,
        artifact_ref: str,
    ) -> tuple[dict[str, Tensor], dict[str, str]]:
        """server-owned safetensors artifact ref를 tensor mapping으로 읽는다."""

        with self._client() as client:
            response = client.get(
                "/api/v1/fl/rounds/aggregation-artifacts/safetensors",
                params={"artifact_ref": artifact_ref},
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Aggregation safetensors response must be an object.")
        raw_tensors = payload.get("tensors")
        if not isinstance(raw_tensors, dict):
            raise ValueError("Aggregation safetensors response requires tensors.")
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError(
                "Aggregation safetensors response metadata must be object."
            )
        return (
            {str(name): torch.tensor(values) for name, values in raw_tensors.items()},
            {str(key): str(value) for key, value in raw_metadata.items()},
        )
