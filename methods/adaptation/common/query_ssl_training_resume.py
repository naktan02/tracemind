"""Query SSL trainer resume checkpoint IO."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from methods.ssl.base import QuerySslAlgorithm
from methods.ssl.model_capabilities import (
    load_query_ssl_auxiliary_module_state_dicts,
    query_ssl_auxiliary_module_state_dicts,
)
from methods.ssl.runtime.lifecycle import configure_query_ssl_algorithm_model
from methods.ssl.state import (
    export_query_ssl_algorithm_state,
    load_query_ssl_algorithm_state,
)

QUERY_SSL_TRAINING_CHECKPOINT_SCHEMA_VERSION = "query_ssl_training_checkpoint.v1"


@dataclass(frozen=True, slots=True)
class QuerySslTrainingResumeState:
    """checkpoint load 후 trainer loop가 이어받아야 하는 상태."""

    completed_steps: int = 0
    history: list[dict[str, Any]] | None = None
    best_checkpoint_state: dict[str, Any] | None = None


def load_query_ssl_training_checkpoint(
    *,
    path: str | Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    algorithm: QuerySslAlgorithm,
    categories: list[str],
    device: str,
    auxiliary_modules: Mapping[str, nn.Module] | None = None,
) -> QuerySslTrainingResumeState:
    """Query SSL local trainer checkpoint를 복원한다."""

    if path is None or not str(path).strip():
        return QuerySslTrainingResumeState()
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if (
        str(checkpoint.get("schema_version"))
        != QUERY_SSL_TRAINING_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported Query SSL training checkpoint schema_version: "
            f"{checkpoint.get('schema_version')!r}."
        )
    if str(checkpoint.get("algorithm_name")) != algorithm.algorithm_name:
        raise ValueError("Checkpoint algorithm_name does not match current method.")
    if list(checkpoint.get("categories", [])) != list(categories):
        raise ValueError("Checkpoint categories do not match current dataset.")

    model.load_state_dict(checkpoint["model_state_dict"])
    load_query_ssl_auxiliary_module_state_dicts(
        {} if auxiliary_modules is None else auxiliary_modules,
        checkpoint.get("auxiliary_module_state_dicts", {}),
    )
    configure_query_ssl_algorithm_model(
        algorithm,
        model=model,
        device=torch.device(device),
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    load_query_ssl_algorithm_state(algorithm, checkpoint.get("algorithm_state", {}))
    restore_query_ssl_rng_state(checkpoint.get("rng_state", {}))
    print(f"query_ssl_resume=loaded checkpoint={checkpoint_path}", flush=True)
    return QuerySslTrainingResumeState(
        completed_steps=int(checkpoint.get("completed_steps", 0)),
        history=checkpoint.get("history", []),
        best_checkpoint_state=checkpoint.get("best_checkpoint_state"),
    )


def save_query_ssl_training_checkpoint(
    *,
    checkpoint_output_dir: Path,
    algorithm: QuerySslAlgorithm,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    total_train_steps: int,
    history: list[dict[str, Any]],
    best_checkpoint_state: dict[str, Any],
    categories: list[str],
    auxiliary_modules: Mapping[str, nn.Module] | None = None,
) -> None:
    """Query SSL local trainer checkpoint를 latest file로 저장한다."""

    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_output_dir / "latest_training_checkpoint.pt"
    torch.save(
        {
            "schema_version": QUERY_SSL_TRAINING_CHECKPOINT_SCHEMA_VERSION,
            "algorithm_name": algorithm.algorithm_name,
            "categories": list(categories),
            "completed_steps": int(completed_steps),
            "total_train_steps": int(total_train_steps),
            "model_state_dict": model.state_dict(),
            "auxiliary_module_state_dicts": query_ssl_auxiliary_module_state_dicts(
                {} if auxiliary_modules is None else auxiliary_modules
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "algorithm_state": dict(export_query_ssl_algorithm_state(algorithm)),
            "rng_state": capture_query_ssl_rng_state(),
            "history": list(history),
            "best_checkpoint_state": best_checkpoint_state,
        },
        checkpoint_path,
    )
    print(
        "query_ssl_resume=saved "
        f"checkpoint={checkpoint_path} completed_steps={completed_steps}",
        flush=True,
    )


def capture_query_ssl_rng_state() -> dict[str, Any]:
    """resume equivalence에 필요한 RNG state를 캡처한다."""

    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_query_ssl_rng_state(state: Mapping[str, Any]) -> None:
    """checkpoint에 저장된 RNG state를 현재 process에 복원한다."""

    python_random = state.get("python_random")
    if python_random is not None:
        random.setstate(python_random)
    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    cuda_state = state.get("torch_cuda_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
