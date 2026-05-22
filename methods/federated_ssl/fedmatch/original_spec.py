"""FedMatch 원본 구현에서 가져온 설정 source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

FEDMATCH_ORIGINAL_REPOSITORY = "https://github.com/wyjeong/FedMatch.git"
FEDMATCH_ORIGINAL_COMMIT = "4947aa255d59bd37915e25a719763aaaf5d7e067"
FEDMATCH_ORIGINAL_PAPER = "arXiv:2006.12097"

FEDMATCH_SCENARIO_LABELS_AT_CLIENT = "labels-at-client"
FEDMATCH_SCENARIO_LABELS_AT_SERVER = "labels-at-server"
DEFAULT_ORIGINAL_SCENARIO = FEDMATCH_SCENARIO_LABELS_AT_CLIENT

FEDMATCH_ORIGINAL_SOURCE_FILES = MappingProxyType(
    {
        "config": "config.py",
        "client_objective": "models/fedmatch/client.py",
        "server_policy": "models/fedmatch/server.py",
        "train_loop": "modules/train.py",
        "federated_loop": "modules/federated.py",
        "parameter_init": "modules/nets.py",
        "decomposed_layers": "modules/layers.py",
    }
)


@dataclass(frozen=True, slots=True)
class FedMatchOriginalRunDefaults:
    """원본 FedMatch script/config.py의 실행 기본값."""

    model_name: str = "fedmatch"
    architecture_name: str = "resnet9"
    dataset_name: str = "cifar_10"
    num_clients: int = 100
    num_rounds: int = 200
    num_tasks: int = 1
    client_fraction: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size_test: int = 100
    num_valid: int = 2000
    num_test: int = 2000


@dataclass(frozen=True, slots=True)
class FedMatchOriginalScenarioSpec:
    """labels-at-client/server별 원본 FedMatch 설정."""

    scenario_name: str
    task_prefix: str
    num_labels_per_class: int
    client_epochs: int
    client_batch_size: int
    server_epochs: int
    server_pretrain_epochs: int
    server_batch_size: int
    lr_decay_factor: float
    lr_patience: int
    lr_min: float
    lambda_s: float
    lambda_i: float
    lambda_a: float
    lambda_l2: float
    lambda_l1: float
    l1_threshold: float
    delta_threshold: float


@dataclass(frozen=True, slots=True)
class FedMatchOriginalMethodDefaults:
    """FedMatch model 공통 hyperparameter."""

    num_helpers: int = 2
    confidence_threshold: float = 0.75
    psi_factor: float = 0.2
    helper_refresh_interval: int = 10


@dataclass(frozen=True, slots=True)
class FedMatchOriginalSpec:
    """원본 구현을 TraceMind method core가 참조하는 단일 snapshot."""

    repository: str
    commit: str
    paper: str
    source_files: Mapping[str, str]
    run_defaults: FedMatchOriginalRunDefaults
    method_defaults: FedMatchOriginalMethodDefaults
    labels_at_client: FedMatchOriginalScenarioSpec
    labels_at_server: FedMatchOriginalScenarioSpec


FEDMATCH_ORIGINAL_RUN_DEFAULTS = FedMatchOriginalRunDefaults()
FEDMATCH_ORIGINAL_METHOD_DEFAULTS = FedMatchOriginalMethodDefaults()
FEDMATCH_ORIGINAL_LABELS_AT_CLIENT = FedMatchOriginalScenarioSpec(
    scenario_name=FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
    task_prefix="lc",
    num_labels_per_class=5,
    client_epochs=1,
    client_batch_size=10,
    server_epochs=0,
    server_pretrain_epochs=0,
    server_batch_size=0,
    lr_decay_factor=3.0,
    lr_patience=5,
    lr_min=1e-20,
    lambda_s=10.0,
    lambda_i=1e-2,
    lambda_a=1e-2,
    lambda_l2=10.0,
    lambda_l1=1e-4,
    l1_threshold=5e-6,
    delta_threshold=5e-5,
)
FEDMATCH_ORIGINAL_LABELS_AT_SERVER = FedMatchOriginalScenarioSpec(
    scenario_name=FEDMATCH_SCENARIO_LABELS_AT_SERVER,
    task_prefix="ls",
    num_labels_per_class=100,
    client_epochs=1,
    client_batch_size=100,
    server_epochs=1,
    server_pretrain_epochs=1,
    server_batch_size=100,
    lr_decay_factor=3.0,
    lr_patience=20,
    lr_min=1e-20,
    lambda_s=10.0,
    lambda_i=1e-2,
    lambda_a=1e-2,
    lambda_l2=10.0,
    lambda_l1=1e-5,
    l1_threshold=1e-5,
    delta_threshold=1e-5,
)
FEDMATCH_ORIGINAL_SPEC = FedMatchOriginalSpec(
    repository=FEDMATCH_ORIGINAL_REPOSITORY,
    commit=FEDMATCH_ORIGINAL_COMMIT,
    paper=FEDMATCH_ORIGINAL_PAPER,
    source_files=FEDMATCH_ORIGINAL_SOURCE_FILES,
    run_defaults=FEDMATCH_ORIGINAL_RUN_DEFAULTS,
    method_defaults=FEDMATCH_ORIGINAL_METHOD_DEFAULTS,
    labels_at_client=FEDMATCH_ORIGINAL_LABELS_AT_CLIENT,
    labels_at_server=FEDMATCH_ORIGINAL_LABELS_AT_SERVER,
)


def resolve_original_scenario_spec(
    scenario_name: str,
) -> FedMatchOriginalScenarioSpec:
    normalized = scenario_name.replace("_", "-")
    if normalized == FEDMATCH_SCENARIO_LABELS_AT_CLIENT:
        return FEDMATCH_ORIGINAL_LABELS_AT_CLIENT
    if normalized == FEDMATCH_SCENARIO_LABELS_AT_SERVER:
        return FEDMATCH_ORIGINAL_LABELS_AT_SERVER
    raise ValueError(f"Unsupported FedMatch scenario: {scenario_name!r}")


def fedmatch_original_parameter_mapping(
    *,
    scenario_name: str = FEDMATCH_SCENARIO_LABELS_AT_CLIENT,
) -> dict[str, object]:
    """Hydra descriptor와 tests가 공유하는 원본 FedMatch parameter mapping."""

    scenario = resolve_original_scenario_spec(scenario_name)
    method_defaults = FEDMATCH_ORIGINAL_METHOD_DEFAULTS
    run_defaults = FEDMATCH_ORIGINAL_RUN_DEFAULTS
    return {
        "scenario": scenario.scenario_name,
        "task_prefix": scenario.task_prefix,
        "num_clients": run_defaults.num_clients,
        "num_rounds": run_defaults.num_rounds,
        "num_tasks": run_defaults.num_tasks,
        "client_fraction": run_defaults.client_fraction,
        "client_epochs": scenario.client_epochs,
        "client_batch_size": scenario.client_batch_size,
        "server_epochs": scenario.server_epochs,
        "server_pretrain_epochs": scenario.server_pretrain_epochs,
        "server_batch_size": scenario.server_batch_size,
        "num_helpers": method_defaults.num_helpers,
        "confidence_threshold": method_defaults.confidence_threshold,
        "psi_factor": method_defaults.psi_factor,
        "helper_refresh_interval": method_defaults.helper_refresh_interval,
        "lambda_s": scenario.lambda_s,
        "lambda_i": scenario.lambda_i,
        "lambda_a": scenario.lambda_a,
        "lambda_l2": scenario.lambda_l2,
        "lambda_l1": scenario.lambda_l1,
        "l1_threshold": scenario.l1_threshold,
        "delta_threshold": scenario.delta_threshold,
    }


def original_parameter_mapping(
    *,
    scenario_name: str = DEFAULT_ORIGINAL_SCENARIO,
) -> dict[str, object]:
    """generic method parameter resolver가 호출하는 표준 entrypoint."""

    return fedmatch_original_parameter_mapping(scenario_name=scenario_name)
