# TraceMind Test Strategy

이 문서는 2026-04-25 기준 TraceMind 테스트 구성을 설명한다.

현재 기준 수치:

| 항목 | 수치 |
|---|---:|
| Python test modules | 90 |
| `def test_*` test cases | 439 |

정확한 수치는 계속 변할 수 있다. 전략의 핵심은 contract drift, local/server boundary drift, 실험 entrypoint/config drift를 빠르게 잡는 것이다.

## 1. 테스트 철학

TraceMind의 주요 위험은 아래에 있다.

- shared payload 의미와 producer/consumer 해석 drift
- agent local state와 server orchestration 책임 혼합
- query adaptation/FL 실험 config drift
- artifact manifest와 runtime state mismatch
- API route와 app consumer type drift
- GPU/model cache 의존 실행을 deterministic test로 오해하는 문제

따라서 테스트는 다음 순서로 두껍게 둔다.

1. `shared` contract와 canonical helper
2. `agent` local inference/training/query buffer/wellbeing service
3. `main_server` round/prototype/experiment orchestration
4. script entrypoint, Hydra config, artifact IO
5. cross-boundary integration과 architecture guard

## 2. 테스트 위치

| 위치 | 책임 |
|---|---|
| `shared/tests/unit` | shared contract, prototype contract, training defaults, generated UI contract |
| `agent/tests/unit` | local inference, query buffer, query adaptation, training, wellbeing API/service |
| `main_server/tests/unit` | FL round lifecycle, aggregation, prototype publication, experiment workspace API/service |
| `tests/unit` | scripts, Hydra config, prototype builder/projection, generated app types |
| `tests/integration` | cross-boundary integration |
| `tests/federation/e2e` | multi-agent federation scenario용 위치 |
| `tests/architecture` | dependency direction과 layer rule guard |

## 3. 기본 명령

전체:

```bash
uv run pytest
```

Package별:

```bash
uv run pytest shared/tests
uv run pytest agent/tests
uv run pytest main_server/tests
```

Root tests:

```bash
uv run pytest tests/unit
uv run pytest tests/integration
uv run pytest tests/architecture
```

Lint/format check:

```bash
uv run ruff check main_server/src agent/src shared/src scripts tests
uv run ruff format --check main_server/src agent/src shared/src scripts tests
```

## 4. 보호 범위

### Shared Contract

대표 파일:

| 파일 | 보호 범위 |
|---|---|
| `shared/tests/unit/test_adapter_contracts.py` | adapter family state/update payload |
| `shared/tests/unit/test_new_training_contracts.py` | training task/update/feedback signal |
| `shared/tests/unit/test_prototype_contracts.py` | prototype pack serialization/helper |
| `shared/tests/unit/test_workspace_manifest_contracts.py` | experiment workspace manifest/compile contract |
| `shared/tests/unit/test_family_access_contracts.py` | family setup/unlock contract |
| `shared/tests/unit/test_wellbeing_signal_contracts.py` | wellbeing summary/timeseries contract |

Contract 변경은 producer, consumer, serialization/compatibility test, 관련 docs를 같은 변경에서 닫는다.

### Agent Runtime

대표 파일:

| 파일 | 보호 범위 |
|---|---|
| `agent/tests/unit/test_inference_pipeline.py` | local inference pipeline |
| `agent/tests/unit/test_scoring_service.py` | scoring backend/policy |
| `agent/tests/unit/test_query_buffer_repository.py` | query buffer local persistence |
| `agent/tests/unit/test_query_buffer_selection_service.py` | query buffer selection |
| `tests/unit/test_methods_fixmatch.py` | reusable FixMatch method objective |
| `tests/unit/test_methods_diagonal_scale_heuristic_update.py` | reusable diagonal-scale local update method core |
| `tests/unit/test_methods_federated_ssl.py` | reusable FL SSL method descriptor |
| `tests/unit/test_methods_federated_shard_policy.py` | reusable FL shard policy method core |
| `tests/unit/test_methods_prototype_scoring.py` | reusable prototype scoring method core |
| `tests/unit/test_methods_prototype_evidence.py` | reusable prototype evidence method core |
| `tests/unit/test_methods_prototype_training_inputs.py` | reusable prototype training input method core |
| `agent/tests/unit/test_local_training_service.py` | local training execution |
| `agent/tests/unit/test_training_api.py` | agent training route |
| `agent/tests/unit/test_wellbeing_api.py` | family/wellbeing route |

Agent 테스트는 raw text, local retention, private state가 server boundary로 새지 않는지 확인해야 한다.

### Main Server Runtime

대표 파일:

| 파일 | 보호 범위 |
|---|---|
| `main_server/tests/unit/test_round_lifecycle_service.py` | round open/update/finalize |
| `main_server/tests/unit/test_round_manager_service.py` | round orchestration facade |
| `main_server/tests/unit/test_aggregation_service.py` | aggregation backend |
| `tests/unit/test_methods_fedavg.py` | FedAvg generic/family aggregation core |
| `main_server/tests/unit/test_fl_rounds_api.py` | FL round API |
| `main_server/tests/unit/test_prototype_pack_service.py` | prototype pack publication |
| `main_server/tests/unit/test_experiment_workspace_service.py` | workspace save/compile |
| `main_server/tests/unit/test_experiment_run_service.py` | local experiment run orchestration |

Server 테스트는 round state, aggregation policy, publication side effect가 agent local concern과 섞이지 않는지 확인해야 한다.

### Scripts and Experiment Surface

대표 파일:

| 파일 | 보호 범위 |
|---|---|
| `tests/unit/test_scripts_hydra_configs.py` | Hydra config group drift |
| `tests/unit/test_script_entrypoint_imports.py` | script entrypoint import |
| `tests/unit/test_fixed_classifier_runner.py` | fixed classifier seed runner |
| `tests/unit/test_lora_supervised_runner.py` | LoRA supervised runner |
| `tests/unit/test_lora_fixmatch_runner.py` | FixMatch runner |
| `tests/unit/test_methods_ssl_hooks.py` | reusable SSL pseudo-labeling/masking/selection hooks |
| `tests/unit/test_run_federated_simulation.py` | FL simulation entrypoint |
| `tests/unit/test_experiment_web_type_generation.py` | experiment web generated type drift |
| `tests/unit/test_family_extension_type_generation.py` | family extension generated type drift |

Script 테스트는 실제 GPU 학습 성능을 보장하지 않는다. 실행 표면, config wiring, deterministic IO를 보호한다.

### Cross-Boundary and Architecture

대표 파일:

| 파일 | 보호 범위 |
|---|---|
| `tests/integration/test_fl_round_e2e.py` | FL round e2e flow |
| `tests/architecture/test_layer_dependencies.py` | dependency direction과 layer rule |

경계를 넘는 변경은 root `tests/`에 검증을 남긴다.

## 5. 테스트 추가 기준

| 변경 | 테스트 기준 |
|---|---|
| shared contract 변경 | shared unit, producer/consumer unit, 필요 시 integration |
| agent local state 변경 | agent unit과 local repository fixture |
| main_server round/aggregation 변경 | main_server unit과 e2e scenario |
| script config 변경 | Hydra config test와 entrypoint import test |
| API route 변경 | package API unit, `docs/api/api-surface.md`, generated UI type test |
| UI contract consumer 변경 | app build와 generated type test |
| architecture rule 변경 | `tests/architecture` 갱신 |

## 6. GPU/Slow 실행 정책

GPU나 online model download가 필요한 실행은 기본 unit test의 전제 조건으로 두지 않는다.

- 빠른 deterministic 검증은 local fixture, fake model, hash/debug embedding을 우선한다.
- GPU 실험은 `docs/operations/local-runbook.md`의 preflight를 먼저 수행한다.
- long-running experiment 결과는 `runs/` report와 manifest로 남기고, test에는 재현 가능한 최소 invariant만 둔다.

## 7. Coverage 기준

Repo guideline은 core service statement coverage 90% 이상을 목표로 둔다. 현재 문서는 측정값을 canonical로 고정하지 않는다.

Coverage를 측정할 때:

```bash
uv run pytest --cov=shared --cov=agent --cov=main_server --cov=scripts
```

측정 결과를 정책으로 승격하려면 CI 또는 별도 quality gate 문서를 함께 추가한다.
