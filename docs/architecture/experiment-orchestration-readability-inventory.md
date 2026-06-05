# Experiment Orchestration Readability Inventory

## 목적

중앙 supervised control, 중앙 SSL control, FL SSL/FSSL 실행 경로를 같은 기준으로
비교한다. 이 문서는 리팩터링 설계 문서가 아니라 현재 코드의 읽기 경로 inventory다.
코드 의미, 실험 기본값, artifact schema는 이 문서가 소유하지 않는다.

## 공통 관찰 기준

각 실행 경로를 다음 항목으로 본다.

```text
entrypoint
typed request 또는 run context
runtime preparation
execution loop
artifact/report writer
method-specific 분기 위치
처음 읽는 주요 파일
```

## 현재 실행 경로

| rail | entrypoint | request/context | runtime preparation | execution loop | artifact/report | 첫 읽기 파일 |
|---|---|---|---|---|---|---|
| central supervised PEFT | `scripts/experiments/central/ssl_control/run_peft_supervised_control.py` | `TextEncoderRunContext` in `scripts/support/query_ssl_text_encoder/text_encoder_run_context.py` | `prepare_text_encoder_run_context()` | `run_supervised_text_encoder_baseline()` -> PEFT training loop | `write_run_artifacts()` | 4-5 |
| central Query SSL PEFT | `scripts/experiments/central/ssl_control/run_peft_ssl_control.py` | `QuerySslRunContext` in `scripts/support/query_ssl_text_encoder/query_ssl/run_context.py` | unlabeled view preparation + `prepare_query_ssl_run_context()` + descriptor capability validation | `run_consistency_query_ssl_peft_baseline()` -> PEFT Query SSL loop | `write_run_artifacts()` | 5-6 |
| FL SSL/FSSL | `scripts/experiments/fl_ssl/run_federated_simulation.py` | `SimulationRunRequest` in `scripts/experiments/fl_ssl/federated_simulation/models.py` | `build_simulation_request_from_config()` + `run_simulation_request()` runtime validation + `bootstrap_simulation()` | `run_simulation_request()` round loop -> `run_one_round()` -> client runtime bridge | `build_simulation_result()` + `SimulationReportBuilder` | 6-8 |

## Central Supervised Control

첫 진입점은 매우 얇다.

```text
run_peft_supervised_control.py
-> run_supervised_peft_baseline()
-> run_supervised_text_encoder_baseline()
-> prepare_text_encoder_run_context()
-> train_classifier()
-> evaluate_text_encoder_run_context()
-> write_run_artifacts()
```

현재 장점:

- entrypoint가 config 의미를 거의 소유하지 않는다.
- supervised baseline의 공통 흐름은 `run_supervised_text_encoder_baseline()`에 모여 있다.
- artifact writer가 runner 밖 helper로 분리되어 있다.

현재 읽기 비용:

- typed request는 없고 context object가 request와 prepared runtime 역할을 함께 한다.
- output dir 결정은 artifact path helper와 writer 쪽으로 흩어져 있어 entrypoint 첫 화면에서는 보이지 않는다.
- runner가 train/eval/artifact를 한 함수에서 모두 보여준다.

다음에 작게 볼 후보:

- central supervised entrypoint를 바꾸기 전에 `run_supervised_text_encoder_baseline()`의 phase 이름만 확인한다.
- 별도 typed request를 바로 만들지는 않는다. 중앙 rail 전체에서 필요성이 확인된 뒤 검토한다.

## Central Query SSL Control

첫 진입점은 supervised와 같이 얇지만, runner 내부가 더 많은 의미를 갖는다.

```text
run_peft_ssl_control.py
-> run_query_ssl_peft_baseline()
-> run_consistency_query_ssl_peft_baseline()
-> prepare_query_ssl_unlabeled_rows()
-> prepare_query_ssl_run_context()
-> descriptor capability validation
-> build unlabeled loader / algorithm
-> train_query_ssl_classifier()
-> evaluate_query_ssl_run_context()
-> write_run_artifacts()
```

현재 장점:

- Query SSL descriptor, algorithm state summary, unlabeled view preparation이 runner 안에서 명시적이다.
- central supervised와 같은 artifact writer를 사용한다.
- `QuerySslRunContext`가 labeled/eval/model/tokenizer 준비 결과를 묶는다.

현재 읽기 비용:

- `run_consistency_query_ssl_peft_baseline()`가 view 준비, runtime capability 검증, training, eval, manifest assembly를 모두 가진다.
- method manifest와 artifact manifest assembly가 training flow와 같은 화면에 섞여 있다.
- 중앙 SSL은 FSSL처럼 `SimulationRunRequest`에 해당하는 typed request가 없다.

다음에 작게 볼 후보:

- 먼저 phase helper 이름을 정리한다.
- `prepare`, `train`, `evaluate`, `write` 흐름이 보이게 하되 algorithm 의미는 `methods/ssl`에 둔다.

## FL SSL/FSSL

FSSL은 typed request와 runtime validation이 가장 잘 분리되어 있지만, 파일 수와 phase 수가 많다.

```text
run_federated_simulation.py
-> build_simulation_request_from_config()
-> run_simulation_request()
-> _resolve_execution_plan() / _require_runtime_compatibility()
-> bootstrap_simulation()
-> run_one_round()
-> run_client_round()
-> generic_client_runtime_bridge
-> finalize publication / validation
-> build_simulation_result()
```

현재 장점:

- `SimulationRunRequest`가 Hydra config 해석 결과를 typed payload로 고정한다.
- bootstrap, round loop, result builder가 파일로 나뉘어 있다.
- method-owned runtime과 manual Query SSL runtime이 bridge에서 분리되어 있다.

현재 읽기 비용:

- `run_federated_simulation.py`가 sweep, safety guard, output dir, resume, request build, run, print를 직접 가진다.
- `run_simulation_request()` 첫 화면이 execution plan, capability plan, runtime validation, bootstrap, loop를 모두 보여준다.
- `run_one_round()`가 server step, client selection, peer context, client training, sync state, publication, validation, cleanup, summary assembly를 모두 가진다.
- 최근 round-boundary transient cleanup은 기능 fix이므로, 나중에 lifecycle phase 정리 때 구조적으로 더 잘 배치할 수 있다.

다음에 작게 볼 후보:

- FSSL entrypoint 첫 화면 정리.
- 그 다음 `run_simulation_request()` runtime preparation 분리.
- `round_loop.py` phase 분리는 마지막에 별도 커밋으로 진행한다.

## 공통 차이점

| 항목 | central supervised | central Query SSL | FSSL |
|---|---|---|---|
| typed request | 없음 | 없음 | 있음 |
| prepared context | `TextEncoderRunContext` | `QuerySslRunContext` | `BootstrappedSimulation` + state objects |
| sweep 처리 | entrypoint 밖/해당 없음 | entrypoint 밖/해당 없음 | entrypoint 내부 |
| output dir 결정 | artifact path/writer 쪽 | artifact path/writer 쪽 | entrypoint 내부 |
| runtime compatibility | 약함 | descriptor capability validation | 강함 |
| loop 이름 | training loop | Query SSL training loop | communication round loop |

## 1차 리팩터링 후보

바로 공통 framework를 만들지 않는다. 먼저 각 rail에서 같은 어휘가 보이게 한다.

우선순위:

1. FSSL entrypoint 첫 화면 정리
2. central Query SSL runner phase helper 정리
3. central supervised runner phase helper 정리
4. FSSL `run_simulation_request()` runtime preparation 분리
5. FSSL `run_one_round()` phase 정리

## 보류할 것

- 중앙 control에 새 typed request를 즉시 도입하지 않는다.
- central/FSSL 공통 base runner를 만들지 않는다.
- artifact schema, report schema, run layout은 건드리지 않는다.
- method-specific 분기를 scripts에 새로 늘리지 않는다.
