# FL SSL Execution Audit

작성일: 2026-05-18

이 문서는 현재 FL SSL 목표를 실제 artifact와 검증 명령 기준으로 대조한 감사표다.
장시간/full-budget 실험은 새로 실행하지 않고, 필요한 경우 1-round smoke와 기존
report/summary를 검증했다.

## 판정

전체 목표는 아직 완료가 아니다. 이 감사표의 2026-05-18 artifact 검증은
manual baseline과 LoRA-classifier artifact path를 확인한 기록이다. 이후 FedMatch가
첫 method로 확정되어 method-owned simulation slice가 열렸으므로, method 선택 상태는
`docs/project_execution_plan.md`와 `docs/contracts/fl_ssl_method_capability_matrix.md`를
우선한다.

완료/검증된 항목:

- `FedAvg + FixMatch + LoRA-classifier` PEFT smoke와 artifact-ref delta 경로는
  구현 및 산출물 verifier로 확인됐다.
- `10 clients`, Dirichlet `alpha=0.3`, split seed `42`, `1 round`,
  `gpu_local + mxbai`, `FixMatch + FedAvg + LoRA-classifier` 현재 smoke report는
  verifier로 확인됐다.
- `agent-local://` LoRA/head delta를 server-owned `aggregation_artifact::` ref로
  upload/materialize하는 경로와 compatibility preflight는 구현 및 테스트로 닫혔다.
  current manifest는 shared update 파일 수, server-owned ref target 파일,
  `agent-local://` ref 미노출, 최종 LoRA/head aggregate snapshot까지 검증한다.
- `10 clients`, Dirichlet `alpha=0.3`, split seed `42`, historical `50 rounds`,
  `FixMatch + FedAvg + LoRA-classifier` main baseline report는 round/split/method/delta
  기준 verifier로 확인됐다. 이 report는 runtime metadata 도입 전 산출물이며,
  현재 full-budget preset source of truth는 `30 rounds`다.
- FlexMatch/FreeMatch/PseudoLabel ablation은 5-round reduced run으로 method
  metadata와 실제 local objective 변경을 확인했다.
- Dirichlet `alpha=0.1` final stress는 현재 `runs/fl_ssl` method-first layout 아래
  검증 가능한 report가 없으므로, 마지막 stress 실행 전까지 current manifest 검증
  대상에서 제외한다. 기본/main 비교는 `alpha=0.3`이다.
- `client_count=1..10` sweep은 1-round summary와 하위 report 10개를 확인했다.
- result index는 FL `fl_ssl_main_comparison.report.json`와 client-count sweep 하위
  report를 dashboard/index record로 정규화한다.
- FL SSL runner는 총 예정 communication round가 `30`을 넘으면 기본 차단한다.

후보와 비교 조건 확정 뒤 별도 실행할 항목:

- Dirichlet `alpha=0.1` final stress의 current full-budget `30 rounds` 실행.
- FlexMatch/FreeMatch/PseudoLabel full-budget ablation.
- `client_count=1..10` full sweep. `5 rounds x 10 client-counts`처럼 합계
  `50` rounds인 sweep도 long-run guard 대상이다.

현재 이 감사표 이후 열린 항목:

- FedMatch가 첫 method로 확정됐다.
- `methods/federated_ssl/fedmatch/`와
  `conf/strategy_axes/fl/method_descriptor/fedmatch.yaml`가 source-of-truth method
  package/config다.
- helper peer-context simulation, method-owned local objective,
  `fedmatch_partitioned` server update slice, labels-at-server supervised seed server
  step은 열렸다.
- sparse S2C/C2S는 아직 full parity 후보로 남아 있다.

## Prompt-To-Artifact Checklist

| 요구사항 | 현재 증거 | 상태 |
|---|---|---|
| 작업별 commit/push | 작업 단위 커밋은 `origin/master`에 push했다. 최신 상태는 `git status --short --branch`와 `git log --oneline origin/master -n 12`로 확인한다 | 완료 |
| scripts는 thin wrapper, method core는 methods 소유 | `scripts/experiments/fl_ssl/*`는 config/run/report, SSL algorithm은 `methods/ssl/algorithms/*`, PEFT encoder aggregation projection은 `methods/adaptation/text_classifier/aggregation/*` | 완료 |
| 실제 PEFT 기준 Query SSL local objective 호출 | `tests/unit/test_lora_fixmatch_runner.py`, `tests/unit/test_federated_agent_runtime_adapters.py`, `tests/unit/test_run_federated_simulation.py`, current 1-round smoke report `objective.query_ssl.algorithm_name=fixmatch` | 완료 |
| LoRA/classifier delta가 FedAvg까지 집계 | `methods/adaptation/text_classifier/aggregation/peft_encoder_fedavg_projection.py`, `main_server/tests/unit/test_aggregation_service.py`, artifact-ref verifier가 server-owned update target과 aggregate snapshot 존재를 확인 | 완료 |
| agent-local artifact upload | `scripts/runtime_adapters/federated_agent/local_training.py`, `scripts/runtime_adapters/federated_agent/artifact_store.py`, `upload_agent_local_lora_classifier_update` | 완료 |
| server-owned materialization | `main_server/src/services/federation/rounds/aggregation/artifact_refs.py`, `methods/adaptation/text_classifier/peft_encoder/update/materialization.py` | 완료 |
| manifest/version compatibility | `methods/adaptation/text_classifier/peft_encoder/server_preflight.py`, `methods/adaptation/server_update_materialization.py`, related unit tests | 완료 |
| alpha=0.3 main baseline | 과거 `clients10_rounds50` report는 round/split/method/delta 기준 PASS지만 runtime metadata 도입 전 산출물이다. 현재 source-of-truth full-budget preset은 30 rounds이며, runtime metadata는 current 1-round smoke와 reduced runs에서 검증 | 부분 |
| alpha=0.1 final stress | 현재 `runs/fl_ssl` 아래 검증 가능한 report가 없음. 기본 비교가 아니라 마지막 stress 확인으로 남김 | 대기 |
| FlexMatch/FreeMatch/PseudoLabel ablation | 5-round reduced reports verified. full budget 비교는 후보와 조건 확정 뒤 별도 실행 | 부분 |
| client_count=1..10 sweep | 1-round summary verified and indexed. full sweep은 후보와 조건 확정 뒤 별도 실행 | 부분 |
| seed sweep은 robustness로 분리 | `seed_sweep` runner/summary는 존재. split seed 42 안정화 뒤 별도 실행 | 대기 |
| accidental long-run guard | `scripts/experiments/fl_ssl/run_safety.py`와 `tests/unit/test_fl_ssl_run_safety.py` | 완료 |
| 선택 전 method placeholder 방지 | `tests/architecture/test_layer_dependencies.py`가 `conf/strategy_axes/fl/method_descriptor/*.yaml`과 실제 `methods/federated_ssl/<method>/` 필수 파일 일치를 검증 | 완료 |

## Read-Only Verification Evidence

아래 2026-05-17 historical 항목들은 당시 감사 기록이다. 현재 워크스페이스에는
해당 artifact들이 없어 `docs/operations/fl_ssl_artifact_verification_manifest.current.json`
대상에서 제외했다. 현재 manifest는 2026-05-26 FedMatch reduced artifact만 읽는다.

전체 검증 manifest:

```bash
uv run python scripts/experiments/fl_ssl/verify_federated_report_artifacts.py \
  --manifest docs/operations/fl_ssl_artifact_verification_manifest.current.json
```

2026-05-26 현재 결과: FedMatch reduced artifact 1개 entry `PASS`.
아래 2026-05-17 결과는 당시 기록으로 유지한다.

- `fixmatch_lora_alpha03_10c_1round_current_20260518` current smoke:
  `PASS runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_seed42/clients10_rounds1/20260517T232304Z/reports/fl_ssl_main_comparison.report.json`
  - 조건: `10 clients`, `alpha=0.3`, split seed `42`, `1 round`,
    `gpu_local + mxbai`, `FixMatch + FedAvg + LoRA-classifier`,
    `server_uploaded_artifact_ref`
  - 서버 update: `10`개, 모두 `aggregation_artifact::` ref이며
    `main_server/aggregation_artifacts/versions/lora_classifier/sim_rev_0001/`에
    누적 LoRA/head snapshot을 남겼다.
- `fixmatch_lora_alpha03_10c_50round_20260518` main:
  `PASS runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_seed42/clients10_rounds50/20260517T150549Z/reports/fl_ssl_main_comparison.report.json`
- `flexmatch_lora_alpha03_10c_5round_20260518` reduced ablation:
  `PASS runs/fl_ssl/manual_baselines/flexmatch_usb_v1__lora_classifier__fedavg/alpha03_seed42/clients10_rounds5/20260517T205436Z/reports/fl_ssl_main_comparison.report.json`
- `freematch_lora_alpha03_10c_5round_20260518` reduced ablation:
  `PASS runs/fl_ssl/manual_baselines/freematch_usb_v1__lora_classifier__fedavg/alpha03_seed42/clients10_rounds5/20260517T212002Z/reports/fl_ssl_main_comparison.report.json`
- `pseudolabel_lora_alpha03_10c_5round_20260518` reduced ablation:
  `PASS runs/fl_ssl/manual_baselines/pseudolabel_usb_v1__lora_classifier__fedavg/alpha03_seed42/clients10_rounds5/20260517T214526Z/reports/fl_ssl_main_comparison.report.json`
- `fixmatch_lora_alpha03_1round_20260518` client-count sweep:
  `PASS runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_seed42/sweeps/client_count_rounds1/20260517T193320Z/reports/fl_ssl_client_count_sweep.summary.json`

Result index read-only ingest:

- 2026-05-17 당시 current manifest 기준으로 FL SSL report 5개와 client-count
  sweep summary 1개를 검증했다. 2026-05-26 현재 current manifest는 FedMatch
  reduced report 1개만 재검증한다.
- `runs/fl_ssl` method-first layout의 dashboard/index 정규화는 result-index unit
  tests와 dashboard export path로 검증한다.
- methods: `fixmatch_usb_v1`, `flexmatch_usb_v1`, `freematch_usb_v1`,
  `pseudolabel_usb_v1`
- algorithms: `fixmatch`, `flexmatch`, `freematch`, `pseudolabel`
- historical artifact round budgets: `[1, 5, 50]`
- shard alphas: `[0.3]`

## Next Gate

FedMatch method-owned smoke는 2026-05-22에 current runtime metadata로 확인했다.

- 1-round smoke:
  `method_owned + fedmatch_agreement + fixed_probe_output_knn + fedmatch_partitioned`
  report 생성과 `partitioned_deltas` 제출을 확인했다. round 1은 previous client
  snapshot이 없어 helper count가 0인 것이 정상이다.
- 2-client 2-round smoke:
  round 2에서 `fedmatch_peer_context_helper_count=1.0`,
  `fedmatch_peer_context_refreshed=1.0`, `fedmatch_partitioned_delta_count=2.0`을
  확인했고 report verification CLI가 PASS했다.

다음 실행성 작업은 바로 10-client 5-round reduced를 돌리는 것이 아니라,
LoRA-classifier simulation 병목을 먼저 줄이는 것이다. 확인된 병목은 client/round마다
frozen transformer backbone/tokenizer를 다시 로드하는 것, helper snapshot마다 helper
model을 다시 materialize하는 것, 전체 validation rows를 fixed probe처럼 쓰는 것이다.
다음 구현 게이트는 `lora_classifier` adapter-family simulation runtime에
backbone/tokenizer cache를 추가하고, `fixed_probe_output_knn` probe를 작은 deterministic
probe subset과 manifest/hash로 계약화하는 것이다. 이후 manual FixMatch baseline과 같은
split/seed/local budget으로 FedMatch reduced report를 맞춘다.
