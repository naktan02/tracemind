# FL SSL Execution Audit

작성일: 2026-05-18

이 문서는 현재 FL SSL 목표를 실제 artifact와 검증 명령 기준으로 대조한 감사표다.
장시간/full-budget 실험은 새로 실행하지 않고, 필요한 경우 1-round smoke와 기존
report/summary를 검증했다.

## 판정

전체 목표는 아직 완료가 아니다.

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
- `10 clients`, Dirichlet `alpha=0.3`, split seed `42`, `50 rounds`,
  `FixMatch + FedAvg + LoRA-classifier` main baseline report는 round/split/method/delta
  기준 verifier로 확인됐다. 이 report는 runtime metadata 도입 전 산출물이라
  `gpu_local + mxbai` 여부는 report 자체로는 재검증할 수 없다.
- FlexMatch/FreeMatch/PseudoLabel ablation은 5-round reduced run으로 method
  metadata와 실제 local objective 변경을 확인했다.
- Dirichlet `alpha=0.1` stress는 현재 `runs/fl_ssl` method-first layout 아래
  검증 가능한 report가 없으므로, 새 실행 전까지 current manifest 검증 대상에서
  제외한다.
- `client_count=1..10` sweep은 1-round summary와 하위 report 10개를 확인했다.
- result index는 FL `fl_ssl_main_comparison.report.json`와 client-count sweep 하위
  report를 dashboard/index record로 정규화한다.
- FL SSL runner는 총 예정 communication round가 `49`를 넘으면 기본 차단한다.

현재 사용자 결정에 따라 새로 실행하지 않는 항목:

- Dirichlet `alpha=0.1` stress의 full `50 rounds` 실행.
- FlexMatch/FreeMatch/PseudoLabel full-budget ablation.
- `client_count=1..10` full sweep. `5 rounds x 10 client-counts`처럼 합계
  `50` rounds인 sweep도 long-run guard 대상이다.

결정 대기 항목:

- FedMatch/FedLGMatch/(FL)^2 중 첫 구현 method 선택.
  현재 추천 선택지는 FedMatch이며, 기본 시작점은 `lora_classifier` payload family 유지,
  custom round-state exchange 없음, FedAvg server policy 유지다.
  2026-05-18 사용자 응답으로 첫 method 선택은 아직 확정하지 않는다.

## Prompt-To-Artifact Checklist

| 요구사항 | 현재 증거 | 상태 |
|---|---|---|
| 작업별 commit/push | 작업 단위 커밋은 `origin/master`에 push했다. 최신 상태는 `git status --short --branch`와 `git log --oneline origin/master -n 12`로 확인한다 | 완료 |
| scripts는 thin wrapper, method core는 methods 소유 | `scripts/experiments/fl_ssl/*`는 config/run/report, SSL algorithm은 `methods/ssl/algorithms/*`, LoRA aggregation은 `methods/adaptation/lora_classifier/*` | 완료 |
| 실제 PEFT 기준 Query SSL local objective 호출 | `tests/unit/test_lora_fixmatch_runner.py`, `tests/unit/test_federated_agent_runtime_adapters.py`, `tests/unit/test_run_federated_simulation.py`, current 1-round smoke report `objective.query_ssl.algorithm_name=fixmatch` | 완료 |
| LoRA/classifier delta가 FedAvg까지 집계 | `methods/adaptation/lora_classifier/aggregation/fedavg.py`, `main_server/tests/unit/test_aggregation_service.py`, artifact-ref verifier가 server-owned update target과 aggregate snapshot 존재를 확인 | 완료 |
| agent-local artifact upload | `scripts/runtime_adapters/federated_agent/query_ssl_lora_classifier_trainer.py`, `upload_agent_local_lora_classifier_update` | 완료 |
| server-owned materialization | `main_server/src/services/federation/rounds/aggregation/artifact_refs.py`, `methods/adaptation/lora_classifier/aggregation/materialization.py` | 완료 |
| manifest/version compatibility | `methods/adaptation/lora_classifier/server_preflight.py`, `methods/adaptation/server_update_materialization.py`, related unit tests | 완료 |
| alpha=0.3 main baseline | `runs/fl_ssl/manual_baselines/fixmatch_usb_v1__lora_classifier__fedavg/alpha03_seed42/clients10_rounds50/20260517T150549Z/reports/fl_ssl_main_comparison.report.json`는 round/split/method/delta 기준 PASS. runtime metadata는 current 1-round smoke와 reduced runs에서 검증 | 부분 |
| alpha=0.1 stress | 현재 `runs/fl_ssl` 아래 검증 가능한 report가 없음. 새 reduced/full 실행은 현재 사용자 결정에 따라 하지 않음 | 대기 |
| FlexMatch/FreeMatch/PseudoLabel ablation | 5-round reduced reports verified. 새 full budget 실행은 현재 사용자 결정에 따라 하지 않음 | 부분 |
| client_count=1..10 sweep | 1-round summary verified and indexed. 새 full sweep은 현재 사용자 결정에 따라 하지 않음 | 부분 |
| seed sweep은 robustness로 분리 | `seed_sweep` runner/summary는 존재. split seed 42 안정화 뒤 별도 실행 | 대기 |
| 50-round 재실행 방지 | `scripts/experiments/fl_ssl/run_safety.py`와 `tests/unit/test_fl_ssl_run_safety.py` | 완료 |
| 선택 전 method placeholder 방지 | `tests/architecture/test_layer_dependencies.py`가 `conf/strategy_axes/fl/method_descriptor/*.yaml`과 실제 `methods/federated_ssl/<method>/` 필수 파일 일치를 검증 | 완료 |

## Read-Only Verification Evidence

아래 검증 중 첫 항목은 현재 코드 기준으로 새로 실행한 1-round smoke다. 나머지는
새 round를 실행하지 않고 기존 JSON artifact만 읽었다.

전체 검증 manifest:

```bash
uv run python scripts/experiments/fl_ssl/verify_federated_report_artifacts.py \
  --manifest docs/operations/fl_ssl_artifact_verification_manifest.current.json
```

결과: 6개 artifact entry 모두 `PASS`.

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

- current manifest 기준 FL SSL report 5개와 client-count sweep summary 1개를 검증했다.
- `runs/fl_ssl` method-first layout의 dashboard/index 정규화는 result-index unit
  tests와 dashboard export path로 검증한다.
- methods: `fixmatch_usb_v1`, `flexmatch_usb_v1`, `freematch_usb_v1`,
  `pseudolabel_usb_v1`
- algorithms: `fixmatch`, `flexmatch`, `freematch`, `pseudolabel`
- round budgets: `[1, 5, 50]`
- shard alphas: `[0.3]`

## Next Gate

다음 실행성 작업은 현재 사용자 결정에 따라 진행하지 않는다. 구현성 작업의 다음
게이트는 FedMatch/FedLGMatch/(FL)^2 중 첫 method 선택이지만, 2026-05-18
사용자 응답으로 이 선택은 아직 보류한다.
`docs/contracts/fl_ssl_method_capability_matrix.md`와
`methods/federated_ssl/NEW_METHOD.md`는 이미 선택 전/선택 후 경계를 나눠 둔 상태다.
현재 권장 첫 후보는 payload family를 바꾸지 않는 FedMatch method-owned local
objective다.
