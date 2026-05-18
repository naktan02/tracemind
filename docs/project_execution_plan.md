# TraceMind Execution Plan

현재 active decision만 유지한다. 세부 구현 지도는 `docs/architecture/system-overview.md`,
`docs/staged_execution_roadmap.md`, `docs/fl_runtime_implementation_checklist.md`,
`docs/contracts/*`를 본다.

## Active Goal

1. 원문 텍스트와 개인 해석 상태는 agent 로컬에 남긴다.
2. 공통 의미 표현 공간은 전역 모델과 shared artifact로 유지한다.
3. 최종 판단은 로컬 개인화 상태와 decision policy가 수행한다.
4. 초기 seed는 중앙집중형 `fixed embedding + classifier`로 만든다.
5. 중앙 SSL은 pooled/offline control로 비교한다.
6. 논문 메인 비교는 `FL SSL under non-IID`로 둔다.
7. winner는 runtime/privacy 제약에 맞게 후행 translation 한다.

```text
central fixed embedding + classifier seed
-> central SSL pooled/offline control
-> FL SSL non-IID main comparison
-> FL/runtime translation
```

## Fixed Decisions

- `WindowSummary`, `NormPack`은 활성 경로가 아니다.
- `PrototypePack`은 bootstrap/comparison/reference artifact이며 메인 판정기가 아니다.
- prototype 기반 pseudo-label/SSL은 SSL 비교군 중 하나로 다룬다.
- canonical seed artifact는 `clf_2026_04_11_143138`이다.
- seed model: `data/processed/classifier_heads/clf_2026_04_11_143138.pt`
- seed manifest: `data/processed/classifier_heads/clf_2026_04_11_143138.manifest.json`
- seed report: `runs/train_classifier/clf_2026_04_11_143138/reports/report.json`
- query-domain 적응 단계에서만 `LoRA + classifier`를 연다.
- 중앙 SSL은 FL client partition 없는 control table이다. seed full replay 기본값이 아니다.
- 중앙 canonical 규약은 `seed checkpoint 1회 생성 -> new accepted query-derived rows only continual adaptation`이다.
- `FedMatch`, `FedLGMatch`, `(FL)^2`는 FL SSL non-IID 메인 비교군이다.
- 논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다.
  method-only local/server/aggregation 변형은 이 폴더에 남기고, 두 개 이상
  방법론에서 공유되는 계산만 축별 `methods` 패키지로 승격한다.
- FL SSL에서 `LoRA + classifier`를 shared family로 승격할 때의 canonical family
  이름은 `lora_classifier`로 둔다. `classifier_head`에 LoRA 옵션을 섞거나
  bare `lora` family로 head 의미를 숨기지 않는다.
- `lora_classifier`의 1차 범위는 FL simulation research path이고, live
  `agent`/`main_server` runtime translation은 2차 범위다.
- `lora_classifier` 비교의 고정 조건은 `mxbai_encoder`, tokenizer, LoRA
  `rank=8/alpha=16/dropout=0.1/target_modules=all-linear`, canonical seed
  checkpoint, label schema, non-IID split, seed, metric으로 둔다. 이 중 하나를
  바꾸면 method 비교가 아니라 scaffold 비교로 기록한다.
- FL SSL main split은 `10 clients`, Dirichlet label-skew `alpha=0.3`, `seed=42`
  materialized manifest로 우선 고정한다.
- FL SSL 기본/main split은 `alpha=0.3`이다.
- `alpha=0.1`은 기본 비교가 아니라 마지막 stress/robustness 확인 요소로 둔다.
- materialized FL split은 선택된 labeled source 전체와 unlabeled source 전체를
  client에 분배한다. labeled source는 `ourafla_reddit` 또는 `szegeelim_general4`
  중에서 `query_data_selection.labeled`로 고른다.
- 라벨 데이터를 일부만 쓰는 ablation은 materialized split 생성 시
  `fl_client_split_materialization.labeled_policy`로 명시하고, 기본값은
  `mode=all`이다.
- FL SSL archived main budget은 `50 communication rounds`, `local_epochs=1`,
  `max_steps=50`이었다. 현재 실행 정책은 새 `50-round`/full-budget 재실행 금지이며,
  기존 alpha=0.3 full report는 read-only evidence로만 사용한다.
- smoke budget은 실행 확인용으로 `3 rounds`를 쓴다.
- winner 1차 기준은 `macro-F1 + worst-client macro-F1`이다.
- tie-breaker/risk 지표는 `loss`, `weighted-F1`, `balanced accuracy`,
  worst-category F1, `ECE/max-ECE`, communication cost, per-client variance다.
- FL SSL report는 `fl_ssl_main_comparison` track으로 저장하고 중앙 SSL control report와 같은 ranking으로 합치지 않는다.
- FL SSL report는 round progression, round delta, client split label
  distribution, aggregation proxy diagnostics를 함께 남긴다. `theta` 같은
  method 내부 파라미터는 기본 report에 노출하지 않는다.
- 신규 FL SSL 실행 산출물은
  `runs/fl_ssl/<method_family>/<method_composition>/<split>/<clients_rounds>/...`
  아래에서 method composition을 먼저 고르고 split/client/round 변수를 그 아래에
  쌓는다. 기존 `runs/federated_simulation*` 산출물도 같은 구조로
  마이그레이션했다.
- 현재 FL SSL 기본 실행 조합은 descriptor 없는
  `manual + FixMatch + FedAvg + LoRA-classifier`다.
- `manual` mode는 논문 method가 아니라
  `query_ssl_method/round_runtime.*` 조합 baseline/ablation용이다.
  report/index에는 `execution_role=manual_baseline`으로 기록하고
  `descriptor_name`은 비워 둔다.
  FedMatch/FedLGMatch/(FL)^2처럼 client objective/server policy/round-state
  요구사항을 함께 소유하는 상위 method는 `method_owned`로 선택한다.
- FL `security_policy`는 method identity가 아니라 runtime capability 축이며, 현재
  simulation은 `plaintext`만 지원한다.
- FedMatch/FedLGMatch/(FL)^2 같은 논문 method 구현은 후보 비교 후 확정된 method부터 연다.
- FL SSL smoke/main/sweep 실행 기본 runtime은 `execution_context/runtime_env=gpu_local`,
  embedding adapter는 `mxbai`다. `gpu_online`은 cache warm-up/최초 다운로드용이고,
  `cpu_local`과 `hash_debug`는 wiring smoke나 단위 검증용으로만 사용한다.
- 시스템 v1 baseline은 `embedding -> global classifier -> local interpretation`이다.
- `v2`에서만 private adapter/head 기반 표현 개인화를 연다.

## Active Rails

Seed:

```text
Reddit Labeled Data -> Fixed Embedding -> Classifier Seed -> Local Deployment
```

Central SSL control:

```text
Query Buffer -> Selection -> Accepted Rows -> LoRA + Classifier -> Central Evaluation
```

비교 family는 `supervised`, `pseudo-label`, `prototype SSL`, `FixMatch`,
`R-Drop`, `MixText`이고 `TAPT`는 optional preadaptation이다. 같은 table 안에서는
backbone, tokenizer, label schema, LoRA spec, initial checkpoint, query selection
rule, seed를 고정한다.

FL SSL non-IID:

```text
Client Signal -> Local SSL Training -> Shared Update -> Aggregation -> New Manifest
```

중앙 SSL 결과는 sanity check와 pooled/offline control로만 해석한다.

고정 조건:

- clients: `10`
- main non-IID: Dirichlet label-skew `alpha=0.3`
- final stress non-IID: Dirichlet label-skew `alpha=0.1`
- split seed: `42`
- archived full round budget: `50`
- current execution policy: 새 `50-round`/full-budget run은 실행하지 않고,
  wiring/method 검증은 `1-round` smoke 또는 `5-round` reduced run으로 제한한다.
- local update budget: `local_epochs=1`, `max_steps=50`
- labeled/unlabeled source: 기본은 labeled source 전체와 unlabeled source 전체를
  client에 분배한다. 일부 labeled source만 쓰는 경우 `labeled_policy`를 manifest에
  기록하고, 실제 ratio는 report count로 기록한다.
- primary metrics: `macro-F1`, `worst-client macro-F1`
- secondary metrics: `loss`, `weighted-F1`, `balanced accuracy`,
  worst-category F1, `ECE/max-ECE`, communication cost, per-client variance
- progression diagnostics: round별 validation delta, best macro-F1 round,
  best loss round
- split/aggregation diagnostics: client별 label distribution, entropy,
  accepted-count 기반 aggregation weight proxy
- report separation: central SSL control table과 FL SSL main comparison table을 같은
  ranking으로 합치지 않는다.
- method selection: 기본 baseline은 `fl_method.composition_mode=manual`,
  `strategy_axes/ssl/consistency_method=fixmatch_usb_v1`,
  `round_runtime.adapter_family_name=lora_classifier`,
  `round_runtime.aggregation_backend_name=fedavg`다.
- runtime: 기본 실행은 `gpu_local + mxbai`로 본다. CPU/hash debug 결과는
  성능 숫자나 논문 비교 근거로 쓰지 않는다.

Runtime translation:

- FL SSL winner를 현재 `ModelManifest`나 `TrainingUpdateEnvelope`에 바로 넣지 않는다.
- 필요한 shared family와 state/update payload를 먼저 정의한다.
- 현재 1순위 translation 후보는 `lora_classifier` family다.
- `lora_classifier` state/update payload는 LoRA adapter state와 classifier head
  state를 함께 표현해야 하며, LoRA weight는 inline JSON vector만 가정하지 않고
  artifact-ref 기반 전송/집계 경로를 열어 둔다.

## Source Of Truth

1. `shared/src/contracts/*.py`
2. `shared/src/domain/entities/*`
3. `shared/src/contracts/README.md`

보조 문서:

- query buffer: `docs/contracts/query_buffer_v1.md`
- central SSL scaffold: `docs/contracts/central_lora_classifier_trainer_contract.md`
- FL runtime worklist: `docs/fl_runtime_implementation_checklist.md`
- evaluation metrics: `methods/evaluation/README.md`
- runtime overview: `docs/architecture/system-overview.md`
- strategy axes: `docs/strategy_surface_map.md`

## Boundaries

- 공용 contract, domain entity, canonical payload 해석은 `shared`에 둔다.
- 교체 가능한 algorithm/method 계산은 `methods`에 둔다.
- 논문 method의 descriptor, recipe metadata 또는 optional recipe,
  local/server/round policy, method-only aggregation 변형은
  `methods/federated_ssl/<method>/`에 둔다.
- 실행 조합과 파라미터는 루트 `conf`에 둔다.
- agent-owned local training/inference runtime은 `agent`에 둔다.
- server-owned round/rebuild/publication orchestration은 `main_server`에 둔다.
- `scripts`는 실험 조합과 실행 표면만 소유한다.
- 운영 후보 로직을 `scripts`에 먼저 만들고 나중에 복사하지 않는다.
- query-domain 중앙 trainer는 시스템 runtime contract를 오염시키지 않는 별도 레일이다.

구조 목표는 `shared -> methods -> agent/main_server/scripts` 방향으로 읽힌다.
`agent`, `main_server`, `scripts` 사이 직접 의존은 금지하고, scripts가 runtime을
재사용해야 할 때만 `scripts/runtime_adapters/`에 명시 bridge를 둔다.

## Phase Map

1. Phase 0: contract 정리.
2. Phase 1: `central fixed embedding + classifier` seed 완료.
3. Phase 2: query buffer, threshold/policy, manual label hook 고정.
4. Phase 3: central SSL pooled/offline control 비교.
5. Phase 4: 고정된 `10 clients`, `alpha=0.3/0.1`, `seed=42` materialized split 조건에서 FL SSL non-IID 메인 비교.
6. Phase 5: FL SSL winner runtime translation.
7. Phase 6: 필요 시 richer shared adapter.
8. Phase 7: clipping, secure aggregation, DP, 필요 시 HE.

## Next Priorities

현재 체크포인트:

- 실제 PEFT executor 기준 `FedAvg + FixMatch + LoRA-classifier` 1-round smoke는
  완료했다. report metadata는 `query_ssl_method.algorithm_name=fixmatch`로
  `methods/ssl/algorithms/*`가 실제 local objective를 소유함을 남긴다.
- LoRA/classifier delta artifact 경로는 `agent-local://` ref를 server-owned
  `aggregation_artifact::` ref로 upload/materialize할 수 있게 닫았다.
  server direct accept는 server-owned ref와 inline debug payload만 수락한다.
- `10 clients`, Dirichlet `alpha=0.3`, split `seed=42`, `50 rounds` main
  baseline report는 round/split/method/delta 기준으로 검증했다. 이 report는
  runtime metadata 도입 전 산출물이라 `gpu_local + mxbai` 여부는 report 자체로
  재검증할 수 없고, 현재 코드 기준 runtime metadata는 같은 split의 1-round smoke와
  reduced runs로 확인했다.
- Dirichlet `alpha=0.1` final stress, full-budget
  FlexMatch/FreeMatch/PseudoLabel ablation, full-budget `client_count=1..10` sweep은
  현재 사용자 결정에 따라 새로 실행하지 않는다. 현재는 `alpha=0.3` 기준
  FlexMatch/FreeMatch/PseudoLabel ablation을 5-round reduced run으로 확인했고,
  `client_count=1..10` sweep은 1-round summary로 확인했다. `alpha=0.1`은 마지막
  stress 확인으로 남겨 두며 current manifest 검증 대상이 아니다.
- FL SSL runner는 총 예정 communication round가 49를 넘으면 기본 차단한다.
  단일 run은 `rounds`, seed/client-count sweep은 `rounds * sweep 항목 수`로
  계산하며, 장시간 실행은 `run_safety.allow_long_run=true`와
  `run_safety.long_run_ack=ALLOW_FL_SSL_LONG_RUN`을 명시한 경우에만 시작된다.
- 기존 smoke/main/reduced ablation/1-round sweep 산출물은
  `scripts/experiments/fl_ssl/verify_federated_report_artifacts.py`로 round budget,
  client count, SSL method, adapter family, aggregation, delta format metadata를
  재검증할 수 있다. 현재 감사용 manifest는
  `docs/operations/fl_ssl_artifact_verification_manifest.current.json`다.

다음 우선순위:

1. FedMatch/FedLGMatch/(FL)^2 중 실제 구현할 첫 method를 확정하고, 필요한
   round-state exchange/server policy capability를 먼저 문서화한다.
   선택 전 capability matrix는 `docs/contracts/fl_ssl_method_capability_matrix.md`에
   있으며, 현재 구현 순서 추천은 FedMatch -> FedLGMatch -> (FL)^2다.
   2026-05-18 사용자 응답으로 첫 method 선택은 아직 보류한다.
2. 확정 method부터 `methods/federated_ssl/<method>/`, `conf`, 필요한 runtime
   capability adapter, test 순서로 추가한다.
3. 확정 method는 먼저 `1-round` smoke와 필요 시 `5-round` reduced run으로
   method metadata와 실제 local/server policy 변경을 검증한다.
4. full ablation, full `client_count=1..10` sweep, 새 `50-round` main rerun은
   현재 보류한다. `alpha=0.1`은 기본 비교가 아니라 최후 stress 확인으로 남기고,
   `alpha=0.3` 기준 후보 비교가 정리된 뒤에만 같은 split seed 42와 같은 local
   budget으로 연다.
5. winner를 `lora_classifier` family 또는 현실적인 fallback family로 translation 한다.

## Validation Criteria

- Seed: canonical seed artifact, confusion, confidence distribution이 재현 가능하다.
- Central SSL: 같은 고정 조건에서 control table과 output metadata가 남는다.
- FL SSL: client partition, non-IID 정도, labeled/unlabeled source/policy,
  metric, split seed, local/round budget이 고정돼 있다.
- Runtime: update base revision, aggregation, publication, artifact rebuild가 일관된다.
- Privacy: raw text는 서버로 올라가지 않고 privacy layer는 training logic과 분리된다.

## User Decisions Still Needed

1. query buffer raw text retention 기본값.
2. LoRA target module/rank/alpha/dropout 변경 여부. 기본 비교 scaffold는
   `rank=8`, `alpha=16`, `dropout=0.1`, `target_modules=all-linear`로 고정한다.
3. FL 범위를 `lora_classifier` family에서 LoRA adapter, classifier head,
   aggregation artifact까지 어디까지 열지.
4. private adapter/head 도입 시점.
5. secure aggregation과 DP 도입 시점.
6. multi-prototype runtime 확장 여부.
