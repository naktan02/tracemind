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
- FL SSL에서 `LoRA + classifier`를 shared family로 승격할 때의 canonical family
  이름은 `lora_classifier`로 둔다. `classifier_head`에 LoRA 옵션을 섞거나
  bare `lora` family로 head 의미를 숨기지 않는다.
- `lora_classifier`의 1차 범위는 FL simulation research path이고, live
  `agent`/`main_server` runtime translation은 2차 범위다.
- `lora_classifier` 비교의 고정 조건은 `mxbai_encoder`, tokenizer, LoRA
  `rank=8/alpha=16/dropout=0.1/target_modules=all-linear`, canonical seed
  checkpoint, label schema, non-IID split, seed, metric으로 둔다. 이 중 하나를
  바꾸면 method 비교가 아니라 scaffold 비교로 기록한다.
- FL SSL main split은 `10 clients`, Dirichlet label-skew `alpha=0.3`, `3 seeds`로 고정한다.
- FL SSL stress split은 같은 조건에서 Dirichlet label-skew `alpha=0.1`로 둔다.
- 각 client pool은 기본적으로 `10% labeled / 90% unlabeled`로 나눈다.
- FL SSL main budget은 `50 communication rounds`, `local_epochs=1`, `max_steps=50`으로 고정한다.
- smoke budget은 실행 확인용으로 `3 rounds`를 쓴다.
- winner 1차 기준은 `macro-F1 + worst-client macro-F1`이다.
- tie-breaker/risk 지표는 `ECE`, communication cost, per-client variance다.
- FL SSL report는 `fl_ssl_main_comparison` track으로 저장하고 중앙 SSL control report와 같은 ranking으로 합치지 않는다.
- 현재 FL SSL method 축의 활성 baseline은 `fedavg_pseudo_label`이다.
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
- stress non-IID: Dirichlet label-skew `alpha=0.1`
- seeds: `3`
- round budget: `50`
- local update budget: `local_epochs=1`, `max_steps=50`
- labeled/unlabeled ratio: `10% / 90%` per client
- primary metrics: `macro-F1`, `worst-client macro-F1`
- secondary metrics: `ECE`, communication cost, per-client variance
- report separation: central SSL control table과 FL SSL main comparison table을 같은
  ranking으로 합치지 않는다.
- method selection: `strategy_axes/fl/method_descriptor=fedavg_pseudo_label` baseline만 현재 active runtime이다.
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
- runtime overview: `docs/architecture/system-overview.md`
- strategy axes: `docs/strategy_surface_map.md`

## Boundaries

- 공용 contract, domain entity, canonical payload 해석은 `shared`에 둔다.
- 교체 가능한 algorithm/method 계산은 `methods`에 둔다.
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
5. Phase 4: 고정된 `10 clients`, `alpha=0.3/0.1`, `10/90`, `3 seeds` 조건에서 FL SSL non-IID 메인 비교.
6. Phase 5: FL SSL winner runtime translation.
7. Phase 6: 필요 시 richer shared adapter.
8. Phase 7: clipping, secure aggregation, DP, 필요 시 HE.

## Next Priorities

1. query buffer 필드와 retention을 고정한다.
2. threshold/policy selection과 manual label override hook을 고정한다.
3. central SSL control의 supervised baseline을 연다.
4. 같은 scaffold에서 pseudo-label, prototype SSL, FixMatch, R-Drop, MixText를 비교한다.
5. FL SSL main comparison smoke를 `strategy_axes/fl/shard_policy=dirichlet_alpha03`와
   `strategy_axes/fl/method_descriptor=fedavg_pseudo_label`로 실행해 report를 확인한다.
6. 후보 논문 method를 비교해 실제 구현할 FL SSL method를 확정한다.
7. `lora_classifier` family를 먼저 FL simulation research path에 얇게 열고,
   contract와 aggregation shape를 smoke로 확인한다.
8. 확정된 method부터 `agent` local runtime과 필요한 `main_server` round/aggregation
   경계에 구현한다.
9. 고정 조건에서 확정 method들을 메인 비교로 실행한다.
10. winner를 `lora_classifier` family 또는 현실적인 fallback family로 translation 한다.

## Validation Criteria

- Seed: canonical seed artifact, confusion, confidence distribution이 재현 가능하다.
- Central SSL: 같은 고정 조건에서 control table과 output metadata가 남는다.
- FL SSL: client partition, non-IID 정도, labeled/unlabeled ratio, metric, seed, local/round budget이 고정돼 있다.
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
