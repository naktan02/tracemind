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

최종 method/runtime 구조 판단은
`docs/architecture/target-method-runtime-structure.md`를 우선한다. 기존 문서의
`lora_classifier`, `adapter_family_name`, `fedmatch_agreement` 같은 이름은 old artifact나
legacy compatibility 표면일 수 있으며, 새 실행 config 판단에서는 target 문서의
`payload_adapter_kind`, `update_family`, `trainable_state`, `method descriptor`,
`runtime capability` 용어를 기준으로 삼는다.

## Fixed Decisions

- `WindowSummary`, `NormPack`은 활성 경로가 아니다.
- `PrototypePack`은 bootstrap/comparison/reference artifact이며 메인 판정기가 아니다.
- prototype 기반 pseudo-label/SSL은 SSL 비교군 중 하나로 다룬다.
- canonical seed artifact는 `clf_2026_04_11_143138`이다.
- seed model: `data/processed/classifier_heads/clf_2026_04_11_143138.pt`
- seed manifest: `data/processed/classifier_heads/clf_2026_04_11_143138.manifest.json`
- seed report: `runs/train_classifier/clf_2026_04_11_143138/reports/report.json`
- query-domain 적응 단계에서만 `PEFT encoder classifier`를 연다.
- 중앙 SSL은 FL client partition 없는 control table이다. seed full replay 기본값이 아니다.
- 중앙 canonical 규약은 `seed checkpoint 1회 생성 -> new accepted query-derived rows only continual adaptation`이다.
- `FedMatch`, `FedLGMatch`, `(FL)^2`는 FL SSL non-IID 메인 비교군이다.
- 논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다.
  method-only local/server/aggregation 변형은 이 폴더에 남기고, 두 개 이상
  방법론에서 공유되는 계산만 축별 `methods` 패키지로 승격한다.
- FL SSL에서 `PEFT encoder classifier`를 공유 가능한 trainable state로 승격할 때의
  target canonical update family 이름은 `peft_text_classifier`로 둔다.
  `lora_classifier`는 v1 contract, old artifact, direct import compatibility 이름으로만
  유지한다. LoRA는 PEFT adapter mechanism이고, classifier head는 linear-head
  primitive이므로 두 개념을 `classifier_head` 옵션이나 bare `lora` family로 숨기지
  않는다.
- `peft_text_classifier`의 1차 범위는 FL simulation research path이고, live
  `agent`/`main_server` runtime translation은 2차 범위다. 현행 v1 실행 field는
  migration window 동안 `lora_classifier` 이름을 쓸 수 있다.
- PEFT text-classifier 비교의 고정 조건은 `mxbai_encoder`, tokenizer, LoRA
  `rank=8/alpha=16/dropout=0.1/target_modules=all-linear`, canonical seed
  checkpoint, label schema, non-IID split, seed, metric으로 둔다. 이 중 하나를
  바꾸면 method 비교가 아니라 scaffold 비교로 기록한다.
- FL SSL main split은 `10 clients`, Dirichlet label-skew `alpha=0.3`, `seed=42`
  materialized manifest로 우선 고정한다.
- FL SSL 기본/main split은 `alpha=0.3`이다.
- `alpha=0.1`은 기본 비교가 아니라 마지막 stress/robustness 확인 요소로 둔다.
- materialized FL split은 labeled source 선택량과 labeled exposure 위치를 분리한다.
  현재 entrypoint 기본값은 `shared_client_seed`로, 같은 public labeled seed를
  모든 client가 local SSL에 쓰고,
    unlabeled source만 client별 non-IID split으로 둔다. materialized split
    생성/로드 경로를 지원한다.
  - `client_local_split`: legacy/ablation 값으로, 선택된 labeled source를
    bootstrap/server seed와 client-local labeled pool에 나눠 둔다.
  - `server_only_seed`: labeled source는 server/bootstrap boundary에만 두고,
    client는 local unlabeled shard만 가진다. client labeled batch를 요구하는 method는
    compatibility validator에서 실행 전에 막는다. materialized artifact와 request
    metadata를 지원하고, `server_step_policy=supervised_seed_step` 조합에서는
    round open 전 supervised server seed step을 실행한다.
  labeled source는 `ourafla_reddit` 또는 `szegeelim_general4` 중에서
  `query_data_selection.labeled`로 고른다.
- 라벨 데이터를 일부만 쓰는 ablation은 materialized split 생성 시
  `fl_client_split_materialization.labeled_policy`로 명시하고, 기본값은
  `mode=all`이다. 이 값은 labeled source에서 얼마나 고를지의 문제이며,
  고른 labeled rows를 client/server 어디에 노출할지는 별도
  `labeled_exposure_policy` 축이 소유한다.
- FL SSL smoke 산출물은 `runs/_smoke/fl_ssl` 아래에 둬서 논문/웹용
  `runs/fl_ssl` 산출물과 섞지 않는다.
- FL SSL reduced preset은 검증 실험용 `10 clients`, `5 communication rounds`로 둔다.
- FL SSL full-budget preset은 `30 communication rounds`, `local_epochs=1`,
  `max_steps=20`이다. 새 method/wiring은 먼저 smoke/reduced run으로 확인한 뒤
  full-budget 비교로 올린다.
- smoke budget은 실행 확인용으로 `3 rounds`를 쓴다.
- winner 1차 기준은 `macro-F1 + worst-client macro-F1`이다.
- tie-breaker/risk 지표는 `loss`, `weighted-F1`, `balanced accuracy`,
  worst-category F1, `ECE/max-ECE`, communication cost, per-client variance다.
- FL SSL report는 `fl_ssl_main_comparison` track으로 저장하고 중앙 SSL control report와 같은 ranking으로 합치지 않는다.
- FL SSL report는 round progression, round delta, client split label
  distribution, aggregation proxy diagnostics를 함께 남긴다. `theta` 같은
  method 내부 파라미터는 기본 report에 노출하지 않는다.
- 신규 FL SSL reduced/main 산출물은
  `runs/fl_ssl/<method_family>/<method_composition>/<split>/<clients_rounds>/...`
  아래에서 method composition을 먼저 고르고 split/client/round 변수를 그 아래에
  쌓는다. smoke 산출물은 같은 하위 구조를 `runs/_smoke/fl_ssl/...` 아래에
  쌓는다. 기존 `runs/federated_simulation*` 산출물도 같은 구조로 마이그레이션했다.
- 현재 FL SSL 기본 실행 조합은 descriptor 없는
  `manual + FixMatch + FedAvg + PEFT-classifier`다.
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
- full round budget preset: `30`
- execution policy: 새 wiring/method 검증은 먼저 `1-round` smoke 또는 `5-round`
  reduced run으로 확인하고, full-budget 비교는 후보와 조건이 확정된 뒤 실행한다.
- local update budget: main fair comparison은 `local_epochs=1`, `max_steps=20`
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
- pseudo-label diagnostics: client full unlabeled pool 크기는 `candidate_count`로
  유지하고, 품질 진단은 `diagnostic_view` deterministic subset의
  `diagnostic_candidate_count`로 별도 기록한다. global/client 성능 평가는
  validation/test split 기준이다.
- runtime diagnostics: client round report에 `timing_breakdown`을 남긴다. 이 값은
  stdout 로그가 아니라 report metadata이며, model prepare, training loop,
  pseudo-label diagnostics, update materialization, server submit 시간을 구간별로
  확인하는 데만 쓴다.
- artifact persistence: FL SSL simulation의 canonical update source는 server-owned
  aggregation artifact다. agent-local update 사본은 기본 저장하지 않고
  `protocol.artifact_persistence.persist_agent_local_updates=false`로 기록한다.
- report separation: central SSL control table과 FL SSL main comparison table을 같은
  ranking으로 합치지 않는다.
- method selection: 현재 기본 baseline은 v1 실행 field 기준
  `fl_method.composition_mode=manual`,
  `strategy_axes/ssl/consistency_method=fixmatch_usb_v1`,
  `strategy_axes/trainable_state/update_family=peft_text_classifier`,
  `round_runtime.update_family_name=peft_text_classifier`,
  `round_runtime.aggregation_backend_name=fedavg`다. `lora_classifier`는 v1
  artifact/report reader compatibility 이름으로만 남긴다.
- runtime: 기본 실행은 `gpu_local + mxbai`로 본다. CPU/hash debug 결과는
  성능 숫자나 논문 비교 근거로 쓰지 않는다.

Runtime translation:

- FL SSL winner를 현재 `ModelManifest`나 `TrainingUpdateEnvelope`에 바로 넣지 않는다.
- 필요한 shared family와 state/update payload를 먼저 정의한다.
- 현재 1순위 translation 후보는 target 기준 `peft_text_classifier` update family다.
  v1 contract/report compatibility에서는 `lora_classifier` 이름이 남을 수 있다.
- `peft_text_classifier` state/update payload는 PEFT adapter state와 linear classifier
  head state를 함께 표현해야 하며, PEFT weight는 inline JSON vector만 가정하지 않고
  artifact-ref 기반 전송/집계 경로를 열어 둔다.

## Source Of Truth

1. `shared/src/contracts/*.py`
2. `shared/src/domain/entities/*`
3. `shared/src/contracts/README.md`

보조 문서:

- query buffer: `docs/contracts/query_buffer_v1.md`
- central SSL scaffold: `docs/contracts/central_peft_classifier_trainer_contract.md`
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

- 실제 PEFT executor 기준 `FedAvg + FixMatch + PEFT-classifier` 1-round smoke는
  완료했다. report metadata는 `query_ssl_method.algorithm_name=fixmatch`로
  `methods/ssl/algorithms/*`가 실제 local objective를 소유함을 남긴다.
- PEFT adapter/classifier-head delta artifact 경로는 `agent-local://` ref를 server-owned
  `aggregation_artifact::` ref로 upload/materialize할 수 있게 닫았다.
  server direct accept는 server-owned ref와 inline debug payload만 수락한다.
- `10 clients`, Dirichlet `alpha=0.3`, split `seed=42`, `30 rounds` main
  baseline report는 round/split/method/delta 기준으로 검증했다. 이 report는
  runtime metadata 도입 전 산출물이라 `gpu_local + mxbai` 여부는 report 자체로
  재검증할 수 없고, 현재 코드 기준 runtime metadata는 같은 split의 1-round smoke와
  reduced runs로 확인했다.
- Dirichlet `alpha=0.1` final stress, full-budget
  FlexMatch/FreeMatch/PseudoLabel ablation, full-budget `client_count=1..10` sweep은
  후보와 비교 조건을 확정한 뒤 별도 실행한다. 현재는 `alpha=0.3` 기준
  FlexMatch/FreeMatch/PseudoLabel ablation을 5-round reduced run으로 확인했고,
  `client_count=1..10` sweep은 1-round summary로 확인했다. `alpha=0.1`은 마지막
  stress 확인으로 남겨 두며 current manifest 검증 대상이 아니다.
- FL SSL runner는 총 예정 communication round가 30을 넘으면 기본 차단한다.
  단일 run은 `rounds`, seed/client-count sweep은 `rounds * sweep 항목 수`로
  계산하며, 장시간 실행은 `run_safety.allow_long_run=true`와
  `run_safety.long_run_ack=ALLOW_FL_SSL_LONG_RUN`을 명시한 경우에만 시작된다.
- 현재 워크스페이스의 감사용 manifest
  `docs/operations/fl_ssl_artifact_verification_manifest.current.json`는
  2026-05-26 FedMatch reduced report를 round budget, client count, method-owned
  FedMatch protocol, partitioned delta artifact ref, sparse S2C posthoc estimate까지
  재검증한다. 2026-05-17 manual baseline 계열 historical artifact는 현재 로컬에
  없어 current manifest 대상에서 제외했다.
- FedMatch method-owned smoke는 `peer_context=fixed_probe_output_knn`,
  `server_update_policy=fedmatch_partitioned`,
  `method_descriptor=fedmatch`에서 파생되는 `fedmatch_agreement` 조합으로
  2026-05-22에 확인했다. 현재 target 구조에서는 `fedmatch_agreement`를 generic
  `local_ssl_policy` leaf가 아니라 `methods/federated_ssl/fedmatch/`가 소유하는
  method-local objective로 둔다.
  1-round smoke는 previous client snapshot이 없어 helper count가 0인 것이 정상이고,
  2-client 2-round smoke에서는 round 2에서 helper count/refreshed가 1.0으로 기록됐다.
  report 검증 CLI도 PASS했다. 다만 `2 clients x 2 rounds x max_steps=1`도 약 10분
  걸려 reduced run 전 runtime 병목 개선이 필요하다.

다음 우선순위:

1. FedMatch reduced run 전에 PEFT-classifier simulation 병목을 줄인다.
   확인된 병목은 client/round마다 `AutoModel.from_pretrained()`로 frozen backbone을
   재로딩하는 것, helper snapshot마다 helper model을 다시 materialize하는 것,
   매 round 전체 validation/probe를 반복 평가하는 것이다.
2. `fixed_probe_output_knn`의 fixed probe surface는 전체 validation rows가 아니라
   `peer_probe.selection_policy=label_balanced`, `max_rows=128` 기본값의 작은
   deterministic subset으로 계약화했다. Report protocol에는 probe source, row count,
   label distribution, query id hash가 남는다. 이후 reduced run에서는 이 probe surface가
   helper selection vector 계산 입력이다.
3. runtime resource cache seam은 `methods.common` protocol과 simulation run-scoped
   in-memory cache로 열었다. PEFT text-classifier model builder는 cache가 있으면
   tokenizer와 frozen backbone base를 재사용하고, client별 LoRA/head state를 별도 model
   instance에 로드한다. Helper provider도 같은 cache를 통해 backbone/tokenizer 재로딩
   비용을 줄인다.
4. helper snapshot별 materialized helper model cache를 추가했다. 같은 run에서 동일
   helper snapshot이 다시 선택되면 PEFT-classifier model 복원과 parameter load를
   재사용한다.
5. client-local pseudo-label quality 진단은 `diagnostic_view.max_rows=512` 기본값의
   deterministic subset으로 줄인다. 이는 학습 pool을 자르는 정책이 아니라 보고용
   diagnostics 입력만 줄이는 공통 runtime capability이며, manual Query SSL과 FedMatch
   method-owned PEFT-classifier 경로가 같이 사용한다.
6. client round별 `timing_breakdown`을 report에 남긴다. 다음 reduced run에서
   model prepare, training loop, pseudo-label diagnostics, helper/peer snapshot,
   update materialization 중 실제 병목을 숫자로 확인한다.
7. simulation에서는 agent-local update 사본 저장을 기본 끈다. server aggregation과
   verifier는 server-owned artifact를 기준으로 유지하고, report protocol에 저장 정책을
   남긴다.
8. FedMatch처럼 partitioned server update policy가 partition별 material만 소비하는
   경로는 `partitioned_deltas_artifact_ref`를 canonical payload에 남기고, 큰
   partitioned delta material은 server-owned aggregation artifact로 저장한다. 이 경로는
   primary LoRA/head delta artifact도 생략해 shared update payload와 중복 artifact 저장을
   줄인다. Inline `partitioned_deltas`는 smoke/debug compatibility로만 유지한다.
9. `training_view`는 현재 기본 계획에서 제외한다. 학습 pool을 제한하면 model update
   의미가 바뀌므로, runtime이 여전히 과한 경우에만 별도 debug/runtime ablation으로
   검토한다.
10. 최적화 후 FedMatch method-owned reduced run을 다시 닫았다. 확인 대상은 현행
   v1 field 기준
   `method_owned`, `method_descriptor=fedmatch`에서 파생되는 `fedmatch_agreement`,
   `peer_context=fixed_probe_output_knn`, `server_update_policy=fedmatch_partitioned`,
   helper injection, `partitioned_deltas_artifact_ref` 소비, final report metadata였다.
   2026-05-26 `10 clients`, `5 rounds`, `max_steps=20`,
   `local_budget_policy=iteration_capped` run은 posthoc communication backfill 뒤
   verifier PASS했다. 원본 labels-at-client budget은
   `ssl_method.local_budget_policy=original_method`를 명시한 별도 faithful run에서만
   사용한다.
11. 같은 split/seed/budget에서 현행
   `FedAvg + FixMatch + PEFT-classifier` manual baseline과 FedMatch method-owned slice를
   비교 가능한 reduced report로 맞춘다. target 구조에서는 이 manual baseline을
   `update_family=peft_text_classifier`로 표현한다.
12. FixMatch를 `fedmatch_partitioned`의 stateless `psi` objective로 주입하는 hybrid는
   validator와 smoke는 열려 있으므로, FedMatch 기본 slice가 안정된 뒤 ablation으로
   실행한다. FlexMatch/FreeMatch처럼 state surface가 필요한 hybrid는 계속 실행 전에
   막는다.
12. sparse S2C/C2S sync는 full FedMatch parity 후보로 남기되, 현재 다음 실행
   게이트는 아니다.
13. full ablation, full `client_count=1..10` sweep, full-budget main run은 후보와
   비교 조건을 먼저 확정한 뒤 실행한다. `alpha=0.1`은 기본 비교가 아니라 최후
   stress 확인으로 남긴다.
14. winner를 target 기준 `peft_text_classifier` update family 또는 현실적인 fallback
    update family로 translation 한다. v1 `lora_classifier` 이름은 compatibility
    표면에만 남긴다.

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
3. FL 범위를 `peft_text_classifier` update family에서 PEFT adapter, linear head,
   aggregation artifact까지 어디까지 열지.
4. private adapter/head 도입 시점.
5. secure aggregation과 DP 도입 시점.
6. multi-prototype runtime 확장 여부.
