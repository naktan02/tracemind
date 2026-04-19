# TraceMind Execution Plan

## 1. 현재 활성 목표

현재 TraceMind의 활성 목표는 아래 다섯 가지다.

1. 로컬에서 원문을 처리한다.
2. 공통 의미 표현 공간은 전역 모델과 shared artifact로 유지한다.
3. 해석과 최종 판단은 로컬 개인화 상태로 수행한다.
4. 초기 seed는 중앙집중형 `fixed embedding + classifier`로 안정적으로 만든다.
5. query가 충분히 쌓인 뒤 `LoRA + classifier` 적응을 열고, 그 winner를 FL/runtime 제약에 맞게 후행 translation 한다.

중요:

- `WindowSummary`, `NormPack`은 활성 경로가 아니다.
- `PrototypePack`은 유지하지만, 우선은 bootstrap/comparison artifact로 본다.
- 초기 seed baseline은 `central + fixed embedding + classifier`다.
- canonical seed artifact는 `clf_2026_04_11_143138`으로 고정한다.
  - model: `data/processed/classifier_heads/clf_2026_04_11_143138.pt`
  - manifest: `data/processed/classifier_heads/clf_2026_04_11_143138.manifest.json`
  - report: `runs/train_classifier/clf_2026_04_11_143138/reports/report.json`
- query-domain 적응 단계에서만 `LoRA + classifier`를 연다.
- 시스템 트랙의 v1 baseline은 `embedding -> global classifier -> local interpretation`이다.
- 라벨된 데이터셋은 prototype build 전용이 아니라 supervised classifier seed와
  validation/calibration split source로도 직접 사용한다.
- query 적응의 현재 권장 순서는 `query accumulation -> threshold/policy selection -> continual LoRA + classifier adaptation`이다.
- query-domain 적응의 canonical baseline은 `supervised LoRA + classifier`다.
- query-domain supervised baseline의 canonical entrypoint는 `scripts/experiments/train_lora_classifier.py`이고,
  scripts runner는 `scripts/experiments/lora_classifier/runner.py`, 학습 코어는 `agent/src/services/training/query_adaptation/`가 소유한다.
- 첫 SSL 실험선은 `pseudo-label self-training`이다.
- 첫 USB faithful consistency baseline entrypoint는 `scripts/experiments/train_lora_fixmatch.py`다.
- `FixMatch` core는 `agent/src/services/training/query_adaptation/algorithms/fixmatch.py`에 두고,
  trainer loop adapter는 `agent/src/services/training/query_adaptation/training.py`가 소유한다.
- scripts 실행 껍데기는 `scripts/experiments/lora_classifier/query_ssl/common.py`와
  `query_ssl/consistency_runner.py`로 나눠 consistency family 공통 scaffolding을 재사용한다.
- 첫 진입 bootstrap entrypoint는 `scripts/experiments/train_lora_bootstrap_classifier_teacher.py`이며,
  `fixed embedding + classifier` teacher가 unlabeled pool에 pseudo-label을 붙인 뒤
  `LoRA + classifier` student를 학습한다.
- 중앙 query-domain 비교의 canonical 규약은 `seed checkpoint 1회 생성 -> 이후 new accepted query-derived rows only continual adaptation`이다.
- 즉 초기 seed labeled split은 최초 checkpoint를 만드는 데만 쓰고, 이후 같은-family loop에서는 full seed replay를 canonical로 두지 않는다.
- seed full replay 또는 small anchor replay는 필요 시 별도 ablation로만 다룬다.
- `FixMatch`, `R-Drop`, `MixText`는 같은 scaffold 위의 후속 비교축이다.
- `TAPT`는 적응 앞단의 optional preadaptation phase다.
- LoRA는 query-domain 적응과 이후 FL LoRA family translation에 유리하다.
- `FedMatch`, `FedLGMatch`, `(FL)^2`는 central query-domain 비교선이 아니라, 이후 FL stage에서 비교할 논문군으로 둔다.
- 중앙 query-domain 비교의 목적은 FL stage로 가져갈 후보를 줄이는 것이다.
  즉 모든 중앙 알고리즘을 FL로 다시 옮기는 것이 아니라, 중앙에서 의미 있게 남는
  소수의 winner만 translation 대상으로 본다.
- `v1` 시스템은 여전히 `same global representation + different local interpretation`이다.
- `v2`에서만 private adapter/head 기반 표현 개인화를 연다.
- multi-prototype runtime은 v1 필수가 아니라 future option으로 둔다.

## 2. 활성 계약

현재 source of truth는 문서보다 코드 계약이 우선이다.

1. `shared/src/contracts/README.md`
2. `shared/src/contracts/adapter_contracts.py`
3. `shared/src/contracts/training_contracts.py`
4. `shared/src/domain/entities/training/`

현재 활성 객체는 아래다.

1. `PrototypePack`
2. `PrototypeBuildState`
3. `ModelManifest`
4. `SharedAdapterState / SharedAdapterUpdate`
5. `TrainingTask`
6. `TrainingUpdateEnvelope`
7. `DecisionFeedbackSignal`
8. `PersonalizationState`
9. `AssessmentResult`

주의:

- 위 계약은 현재 시스템/FL runtime의 source of truth다.
- query-domain 적응 단계의 `central LoRA classifier` trainer는 별도 실험 레일로 둔다.
- 즉 이 비교선이 곧바로 현재 shared adapter contract나 update envelope을 의미하지는 않는다.
- LoRA 적응 scaffold와 산출물 규칙은 `docs/contracts/central_lora_classifier_trainer_contract.md`를 기준으로 본다.
- query buffer와 selection 입력 경계는 `docs/contracts/query_buffer_v1.md`를 기준으로 본다.

## 3. 활성 아키텍처

### 3-1. 로컬 추론 레일

```text
Raw Event
-> Preprocess / Translation
-> Embedding
-> Global Classifier
-> PersonalizationState
-> Time-Series Accumulator / Persistence
-> DecisionPolicy
-> AssessmentResult
```

비교/확장 레일:

```text
Raw Event
-> Embedding
-> Prototype Scoring or Shared Adapter
-> Local Interpretation
-> AssessmentResult
```

### 3-2. staged seed / adaptation 레일

```text
Reddit Labeled Data
-> Fixed Embedding
-> Classifier Seed
-> Local Deployment
-> Query Buffer (raw text + confidence + predicted label)
-> Threshold / Policy Selection
-> Accepted Query-derived Rows
-> Continue LoRA + Classifier Adaptation
-> Central or Federated Evaluation
```

설명:

1. 초기 seed의 목적은 표현을 함부로 움직이지 않고 안정적인 classifier 기준선을 만드는 것이다.
2. 메인 라벨링은 classifier posterior가 담당하고, prototype은 메인 판정기가 아니다.
3. query-domain 적응은 query가 충분히 쌓인 뒤에만 연다.
4. 첫 pseudo-label bootstrap은 `fixed classifier teacher -> LoRA student`로 두고,
   그 이후 loop에서는 같은 초기 checkpoint에서 출발해 newly accepted query-derived rows만으로 same-family continual adaptation을 연다.
5. baseline 이후에는 `pseudo-label self-training`, `FixMatch`, `R-Drop`, `MixText`를 가능한 한 같은 backbone, 같은 LoRA spec, 같은 query selection 규칙, 같은 초기 checkpoint 위에서 비교한다.
6. `TAPT`는 backbone을 target query 분포로 먼저 적응시키는 별도 preadaptation 단계다.
7. raw query text는 LoRA 재학습과 future query adaptation에 필요하므로 로컬에 남겨야 한다.
8. `FedMatch`, `FedLGMatch`, `(FL)^2` 같은 FL-specific 방법론은 중앙 continual protocol이 아니라 FL translation stage에서만 비교한다.

### 3-3. 시스템 FL 레일

```text
Raw Event / Local Signal
-> Query Buffer / Pseudo-label / Feedback Signal
-> Local Training
-> SharedClassifierUpdate or SharedAdapterUpdate
-> Central Aggregation
-> New ModelManifest / PrototypePack pair
```

설명:

1. 시스템 v1에서는 backbone은 고정하고 shared classifier/head family를 우선 baseline으로 다룬다.
2. 현재 구현된 family는 `diagonal_scale`, `classifier_head` 두 개다.
3. query-domain 적응 winner를 FL로 옮길 때는 `lora` family를 새로 추가하는 것이 1순위 translation 후보다.
4. 시스템 기본 비교축은 `global classifier + local interpretation`이고,
   `diagonal_scale`과 prototype scoring은 비교 실험/확장 레일로 유지한다.
5. `diagonal_scale`은 lightweight baseline으로 계속 유지하지만, 메인 query adaptation scaffold는 아니다.
6. `classifier_head` family는 simulation과 row-source multiview 경로까지 닫혔고,
   stored scored event를 쓰는 real agent 경로는 아직 제한이 있다.
7. prototype은 직접 FL 파라미터라기보다 bootstrap/비교용 semantic artifact에 가깝다.

## 4. 구현 배치 규칙

1. 공용 계산 규칙과 canonical 알고리즘은 `shared`에 둔다.
2. agent-owned local training/inference runtime은 `agent`에 둔다.
3. server-owned round/rebuild/publication orchestration은 `main_server`에 둔다.
4. `scripts`는 위 코어를 조합하는 실험층으로만 유지한다.
5. 운영 후보 로직을 `scripts`에 먼저 만들고 나중에 복사하는 흐름은 허용하지 않는다.
6. query-domain 적응용 중앙 trainer는 시스템 runtime contract를 오염시키지 않게 별도 레일로 둔다.

## 5. 현재 구현 상태

완료 또는 정리된 방향:

1. shared adapter 계약을 일반화했다.
2. training backend와 privacy guard를 분리했다.
3. 서버 aggregation을 family object 기준으로 조합하게 정리했다.
4. 계약 필드 의미를 `shared/src/contracts` 가까이에 두기 시작했다.
5. `main_server`는 round lifecycle, update acceptance policy, prototype rebuild runtime을 소유한다.
6. `agent`는 runtime training example builder를 소유한다.
7. `agent`는 `RoundClient`, `FederationRuntimeService`, training API(`run-current-task`, `status`)를 가진다.
8. `scripts/experiments/federated_simulation`은 `RoundLifecycleService`와 prototype rebuild runtime을 직접 조합한다.
9. `scripts/experiments/prototype_strategy`의 single/kmeans는 shared canonical builder를 재사용한다.
10. 기본 local training objective/selection fallback은 `shared/src/contracts/training_contracts.py`에 canonical builder로 모였다.
11. scorer backend와 example-generation backend를 독립 축으로 분리했다.
12. 서버는 `adapter_family`, `aggregation_backend`를 server-owned config axis로 고른다.
13. secure aggregation 메타데이터는 typed contract로 승격했다.
14. fixed-embedding + linear classifier supervised seed baseline은 이미 실행 가능하다.
15. canonical seed artifact는 `clf_2026_04_11_143138`으로 확정했다.

아직 남은 핵심:

1. query 버퍼, threshold/policy selection, 소량 수동 라벨 개입 지점을 설계한다.
   - local source of truth: `docs/contracts/query_buffer_v1.md`
2. 그 뒤에만 `LoRA + classifier` 적응 레일을 열고 `supervised -> pseudo-label -> FixMatch -> R-Drop -> MixText`를 같은 continual adaptation 규약 아래에서 닫는다.
3. 이후에야 시스템 FL translation 기준선을 선택한다.
4. classifier-first 시스템 baseline을 live agent/runtime 레일까지 안정적으로 확장한다.
5. 아직 두 번째 real aggregation backend는 없다.
6. integration test infra를 안정화하고 multi-agent HTTP 시나리오를 확대한다.
7. secure aggregation / DP / robust aggregation의 실제 runtime 구현을 붙인다.

## 6. Phase 요약

### Phase 0. 계약 정리

- 활성 contract와 보관 contract를 분리한다.

### Phase 1. 중앙집중형 seed baseline

- `central fixed embedding + classifier`를 canonical seed로 닫는다.
- canonical seed artifact: `clf_2026_04_11_143138`

### Phase 2. query 적응 준비

- query 버퍼, threshold/policy selection, 소량 수동 라벨 개입 지점을 정한다.
- query buffer local contract는 `docs/contracts/query_buffer_v1.md`로 고정한다.

### Phase 3. 중앙집중형 적응 비교

- 같은 초기 seed checkpoint와 같은 accepted query-derived rows를 기준으로
  `LoRA + classifier` continual adaptation 위에서
  `supervised -> pseudo-label self-training -> FixMatch -> R-Drop -> MixText`를 비교한다.

### Phase 4. 시스템 FL baseline

- `fixed embedding + classifier_head` 기준의 FL baseline을 닫는다.

### Phase 5. 시스템 FL translation

- 적응 winner를 우선 `LoRA family + classifier` 후보로 FL/runtime 제약에 맞게 옮긴다.

### Phase 6. richer shared adapter

- classifier-first 시스템 baseline이 충분하지 않을 때만 diagonal scale보다 표현력 있는 shared adapter로 확장한다.

### Phase 7. privacy hardening

- clipping, secure aggregation, DP, 필요 시 HE를 붙인다.

## 7. 다음 우선순위

가장 자연스러운 다음 작업은 아래 순서다.

1. query 버퍼에 어떤 필드(raw text, confidence, predicted label, timestamp, model_revision)를 남길지 정한다.
2. threshold/policy selection과 소량 수동 라벨 개입 지점을 설계한다.
3. 그다음 `LoRA + classifier` 적응용 supervised baseline을 연다.
4. 같은 query-domain 적응 scaffold에서 `pseudo-label self-training`, `FixMatch`, `R-Drop`, `MixText`를 같은 continual adaptation 규약으로 비교한다.
5. 필요하면 `TAPT`를 preadaptation 축으로 추가한다.
6. `FedMatch`, `FedLGMatch`, `(FL)^2`는 중앙 비교가 아니라 FL stage 논문군으로 남긴다.
7. 적응 winner를 고른 뒤에 시스템용 `FL supervised classifier` baseline으로 넘어간다.
8. 그 winner를 우선 `lora` family, 필요 시 `classifier_head` 같은 현실적인 시스템 scope로 translation 한다.
9. 그 다음에야 live agent path와 richer FL family를 닫는다.

## 8. 검증 기준

### seed baseline

1. `fixed embedding + classifier` seed baseline이 재현 가능하게 닫혀 있다.
2. 클래스별 confusion과 confidence 분포가 남아 있다.
3. prototype은 메인 라벨러가 아니라 comparison/reference artifact로만 쓰인다.
4. canonical seed artifact는 `clf_2026_04_11_143138`으로 고정한다.

### query-domain 적응

1. query 버퍼와 threshold/policy selection 규칙이 문서로 명확하다.
2. 적응 단계에서 supervised / pseudo-label self-training / FixMatch / R-Drop / MixText 비교가 같은 continual adaptation 규약 아래 가능하다.
3. raw query text retention과 pseudo-label 사용 범위가 명확하다.

### 시스템 FL

1. update가 base revision과 호환된다.
2. aggregation 후 새 revision이 일관되게 발행된다.
3. shared adapter drift가 과도하지 않다.
4. prototype 재생성이 새 revision과 일치한다.

### privacy

1. 원문이 서버로 올라가지 않는다.
2. clipping/secure aggregation/DP는 training logic과 분리된 계층으로 붙는다.

## 9. 사용자 확인이 필요한 결정

1. query 버퍼 raw text retention 기본값은 현재 `30일 + 최신 5000건 유지`로 둔다. 필요 시 로컬 config로 조정한다.
2. threshold/policy로 채택한 query는 현재 `pseudo-label only`로 학습하고, manual label은 이후 같은 dataset shape 위에 override로 얹는다.
3. query-domain 적응에서 LoRA target module/rank를 무엇으로 고정할지
4. FL 범위를 `lora` family/head에서 어디까지 열지
5. private adapter/head를 언제 도입할지
6. secure aggregation과 DP 도입 시점
7. multi-prototype를 runtime까지 확장할지
8. shared adapter를 기본선으로 복귀시킬지, 비교축으로 유지할지
