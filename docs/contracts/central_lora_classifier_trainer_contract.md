# Central LoRA Classifier Trainer Contract

## 목적

이 문서는 TraceMind의 query-domain 적응 단계에서 쓰는 중앙집중형 `LoRA + classifier`
pooled/offline control scaffold를 정의한다.

이 문서가 다루는 것은 현재 시스템/FL runtime contract가 아니라,
`adaptation-track central trainer`가 어떤 고정 조건과 산출물을 가져야 하는지다.
중앙 SSL 결과는 FL SSL 논문 비교의 대조군이며, 최종 메인 랭킹은 non-IID client
split 위의 FL SSL 비교에서 정한다.
여기서 `pooled/offline`은 FL client partition 없이 중앙 control을 만든다는 뜻이며,
seed full replay를 매번 수행하는 offline union retraining을 뜻하지 않는다.

## 범위

이 contract는 아래를 명확히 한다.

1. 초기 seed baseline과 이 적응 scaffold를 어떻게 분리하는지
2. backbone을 어떻게 고정하는지
3. LoRA를 어디에 어떤 설정으로 삽입하는지
4. classifier head를 어떻게 결합하는지
5. 어떤 방법론을 같은 scaffold 위에서 비교하는지
6. 결과 산출물을 어떻게 future FL translation에 넘길지

이 문서는 아래는 직접 다루지 않는다.

1. `TrainingTask`, `TrainingUpdateEnvelope`, `ModelManifest` 같은 현재 시스템/FL runtime 계약
2. live agent stored-event 경로
3. 현재 `classifier_head`, `diagonal_scale` family의 배포/update payload shape

## staged boundary

적응 단계는 아래 seed 단계와 구분한다.

```text
Reddit Labeled Data
-> Fixed Embedding
-> Classifier Seed
```

즉 이 문서는 첫 supervised seed baseline이 아니라,
query가 충분히 쌓인 뒤 그 query 분포에 맞춰 표현과 경계를 함께 조정하는
두 번째 단계 contract다.

## canonical scaffold

query-domain 적응 단계의 기본 scaffold는 아래로 고정한다.

```text
Query Buffer (raw text)
-> Accepted Query-derived Rows
-> Seed or Previous Adaptation Checkpoint
-> Frozen Backbone
-> LoRA Modules
-> Classifier Head
-> Adaptation Objective
-> Central Evaluation
```

핵심 원칙:

1. backbone 본체 가중치는 고정한다.
2. trainable parameter는 LoRA module과 classifier head로 제한한다.
3. 중앙 canonical protocol에서는 initial seed labeled split으로 checkpoint를 한 번 만든 뒤,
   이후 비교는 same checkpoint에서 출발해 new accepted query-derived rows only로 계속 적응한다.
4. 같은 비교표 안에서는 backbone, tokenizer, label schema, LoRA spec, initial checkpoint를 고정한다.
5. 방법론 비교에서는 adaptation objective만 바꾼다.
6. full seed replay는 canonical 경로가 아니라 optional ablation로만 다룬다.
7. raw query text는 로컬에 남겨야 한다. embedding snapshot만으로는 LoRA 재학습과 future query adaptation을 닫을 수 없다.
8. scripts 실험 표면에서는 `strategy_axes/adaptation/initial_checkpoint` selector를 source of truth로 두고, 같은 비교표 안의 run은 같은 초기 checkpoint provenance를 공유해야 한다.

## 중앙 SSL control 비교축

### control family

같은 LoRA adaptation scaffold 위에서 아래를 우선 비교한다.

1. `supervised adaptation`
2. `pseudo-label self-training`
3. `FixMatch`
4. `R-Drop`
5. `MixText`

옵션:

- `TAPT`는 classifier objective 앞단의 optional preadaptation phase로 추가 가능하다.

주의:

- USB `PseudoLabel` baseline은 weak/original view `text`만 사용해 online
  pseudo-label target과 confidence mask를 만들고, strong backtranslation view를
  요구하지 않는다.
- 첫 teacher-bootstrap pseudo-label 진입은 `fixed embedding + classifier` teacher가 unlabeled pool에
  pseudo-label을 붙이고, `LoRA + classifier` student가 이를 학습하는 bootstrap으로
  시작할 수 있다.
- first-stage bootstrap/adaptation에서 LoRA seed artifact가 아직 없으면 canonical fixed classifier seed manifest로 classifier head를 warm-start할 수 있다.
- `FixMatch`는 USB core를 기준으로 weak view에서 pseudo-label/mask를 만들고,
  strong view에 consistency CE를 적용하는 별도 비교축으로 둔다.
- NLP strict USB baseline에서는 `FixMatch` unlabeled input을
  `text + aug_0 + aug_1` canonical shape로 둔다.
  즉 weak는 `text`, strong은 `query_ssl_strong_view_policy`가 선택한
  `aug_0` 또는 `aug_1` 후보를 사용한다. 기본값은 기존 동작 보존을 위해
  `first_aug`다.
  `aug_0`, `aug_1` 생성/caching은 `execution_context/query_view`의 view
  materialization config가 담당한다. 학습의
  `strategy_axes/ssl/augmentation_source`는 이미 저장된
  `text + aug_0 + aug_1` row를 읽을지, 실행 중 생성할지를 고른다.
- 중앙 SSL 알고리즘 비교용 canonical query source는
  `ourafla_ssl_labeled1024_per_class_seed42_v1`이다.
  `data/processed/splits/ourafla_train_split.v1.train.jsonl`에서 각 class별
  1024개를 labeled train으로 뽑고, 나머지 train row 전체를 unlabeled pool로
  둔다. validation/test는 합치지 않고 기존 validation/test split을 유지한다.
  unlabeled artifact에는 audit과 stratified metric을 위해 원 라벨 필드를 보존하지만,
  중앙 SSL unlabeled loader와 algorithm core는 이 라벨을 소비하지 않는다.
- backtranslation strong view는 split과 분리된 artifact로 materialize한다.
  새 dataset은 `data/datasets/<dataset_id>/views/<split>/<augmenter>/`를 기본으로
  쓰고, 기존 ourafla 산출물은 legacy `data/processed/query_ssl_views/*` 경로를
  유지한다.
  SemiLearn/USB 계열 알고리즘 비교에서는 `text`를 weak view로, `aug_0`/`aug_1`을
  strong candidate로 읽고, 알고리즘별 실행 중 backtranslation을 다시 수행하지 않는다.
- 그 이후 반복 loop에서는 같은 initial checkpoint에서 출발해
  newly accepted query-derived rows only로 `LoRA + classifier` same-family continual adaptation을 연다.
- `FedMatch`, `FedLGMatch`, `(FL)^2`는 FL-specific 제약이 핵심이므로
  central query-domain control family에는 넣지 않고, FL SSL non-IID 메인 비교선으로 둔다.

## 고정해야 할 입력 조건

한 실험군 안에서는 아래를 바꾸지 않는다.

1. backbone model id / revision
2. tokenizer
3. LoRA target modules
4. LoRA rank / alpha / dropout
5. classifier head 형태
6. initial seed checkpoint
7. query selection rule
8. adaptation cadence / update budget
9. adaptation objective 세부 하이퍼파라미터

이 중 하나라도 바뀌면 방법론 비교가 아니라 scaffold 비교가 된다.

## 산출물

각 run은 최소 아래 산출물을 남긴다.

1. seed baseline reference
2. initial seed checkpoint reference
3. backbone reference
4. LoRA config snapshot
5. LoRA adapter checkpoint
6. classifier head checkpoint
7. label schema snapshot
8. accepted query-derived row manifest
9. threshold / objective config snapshot
10. validation / test / query-eval report
11. acceptance ratio, confidence, calibration 관련 요약

중요:

- LoRA adapter와 classifier head는 분리된 artifact로 남기는 편이 좋다.
- 그래야 나중에 시스템 FL translation에서 `lora_classifier` family의 LoRA
  adapter와 classifier head 결합 방식을 다시 선택하기 쉽다.

## future FL SSL translation 경계

중앙 control 결과와 FL SSL winner를 시스템/FL로 옮길 때는 아래 원칙을 지킨다.

1. paper/adaptation checkpoint를 현재 `ModelManifest`나 `TrainingUpdateEnvelope`에 바로 우겨넣지 않는다.
2. 먼저 `lora_classifier` family용 state/update payload를 별도 정의한다.
3. `lora_classifier`는 classifier head에 LoRA 옵션을 붙인 것이 아니라, LoRA
   adapter state와 classifier head state를 함께 배포/집계하는 별도 family로 본다.
4. LoRA target module 이름과 rank 같은 핵심 메타데이터는 적응 트랙 산출물에서 보존한다.
5. `diagonal_scale`은 제거하지 않고 lightweight baseline으로 유지한다.
6. `classifier_head` family는 translation fallback으로 남길 수 있다.

기본 scaffold 고정값:

1. backbone/tokenizer: `strategy_axes/adaptation/transformer_backbone=mxbai_encoder`
2. LoRA config: `rank=8`, `alpha=16`, `dropout=0.1`, `target_modules=all-linear`
3. initial checkpoint: 중앙 Query SSL method 비교 기본값은 `none`이다. 기존
   `canonical_fixed_classifier_seed` warm-start는 continual adaptation 또는
   ablation run에서 명시적으로 선택한다.
4. train budget: USB처럼 `max_train_steps` 총 optimizer update 수를 고정한다.
   `epochs`는 selection 평가/history cadence이며 전체 unlabeled pool replay 횟수가 아니다.
5. label schema, split, seed, metric은 FL SSL main comparison 규약을 따른다.

위 값을 바꾸는 run은 method 비교가 아니라 scaffold 비교 또는 ablation로 기록한다.

즉, 이 문서는 중앙 pooled/offline 조건에서 `LoRA + classifier` SSL objective가
기본적으로 성립하는지 검증하고, 메인 논문 비교는 FL SSL non-IID 조건에서 별도로
닫는다. 시스템 트랙은 FL SSL winner를 `어떤 shared family로 배포/집계할지`를 다시 설계한다.

## central continual protocol note

이 문서의 canonical 비교는 `offline union retraining`이 아니라
`seed checkpoint 1회 생성 -> 이후 newly accepted query-derived rows only continual adaptation`
으로 본다.

즉 accepted query가 들어올 때마다 아래 질문을 같은 규약 아래 비교한다.

1. supervised-style update가 가장 낫나
2. pseudo-label self-training이 가장 낫나
3. USB FixMatch consistency objective가 가장 낫나
4. R-Drop 같은 regularized objective가 가장 낫나
5. 필요 시 MixText/TAPT가 incremental adaptation에서도 이점을 주나

seed full replay 또는 small anchor replay는 drift/forgetting 분석을 위한 ablation로는 가능하지만,
canonical central table의 기본 규약으로 두지 않는다.

## phase 분리

`TAPT`는 분류 objective와 다른 phase로 다룬다.

예:

- `TAPT -> supervised LoRA`
- `TAPT -> pseudo-label self-training`
- `TAPT -> FixMatch`
- `TAPT -> MixText`

즉 `TAPT`는 메인 adaptation objective 자체라기보다,
target query 분포에 backbone을 먼저 맞추는 앞단 preadaptation 단계다.
