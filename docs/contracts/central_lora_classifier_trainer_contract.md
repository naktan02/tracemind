# Central LoRA Classifier Trainer Contract

## 목적

이 문서는 TraceMind의 query-domain 적응 단계에서 쓰는 중앙집중형 `LoRA + classifier`
비교 scaffold를 정의한다.

이 문서가 다루는 것은 현재 시스템/FL runtime contract가 아니라,
`adaptation-track central trainer`가 어떤 고정 조건과 산출물을 가져야 하는지다.

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
-> Frozen Backbone
-> LoRA Modules
-> Classifier Head
-> Adaptation Objective
-> Central Evaluation
```

핵심 원칙:

1. backbone 본체 가중치는 고정한다.
2. trainable parameter는 LoRA module과 classifier head로 제한한다.
3. 같은 비교표 안에서는 backbone, tokenizer, label schema, LoRA spec을 고정한다.
4. 방법론 비교에서는 adaptation objective만 바꾼다.
5. raw query text는 로컬에 남겨야 한다. embedding snapshot만으로는 LoRA 재학습과 future query adaptation을 닫을 수 없다.

## 적응 비교축

### 메인 비교 family

같은 LoRA adaptation scaffold 위에서 아래를 우선 비교한다.

1. `supervised adaptation`
2. `pseudo-label self-training`
3. `R-Drop`
4. `MixText`

옵션:

- `TAPT`는 classifier objective 앞단의 optional preadaptation phase로 추가 가능하다.

## 고정해야 할 입력 조건

한 실험군 안에서는 아래를 바꾸지 않는다.

1. backbone model id / revision
2. tokenizer
3. LoRA target modules
4. LoRA rank / alpha / dropout
5. classifier head 형태
6. query selection rule
7. adaptation objective 세부 하이퍼파라미터

이 중 하나라도 바뀌면 방법론 비교가 아니라 scaffold 비교가 된다.

## 산출물

각 run은 최소 아래 산출물을 남긴다.

1. seed baseline reference
2. backbone reference
3. LoRA config snapshot
4. LoRA adapter checkpoint
5. classifier head checkpoint
6. label schema snapshot
7. threshold / objective config snapshot
8. validation / test / query-eval report
9. acceptance ratio, confidence, calibration 관련 요약

중요:

- LoRA adapter와 classifier head는 분리된 artifact로 남기는 편이 좋다.
- 그래야 나중에 시스템 FL translation에서 `lora family`와 `classifier` 결합 방식을 다시 선택하기 쉽다.

## future FL translation 경계

적응 트랙 결과를 시스템/FL로 옮길 때는 아래 원칙을 지킨다.

1. paper/adaptation checkpoint를 현재 `ModelManifest`나 `TrainingUpdateEnvelope`에 바로 우겨넣지 않는다.
2. 먼저 `lora` family용 state/update payload를 별도 정의한다.
3. LoRA target module 이름과 rank 같은 핵심 메타데이터는 적응 트랙 산출물에서 보존한다.
4. `diagonal_scale`은 제거하지 않고 lightweight baseline으로 유지한다.
5. `classifier_head` family는 translation fallback으로 남길 수 있다.

즉, 이 문서는 query-domain 적응에서 `LoRA + classifier`가 실제로 도움이 되는지 검증하고,
시스템 트랙은 그 winner를 `어떤 shared family로 배포/집계할지`를 다시 설계한다.

## phase 분리

`TAPT`는 분류 objective와 다른 phase로 다룬다.

예:

- `TAPT -> supervised LoRA`
- `TAPT -> pseudo-label self-training`
- `TAPT -> MixText`

즉 `TAPT`는 메인 adaptation objective 자체라기보다,
target query 분포에 backbone을 먼저 맞추는 앞단 preadaptation 단계다.
