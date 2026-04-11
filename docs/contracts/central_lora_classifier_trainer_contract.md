# Central LoRA Classifier Trainer Contract

## 목적

이 문서는 TraceMind의 `논문 트랙`에서 쓰는 중앙집중형 `LoRA + classifier`
비교 scaffold를 정의한다.

이 문서가 다루는 것은 현재 시스템/FL runtime contract가 아니라,
`paper-track central trainer`가 어떤 고정 조건과 산출물을 가져야 하는지다.

## 범위

이 contract는 아래를 명확히 한다.

1. backbone을 어떻게 고정하는지
2. LoRA를 어디에 어떤 설정으로 삽입하는지
3. classifier head를 어떻게 결합하는지
4. 어떤 방법론을 같은 scaffold 위에서 비교하는지
5. 결과 산출물을 어떻게 future FL translation에 넘길지

이 문서는 아래는 직접 다루지 않는다.

1. `TrainingTask`, `TrainingUpdateEnvelope`, `ModelManifest` 같은 현재 시스템/FL runtime 계약
2. live agent stored-event 경로
3. 현재 `classifier_head`, `diagonal_scale` family의 배포/update payload shape

## canonical scaffold

논문 트랙의 기본 scaffold는 아래로 고정한다.

```text
Text
-> Frozen Backbone
-> LoRA Modules
-> Classifier Head
-> Pseudo-label / SSL Rule
-> Central Evaluation
```

핵심 원칙:

1. backbone 본체 가중치는 고정한다.
2. trainable parameter는 LoRA module과 classifier head로 제한한다.
3. 같은 비교표 안에서는 backbone, tokenizer, label schema, LoRA spec을 고정한다.
4. 방법론 비교에서는 pseudo-label/consistency rule만 바꾼다.

## 논문 비교축

### 메인 비교 family

같은 LoRA scaffold 위에서 아래를 우선 비교한다.

1. `supervised`
2. `FixMatch`
3. `FreeMatch`
4. `PabLO`

옵션:

- `SAT`는 weak/strong self-training 보조 비교축으로 추가 가능하다.

### 보조 비교 family

아래는 메인 FixMatch family와 분리된 PEFT/self-training 계열로 다룬다.

1. `UPET`
2. `LiST`

이 둘은 같은 표에 기계적으로 섞기보다,
`PEFT self-training family`로 따로 읽는 것이 더 정직하다.

## 고정해야 할 입력 조건

한 실험군 안에서는 아래를 바꾸지 않는다.

1. backbone model id / revision
2. tokenizer
3. LoRA target modules
4. LoRA rank / alpha / dropout
5. classifier head 형태
6. labeled / unlabeled split
7. weak/strong augmentation recipe

이 중 하나라도 바뀌면 방법론 비교가 아니라 scaffold 비교가 된다.

## 산출물

각 run은 최소 아래 산출물을 남긴다.

1. backbone reference
2. LoRA config snapshot
3. LoRA adapter checkpoint
4. classifier head checkpoint
5. label schema snapshot
6. augmentation / threshold config snapshot
7. validation / test report
8. acceptance ratio, confidence, calibration 관련 요약

중요:

- LoRA adapter와 classifier head는 분리된 artifact로 남기는 편이 좋다.
- 그래야 나중에 시스템 FL translation에서 `lora family`와 `classifier` 결합 방식을 다시 선택하기 쉽다.

## future FL translation 경계

논문 트랙 결과를 시스템/FL로 옮길 때는 아래 원칙을 지킨다.

1. paper checkpoint를 현재 `ModelManifest`나 `TrainingUpdateEnvelope`에 바로 우겨넣지 않는다.
2. 먼저 `lora` family용 state/update payload를 별도 정의한다.
3. LoRA target module 이름과 rank 같은 핵심 메타데이터는 논문 트랙 산출물에서 보존한다.
4. `diagonal_scale`은 제거하지 않고 lightweight baseline으로 유지한다.
5. `classifier_head` family는 translation fallback으로 남길 수 있다.

즉, 논문 트랙은 `LoRA가 잘 되는지`를 검증하고,
시스템 트랙은 그 winner를 `어떤 shared family로 배포/집계할지`를 다시 설계한다.

## full 비교 위치

`full`은 기본 scaffold가 아니라 upper-bound 비교축으로 둔다.

예:

- `FixMatch-LoRA` vs `FixMatch-full`
- `FreeMatch-LoRA` vs `FreeMatch-full`

이 비교는 가능하지만, 메인 비교표보다 뒤에 두는 편이 좋다.
