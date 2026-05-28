# Query Buffer v1

## 1. 목적

`QueryBuffer`는 agent 로컬에만 남는 query-domain 적응 준비 계층이다.

이 계층의 목적은 아래 세 가지다.

1. 초기 `fixed embedding + classifier` seed 이후 실제 query를 로컬에 축적한다.
2. threshold/policy selection 전에 필요한 원문과 예측 스냅샷을 보존한다.
3. 이후 `PEFT encoder classifier` 적응 또는 FL translation으로 넘길 후보를 만든다.

중요:

- `QueryBuffer`는 서버로 직접 올라가지 않는다.
- raw text는 로컬에만 남는다.
- selection 결과는 기존 `PseudoLabelEvidence`, `PseudoLabelCandidate`,
  `DecisionFeedbackSignal`로 연결하고, query buffer 자체가 그 역할을 겸하지 않는다.

## 2. 소유 경계

`QueryBuffer`는 `agent` 소유 경계다.

- `shared`
  - 공통 training contract, pseudo-label evidence/candidate 해석만 소유한다.
- `agent`
  - raw query text retention, local buffer lifecycle, threshold/policy selection 입력을 소유한다.
- `main_server`
  - query buffer 원문이나 개별 레코드를 읽지 않는다.

즉 `QueryBuffer`는 shared payload가 아니라 local persistence/runtime state다.

## 3. 왜 별도 계층이 필요한가

현재 저장소에는 이미 `ScoredEventRepository`가 있다.
하지만 이 저장소는 주로 아래를 위한 것이다.

1. scored event 보관
2. base embedding 재사용
3. row-source/stored-event training example 재구성

query-domain 적응 단계에서는 이것만으로 부족하다.
이유는 다음과 같다.

1. PEFT text encoder 적응과 weak/strong augmentation에는 raw text가 필요하다.
2. 나중에 backbone/PEFT adapter가 바뀌면 과거 embedding은 stale해질 수 있다.
3. selection policy는 prediction snapshot과 retention lifecycle을 함께 관리해야 한다.
4. 같은 query라도 policy나 revision이 바뀌면 다시 selection할 수 있어야 한다.

따라서 v1에서는 `ScoredEventRepository`와 `QueryBuffer`를 구분한다.

- `ScoredEventRepository`
  - translated text, category scores, base embedding 저장
- `QueryBuffer`
  - raw text, locale, source_type, model revision, top-1/top-2 snapshot 저장

## 4. v1 설계 원칙

1. raw query text는 로컬에만 저장한다.
2. query buffer record는 event snapshot이며, selection 결과를 직접 내장하지 않는다.
3. selection은 재실행 가능해야 한다.
4. selection output은 새 전용 shape를 만들기보다 기존
   `PseudoLabelEvidence` / `PseudoLabelCandidate`를 재사용한다.
5. retention policy는 config-driven으로 열어 두고, code path에 하드코딩하지 않는다.

## 5. QueryBufferRecord canonical field

v1에서 권장하는 최소 필드는 아래다.

1. `schema_version`
   - 예: `query_buffer_record.v1`
2. `query_id`
   - `QueryEvent.query_id`와 동일한 참조 키
3. `occurred_at`
4. `raw_text`
5. `locale`
6. `source_type`
7. `model_revision`
   - 해당 예측이 어떤 seed/adaptation revision에서 나왔는지
8. `predicted_label`
9. `confidence`
10. `margin`
11. `runner_up_label`
12. `runner_up_score`
13. `confidence_kind`
   - 예: `softmax_top1`, `cosine_top1`
14. `metadata`
   - 실험용 보조 정보

포함하지 않을 것:

1. full category score vector
   - `ScoredEventRepository`에서 읽는다.
2. base embedding
   - `ScoredEventRepository`에서 읽는다.
3. server aggregation metadata
4. 개인 식별자

## 6. selection과의 연결 방식

query-domain 적응 준비 흐름은 아래처럼 해석한다.

```text
QueryEvent
-> InferencePipeline
-> ScoredEventRepository 저장
-> QueryBufferRecord 저장
-> selection run
-> PseudoLabelEvidence
-> AcceptancePolicy
-> PseudoLabelCandidate / DecisionFeedbackSignal
-> adaptation dataset assembly
```

즉 `QueryBufferRecord`는 selection 입력의 local source이고,
selection 결과는 existing training domain object로 연결한다.

## 7. retention 원칙

v1 기본 방향:

1. raw text retention은 로컬 config로 제어한다.
2. retention 만료 전까지는 selection 재실행이 가능해야 한다.
3. raw text가 만료돼도 selection summary와 update metric은 남길 수 있다.
4. raw text를 서버나 shared artifact로 승격하지 않는다.

현재 local 기본 구현값:

1. raw text retention: 30일
2. capacity purge: 최신 5000건 유지
3. adaptation dataset label 정책: `pseudo_label_only`
4. future manual label override는 열어 두되, 기본 경로에서는 사용하지 않는다.

즉 v1은 pseudo-label only를 기본으로 두고,
manual label은 이후 같은 dataset shape 위에 override로 얹는 방향이다.

주의:

- 여기서 `pseudo_label_only`는 accepted query-derived adaptation dataset의
  label source를 뜻한다.
- 이 값은 중앙 학습에서 initial seed labeled rows를 full replay할지 여부를
  직접 의미하지 않는다.
- 현재 central canonical protocol은 `seed checkpoint 1회 생성 -> 이후 newly accepted query-derived rows only continual adaptation`으로 본다.

## 8. v1 구현 순서

1. `agent` 로컬 저장소에 query buffer record 저장 경로 추가
2. ingest/inference 후 scored event와 같은 `query_id`로 query buffer append
3. query buffer -> `PseudoLabelEvidence` projection helper 추가
4. threshold/policy selection runner 추가
5. accepted candidate를 adaptation dataset builder로 연결

## 9. 검증 기준

1. raw text가 로컬에만 남는다.
2. 같은 query를 다른 policy로 재평가할 수 있다.
3. selection 결과가 `PseudoLabelEvidence` / `PseudoLabelCandidate`로 연결된다.
4. PEFT text encoder 적응에 필요한 원문이 retention window 안에서 보존된다.
5. FL stage에서도 서버는 query buffer 원문을 읽지 않는다.
