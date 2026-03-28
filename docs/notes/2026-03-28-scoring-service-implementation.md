# 2026-03-28 Scoring Service Implementation

## User

그럼 공부보다는 일단 tracemind 구현 다음단계로 넘어가면 되나? 지금 어디 해야하지? 윈도우 서머리만 한건가?

## Assistant

현재 상태는 거의 잡혔습니다. 결론부터 말하면 `WindowSummary`만 한 건 아니고, "도메인 계약 + 로컬 파이프라인 골격"까지는 올라와 있습니다. 다만 실제로 값을 계산해서 중앙까지 흐르게 만드는 핵심 구현은 아직 비어 있습니다.

지금 되어 있는 것:

1. 공용 도메인 객체는 꽤 잡혀 있다.
2. agent 쪽 서비스 골격도 있다.
3. 하지만 실제 계산 핵심은 아직 비어 있다.
4. 모델 어댑터도 아직 미구현이다.
5. 중앙 집계도 아직 거의 시작 전이다.

지금 해야 할 다음 순서는 이거다.

1. agent에서 `ScoringService`를 구현해서 `embedding -> category_scores`를 닫기
2. `WindowingService`를 구현해서 `ScoredEvent[] -> WindowSummary`를 만들기
3. fixture 입력 기준으로 deterministic `WindowSummary` 테스트 붙이기
4. 그 다음에야 main-server의 `summary upload -> cohort aggregation -> NormPack`으로 넘어가기

## User

임베딩 모델 카테고리 나눌 데이터셋 가져오는건 나중에 하는건가?
1번부터 진행하려는데 스코어링서비스 구현한다는게 뭐지? 임베딩 모델에 데이터셋 넣어서 카테고리 중심값 구한다는 말인가?

## Assistant

맞다. `ScoringService`는 "데이터셋을 넣어서 중심값을 만드는 서비스"가 아니다. 그건 별도 `PrototypePack` 빌드 단계다.

구조를 분리해서 보면 이렇다.

1. `LabeledQuerySet` + 임베딩 모델 + 라벨 매핑 규칙으로 카테고리별 중심값을 만든다.
   이게 `PrototypePack` 생성 단계다.
2. agent는 이미 만들어진 `PrototypePack`을 내려받는다.
3. `ScoringService`는 각 query의 embedding과 각 카테고리 centroid 사이의 유사도를 계산한다.

즉 지금 구현할 것은 "이미 만들어진 중심값으로 점수를 매기는 서비스"다.

## User

cosine기준으로 하는 이유가 있나? 시그모이드 함수같은걸로 안하고?

## Assistant

`cosine`은 "임베딩과 centroid가 얼마나 비슷한가"를 재는 similarity이고, `sigmoid`는 나온 점수를 0~1 범위로 변환하는 calibration 함수다. 지금 단계의 `ScoringService`는 similarity 자체를 계산하는 역할이므로 `cosine`이 맞다.

정리:

1. `cosine similarity`
   - query embedding과 category centroid의 방향 유사도를 잰다.
   - prototype 기반 scoring에 자연스럽다.
2. `sigmoid`
   - 나온 점수를 후처리해서 0~1로 누르는 함수다.
   - 자체적으로 벡터 간 similarity를 계산하지 않는다.

## User

기존 계획대로 진행해보자

## Assistant

기존 순서대로 진행합니다. 먼저 `ScoringService` 구현에 필요한 도메인과 테스트 구조를 확인한 뒤, cosine scoring과 fixture 기반 unit test까지 바로 넣겠습니다.

작업 내용:

1. `agent/src/services/scoring_service.py`
   - cosine similarity 계산 구현
   - unsupported metric, 차원 불일치, zero-norm 입력 검증 추가
2. `agent/tests/unit/fixtures/prototype_pack_v1.json`
   - contract 형태의 prototype fixture 추가
3. `agent/tests/unit/test_scoring_service.py`
   - scoring 정상 케이스
   - unsupported metric
   - dimension mismatch
   - zero-norm embedding / prototype 검증 추가

검증:

1. `python3.11 -m compileall agent/src/services/scoring_service.py agent/tests/unit/test_scoring_service.py`
   - 성공
2. `python3.11 -m pytest agent/tests/unit/test_scoring_service.py`
   - 실패: 현재 환경에 `pytest` 모듈 없음
3. `python3.11 -c ...`
   - fixture를 읽어 `ScoringService` 핵심 cosine assertion 직접 검증
   - 성공: `scoring-service-check: ok`
