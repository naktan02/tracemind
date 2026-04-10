# TraceMind Plan

## 1. 문제 정의

TraceMind는 아동·청소년의 온라인 위험 신호를 다룬다.
기존 접근의 핵심 문제는 두 가지다.

1. 원문을 중앙에 모으는 감시형 구조
2. 모든 사용자에게 같은 기준을 적용하는 절대적 해석

이 도메인에서는 같은 표현도 사용자 맥락에 따라 의미가 달라질 수 있다.
따라서 공통 의미 표현은 공유할 수 있어도, 최종 해석 기준까지 전역으로
고정하는 것은 부적절하다.

## 2. 핵심 가설

1. 공통 의미 표현 공간은 전역 모델로 공유할 수 있다.
2. 최종 해석은 로컬 개인화 상태에서 수행해야 한다.
3. 중앙은 개인 상태를 판정하는 서버가 아니라, 전역 shared artifact와
   학습 라운드를 조정하는 서버가 되어야 한다.

즉 활성 방향은
`normative modeling + peer norm`
이 아니라
`privacy-preserving personalized local inference + federated shared model improvement`
이다.

## 3. 핵심 설계 결정

1. `WindowSummary`와 `NormPack`은 활성 아키텍처에서 제외한다.
2. `PrototypePack`은 유지하지만, cohort norm artifact가 아니라
   bootstrap semantic artifact이자 비교 실험용 semantic artifact로 해석한다.
3. shared representation은 전역에서 관리한다.
4. 개인화 해석과 최종 판단은 로컬에서 수행한다.
5. v1의 우선 baseline은 `embedding -> global classifier -> local interpretation`으로 둔다.
6. shared adapter와 prototype scoring은 비교 실험 및 확장 축으로 유지한다.
7. single prototype baseline은 허용하고, multi-prototype runtime은 필요성이 확인될 때 다시 연다.
8. full encoder FL은 마지막 확장 옵션으로 둔다.

## 4. 구조 원칙

### 4-1. 전역에서 유지하는 것

1. backbone 또는 shared classifier/adapter의 공통 파라미터
2. `PrototypePack`
3. `ModelManifest`
4. `TrainingTask`
5. 라운드 상태와 model revision

### 4-2. 로컬에서만 유지하는 것

1. 원문 텍스트와 이벤트
2. `PersonalizationState`
3. 개인 baseline
4. 개인 threshold
5. personal prototype 또는 개인 보정 상태(필요 시)
6. persistence와 시계열 누적 상태
7. 로컬 학습 후보와 체크포인트
8. 최종 `AssessmentResult`

핵심 문장은 아래와 같다.

> 공통 의미 표현은 전역에서 관리하고,
> 해석과 적응은 로컬에서 수행한다.

## 5. 프라이버시 원칙

1. 원문 텍스트는 로컬에 남긴다.
2. 서버는 개인 위험 판정을 하지 않는다.
3. 로컬 개인화 상태는 기본적으로 서버에 보내지 않는다.
4. 서버로 올라가는 것은 shared model update와 최소 메타데이터다.
5. update 경로는 자동으로 안전하지 않으므로 clipping, secure aggregation,
   DP, 필요 시 HE를 별도 계층으로 다룬다.

## 6. 구현 원칙

1. 계약과 도메인을 먼저 고정한다.
2. 로컬 추론 폐회로를 먼저 안정화한다.
3. 그 다음 로컬 update 생성과 중앙 aggregation을 붙인다.
4. heuristic과 gradient, runtime과 aggregation, privacy와 training을 분리한다.
5. source of truth는 코드 가까이에 둔다.
6. raw registry보다 strategy/factory/family object를 우선한다.

## 7. 연구 메시지

TraceMind의 메시지는 아래와 같다.

> 본 연구는 원문을 중앙 서버에 보내지 않고,
> 공통 의미 표현 공간과 로컬 개인화 상태를 결합해
> 사용자별 변화와 시계열 맥락을 해석하고,
> 필요 시 연합학습으로 공통 표현 계층을 점진적으로 개선하는
> privacy-preserving personalized local inference 시스템을 제안한다.

## 8. 현재 결론

1. shared representation과 shared class evidence는 전역으로 관리할 수 있다.
2. 최종 판단까지 전역 하나로 밀면 drift와 개인 해석 손실이 커진다.
3. 따라서 v1의 기본선은 `global classifier + local personalized interpretation`으로 두되,
   최종 구조 원칙은 계속 `global shared representation + local personalized interpretation`
   이어야 한다.
4. shared adapter와 multi-prototype은 v1 필수가 아니라 비교/확장 축으로 두는 편이 실용적이다.
