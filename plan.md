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
4. 초기 seed, 중앙 SSL control, FL SSL non-IID 비교는 같은 방식으로 닫히지 않는다.
   초기에는 중앙집중형 `fixed embedding + classifier` seed를 만들고,
   query가 충분히 쌓인 뒤에는 중앙 SSL을 pooled/offline control로 비교한 뒤,
   논문 메인 비교는 non-IID 제약이 들어간 FL SSL 방법론에서 연다.

즉 활성 방향은
`normative modeling + peer norm`
이 아니라
`privacy-preserving personalized local inference + federated shared model improvement`
이다.

## 3. 핵심 설계 결정

1. `WindowSummary`와 `NormPack`은 활성 아키텍처에서 제외한다.
2. `PrototypePack`은 유지하지만, cohort norm artifact가 아니라
   bootstrap, comparison baseline, semantic reference artifact로 해석한다.
   메인 classifier를 대체하지는 않는다.
3. 활성 연구 로드맵은 staged 구조로 분리한다.
   - Seed 단계: `central + fixed embedding + classifier`
   - 중앙 control 단계: `query accumulation -> threshold/policy selection -> pooled/offline SSL control`
   - 메인 논문 비교: `FL SSL under non-IID`
   - 시스템 트랙: FL SSL winner를 `runtime/privacy` 제약에 맞게 translation
4. 중앙 SSL 비교는 메인 논문 랭킹이 아니라 pooled/offline control table이다.
5. query-domain 적응의 메인 baseline은 `supervised LoRA + classifier`다.
6. 중앙 SSL control의 첫 실험선은 `pseudo-label self-training`, `FixMatch`, `R-Drop`, `MixText` 순으로 둔다.
7. `TAPT`는 분류 objective와 분리된 optional preadaptation phase로 둔다.
8. FL SSL 메인 비교에서는 `FedMatch`, `FedLGMatch`, `(FL)^2` 같은 non-IID 대응 방법을 우선 후보군으로 둔다.
9. 시스템 트랙의 v1 baseline은 여전히
   `embedding -> global classifier -> local interpretation`으로 둔다.
10. 라벨된 데이터셋은 prototype build뿐 아니라 supervised classifier seed와
   validation/calibration split source로도 직접 사용한다.
11. query 적응을 위해 원문 query, confidence, predicted label 같은 로컬 버퍼를 유지한다.
12. diagonal scale adapter와 prototype scoring은 비교 실험 및 확장 축으로 유지한다.
13. single prototype baseline은 허용하고, multi-prototype runtime은 필요성이 확인될 때 다시 연다.
14. full encoder FL은 upper-bound 또는 마지막 확장 옵션으로 둔다.

## 4. 구조 원칙

### 4-1. 전역에서 유지하는 것

1. backbone 또는 shared classifier/adapter의 공통 파라미터
2. `PrototypePack`
3. `ModelManifest`
4. `TrainingTask`
5. 라운드 상태와 model revision

### 4-2. 로컬에서만 유지하는 것

1. 원문 텍스트와 이벤트
2. 로컬 baseline
3. persistence와 시계열 누적 상태
4. 로컬 query buffer와 학습 후보
5. 최종 `AssessmentResult`

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
2. seed baseline과 query-domain 적응 단계를 섞지 않는다.
3. 중앙 SSL control로 기본 동작을 확인한 뒤 FL SSL non-IID 메인 비교로 넘어간다.
4. heuristic과 gradient, runtime과 aggregation, privacy와 training을 분리한다.
5. source of truth는 코드 가까이에 둔다.
6. raw registry보다 strategy/factory/family object를 우선한다.
7. 논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 두고,
   method-only 변형은 그 안에 남긴다. 두 개 이상 방법론에서 공유되는 계산만
   축별 `methods` 패키지로 승격한다.

## 7. 연구 메시지

TraceMind의 메시지는 아래와 같다.

> 본 연구는 원문을 중앙 서버에 보내지 않고,
> 공통 의미 표현 공간과 로컬 개인화 상태를 결합해
> 사용자별 변화와 시계열 맥락을 해석하고,
> 필요 시 연합학습으로 공통 표현 계층을 점진적으로 개선하는
> privacy-preserving personalized local inference 시스템을 제안한다.
>
> 또한 초기 seed는 중앙집중형 `fixed embedding + classifier`로 안정적으로 만들고,
> 중앙 SSL은 pooled/offline control로 비교하며, 핵심 논문 비교는 non-IID 환경의
> FL SSL 방법론으로 둔다. 시스템 구현선에서는 그 결과를 runtime/privacy 제약에
> 맞게 재설계한다.

## 8. 현재 결론

1. 초기 기준선은 `central fixed embedding + classifier` seed다.
2. query가 충분히 쌓이기 전에는 표현을 함부로 움직이지 않는 편이 안전하다.
3. 중앙 SSL 비교는 같은 seed, split, query selection, LoRA spec을 둔 pooled/offline control로 제한한다.
4. 메인 논문 비교는 non-IID client split 위의 FL SSL 방법론 비교로 둔다.
5. 시스템 트랙은 FL SSL winner를 runtime/privacy 제약에 맞게 옮기는 후행 단계다.
6. `head_only` classifier family와 `diagonal_scale` adapter는 여전히 시스템 translation과 lightweight baseline에는 유용하다.
7. prototype은 메인 라벨러가 아니라 bootstrap/comparison/reference artifact다.
8. shared adapter와 multi-prototype은 v1 필수가 아니라 비교/확장 축으로 두는 편이 실용적이다.
