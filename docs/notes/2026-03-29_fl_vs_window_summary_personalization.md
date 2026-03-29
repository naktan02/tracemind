# 2026-03-29 FL vs WindowSummary Personalization 논의

## User

진짜 fl처럼 window summary,norm pack안하고 전역에서 모델 내려주고 로컬에서 학습 진행하는데 전역파라미터+로컬파라미터로 로컬파라미터는 개인 각각 조금씩 퇴는거 보정값으로해서 되나? 서머시값들을 주고 받으면 결국 fl로 하는 개인 프라이버시가 좀 퇴색되잖아 그리고 gpt랑 이야기한건데 이거는 어떻게 생각해?

🔥 전체 구조 한눈에 (핵심 아키텍처)
[입력]
쿼리 (텍스트)

→ [FL 기반 encoder or fine-tuned encoder]
→ embedding

→ [global prototype 비교]
→ category score (cosine)

→ [개인화 보정]
→ personal threshold / baseline

→ [시계열 누적]
→ 상태 판단

→ [pseudo-label 생성 (선별)]
→ 로컬 학습 (개인화 + FL 업데이트)
🔥 1. 초기 단계 (배포 전)
✔ 목표
안정적인 기준 만들기
구성
1. 임베딩 모델
pretrained model + 파인튜닝 (중앙 데이터)
(FL은 이 단계에서 필수 아님)
2. 글로벌 프로토타입 생성
proto[c] = mean(embedding(label=c))
카테고리별 중심
필요하면 multi-prototype (DBSCAN)
3. 기본 score 계산
score_c = cosine(x, proto[c])
4. 초기 threshold 설정
top1 score
margin(top1-top2)
🔥 2. 온라인 추론 (핵심 파이프라인)
✔ Step-by-step
① 입력 → embedding
query → encoder → embedding
② global prototype 비교
scores = {
    c: cosine(x, proto[c])
}
③ 개인화 보정
adjusted_score = score - user_baseline

또는

if score > user_threshold:
    ...
④ 시계열 누적
(중요)
S_c(t) = α * current_score + (1-α) * S_c(t-1)

또는

최근 N개 평균
상승 추세
연속 high score count
⑤ 최종 판단
NORMAL / WARNING / RISK
🔥 3. 개인화 구조
✔ 반드시 포함
1. personal baseline
baseline = mean(user_scores)
2. personal threshold
user A: 0.7
user B: 0.85
3. (선택) personal prototype
proto_user[c] = mean(user_data[c])
🔥 4. 라벨 없는 데이터 학습 (핵심)
✔ pseudo-label pipeline
① 후보 생성
pred = argmax(scores)
margin = top1 - top2
② 필터링 (필수)
if score > 0.8 and margin > 0.15:
    use as pseudo-label
else:
    discard
③ 로컬 학습
embedding 모델 미세 조정 (optional)
prototype 업데이트 (추천)
🔥 5. FL 구조 (운영 단계)
✔ 역할 분리
클라이언트
- 로컬 데이터 유지
- pseudo-label 생성
- 개인화 학습
- 일부 파라미터만 서버로 전송
서버
- global encoder 업데이트 (집계)
- 모델 배포
핵심 구조
global encoder (공통)
+
local personalization (개인)
🔥 6. 프로토타입 관리 (중요)
❗ 규칙
실시간 업데이트 ❌
batch 업데이트 ✔
✔ 안전한 흐름
1. pseudo-label 데이터 수집
2. high confidence만 선택
3. 새 prototype 생성
4. 기존과 비교
5. 교체 or 유지
🔥 7. 시계열 안정성 (핵심)
❗ 문제
모델/프로토타입 바뀌면 값 변함
✔ 해결
방법 1: 버전 관리
encoder_v1 + proto_v1 → 시계열 v1
encoder_v2 + proto_v2 → 시계열 v2
방법 2: baseline 중심 판단
delta = score - user_mean
방법 3: 변화량 기반
증가 추세
slope
variance
🔥 8. 전체 흐름 요약 (진짜 핵심)
✔ 초기
중앙 데이터로 파인튜닝
global prototype 생성
✔ 운영
embedding → score → 시계열
개인 baseline / threshold 적용
✔ 학습 (라벨 없음)
global prototype → pseudo-label 생성
high confidence만 사용
로컬 학습
✔ FL
encoder만 공유
개인화는 로컬 유지
🔥 9. 최종 구조 한 줄

👉 “공통 embedding 공간 위에서,
global prototype으로 기준을 잡고,
개인 baseline과 시계열로 상태를 해석하며,
pseudo-label로 지속적으로 개인화 학습하는 구조”

## Assistant

가능은 하다. 그 방식은 사실상 `personalized FL` 또는 `federated fine-tuning` 계열이다.
즉 `global parameter + local correction` 구조 자체는 이상하지 않다.

다만 TraceMind 문맥에서는 그 구조가 `WindowSummary/NormPack`을 대체하는 순간, 단순 구현 선택이 아니라 문제 정의가 바뀐다.

핵심은 세 가지다.

1. `전역 모델 + 로컬 보정값`은 성립한다.
   - 예를 들어 `theta_user = theta_global + delta_user`처럼 둘 수 있다.
   - 하지만 이 프로젝트에서는 전체 encoder를 개인별로 조금씩 틀어버리기보다, `personal threshold`, `personal baseline`, `personal prototype`, 작은 adapter/head 같은 얇은 개인화 계층이 더 안정적이다.
   - 특히 이 도메인에서는 아래처럼 점수 계층에서 보정하는 쪽이 해석 가능성이 높다.
   - `score_c = λ * cos(z, proto_global[c]) + (1-λ) * cos(z, proto_user[c]) - b_user[c]`

2. `summary를 주고받으면 privacy가 퇴색하고, FL은 더 안전하다`는 식의 이분법은 맞지 않는다.
   - `WindowSummary`도 개인 행동에서 나온 값이므로 privacy cost가 0은 아니다.
   - 하지만 FL의 gradient/update도 민감 정보가 새어 나갈 수 있다.
   - secure aggregation, clipping, DP 없이 parameter update를 모으면 오히려 coarse summary보다 더 위험할 수도 있다.
   - 즉 privacy는 `summary냐 FL이냐`보다, 무엇을 얼마나 자주 어떤 보호장치와 함께 보내느냐의 문제다.

3. `WindowSummary/NormPack`을 빼면 peer norm이 사라진다.
   - TraceMind의 핵심은 `Self-baseline + Peer norm` 분리다.
   - 요약/규범 팩 없이 pure FL만 하면, 시스템은 개인화 anomaly detection이나 personalized classifier 쪽으로 바뀐다.
   - 그 경우 “또래 대비 드묾”이라는 축이 약해진다.

GPT와 이야기한 구조에 대한 평가는 아래처럼 정리할 수 있다.

- 맞는 부분
  - `global prototype + personal baseline + temporal accumulation` 조합은 좋다.
  - pseudo-label은 high-confidence만 쓰고 batch update만 하는 게 맞다.
  - 모델/프로토타입 버전 관리가 꼭 필요하다는 판단도 맞다.

- 조심할 부분
  - pseudo-label로 곧바로 encoder를 FL 업데이트하는 건 너무 이르다.
  - 라벨 없는 상태에서 encoder까지 계속 미세조정하면 drift와 자기확증 오류가 커질 수 있다.
  - 이 프로젝트에서는 먼저 `prototype/baseline/decision policy`를 안정화하고, 나중에 feedback이 생기면 작은 `decision head`만 FL로 여는 게 더 자연스럽다.

내 판단은 이렇다.

- `global encoder + local personalization`은 채택 가능하다.
- 하지만 `WindowSummary/NormPack`을 없애는 대체재로 쓰는 건 현재 TraceMind 방향과 안 맞는다.
- 더 맞는 구조는 하이브리드다.

추천 하이브리드:

1. 전역:
   - 중앙이 `encoder version`과 `PrototypePack`을 배포

2. 로컬 전용:
   - `personal baseline`
   - `personal threshold`
   - 필요 시 `personal prototype` 또는 작은 local adapter

3. 중앙 업로드:
   - privacy-safe `WindowSummary`만 전송
   - 단, `self-baseline`, 개별 이벤트 점수, 개인 판단 결과는 보내지 않음

4. 나중 확장:
   - self-report / feedback이 생기면 작은 `decision head`만 FL
   - 그때 secure aggregation + clipping + 필요 시 DP 추가

즉 한 줄로 정리하면:

`전역 의미 공간 + 로컬 보정 + 중앙 norm 학습`은 같이 갈 수 있고, 오히려 이 프로젝트에는 그 조합이 제일 자연스럽다.

참고로 현재 프로젝트 문서도 같은 방향이다.
- `WindowSummary/NormPack`은 analytics 산출물
- FL은 나중의 optional decision-model 산출물
- `NormPack`은 global model parameter가 아님

## User

window summary랑 norm summary안하고 임베딩 모델을  fl로 하는걸로 하려고 하는데 어떻게 생각해
지금 구조를 뜯어고쳐도 돼

## Assistant

가능은 하지만, 현재 TraceMind의 핵심 목표를 유지할 생각이면 권하지 않는다.

이유는 단순하다.

1. 지금 구조의 중심은 `Self-baseline + Peer norm`이다.
   - `WindowSummary`와 `NormPack`을 빼면 `peer norm` 축이 거의 사라진다.
   - 그러면 시스템은 "또래 대비 드묾을 보는 normative analytics"가 아니라, "개인별 임베딩 모델을 연합학습하는 personalized FL 시스템"으로 바뀐다.

2. 임베딩 모델 FL은 학습 신호 정의가 더 어렵다.
   - `WindowSummary/NormPack` 경로는 feedback 없이도 성립한다.
   - 반면 embedding FL은 무엇을 loss로 학습할지부터 다시 정해야 한다.
   - supervised label, contrastive pair, pseudo-label, distillation 중 무엇을 쓸지 명확해야 한다.
   - 특히 pseudo-label만으로 encoder를 지속 업데이트하면 drift와 confirmation bias 위험이 크다.

3. privacy도 자동으로 좋아지지 않는다.
   - summary를 안 보내면 좋아 보일 수 있지만, gradient/update도 민감 정보 누출 가능성이 있다.
   - secure aggregation, clipping, DP 없이 embedding update를 모으면 privacy 측면에서 오히려 더 까다롭다.

4. 구조를 뜯는 범위가 크다.
   - 현재 문서는 `WindowSummary/NormPack`을 analytics 산출물로, FL은 나중의 decision-model 확장으로 정의한다.
   - 이걸 바꾸면 계약, phase 순서, 검증 기준, 서버 책임, privacy 설명까지 전부 다시 써야 한다.

그래서 결론은 아래 둘 중 하나다.

1. 현재 연구 질문을 유지한다.
   - 이 경우 구조를 뜯지 않는 게 맞다.
   - `WindowSummary -> NormPack -> local decision`을 유지하고, FL은 나중의 optional decision head로 둔다.

2. 연구 질문을 바꾼다.
   - 이 경우 구조를 뜯어고쳐도 된다.
   - 하지만 그건 리팩터링이 아니라 아키텍처 재정의다.
   - 프로젝트 소개 문장부터 "normative modeling"이 아니라 "privacy-preserving personalized representation learning" 쪽으로 바뀌어야 한다.

실무적으로는 완전 전환보다 아래 절충안이 더 안전하다.

1. `WindowSummary/NormPack` 경로는 유지
2. embedding model은 중앙 사전학습/오프라인 업데이트 중심
3. 나중에 정말 FL을 붙이고 싶으면 encoder 전체가 아니라 작은 adapter 또는 decision head만 FL

즉, 지금 당장 구조를 뜯어도 되냐는 질문에는:

- 현재 목표를 유지하면 `아니오`
- 목표 자체를 바꾸겠다면 `예, 하지만 문서와 평가 기준까지 전면 수정해야 함`

## User

얘기한건 밑에와 같고 전화하려고 해 어떻게 할까? 지금 데이터셋 가져온것,임베딩 모델은 그대로 두고 폴더구조도 크게 바뀌지는 않지 않나? 계약밖에 현재 진행 안했으니까 앞으로 어떻게 할지 계획만 자세히 다시 정하면 될것 같은데 밑에 말이 맞는거지?

🔥 1. “절대적 기준”이 맞는 경우 vs 아닌 경우
✔ 절대적인 경우
고양이 vs 개
숫자 인식
물체 분류

👉 같은 입력이면 누구에게나 같은 결과

❌ 절대적이지 않은 경우 (너 케이스)
감정
상태
위험도
심리

👉 해석이 사람마다 다름

🔥 2. 왜 같은 문장이 다르게 해석될 수 있냐
예시

"요즘 너무 힘들다"

사용자 A
평소 표현 강함
그냥 피곤한 상태

👉 normal / mild

사용자 B
평소 거의 긍정적
이런 말 거의 안 함

👉 depression 위험

👉 같은 문장
👉 다른 의미

🔥 3. 그래서 “절대 거리”가 깨지는 이유

너 지금 구조:

embedding
prototype 거리

👉 이 거리 자체는 수학적으로는 동일하지만

👉 “해석 기준”이 달라짐

🔥 4. 개인화의 진짜 의미

👉 embedding 공간은 공통

👉 해석은 개인화

구조로 보면
embedding = 공통
distance = 공통 계산
판단 기준 = 개인별
🔥 5. 장점 (핵심 질문 답)
✅ 1. False Positive 줄임
표현이 강한 사람
→ 항상 위험으로 분류되는 문제 방지
✅ 2. False Negative 줄임
평소 조용한 사람
→ 작은 변화도 감지 가능
✅ 3. 변화 감지 가능 (중요)

👉 절대값보다 중요한 건:

👉 “변화”

예
평소 depression score: 0.3
최근: 0.6

👉 위험 상승

근데 다른 사람:

평소: 0.6
현재: 0.6

👉 정상

👉 절대값 같지만 의미 다름

✅ 4. 개인 baseline 반영

👉 핵심:

👉 “이 사람이 평소 어떤 사람인가”

🔥 6. 네 질문 핵심 답

각 쿼리는 어디 분류에 가까운지는 절대적인 기준 아니야?

👉 ❌ 아니다 (너 문제에서는)

👉 ✔ “공통 공간에서의 위치”는 동일하지만
“그걸 어떻게 해석할지는 개인마다 다름”

🔥 7. 변해도 되는거냐?

👉 ✔ “변해야 한다”

단, 중요한 조건 있음:

❗ 변하면 안 되는 것
embedding 공간 자체 (너무 자주 바꾸면 안됨)
모델 기준
✔ 변해야 하는 것
개인 기준
threshold
baseline
prototype 일부
🔥 8. 핵심 구조 정리
[공통]
embedding 공간 (FL / fine-tuned)

[개인]
- prototype
- threshold
- baseline

→ 해석
🔥 9. 직관적인 비유

👉 체온 예시

37도 = 정상 (일반 기준)
어떤 사람은 평소 36도
→ 37도면 이상

👉 기준은 같지만 해석은 다름

## Assistant

밑에 설명은 핵심 절반은 맞고, 결론 절반은 비약이다.

맞는 부분:

1. 같은 문장도 개인 과거 맥락에 따라 해석이 달라진다.
2. 따라서 `personal baseline`, `personal threshold`, 필요 시 `personal prototype`이 중요하다.
3. 절대 점수보다 변화량과 지속성이 더 중요하다는 판단도 맞다.

비약인 부분:

1. 개인 해석이 중요하다는 사실이 곧바로 `WindowSummary/NormPack` 제거를 뜻하지는 않는다.
2. 개인화가 필요하다는 사실이 곧바로 `embedding model FL`을 뜻하지도 않는다.
3. 이 논리는 우선 `해석 계층 개인화`를 강하게 지지하는 것이지, 반드시 `표현 계층 FL`을 지지하는 것은 아니다.

현재 저장소 상태를 보면, 아직 방향 전환이 불가능한 수준은 아니다.
하지만 "계약밖에 안 했다"도 정확하지 않다.

이미 있는 것:

1. `agent/src/services/scoring_service.py`
2. `agent/src/services/windowing_service.py`
3. `agent/src/services/baseline_service.py` 초안
4. `shared/src/domain/entities/window_summary.py`
5. `shared/src/domain/entities/norm_pack.py`
6. `shared/src/contracts/sync_contracts.py` 초안
7. `main-server/src/services/norm_pack_service.py` 초안
8. `main-server/src/services/round_manager_service.py` 초안

즉 지금은 "완성 직전"은 아니지만, 이미 analytics 경로의 골격은 들어가 있다.

그래도 지금 바꾸는 것은 가능하다.
이유는 아래와 같다.

1. `NormPackService`, `RoundManagerService`, `training` 라우터는 아직 자리표시자 수준이다.
2. `WindowSummary` 경로도 핵심 알고리즘이 깊게 누적된 상태는 아니다.
3. 데이터셋과 임베딩 어댑터는 그대로 재사용 가능하다.
4. `agent / main-server / shared` 경계도 그대로 유지 가능하다.

따라서 크게 보면 아래가 맞다.

1. 데이터셋은 대부분 유지 가능
2. 임베딩 모델도 유지 가능
3. 폴더 구조도 크게 안 바꿔도 됨
4. 하지만 "계획만 다시 쓰면 끝"은 아님
5. 문서, 계약, 서비스 책임, 테스트 기준을 함께 다시 정해야 함

가장 중요한 판단은 이것이다.

### A. 지금도 `peer norm`을 연구의 핵심으로 둘 것인가

그렇다면 `WindowSummary/NormPack`을 버리면 안 된다.

### B. 연구를 `개인화된 의미 해석과 로컬 적응` 쪽으로 바꿀 것인가

그렇다면 지금 전환해도 된다.

이 경우 추천 전환은 "encoder 전체 FL"보다 아래 순서가 더 안전하다.

1. global embedding model은 우선 freeze 또는 매우 느리게만 갱신
2. local personalization은 `baseline`, `threshold`, `personal prototype`, 작은 adapter/head 중심
3. 충분한 로컬 signal이 생긴 뒤에만 adapter/head FL
4. encoder full FL은 가장 마지막 옵션

즉 전환안의 핵심은:

- 공통 표현 공간은 유지
- 해석은 개인화
- 중앙은 norm server 대신 model/version/update coordinator

전화나 논의 자리에서는 아래처럼 말하면 된다.

1. "우리는 개인 해석이 중요한 문제를 다루고 있다."
2. "그래서 개인화 계층 강화는 맞다."
3. "다만 그게 곧바로 encoder full FL을 의미하지는 않는다."
4. "현재 시점에서는 데이터셋과 임베딩 모델, 폴더 구조는 대부분 유지한 채 문서와 계약을 새 목표에 맞게 바꾸는 전환이 가능하다."
5. "단, 그렇게 하면 연구 메시지는 normative modeling에서 personalized local adaptation 쪽으로 이동한다."

전환을 실제로 하려면 다음 순서로 문서를 다시 잡는 것이 맞다.

1. `plan.md`의 핵심 문제 정의를 수정
2. `docs/project_execution_plan.md`의 phase를 `WindowSummary/NormPack` 중심에서 `personalization + FL runtime` 중심으로 재작성
3. `docs/staged_execution_roadmap.md`의 검증 기준을 `summary 생성`이 아니라 `local personalized inference + update cycle`로 수정
4. `shared` 계약을 `TrainingTask`, `TrainingUpdateEnvelope`, `ModelManifest`, `PersonalizationState` 중심으로 재정의
5. 그 다음 코드 정리를 시작

## User

WindowSummary/NormPack 완번히 버리는쪽으로 해서 전황안으로 진행하자

## Assistant

사용자가 `WindowSummary/NormPack`을 완전히 버리는 전환안을 명시적으로 선택했다.
이에 따라 문서 기준을 아래처럼 재작성했다.

1. `plan.md`
   - normative modeling 중심 메시지를 버리고
   - `privacy-preserving personalized local adaptation + FL runtime` 중심 메시지로 전환

2. `docs/project_execution_plan.md`
   - `WindowSummary/NormPack`을 보관 경로로 내리고
   - `ModelManifest`, `TrainingTask`, `TrainingUpdateEnvelope`, `DecisionFeedbackSignal`, `PersonalizationState` 중심 phase로 재작성

3. `docs/staged_execution_roadmap.md`
   - active execution rail을
     `Personalized Inference`와 `Federated Training` 두 레일로 재정의
   - 단계별 검증 기준도 이에 맞게 수정

4. `docs/execution_index.md`
   - 읽는 순서와 활성 contract 목록을 새 방향으로 갱신

5. 새 contract 문서 추가
   - `docs/contracts/model_manifest_v1.md`
   - `docs/contracts/training_task_v1.md`
   - `docs/contracts/training_update_envelope_v1.md`
   - `docs/contracts/decision_feedback_signal_v1.md`
   - `docs/contracts/personalization_state_v1.md`

6. `docs/contracts/window_summary_v1.md`
   - 활성 source of truth가 아니라는 보관 표시 추가

현재 기준에서 유지하는 것:

1. 데이터셋 경로와 처리 자산
2. 임베딩/번역 어댑터
3. `PrototypePack`
4. `agent / main-server / shared` 구조

현재 기준에서 내려가는 것:

1. `WindowSummary`
2. `NormPack`
3. cohort normative aggregation

앞으로의 구현 순서는 아래로 고정했다.

1. active contract 확정
2. local personalized inference 폐회로
3. local update generation
4. central FL coordinator
5. federated e2e
6. selective encoder FL
7. privacy hardening

문서 정리 뒤, 코드 뼈대도 아래처럼 추가로 맞췄다.

1. `shared/src/domain/entities/model_manifest.py`
2. `shared/src/domain/entities/training_task.py`
3. `shared/src/domain/entities/training_update.py`
4. `shared/src/domain/entities/decision_feedback_signal.py`
5. `shared/src/domain/entities/personalization_state.py`
6. `shared/src/contracts/model_contracts.py`
7. `shared/src/contracts/training_contracts.py`
8. `shared/src/contracts/personalization_contracts.py`

기존 타입 중 새 방향과 충돌하는 부분도 조정했다.

1. `AssessmentResult`에서 `peer_deviation` 중심 의미를 제거
2. `DecisionService` 설명을 `personalization state` 기준으로 수정
3. `sync_contracts.py`를 보관 경로 설명으로 변경

검증:

- `PYTHONPATH=. .venv/bin/pytest shared/tests/unit/test_new_training_contracts.py -q`
- 결과: `3 passed`

## User

지금은 들어온 쿼리에 대해 fl로 임베딩 하는거고 시계열로 누적해서 최종 판단은 어떻게 할지 고민이긴한데 그런 내용도 들어있지? 마음 이상 탐지 조치가 목표인것은 변함이 없어

## Assistant

사용자 уточнение에 맞춰 문서에서 아래를 더 명시적으로 보강했다.

1. 목표는 여전히 마음 이상 신호 탐지와 조치 제안이라는 점
2. 단일 이벤트 점수 자체가 아니라 시계열 누적과 persistence를 통해 최종 판단한다는 점
3. FL 기반 임베딩/공통 표현 공간 위에 로컬 `baseline + personalization + time-series accumulator`가 올라간다는 점

반영 위치:

1. `plan.md`
2. `docs/project_execution_plan.md`
3. `docs/staged_execution_roadmap.md`

핵심 문구:

- 공통 임베딩 공간은 전역에서 관리
- 로컬은 시계열 누적과 변화 감지를 수행
- 최종 `AssessmentResult`는 지원 필요 여부와 조치 레벨 판단에 사용

## User

1. 결국 임베딩 모델은 각 에이전트마다 다르게 학습하는거지? non-iid? 같이
2. 다음 진행해야 하는데 나의 의견이 필요한 부분에는 나에게 먼저 묻고 진행해 혼자서 진행하지 말고, 그리고 단일 클래스로 해서 빠른 해결을 하는것이 아니라 갈아끼울 수 있는 미래지향적인 설계를 난 지행해

## Assistant

1번에 대한 정리:

- 초기 상태는 모든 agent가 같은 전역 모델에서 시작한다.
- 하지만 각 agent는 자기 로컬 데이터로 학습하므로, round 내부에서는 각자 다른 방향으로 업데이트된다.
- 이 차이는 곧 non-IID 조건을 전제로 한다는 뜻이다.
- 이후 중앙은 "공유할 파라미터 subset"만 집계해 새 전역 모델을 만든다.
- 동시에 로컬에만 남기는 상태도 있다.
  - `baseline`
  - `threshold`
  - `personal prototype`
  - 시계열 누적/persistence 상태
  - 필요 시 private adapter/head

즉 정확히는:

- `shared/global parameter`는 다시 모인다.
- `private/local parameter or state`는 agent마다 다르게 남는다.

2번에 대한 작업 원칙:

- 사용자 의견이 필요한 구조 결정은 먼저 확인하고 진행한다.
- 단일 클래스 임시 해결은 지양한다.
- 교체 가능한 미래지향적 설계를 우선한다.

다음 구현 전 사용자 확인이 필요한 대표 항목:

1. 1차 FL 범위를 어디까지 열지
   - `head only`
   - `adapter only`
   - `selected encoder block`

2. 로컬 private 상태 범위를 어디까지 둘지
   - `baseline + threshold`
   - 여기에 `personal prototype`까지 포함할지

3. 최종 판단 v1을 무엇으로 시작할지
   - 규칙 기반 `DecisionPolicy`
   - 작은 decision model

앞으로는 이런 항목을 먼저 사용자에게 확인한 뒤 진행하는 것이 맞다고 정리했다.

## User

잠만 진행하기 전에 global embedding + 다른 local interpretation 이걸로 한다고 했는데 이것도 fl은 맞아?

## Assistant

정확히는 아래처럼 구분된다고 답했다.

1. 같은 global embedding을 쓰기만 하고 중앙으로 update를 보내지 않으면 FL은 아니다.
2. 로컬에서 학습한 update를 중앙이 집계해 global embedding model을 다시 배포하면 FL 맞다.

즉 FL 여부는 "같은 global embedding을 쓰느냐"가 아니라,
"로컬 update를 중앙이 집계해 다시 global model을 갱신하느냐"로 결정된다고 정리했다.

현재 방향은:

- `v1`: 같은 global embedding + 다른 local interpretation
- `v2`: private adapter를 붙여 embedding도 일부 개인화

## User

그리고 개인이 입력하는 쿼리는 라벨링이 안되어 있으니까 신뢰도가 높은 것에는 프로토타입으로 카테고리를 붙이고 학습 진행하는것도 맞지?

## Assistant

맞다. 그건 현재 구조에서 자연스러운 `pseudo-label` 경로다.

다만 아래 경계가 중요하다고 정리했다.

1. prototype 기반 pseudo-label은 가능하다.
2. 하지만 ground truth처럼 취급하면 안 된다.
3. high-confidence와 margin filtering이 꼭 필요하다.
4. cold start나 warm-up 구간에서는 더 보수적으로 써야 한다.
5. 실시간 즉시 prototype/encoder를 바꾸기보다 batch 단위 누적 후 반영하는 것이 안전하다.

초기 권장 사용처:

1. local training candidate 생성
2. adapter/head 또는 calibration 계층 학습
3. 충분히 선별된 경우에만 FL update 생성

초기 비권장:

1. low-confidence pseudo-label 사용
2. pseudo-label만으로 full encoder를 바로 크게 업데이트
3. pseudo-label을 곧바로 global prototype source of truth처럼 취급

## User

오케이 1번부터 진행하고 v2에서 추가하자
그리고 BaselineService와 DecisionService 도 이제 다시 하는건가? 이건 어떻게 할건데?

## Assistant

`v1`은 `same global embedding + different local interpretation`,
`v2`는 private adapter 추가로 정리하고,
지금은 `minimal local interpretation`을 먼저 구현하는 것이 맞다고 설명했다.

구조는 단일 클래스 하나로 몰지 않고 아래처럼 나눈다고 정리했다.

1. `BaselineService`
2. `TimeSeriesAccumulator`
3. `DecisionPolicy`
4. `DecisionService`

즉:

- `BaselineService`는 평소값 계산기
- `TimeSeriesAccumulator`는 변화 누적기
- `DecisionPolicy`는 판단 규칙
- `DecisionService`는 orchestration

## User

판단부분까지 지금 바로 하는게 좋은거야?
지금은 임베딩 fl 구조부터 만들어야 하는거 아니야?

## Assistant

완성형 판단 엔진 전체를 지금 다 만드는 건 이르지만,
`minimal local interpretation layer`는 FL runtime보다 먼저 필요하다고 설명했다.

이유:

1. 어떤 샘플을 pseudo-label로 채택할지 기준이 필요하다.
2. baseline 대비 변화와 persistence가 있어야 학습 후보 선별이 가능하다.
3. 그래서 `BaselineService`, `TimeSeriesAccumulator`, 얇은 `DecisionPolicy`, `DecisionService`는 먼저 있어야 한다.

## User

진행해

## Assistant

아래 구현을 추가했다.

### 새 entity / policy

1. `shared/src/domain/entities/time_series_state.py`
2. `shared/src/domain/policies/decision_policy.py`

### 보강된 entity

1. `shared/src/domain/entities/baseline_profile.py`
   - `observed_days`
   - `event_count`
   - `category_counts`
   - `category_latest`
   - `computed_at`

2. `shared/src/domain/entities/assessment_result.py`
   - `focus_category` 추가

### 새/갱신된 agent 서비스

1. `agent/src/services/baseline_service.py`
   - lookback/warm-up 기반 baseline profile 계산

2. `agent/src/services/time_series_service.py`
   - EWMA delta
   - elevated streak
   - event count 누적

3. `agent/src/services/decision_service.py`
   - accumulator + policy orchestration
   - `AssessmentResult`와 갱신된 `TimeSeriesState`를 함께 반환

### 테스트

1. `agent/tests/unit/test_local_interpretation_services.py`
   - baseline profile 계산
   - time-series streak 누적
   - single spike -> `WATCH`
   - sustained change -> `SUPPORT` / `RISK`

검증 결과:

- `PYTHONPATH=. .venv/bin/pytest agent/tests/unit shared/tests/unit -q`
- 결과: `17 passed`

현재 상태 해석:

1. legacy `WindowSummary/NormPack` 경로는 아직 파일로는 남아 있지만 활성 구현 우선순위는 아니다.
2. 새 active local interpretation 경로는 이제 코드 뼈대와 테스트를 가진 상태다.
3. 다음 단계는 `pseudo-label candidate -> TrainingTask -> TrainingUpdateEnvelope` 경로를 여는 것이다.

## User

pseudo-label candidate -> TrainingTask -> TrainingUpdateEnvelope까지 열고 데이터셋 train을 split한것을 가지고 할 수 있을까? train은 프로토타입으로 만들고 test 데이터셋으로 여러 에이전트에게 나눈다음 학습 진행 테스트 진행하는거 어때? 결국 이 구글 코드랑 비슷한건가? 구조?

## Assistant

가능하다고 답했다. 다만 `test`를 agent 학습 데이터로 쓰는 것은 비권장이라고 정리했다.

권장 데이터 사용 방식:

1. `train`에서 bootstrap subset 또는 현재 train split으로 `PrototypePack` 생성
2. 같은 train 계열 데이터의 나머지를 여러 agent shard로 non-IID 분할
3. client 쪽에서는 라벨을 가린 상태로 pseudo-label 기반 local training simulation 수행
4. `validation`은 confidence threshold, margin, policy tuning에 사용
5. 원본 `test`는 최종 holdout 평가로 유지

즉:

- `test`를 학습에 쓰지 않는다
- `train` 내부를 `prototype bootstrap`과 `client simulation` 용도로 다시 나누는 편이 맞다

현재 저장소 근거:

1. `data/processed/splits/ourafla_train_split.v1.train.jsonl`
2. `data/processed/splits/ourafla_train_split.v1.validation.jsonl`
3. `data/processed/labeled_query_sets/ourafla_mental_health_text_classification_test.v1.jsonl`
4. `scripts/experiments/run_federated_simulation.py`는 아직 미구현이라 지금부터 이 흐름으로 채우면 된다

TensorFlow Federated와의 관계는 아래처럼 정리했다.

비슷한 점:

1. 여러 client local dataset
2. round 기반 local training
3. server aggregation
4. 새 global model 배포
5. single-machine simulation 가능

다른 점:

1. TraceMind는 `agent / main-server / shared` 애플리케이션 구조를 직접 설계한다
2. bootstrap `PrototypePack`과 local interpretation 계층이 있다
3. 로컬 private state(`baseline`, `threshold`, `persistence`)를 함께 다룬다
4. TFF는 프레임워크이고, TraceMind는 도메인 특화 시스템 구조다

따라서 "개념적 FL 루프"는 비슷하지만,
"코드 구조와 책임 분리"는 TFF보다 TraceMind가 훨씬 도메인 지향적이라고 정리했다.

## User

그렇게 진행하자

## Assistant

다음 단계 구현을 진행하기로 했다.

핵심 전제는 아래처럼 정리했다.

1. `v1`은 `round active pair only`
2. 즉 각 round마다 `embedding revision + prototype version`이 함께 움직인다
3. 추론, pseudo-label 생성, local training은 모두 같은 active pair를 본다
4. `training candidate model`과 `inference model`을 따로 두는 구조는 현재 단계에서는 쓰지 않는다

## User

전역에서 내려주는 공통 임베딩 모델은 계속 업데이트 되는데 기준점이 프로토탑이 안바뀌면 결과값이 틀리게 되는거 아냐?

## Assistant

그 우려가 맞다고 답했다.

정리:

1. active runtime에서는 `embedding model`과 `PrototypePack`이 항상 한 쌍이어야 한다
2. 따라서 `embedding_rev_B`인데 `prototype_pack_A`를 쓰는 mismatch는 허용하면 안 된다
3. v1에서는 round마다 새 global revision이 생기면 대응하는 `PrototypePack`도 같이 만들어 active pair로 배포하는 쪽이 맞다

## User

중앙에서 모으다가 일정시점에 모델 파라미터를 다시 내려주면서 프로토타입도 동시에 내려주는거지? 새로?

## Assistant

그렇다고 답했다.

흐름을 아래처럼 정리했다.

1. 각 agent가 같은 active pair에서 시작
2. pseudo-label 기반 local update 생성
3. 중앙 aggregation
4. 새 global revision 결정
5. 그 revision으로 새 `PrototypePack` 생성
6. 새 `ModelManifest + PrototypePack` pair 재배포

## User

원래 fl에서는 라운드마다 내려주나? 아니면 쌓였을때?

## Assistant

둘 다 가능한데, 현재 TraceMind v1에서는 아래처럼 분리하지 않기로 정리했다.

1. 고전적 FL은 round마다 다음 global model을 배포한다
2. 하지만 우리 구조는 prototype mismatch를 피해야 하므로
3. 지금 단계에서는 round마다 synchronized active pair를 같이 배포하는 쪽으로 가기로 했다

## User

agent들이 v2를 받았는데 실제 추론은 v1으로 하고 있으면 v2 학습 진행이 안되잖아

## Assistant

그 지적이 맞다고 답했고, 그래서 분리형 candidate model 구조를 현 단계에서 쓰지 않기로 했다.

최종 정리:

1. `v1`: `round active pair only`
2. 같은 pair로 추론, pseudo-label 생성, local training을 모두 수행
3. 이후 round aggregation 뒤 새 pair를 다시 배포

## User

잠만 진행하기 전에 global embedding + 다른 local interpretation 이걸로 한다고 했는데 이것도 fl은 맞아?

## Assistant

아래처럼 정리했다.

1. local update를 중앙이 모아 global embedding을 다시 배포하면 FL 맞다
2. 다만 같은 query에 대해 agent별 raw embedding이 당장 달라지는 것은 아니다
3. v1에서는 같은 global embedding 위에서 local interpretation이 달라진다
4. v2에서 private adapter를 붙이면 embedding 자체도 약하게 달라질 수 있다

## User

그리고 개인이 입력하는 쿼리는 라벨링이 안되어 있으니까 신뢰도가 높은 것에는 프로토타입으로 카테고리를 붙이고 학습 진행하는것도 맞지?

## Assistant

맞다고 답했다.

v1 규칙:

1. `global embedding -> prototype score`
2. `score threshold + margin`으로 high-confidence pseudo-label 선택
3. accepted sample만 local training에 사용
4. low-confidence sample은 버린다

## User

오케이 1번부터 진행하고 v2에서 추가하자
그리고 BaselineService와 DecisionService 도 이제 다시 하는건가? 이건 어떻게 할건데?

## Assistant

`BaselineService`, `TimeSeriesAccumulator`, `DecisionPolicy`, `DecisionService`를 분리해 구현하는 방향을 설명했다.

## User

판단부분까지 지금 바로 하는게 좋은거야?
지금은 임베딩 fl 구조부터 만들어야 하는거 아니야?

## Assistant

완성형 조치 엔진까지는 아니지만, local interpretation 골격은 먼저 있어야 pseudo-label과 FL update의 기준이 생긴다고 설명했다.

## User

진행해

## Assistant

그 결과, local interpretation 경로를 먼저 구현했다.

추가된 것:

1. `BaselineService`
2. `TimeSeriesAccumulator`
3. `RuleBasedDecisionPolicy`
4. `DecisionService`
5. 관련 unit test

검증:

- `PYTHONPATH=. .venv/bin/pytest agent/tests/unit shared/tests/unit -q`
- 결과: `17 passed`

## User

그리고 지금은 들어온 쿼리에 대해 fl로 임베딩 하는거고 시계열로 누적해서 최종 판단은 어떻게 할지 고민이긴한데 그런 내용도 들어있지? 마음 이상 탐지 조치가 목표인것은 변함이 없어

## Assistant

그 방향이 문서에 반영되어 있다고 답했다.

## User

1. 그리고 fl로 인해 전역 공통 임베딩 공간이 업데이트 되는거지? 그러면서 전역파라미터+로컬 파라미터 같은 부분이 있는거고?
2. 남아있는 legacy 제거하고  BaselineService와 DecisionService를 실구현하는게 다음 순서인가?

## Assistant

1번은 `global shared parameter + local private state`로 설명했고,
2번은 legacy를 더 확장하지 않고 `BaselineService -> TimeSeriesAccumulator -> DecisionService -> FL update path` 순으로 가는 게 맞다고 정리했다.

## User

1. 결국 임베딩 모델은 각 에이전트마다 다르게 학습하는거지? non-iid? 같이
2. 다음 진행해야 하는데 나의 의견이 필요한 부분에는 나에게 먼저 묻고 진행해 혼자서 진행하지 말고, 그리고 단일 클래스로 해서 빠른 해결을 하는것이 아니라 갈아끼울 수 있는 미래지향적인 설계를 난 지행해

## Assistant

1. `non-IID` 전제로 보되, v1에서는 같은 global model에서 시작하고 local private state로 해석이 달라진다고 설명했다.
2. 사용자 판단이 필요한 항목은 먼저 묻고, 구현은 조합 가능한 구조로 가겠다고 답했다.

## User

WindowSummary/NormPack 완번히 버리는쪽으로 해서 전황안으로 진행하자

## Assistant

문서와 공용 타입을 새 방향으로 옮기고,
`WindowSummary/NormPack`은 active path에서 제외했다.

## User

그렇게 진행하자

## Assistant

이번 턴 구현 결과:

### 새 공용 도메인/contract

1. `shared/src/domain/entities/pseudo_label_candidate.py`
2. `shared/src/domain/entities/vector_adapter_state.py`
3. `shared/src/domain/entities/vector_adapter_delta.py`
4. `shared/src/contracts/adapter_contracts.py`

### agent 로컬 학습 경로

1. `agent/src/infrastructure/repositories/training_artifact_repository.py`
2. `agent/src/services/pseudo_label_service.py`
3. `agent/src/services/local_training_service.py`

구조:

1. `PseudoLabelSelectionService`
2. `LocalTrainingService`
3. `SyntheticVectorAdapterTrainingBackend`
4. `TrainingArtifactRepository`

### main-server FL coordinator 경로

1. `main-server/src/infrastructure/repositories/vector_adapter_state_repository.py`
2. `main-server/src/services/aggregation_service.py`
3. `main-server/src/services/round_manager_service.py`

구조:

1. `AggregationService`
2. `RoundManagerService`
3. `RoundPublication`

### simulation runner

1. `scripts/experiments/run_federated_simulation.py`

핵심:

1. `train` 데이터를 `bootstrap subset + client shard`로 분리
2. bootstrap subset으로 active `PrototypePack` 생성
3. client shard에서 pseudo-label 기반 local update 생성
4. 중앙에서 vector adapter delta 집계
5. 새 `model_revision + prototype_version` pair 발행
6. `validation`으로 round 결과 확인

### 테스트

추가:

1. `agent/tests/unit/test_local_training_service.py`
2. `main-server/tests/unit/test_round_manager_service.py`
3. `shared/tests/unit/test_adapter_contracts.py`
4. `tests/unit/test_run_federated_simulation.py`

검증:

1. `PYTHONPATH=. .venv/bin/pytest agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q`
2. 결과: `34 passed`

실데이터 smoke run:

1. `scripts/experiments/run_federated_simulation.py`
2. 입력: `data/processed/splits/ourafla_train_split.v1.train.jsonl`, `data/processed/splits/ourafla_train_split.v1.validation.jsonl`
3. 설정: `hash_debug`, `rounds=1`, 느슨한 threshold로 round 폐회로 확인
4. 결과:
   - `initial_model_revision=sim_rev_0000`
   - `initial_prototype_version=proto_sim_0000`
   - `final_model_revision=sim_rev_0001`
   - `final_prototype_version=proto_sim_0001`
   - `round_count=1`
## 2026-03-29 추가 기록: legacy summary/norm 코드 제거

### 사용자

지금 이제 사용 안하는 파일들이 아직 좀 많이 사용되는거 같은데 맞나?

버리기로 한거는 지우자

### 어시스턴트

`WindowSummary/NormPack` 축이 active code path에서 완전히 분리되지 않았다고 설명했다.
구체적으로는 아래 범주가 남아 있었다.

1. legacy 서비스/엔티티
   - `agent/src/services/windowing_service.py`
   - `main-server/src/services/norm_pack_service.py`
   - `shared/src/domain/entities/window_summary.py`
   - `shared/src/domain/entities/norm_pack.py`
2. old summary upload 진입점
   - `main-server/src/api/routers/norms.py`
   - `main-server/src/api/routers/sync.py`
   - `main-server/src/services/ingestion_service.py`
3. legacy 테스트/계약 placeholder
   - `agent/tests/unit/test_window_summary_builder.py`
   - `shared/src/contracts/sync_contracts.py`

사용자가 버리기로 한 것은 지우자고 명시적으로 요청했다.

### 적용한 변경

active code path에서 아래를 제거했다.

1. `main-server/src/api/routers/norms.py`
2. `main-server/src/api/routers/sync.py`
3. `main-server/src/services/ingestion_service.py`
4. `main-server/src/services/norm_pack_service.py`
5. `agent/src/services/windowing_service.py`
6. `agent/tests/unit/test_window_summary_builder.py`
7. `shared/src/domain/entities/window_summary.py`
8. `shared/src/domain/entities/norm_pack.py`
9. `shared/src/contracts/sync_contracts.py`

같이 정리한 부분:

1. `main-server/src/api/main.py`에서 legacy router mount 제거
2. `shared/src/domain/value_objects/schema_version.py`에서 legacy schema constant 제거
3. `plan.md`, `docs/execution_index.md`, `docs/project_execution_plan.md`에 code removal 반영

### 검증

다음 명령으로 회귀를 확인했다.

```bash
PYTHONPATH=. .venv/bin/pytest agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q
```

결과:

- `30 passed`

---

## 2026-03-30 실행 UX 개선

사용자는 `seed_prototypes.py` 같은 실행 명령에서
`--backend transformers_mxbai`와
`--embedding-model-id mixedbread-ai/mxbai-embed-large-v1`가
동시에 반복되는 것이 UX상 비효율적이라고 지적했다.

또한 이후 `mxbai`가 다른 실제 모델로 바뀔 수 있으니,
사용자 입장에서는 데이터 경로와 모델명만 바꾸면 되는 형태가 좋다고 요청했다.

이에 따라 실행 계층에 아래 원칙을 적용했다.

1. 사용자 실행 기본값은 `backend/model_id/revision` 개별 인자가 아니라 `embedding profile`로 받는다.
2. 정말 필요할 때만 low-level override로 `backend`, `embedding-model-id`, `embedding-model-revision`을 덮어쓴다.
3. config 기반 실험 스크립트도 같은 개념을 `embedding.profile`로 사용한다.
4. simulation처럼 publication용 model id와 base embedding model id가 다른 경우는 인자를 분리한다.

### 추가한 공통 모듈

새 파일:

- `scripts/common/embedding_profiles.py`

이 모듈에 아래를 넣었다.

1. `mxbai`, `hash_debug` preset registry
2. `ResolvedEmbeddingSettings`
3. `add_embedding_profile_arguments()`
4. `resolve_embedding_settings()`
5. `resolve_embedding_settings_from_args()`
6. `infer_embedding_profile()`

즉 이제 주요 스크립트는 공통적으로
`--embedding-profile mxbai`
또는
`--embedding-profile hash_debug`
를 기본 진입점으로 쓴다.

### 바꾼 주요 스크립트

1. `scripts/prototypes/seed_prototypes.py`
   - `--backend`, `--embedding-model-id`, `--embedding-model-revision` 직접 기본 노출을 없애고
     `--embedding-profile` 기반으로 전환
   - prototype 생성 기본 UX를 짧게 정리

2. `scripts/prototypes/evaluate_prototype_pack.py`
   - `--embedding-profile` 기반으로 전환
   - `PrototypePack`의 `embedding_model_id`를 기준으로 가능한 경우 profile을 추론해
     `hash_debug` pack 평가도 기본 동작이 덜 헷갈리게 보정

3. `scripts/experiments/train_softmax_classifier.py`
   - classifier baseline 실행도 `--embedding-profile` 기반으로 전환

4. `scripts/experiments/run_federated_simulation.py`
   - simulation 임베딩 경로도 `--embedding-profile` 기반으로 전환
   - 기존 `--model-id`는 사실 publication되는 active pair의 model id와
     base embedding model id를 혼동시키는 이름이었기 때문에
     `--published-model-id`로 의미를 분리하고 기존 `--model-id`는 alias로 유지

5. `scripts/experiments/configs/prototype_strategy/default.yaml`
6. `scripts/experiments/configs/prototype_threshold_sweep/default.yaml`
   - `embedding.backend`, `embedding.model_id`, `embedding.revision` 직접 명시 대신
     `embedding.profile: mxbai` 기준으로 해석하도록 변경

### 문서 갱신

`scripts/README.md`를 업데이트해서 아래를 문서화했다.

1. 공통 embedding profile 사용법
2. `mxbai`와 `hash_debug` preset 의미
3. 같은 backend 안에서 모델만 바꾸는 override 예시
4. prototype 생성 / 평가 / classifier baseline / federated simulation 실행 예시 단순화
5. config 기반 실험에서 `embedding.profile=...` override를 쓰는 방식

### 새 실행 예시

prototype 생성:

```bash
uv run python scripts/prototypes/seed_prototypes.py \
  --input-jsonl data/processed/splits/ourafla_train_split.v1.train.jsonl \
  --embedding-profile mxbai \
  --device cuda \
  --local-files-only
```

같은 backend 안에서 모델만 바꾸는 경우:

```bash
uv run python scripts/prototypes/seed_prototypes.py \
  --input-jsonl data/processed/splits/ourafla_train_split.v1.train.jsonl \
  --embedding-profile mxbai \
  --embedding-model-id intfloat/e5-large-v2 \
  --embedding-model-revision main \
  --device cuda \
  --local-files-only
```

federated simulation:

```bash
uv run python scripts/experiments/run_federated_simulation.py \
  --train-jsonl data/processed/splits/ourafla_train_split.v1.train.jsonl \
  --validation-jsonl data/processed/splits/ourafla_train_split.v1.validation.jsonl \
  --embedding-profile hash_debug \
  --published-model-id tracemind-embed-sim
```

### 검증

다음 명령으로 회귀를 확인했다.

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/pytest agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q
```

결과:

- `41 passed`

## 2026-03-30 추가 진행

### 사용자 결정

1. `single` 기준 threshold sweep도 실제로 돌려 비교한다.
2. 리팩터링은 전략+팩토리 조합을 늘리지 않고 진행한다.
3. pseudo-label 정책은 `precision-first` 성격의 보수적 기본값으로 옮긴다.

### 코드 변경

1. 기본 pseudo-label threshold를 보수적으로 조정했다.
   - `agent/src/services/training/pseudo_label_service.py`
     - `confidence_threshold: 0.6`
     - `margin_threshold: 0.02`
   - `main-server/src/services/rounds/round_manager_service.py`
     - training task 기본 objective도 `0.6 / 0.02`로 변경
   - `scripts/experiments/configs/prototype_strategy/default.yaml`
     - 실험 기본 threshold도 `0.6 / 0.02`로 변경
   - `scripts/experiments/run_federated_simulation.py`
     - CLI 기본 margin threshold를 `0.02`로 맞춤

2. placeholder API mount를 active app에서 제거했다.
   - `agent/src/api/main.py`
     - `assessment`, `ingest`, `training` router include 제거
   - `main-server/src/api/main.py`
     - `agents`, `fl_rounds` router include 제거

3. wide signature 축소를 위해 Request 객체를 추가했다.
   - `agent/src/services/training/local_training_service.py`
     - `LocalTrainingRequest`
     - `run(request)` 추가
     - 기존 `run_task(...)`는 wrapper로 유지
   - `main-server/src/services/rounds/round_manager_service.py`
     - `TrainingTaskRequest`
     - `RoundPublicationRequest`
     - `create_training_task(request)`
     - `publish_next_pair(request)`
   - `scripts/experiments/prototype_strategy/runner.py`
     - `PrototypeExperimentRequest`
   - `scripts/experiments/prototype_strategy/sweep.py`
     - `ThresholdSweepRequest`

4. threshold sweep selection policy를 보수적으로 변경했다.
   - `scripts/experiments/prototype_strategy/sweep.py`
   - `scripts/experiments/prototype_strategy/sweep_config.py`
   - `scripts/experiments/configs/prototype_threshold_sweep/default.yaml`
   - 새 기본은 `minimum_accepted_ratio = 0.5`
   - 그 범위를 만족하는 후보들 중 `accepted_accuracy`를 우선하고,
     그다음 `accepted_correct_ratio`, `accepted_ratio` 순으로 고른다.
   - 의도:
     - 너무 적은 샘플만 받아들이는 극단적 threshold는 제외
     - coverage보다 precision을 우선

### 단위 테스트

다음 명령으로 회귀를 확인했다.

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/pytest agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q
```

결과:

- `36 passed`

### 전체셋 GPU threshold sweep 재실행

#### 1. kmeans + conservative selection policy

실행:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. uv run python scripts/experiments/prototype_threshold_sweep.py \
  --override embedding.device=cuda \
  --override embedding.local_files_only=true
```

출력:

- `data/processed/evaluations/prototype_threshold_sweeps/20260329T163955Z`

결과:

- strategy: `kmeans`
- selected thresholds: `confidence >= 0.60`, `margin >= 0.02`
- validation
  - `accepted_count = 2492`
  - `accepted_ratio = 0.5023`
  - `accepted_accuracy = 0.8808`
  - `accepted_correct_ratio = 0.4425`
- test
  - `accepted_count = 462`
  - `accepted_ratio = 0.4657`
  - `accepted_accuracy = 0.8745`
  - `accepted_correct_ratio = 0.4073`

해석:

- 보수적 policy가 적용되면서 예전 `0.60 / 0.00` 대신 `0.60 / 0.02`가 선택됐다.
- 채택량은 줄었지만 accepted accuracy가 크게 올라갔다.

#### 2. single + conservative selection policy

실행:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. uv run python scripts/experiments/prototype_threshold_sweep.py \
  --override strategy.name=single \
  --override embedding.device=cuda \
  --override embedding.local_files_only=true
```

출력:

- `data/processed/evaluations/prototype_threshold_sweeps/20260329T164225Z`

결과:

- strategy: `single`
- selected thresholds: `confidence >= 0.60`, `margin >= 0.02`
- validation
  - `accepted_count = 2694`
  - `accepted_ratio = 0.5430`
  - `accepted_accuracy = 0.8441`
  - `accepted_correct_ratio = 0.4584`
- test
  - `accepted_count = 507`
  - `accepted_ratio = 0.5111`
  - `accepted_accuracy = 0.8580`
  - `accepted_correct_ratio = 0.4385`

해석:

- 보수적 policy 기준에서도 `single`은 `0.60 / 0.02`가 선택됐다.
- `kmeans`보다 purity는 낮지만 accepted sample volume은 더 많다.
- 따라서 self-training 관점에서는
  - `kmeans`: 더 깨끗한 pseudo-label
  - `single`: 더 많은 pseudo-label
  라는 trade-off가 보인다.

### 현재 해석

1. base classification quality는 여전히 `kmeans`가 우세하다.
   - 이전 strategy experiment validation accuracy:
     - `single = 0.7428`
     - `kmeans = 0.7593`
2. conservative pseudo-label selection까지 포함하면,
   - `kmeans`는 precision 쪽이 더 강하고
   - `single`은 accepted volume 쪽이 더 강하다.
3. 따라서 다음 제품/연구 판단은 아래 둘 중 하나다.
   - 초기 drift를 더 경계한다: `kmeans + 0.60 / 0.02`
   - data volume 확보를 더 중시한다: `single + 0.60 / 0.02`

현재까지는 사용자와 합의한 방향상 `보수적`이 우선이므로,
기본값은 `kmeans`보다는 threshold 정책 기준에서 `0.60 / 0.02`를 유지하는 것이 자연스럽다.

## 2026-03-30 문서화 추가

사용자 요청:

1. script README에 각 실험 실행 방법을 적어둘 것
2. 지금까지 실험한 내용 전체를 별도 md 파일로 정리할 것

수행 내용:

1. `scripts/README.md` 추가
   - `uv sync --extra dev --extra experiments`
   - `seed_prototypes.py`
   - `evaluate_prototype_pack.py`
   - `prototype_strategy_experiment.py`
   - `prototype_threshold_sweep.py`
   - `train_softmax_classifier.py`
   - `run_federated_simulation.py`
   - `run_local_demo.py` 현재 상태
   를 모두 실행 예시와 출력 경로 중심으로 정리했다.

2. `docs/experiment_results.md` 추가
   - single-centroid `PrototypePack` baseline
   - fixed embedding + softmax classifier head baseline
   - `single / kmeans / dbscan` prototype strategy comparison
   - coverage-first threshold sweep 결과
   - conservative threshold sweep 결과
   - smoke / federated simulation smoke 결과
   - 현재 권장 기본 조합
   를 한 문서에 정리했다.

의도:

1. 실행 방법 문서는 `scripts` 쪽에 둬서 실제 runbook 역할을 하게 한다.
2. 결과 요약 문서는 `docs` 쪽에 둬서 baseline/의사결정 참고 문서 역할을 하게 한다.

## Prototype 전략 비교 실험 추가

`single / k-means / DBSCAN` prototype 전략을
`train -> validation 선택 -> test 최종 평가` 구조로 비교하는
실험 runner를 추가했다.

### 결정 사항

1. `validation`은 agent 학습용이 아니라 전략/하이퍼파라미터 선택용으로 둔다.
2. `test`는 최종 보고용 holdout으로 유지한다.
3. `k-means`는 elbow가 아니라 `silhouette` 기반으로 `k=2..5` 범위에서 고른다.
4. `DBSCAN`은 `eps/min_samples` grid search와 coverage 조건으로 선택한다.
5. 시각화는 `PCA`와 `UMAP` 둘 다 저장하되, 최종 선택은 그림이 아니라 `validation` 성능으로 한다.

### 구현한 파일

1. `scripts/experiments/prototype_strategy_experiment.py`
   - `PrototypeBuildStrategy`
   - `SinglePrototypeStrategy`
   - `KMeansPrototypeStrategy`
   - `DbscanPrototypeStrategy`
   - `ProjectionService`
   - `StrategySelectionPolicy`
   - `PrototypeExperimentRunner`
2. `tests/unit/test_prototype_strategy_experiment.py`
   - single, k-means, DBSCAN, selection policy 단위 테스트

### 의존성 정리

실험 코드는 우회 구현하지 않고 `uv` extra로 명시했다.

`pyproject.toml`

1. `experiments`
   - `matplotlib`
   - `numpy`
   - `scikit-learn`
   - `umap-learn`

실행 기준:

```bash
uv sync --extra dev --extra experiments
```

### 검증

단위 테스트:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/pytest tests/unit/test_prototype_strategy_experiment.py -q
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/pytest agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q
```

결과:

1. prototype strategy unit test `3 passed`
2. 전체 회귀 `33 passed`

### smoke run

작은 stratified subset으로 두 번 확인했다.

1. `hash_debug` backend
   - 출력: `tmp/prototype_strategy_smoke/output/20260329T151656Z`
   - 선택 전략: `kmeans`
2. 실제 `mxbai` backend + GPU
   - 출력: `tmp/prototype_strategy_real_backend_smoke_gpu/20260329T152304Z`
   - 모델 로드 장치: `cuda`
   - 선택 전략: `dbscan`
   - validation:
     - `single accuracy=0.7000`
     - `kmeans accuracy=0.7000`
     - `dbscan accuracy=0.7000`
   - test:
     - `accuracy=0.5500`

산출물 구조는 공통적으로 아래를 포함한다.

1. `projections/train_pca.jsonl`
2. `projections/train_pca.png`
3. `projections/train_umap.jsonl`
4. `projections/train_umap.png`
5. `strategies/*.prototype_index.json`
6. `validation/*.json`
7. `summary.json`

### 주의

내가 중간에 `--device cpu`를 넣어 작은 smoke run을 시도했던 적이 있지만,
저장소 코드 기본값이 CPU로 고정된 것은 아니다.

실제 장치 선택은 `agent/src/infrastructure/runtime.py`의
`resolve_runtime_device(device=\"auto\")`가 담당하며,
CUDA가 보이면 `cuda`를 사용한다.

### 전체 split 본실험

실제 전체 split으로도 GPU 본실험을 수행했다.

실행:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python \
  scripts/experiments/prototype_strategy_experiment.py \
  --backend transformers_mxbai \
  --model-id mixedbread-ai/mxbai-embed-large-v1 \
  --revision main \
  --local-files-only \
  --cache-dir hf_cache \
  --device cuda \
  --batch-size 16
```

출력:

1. `data/processed/evaluations/prototype_strategy_experiments/20260329T152634Z`

validation 비교 결과:

1. `single`
   - accuracy: `0.7427937915742794`
   - accepted_ratio: `0.0`
   - prototype_count: `4`
   - by_category:
     - anxiety `1`
     - depression `1`
     - normal `1`
     - suicidal `1`
2. `kmeans`
   - accuracy: `0.759322717194114`
   - accepted_ratio: `0.0`
   - prototype_count: `8`
   - by_category:
     - anxiety `2`
     - depression `2`
     - normal `2`
     - suicidal `2`
3. `dbscan`
   - accuracy: `0.7397702076194316`
   - accepted_ratio: `0.0006047167909695626`
   - prototype_count: `19`
   - by_category:
     - anxiety `2`
     - depression `11`
     - normal `1`
     - suicidal `5`

선택 결과:

1. `selected_strategy = kmeans`

최종 test 평가:

1. accuracy: `0.7096774193548387`
2. accepted_ratio: `0.0`

해석:

1. 현재 validation 기준에서는 `single`보다 `kmeans`가 분명히 낫다.
2. `DBSCAN`은 prototype 수를 많이 늘렸지만 validation 성능은 오히려 떨어졌다.
3. 현재 pseudo-label threshold (`score >= 0.8`, `margin >= 0.15`)는 매우 보수적이라 accepted ratio가 거의 0이다.
4. 따라서 다음 실험은 prototype 전략 선택과 pseudo-label threshold 탐색을 분리해서 보는 편이 맞다.

## Prototype 전략 실험 구조 리팩터링

사용자는 아래 세 가지 문제를 지적했다.

1. `prototype_strategy_experiment.py`가 너무 길다.
2. `parser.add_argument(...)`가 너무 많아 실수 유발 가능성이 크다.
3. `uv run script.py`로 실행하는데 굳이 두꺼운 CLI가 필요한지 의문이다.

이에 따라 구조를 아래처럼 바꿨다.

### 결정

1. CLI를 없애지 않는다.
2. 대신 CLI는 `--config`, `--override key=value`만 받는 얇은 형태로 줄인다.
3. 실제 source of truth는 YAML config와 typed dataclass config로 옮긴다.
4. 기존 실행 습관을 깨지 않기 위해 `scripts/experiments/prototype_strategy_experiment.py`는 thin entrypoint로 유지한다.

### 새 구조

`scripts/experiments/prototype_strategy/`

1. `config.py`
2. `cli.py`
3. `models.py`
4. `io_utils.py`
5. `strategies.py`
6. `projection.py`
7. `evaluation.py`
8. `runner.py`
9. `main.py`
10. `__init__.py`

추가 설정 파일:

1. `scripts/experiments/configs/prototype_strategy/default.yaml`

### 의미

1. `uv run scripts/experiments/prototype_strategy_experiment.py`는 그대로 가능하다.
2. 다만 실험 기본값은 더 이상 `parse_args()` 코드가 아니라 YAML config에 있다.
3. 필요할 때만 `--override embedding.device=cuda` 같은 식으로 덮어쓴다.

### 검증

단위 테스트:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/pytest tests/unit/test_prototype_strategy_experiment.py -q
```

결과:

1. `3 passed`

wrapper smoke run:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python \
  scripts/experiments/prototype_strategy_experiment.py \
  --override embedding.backend=hash_debug \
  --override embedding.model_id=hash_debug \
  --override embedding.device=cpu \
  --override embedding.local_files_only=true \
  --override dataset.train_jsonl=tmp/prototype_strategy_smoke/train.jsonl \
  --override dataset.validation_jsonl=tmp/prototype_strategy_smoke/validation.jsonl \
  --override dataset.test_jsonl=tmp/prototype_strategy_smoke/test.jsonl \
  --override output.output_dir=tmp/prototype_strategy_refactor_smoke
```

전체 회귀:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/pytest \
  agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q
```

결과:

1. `33 passed`

## Threshold Sweep 실험 추가

사용자는 `3-class merge ablation`보다 먼저
`threshold sweep`을 진행하기로 결정했다.

이유:

1. 현재 prototype 전략은 `kmeans`로 정리됐지만,
2. pseudo-label 기준이 너무 보수적이라 `accepted_ratio`가 거의 `0`이고,
3. 이 상태에서는 FL용 pseudo-label 루프가 사실상 열리지 않기 때문이다.

### 구현

새 파일:

1. `scripts/experiments/prototype_strategy/sweep_config.py`
2. `scripts/experiments/prototype_strategy/sweep.py`
3. `scripts/experiments/prototype_strategy/sweep_main.py`
4. `scripts/experiments/prototype_threshold_sweep.py`
5. `scripts/experiments/configs/prototype_threshold_sweep/default.yaml`
6. `tests/unit/test_prototype_threshold_sweep.py`

추가로 기존 평가 계층도 확장했다.

1. `scripts/experiments/prototype_strategy/models.py`
   - `ScoredPrediction`
   - `EvaluationMetrics.accepted_count`
   - `EvaluationMetrics.accepted_accuracy`
   - `EvaluationMetrics.accepted_correct_ratio`
2. `scripts/experiments/prototype_strategy/evaluation.py`
   - `score_embeddings()`
   - `evaluate_scored_predictions()`

### sweep 방식

선택된 strategy 하나를 고정하고,
`confidence_threshold x margin_threshold` grid를 돈다.

현재 기본은:

1. strategy: `kmeans`
2. confidence:
   - `0.60`
   - `0.65`
   - `0.70`
   - `0.75`
   - `0.80`
3. margin:
   - `0.00`
   - `0.01`
   - `0.02`
   - `0.03`
   - `0.05`
   - `0.10`
   - `0.15`

selection policy는 다음 순서로 정렬한다.

1. `accepted_correct_ratio`
2. `accepted_accuracy`
3. `accepted_ratio`
4. threshold 보수성

즉 "전체 validation 중 올바르게 pseudo-label로 받아들일 수 있는 샘플 비율"을 우선 본다.

### 검증

단위 테스트:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/pytest \
  tests/unit/test_prototype_strategy_experiment.py \
  tests/unit/test_prototype_threshold_sweep.py -q
```

결과:

1. `5 passed`

전체 회귀:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/pytest \
  agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q
```

결과:

1. `35 passed`

### smoke run

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python \
  scripts/experiments/prototype_threshold_sweep.py \
  --override embedding.backend=hash_debug \
  --override embedding.model_id=hash_debug \
  --override embedding.device=cpu \
  --override embedding.local_files_only=true \
  --override dataset.train_jsonl=tmp/prototype_strategy_smoke/train.jsonl \
  --override dataset.validation_jsonl=tmp/prototype_strategy_smoke/validation.jsonl \
  --override dataset.test_jsonl=tmp/prototype_strategy_smoke/test.jsonl \
  --override output.output_dir=tmp/prototype_threshold_sweep_smoke \
  --override threshold_grid.confidence_thresholds=[0.1,0.2,0.3,0.4] \
  --override threshold_grid.margin_thresholds=[0.0,0.01,0.05]
```

결과:

1. output:
   - `tmp/prototype_threshold_sweep_smoke/20260329T155742Z`
2. selected:
   - `confidence >= 0.20`
   - `margin >= 0.01`

### 전체 split GPU 본실험

실행:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. .venv/bin/python \
  scripts/experiments/prototype_threshold_sweep.py \
  --override embedding.device=cuda \
  --override embedding.local_files_only=true
```

출력:

1. `data/processed/evaluations/prototype_threshold_sweeps/20260329T155803Z`

선택 결과:

1. strategy:
   - `kmeans`
2. selected thresholds:
   - `confidence >= 0.60`
   - `margin >= 0.00`

validation selected metrics:

1. accepted_count: `4598`
2. accepted_ratio: `0.9268`
3. accepted_accuracy: `0.7462`
4. accepted_correct_ratio: `0.6916`

test selected metrics:

1. accepted_count: `966`
2. accepted_ratio: `0.9738`
3. accepted_accuracy: `0.7039`
4. accepted_correct_ratio: `0.6855`

top validation 후보 예시:

1. `0.60 / 0.00`
   - accepted_ratio `0.9268`
   - accepted_accuracy `0.7462`
2. `0.60 / 0.01`
   - accepted_ratio `0.6888`
   - accepted_accuracy `0.8273`
3. `0.60 / 0.02`
   - accepted_ratio `0.5023`
   - accepted_accuracy `0.8808`
4. `0.60 / 0.03`
   - accepted_ratio `0.3755`
   - accepted_accuracy `0.9216`

해석:

1. 현재 selection policy는 coverage보다 "올바르게 받아들일 수 있는 pseudo-label 총량"을 더 중시한다.
2. 그래서 `0.60 / 0.00`이 선택됐다.
3. 하지만 precision을 더 중시하면 `0.60 / 0.02` 또는 `0.60 / 0.03`도 유력한 운영 후보가 될 수 있다.
4. 따라서 다음 단계는 threshold를 하나로 확정하기보다, 운영 목적에 따라 `coverage-first`와 `precision-first` 정책을 분리해 보는 것이다.

## 2026-03-29 추가 기록: shared inference entity 병합

### 사용자

shared에서 inference밑에 각 파일들에서 한파일에 한클래스만 있는게 문제 없나?

진행해

### 어시스턴트

서비스 클래스는 그대로 두고, `shared/src/domain/entities/inference`의 작은 dataclass만 병합하는 편이 더 읽기 좋다고 설명했다.

선택한 방향:

1. `state.py`
   - `BaselineProfile`
   - `PersonalizationState`
   - `TimeSeriesState`
2. `events.py`
   - `QueryEvent`
   - `ScoredEvent`
3. `result.py`
   - `AssessmentResult`

### 적용한 변경

기존 파일:

1. `assessment_result.py`
2. `baseline_profile.py`
3. `personalization_state.py`
4. `query_event.py`
5. `scored_event.py`
6. `time_series_state.py`

위 파일들을 제거하고 아래 세 파일로 재구성했다.

1. `shared/src/domain/entities/inference/result.py`
2. `shared/src/domain/entities/inference/state.py`
3. `shared/src/domain/entities/inference/events.py`

같이 정리한 부분:

1. `shared/src/domain/entities/inference/__init__.py`
2. `shared/src/domain/policies/decision_policy.py`
3. `agent` inference/training service import
4. 관련 unit test import
5. `scripts/experiments/run_federated_simulation.py`

### 검증

```bash
PYTHONPATH=. .venv/bin/pytest agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q
```

결과:

- `30 passed`

추가 확인:

- 코드 경로 기준 `window_summary`, `norm_pack`, `sync_contracts`, `norms_router`, `IngestionService` 참조는 제거되었고,
- 남은 언급은 보관 문서와 실행 계획 설명뿐이다.

## 2026-03-29 추가 기록: active path 폴더 구조 리팩터링

### 사용자

근데 이거 너무 클래스 하나당 파일 하나로 되어 있어서 찾기가 더 어려운거 아냐?
또는 세부 폴더를 추가해야 하는거 아냐? 병합하는게 낫지 않나

리펙토링 진행해

### 어시스턴트

단순 병합보다 흐름별 하위 폴더로 재배치하는 편이 안전하다고 설명했다.

선택한 방향:

1. 서비스는 합치지 않고 기능별 패키지로 묶는다.
2. active path 기준으로 `inference / training / prototype / rounds / prototypes`를 만든다.
3. shared entity도 `inference / training / artifacts`로 나눈다.
4. 동작은 바꾸지 않고 import와 테스트만 맞춘다.

### 적용한 구조 변경

`agent/src/services/`

1. `inference/`
   - `baseline_service.py`
   - `time_series_service.py`
   - `decision_service.py`
   - `scoring_service.py`
2. `training/`
   - `pseudo_label_service.py`
   - `local_training_service.py`
3. `prototype/`
   - `runtime_service.py`
   - `sync_service.py`

`main-server/src/services/`

1. `rounds/`
   - `aggregation_service.py`
   - `round_manager_service.py`
2. `prototypes/`
   - `prototype_pack_service.py`
   - `prototype_build_state_service.py`

`shared/src/domain/entities/`

1. `inference/`
   - `assessment_result.py`
   - `baseline_profile.py`
   - `personalization_state.py`
   - `query_event.py`
   - `scored_event.py`
   - `time_series_state.py`
2. `training/`
   - `decision_feedback_signal.py`
   - `pseudo_label_candidate.py`
   - `training_task.py`
   - `training_update.py`
   - `vector_adapter_delta.py`
   - `vector_adapter_state.py`
3. `artifacts/`
   - `model_manifest.py`
   - `prototype_pack.py`
   - `labeled_query.py`

추가로 각 새 폴더에 `__init__.py`를 두고 export를 정리했다.

### 갱신한 참조

다음 종류의 import를 모두 새 경로로 바꿨다.

1. agent API/router
2. main-server router/service
3. unit tests
4. prototype 관련 scripts
5. federated simulation script
6. `shared/src/domain/policies/decision_policy.py`

### 검증

다음 명령으로 회귀를 확인했다.

```bash
PYTHONPATH=. .venv/bin/pytest agent/tests/unit main-server/tests/unit shared/tests/unit tests/unit -q
```

결과:

- `30 passed`
