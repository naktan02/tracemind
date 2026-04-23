# Family Extension Wellbeing Signal MVP Plan

## 목적

이 문서는 `family_extension`의 현재 MVP 구조와 다음 확장 우선순위를 정리한다.

중요:

- 이 문서는 실험 실행 레일이 아니라 product-surface 계획이다.
- 현재 범위는 `이 PC의 로컬 agent만 사용하는 크롬 확장 프로그램`이다.
- 별도 부모 웹 대시보드와 원격 agent 선택 UI는 아직 열지 않는다.

## 현재 canonical 구조

현재 MVP는 아래 흐름을 source of truth로 본다.

```text
Setup
  -> Gate
    -> Child Unlock -> Child View
    -> Parent Unlock -> Parent Detail
```

핵심 원칙:

1. 외부 UI는 내부 카테고리 점수를 노출하지 않는다.
2. 외부 UI는 `wellbeing_signal` 결과만 소비한다.
3. child와 parent는 모두 잠금 경계를 거친다.
4. 현재는 `this_device_only` access mode만 지원한다.
5. role 화면을 벗어나면 즉시 세션을 폐기한다.

## Canonical Contract

### `WellbeingSignalSummaryPayload`

- child/parent 공통 현재 상태 카드 source of truth

주요 필드:

- `computed_at`
- `signal_score`
- `signal_level`
- `signal_label`
- `trend`
- `summary`
- `action_tip`
- `confidence`
- `low_data`

### `WellbeingSignalTimeseriesPayload`

- parent detail의 추이 그래프 source of truth

주요 필드:

- `range`
- `points[].ts`
- `points[].signal_score`

### `FamilyAccess*`

- app-level setup/auth source of truth

주요 payload:

- `FamilySetupStatusPayload`
- `FamilySetupRequestPayload`
- `FamilySetupResponsePayload`
- `FamilyUnlockRequestPayload`
- `FamilyUnlockResponsePayload`

## 로컬 프로그램 API

현재 필요한 API:

- `GET /api/v1/family/setup/status`
- `POST /api/v1/family/setup`
- `POST /api/v1/family/unlock`
- `GET /api/v1/wellbeing/summary`
- `GET /api/v1/wellbeing/timeseries?range=7d|14d|30d`
- `GET /api/v1/system/health`

호환성 경로:

- `POST /api/v1/parent/unlock`
  - 기존 parent-only unlock을 유지하지만 내부적으로는 `family_access parent role`에 위임한다.

## 저장 구조

현재 SQLite source of truth:

- `wellbeing_snapshots`
  - wellbeing summary snapshot 저장
- `family_access_profiles`
  - child/parent role별 PIN hash, 실패 횟수, 잠금 시각
- `wellbeing_settings`
  - role별 session/lock/attempt 기본값

## 확장 라우트

현재 라우트:

- `/setup`
- `/gate`
- `/child/unlock`
- `/parent/unlock`
- `/child`
- `/parent`

가시성 규칙:

- setup 전: `/setup`만 보인다
- setup 완료 후 잠금 상태: `/gate`만 보인다
- child 로그인 중: `/child`만 보인다
- parent 로그인 중: `/parent`만 보인다
- unlock route는 탭이 아니라 진입 흐름으로만 사용한다

## 현재 구현 완료 범위

1. `wellbeing_signal` shared contract
2. `family_access` shared contract
3. local wellbeing mock API에서 실제 projection 기반 API로 전환
4. wellbeing SQLite 저장 구조
5. family access SQLite 저장 구조
6. extension shell
7. child summary view
8. parent detail view
9. health/low-data/stale-data 상태 표시
10. local-only setup + gate + role별 unlock

## 아직 하지 않는 것

1. 원격 agent 페어링/선택
2. 별도 부모 웹 대시보드
3. 다중 자녀 프로필
4. LLM 대화
5. persisted personalization state와의 정밀 연결
6. child/parent별 세션 정책 세분화

## 다음 우선순위

현재 MVP는 한번 멈추고 실제 사용 흐름을 보는 단계까지 왔다.

다음 후보는 아래 둘 중 하나다.

1. `personalization state`를 wellbeing projection에 더 정밀하게 연결
2. `empty-state / no-data-state`를 mock 없이 더 명시적으로 다듬기

즉 다음 확장은 새 화면 추가보다 `로컬 해석 결과의 품질과 의미를 더 정밀하게 연결하는 작업`이 우선이다.
