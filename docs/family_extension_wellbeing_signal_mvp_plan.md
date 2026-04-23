# Family Extension Wellbeing Signal MVP Plan

## 목적

이 문서는 `아이용 화면 + 부모용 PIN 화면 + 부모용 상세 화면`으로 구성된
크롬 확장 프로그램 MVP를 위한 실행 계획이다.

중요:

- 이 문서는 현재 활성 FL/실험 실행 계획을 대체하지 않는다.
- `docs/project_execution_plan.md`의 seed/adaptation/FL 레일과는 별도의
  product-surface 계획이다.
- MVP 범위는 `확장 프로그램 + 로컬 프로그램 API`까지만 본다.
- 별도 부모 웹 대시보드는 future expansion으로 미룬다.

## 이름과 용어

외부 UI와 계약에서는 `risk` 대신 `wellbeing_signal`을 사용한다.

이유:

- `risk`는 과도하게 진단/위험 판정처럼 들릴 수 있다.
- 현재 목표는 내부 feature를 직접 노출하지 않고,
  부모/아이에게 `현재 상태 + 최근 변화`만 전달하는 것이다.

권장 용어:

- 코드/계약: `wellbeing_signal`
- UI 문구: `현재 상태`, `최근 변화`, `최근 추이`

## MVP 범위

MVP에서 포함하는 것:

1. 크롬 확장 프로그램
   - 아이용 기본 화면
   - 부모용 PIN 잠금 화면
   - 부모용 상세 화면
2. 로컬 프로그램 API
   - summary 조회
   - timeseries 조회
   - PIN unlock
   - health check
3. 로컬 저장
   - wellbeing snapshot
   - PIN hash
   - 최소 설정값

MVP에서 하지 않는 것:

1. 별도 부모 웹 대시보드
2. 카테고리별 점수 공개
3. LLM 대화
4. 다중 자녀 프로필
5. 클라우드 동기화
6. 고급 알림/리포트 export

## 제품 화면 구분

### 아이용 화면

- 확장 프로그램 기본 진입 화면
- 현재 상태를 짧고 부담 없이 보여준다
- 숫자/세부 이유보다 메시지와 행동 제안을 우선한다

### 부모용 화면

- 확장 프로그램 안의 보호된 상세 화면
- `PIN` 입력 후 진입
- 전체 wellbeing signal, 최근 추이, 권장 행동을 보여준다

### 별도 웹 화면

- MVP에서는 만들지 않는다
- 나중에 필요하면 장기 통계/리포트/LLM 대화용 부모 대시보드로 분리한다

즉 MVP는:

```text
Chrome Extension
  ├─ Child View
  ├─ Parent Unlock
  └─ Parent Detail

Local Program
  ├─ Wellbeing Summary API
  ├─ Timeseries API
  └─ Parent Auth API
```

## 설계 원칙

1. 외부 UI는 내부 카테고리 구조를 알지 않아야 한다.
2. 외부 UI는 `전체 wellbeing signal`만 본다.
3. 내부 판단 엔진이 바뀌어도 UI 계약은 가능한 유지한다.
4. 확장 프로그램은 로컬 프로그램에 질의만 하고 판단 코어를 소유하지 않는다.
5. 아이용과 부모용은 같은 앱 안에 두되, 권한과 정보량을 분리한다.

## Canonical Output Contract

MVP는 두 개의 읽기 계약과 하나의 인증 계약만 먼저 고정한다.

### 1. `WellbeingSignalSummaryPayload`

역할:
- 확장 프로그램의 현재 상태 카드 source of truth

권장 필드:

- `schema_version`
- `computed_at`
- `signal_score`
- `signal_level`
- `signal_label`
- `trend`
- `summary`
- `action_tip`
- `confidence`
- `low_data`

예시:

```json
{
  "schema_version": "wellbeing_signal_summary.v1",
  "computed_at": "2026-04-24T10:30:00Z",
  "signal_score": 72,
  "signal_level": "high",
  "signal_label": "주의 필요",
  "trend": "rising",
  "summary": "최근 상태가 평소보다 높게 관찰되었습니다.",
  "action_tip": "오늘은 사용 시간을 줄이고 짧게 대화를 시도해 보세요.",
  "confidence": "medium",
  "low_data": false
}
```

### 2. `WellbeingSignalTimeseriesPayload`

역할:
- 부모용 상세 화면의 선 그래프 source of truth

권장 필드:

- `schema_version`
- `range`
- `points`

point 필드:

- `ts`
- `signal_score`

예시:

```json
{
  "schema_version": "wellbeing_signal_timeseries.v1",
  "range": "7d",
  "points": [
    {"ts": "2026-04-20", "signal_score": 34},
    {"ts": "2026-04-21", "signal_score": 39},
    {"ts": "2026-04-22", "signal_score": 51},
    {"ts": "2026-04-23", "signal_score": 61},
    {"ts": "2026-04-24", "signal_score": 72}
  ]
}
```

### 3. `ParentUnlock` Contract

역할:
- 부모용 화면 진입 보호

요청:

- `pin`

응답:

- `granted`
- `session_expires_at`
- `remaining_attempts`
- `locked_until`

## 로컬 프로그램 API

MVP에서 필요한 API는 아래 네 개다.

### `GET /api/v1/wellbeing/summary`

- 현재 wellbeing signal 한 건 반환
- 아이용/부모용 공통

### `GET /api/v1/wellbeing/timeseries?range=7d|14d|30d`

- 부모용 상세 그래프용

### `POST /api/v1/parent/unlock`

- PIN 검증
- 부모용 세션 발급

### `GET /api/v1/system/health`

- 확장 프로그램이 로컬 프로그램 연결 여부를 확인하는 용도

## 저장 구조

MVP는 `SQLite`로 충분하다.

권장 테이블:

### `wellbeing_snapshots`

- `computed_at`
- `signal_score`
- `signal_level`
- `signal_label`
- `trend`
- `summary`
- `action_tip`
- `confidence`
- `low_data`

### `parent_auth`

- `pin_hash`
- `failed_attempt_count`
- `locked_until`
- `updated_at`

### `wellbeing_settings`

- 기본 range
- parent session ttl
- 기타 최소 설정

## 확장 프로그램 라우트

권장 라우트:

- `/child`
- `/unlock`
- `/parent`

기본 흐름:

1. 확장 열면 `/child`
2. `부모용 보기` 누르면 `/unlock`
3. PIN 통과 시 `/parent`

## 화면 명세

### Child View

목표:
- 아이가 상태를 부담 없이 이해하게 한다

구성:

1. 현재 상태 카드
   - `signal_label`
   - 짧은 메시지
2. 한 줄 요약
   - `summary`
3. 행동 제안
   - `action_tip`
4. 마지막 업데이트 시각
5. `부모용 보기` 버튼

하지 않는 것:

- 세부 수치 강조
- 그래프
- 카테고리별 설명
- 모델/근거 세부 설명

### Parent Unlock View

목표:
- 부모용 정보 접근 보호

구성:

1. PIN 입력 필드 또는 keypad
2. 실패 메시지
3. 남은 시도 횟수
4. 잠금 상태 메시지

기본 정책:

- 4~6자리 PIN
- 5회 실패 시 일정 시간 잠금
- 세션 10~15분 유지

### Parent Detail View

목표:
- 부모가 `현재 상태 + 최근 변화 + 대응`을 한눈에 보게 한다

구성:

1. 현재 상태 카드
   - `signal_label`
   - `signal_score`
   - `trend`
2. 전체 signal 추이 그래프
   - `7d / 14d / 30d`
   - 단일 선 그래프
3. 요약 설명
   - `summary`
4. 권장 행동
   - `action_tip`
5. 데이터 부족 상태
   - `low_data`
6. 마지막 업데이트 시각

그래프 규칙:

- 카테고리별 그래프는 넣지 않는다
- `0~100` 단일 축
- 필요하면 배경에 low/medium/high band만 둔다

## 폴더 소유 경계

### `shared`

역할:
- 확장/로컬 프로그램이 같이 쓰는 canonical contract

후보 파일:

- `shared/src/contracts/wellbeing_signal_contracts.py`

### `agent`

역할:
- 로컬 프로그램 API
- snapshot 저장/조회
- parent auth
- 실제 판단 결과를 summary/timeseries로 번역

후보 경로:

- `agent/src/services/wellbeing/`
- `agent/src/api/extension_routes.py`
- `agent/src/infrastructure/repositories/`

### `apps/family_extension`

역할:
- 아이용/부모용 확장 프로그램 UI

권장 구조:

```text
apps/family_extension/
  src/
    pages/
      child/
      unlock/
      parent/
    components/
    api/
    contracts/
    lib/
```

## 구현 순서

### Phase 1. 출력 계약 고정

작업:

1. `shared`에 `wellbeing_signal` 계약 추가
2. unit test 추가
3. mock JSON 기준 serialize/validate 닫기

완료 기준:

- 확장 UI가 이 계약만 보고 화면을 만들 수 있다

### Phase 2. 로컬 mock API

작업:

1. `summary`
2. `timeseries`
3. `unlock`
4. `health`

완료 기준:

- 실제 엔진 없이도 확장 UI 개발 가능

### Phase 3. 로컬 저장 구조

작업:

1. snapshot repository
2. parent auth repository
3. settings repository

완료 기준:

- timeseries 조회 가능
- PIN 저장 가능

### Phase 4. 확장 프로그램 shell

작업:

1. `apps/family_extension` scaffold
2. manifest 구성
3. child/unlock/parent route 구성

완료 기준:

- 확장 로드 가능
- 라우트 전환 가능

현재 상태:

- 완료
- popup entry는 `apps/family_extension/index.html`
- 부모 상세 entry는 `apps/family_extension/parent.html`
- manifest source는 `apps/family_extension/public/manifest.json`
- 공통 route shell은 `apps/family_extension/src/App.tsx`

### Phase 5. 아이용 화면 MVP

작업:

1. 상태 카드
2. 요약 문구
3. 행동 제안
4. 부모용 보기 버튼

완료 기준:

- summary mock으로 child view 완성

현재 상태:

- 완료
- `apps/family_extension/src/pages/child/ChildPage.tsx`가 `/api/v1/wellbeing/summary`를 읽는다
- `apps/family_extension/src/components/WellbeingSignalCard.tsx`가 현재 상태 카드를 렌더링한다
- 프런트 타입은 `apps/family_extension/src/contracts/generated.ts`로 생성한다

### Phase 6. 부모용 잠금 화면

작업:

1. PIN 입력
2. 실패 처리
3. 세션 처리

완료 기준:

- unlock API와 연결 가능

현재 상태:

- 완료
- `apps/family_extension/src/hooks/useParentUnlock.ts`가 `/api/v1/parent/unlock`을 호출한다
- `apps/family_extension/src/pages/unlock/UnlockPage.tsx`가 남은 시도 횟수와 잠금 상태를 보여준다
- `apps/family_extension/src/pages/parent/ParentPage.tsx`는 유효한 부모 세션이 없으면 보호 화면으로 남는다

### Phase 7. 부모용 상세 화면 MVP

작업:

1. 현재 상태 카드
2. line chart
3. range toggle
4. 권장 행동/요약/데이터 부족 표시

완료 기준:

- summary + timeseries mock으로 parent view 완성

현재 상태:

- 완료
- `apps/family_extension/src/pages/parent/ParentPage.tsx`가 부모 세션이 유효할 때 summary와 timeseries를 함께 읽는다
- `apps/family_extension/src/components/ParentWellbeingSummaryCard.tsx`가 부모용 현재 상태 카드를 렌더링한다
- `apps/family_extension/src/components/WellbeingSignalTrendChart.tsx`가 7d / 14d / 30d range toggle과 단일 선 그래프를 렌더링한다

### Phase 8. 연결 실패/오프라인 상태 처리

작업:

1. health polling
2. offline banner
3. stale data 표시
4. low-data 문구 분리

완료 기준:

- 로컬 프로그램이 꺼져 있어도 UX가 명확하다

### Phase 9. 실제 엔진 연결

작업:

1. 내부 판단 결과 -> `WellbeingSignalSummaryPayload`
2. snapshot 누적 저장
3. timeseries를 실제 데이터 기반으로 교체

완료 기준:

- UI를 바꾸지 않고 mock을 실제 데이터로 대체 가능

## 파일 단위 체크리스트

### `shared`

- `shared/src/contracts/wellbeing_signal_contracts.py`
- `shared/tests/unit/test_wellbeing_signal_contracts.py`

### `agent`

- `agent/src/api/extension_routes.py`
- `agent/src/services/wellbeing/summary_service.py`
- `agent/src/services/wellbeing/timeseries_service.py`
- `agent/src/services/wellbeing/auth_service.py`
- `agent/src/infrastructure/repositories/wellbeing_snapshot_repository.py`
- `agent/src/infrastructure/repositories/parent_auth_repository.py`
- `agent/src/infrastructure/repositories/wellbeing_settings_repository.py`
- `agent/tests/unit/test_extension_routes.py`

### `apps/family_extension`

- `apps/family_extension/manifest.json`
- `apps/family_extension/src/App.tsx`
- `apps/family_extension/src/pages/child/ChildPage.tsx`
- `apps/family_extension/src/pages/unlock/UnlockPage.tsx`
- `apps/family_extension/src/pages/parent/ParentPage.tsx`
- `apps/family_extension/src/components/RiskStatusCard.tsx`
- `apps/family_extension/src/components/RiskTrendChart.tsx`
- `apps/family_extension/src/components/PinPad.tsx`
- `apps/family_extension/src/api/client.ts`
- `apps/family_extension/src/api/wellbeing.ts`

## 완료 기준

MVP는 아래가 되면 닫는다.

1. 확장을 열면 아이용 화면이 뜬다.
2. 부모용 상세는 PIN 없이는 못 들어간다.
3. 부모용에서 전체 wellbeing signal 추이를 볼 수 있다.
4. 내부 카테고리 점수는 외부에 보이지 않는다.
5. 로컬 프로그램 연결 실패가 명확히 드러난다.
6. 내부 판단 엔진이 바뀌어도 UI 계약은 유지된다.

## 미래 확장

MVP 이후에만 검토한다.

1. 별도 부모 웹 대시보드
2. 장기 통계
3. 리포트 export
4. LLM 대화
5. 다중 자녀 프로필
6. 클라우드 동기화

## 참고 자료

현재 mockup reference:

- `docs/mockups/family_report_ui/index.html`
- `docs/mockups/family_report_ui/child_support.html`
- `docs/mockups/family_report_ui/parent_dashboard.html`

이 mockup들은 시각 참고용이다.
실제 source of truth는 이 문서와 이후 추가할 `shared` contract로 본다.
