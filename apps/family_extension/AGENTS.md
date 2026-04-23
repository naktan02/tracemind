# Family Extension AGENTS

## 역할

`apps/family_extension`는 아이용 화면과 부모용 보호 화면을 담는
크롬 확장 프로그램 UI shell이다.

중요:

- 이 앱은 wellbeing 계산 owner가 아니다.
- 출력 의미는 `shared/src/contracts/wellbeing_signal_contracts.py`와
  `agent`의 `/api/v1/wellbeing/*`가 소유한다.
- 확장 프로그램은 로컬 프로그램 결과를 소비하고 보여주는 역할만 맡는다.

## 변경 규칙

- 아이용 화면과 부모용 화면의 정보량을 섞지 않는다.
- 카테고리별 내부 점수나 실험용 metadata를 UI에 직접 노출하지 않는다.
- `wellbeing_signal` 계약 의미를 프런트에서 다시 정의하지 않는다.
- popup shell과 부모 상세 entry를 구분하되, 공통 App shell은 재사용 가능하게 둔다.
- 로컬 프로그램 연결 실패 상태를 숨기지 않는다.

## 구현 스타일

- `React + TypeScript + Vite`를 기준으로 유지한다.
- shell 단계에서는 route, manifest, API client 경계를 먼저 세운다.
- heavy styling framework보다 얇은 CSS와 명시적 layout을 우선한다.
- 앱 안에서 inference/training 로직을 추가하지 않는다.
