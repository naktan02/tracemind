# Experiment Dashboard Source

이 앱은 `scripts/workflows/result_index`가 export한 정적 JSON bundle을 읽는
research/developer UI다. metric 의미와 result schema의 source of truth는
`scripts/workflows/result_index`에 있고, 이 폴더는 화면 조합과 로컬 표시 상태만
소유한다.

## 읽기 경로

1. `main.js`: 데이터 로딩, 이벤트 바인딩, track/page 렌더 순서.
2. `features/central_ssl/`: 중앙 SSL control 화면 상태, 필터, selector, 페이지 UI.
3. `features/fl_ssl/`: FL SSL main comparison 화면 상태, 필터, selector, 페이지 UI.
4. `ui/`: track에 묶이지 않는 chart, control, table primitive.
5. `shared/`: HTML escape, 숫자/metric formatting, faceted filter 같은 공통 규칙.
6. `data/`: legacy bundle field 호환 정규화와 JSON 로딩.

## 경계

- `features/central_ssl/logic`과 `features/fl_ssl/logic`은 각 track의 run 해석과 필터 축을 소유한다.
- `features/central_ssl/ui`와 `features/fl_ssl/ui`는 HTML fragment 조합만 맡고, bundle schema 의미를
  새로 정의하지 않는다.
- `ui/`와 `shared/`는 두 track에서 같은 의미로 반복되는 작은 primitive만 둔다.
- localStorage는 alias/color 같은 표시 preference만 저장하며 실험 결과의 source of
  truth가 아니다.
