# Code Expression Guidelines

이 문서는 구조 자체가 아니라 "코드를 읽는 표면"을 가볍게 유지하기 위한 기준이다.
TraceMind의 계층 구조와 contract-first 방향은 유지하되, 모든 파일에 같은 수준의
framework 표현을 퍼뜨리지 않는 것을 목표로 한다.

## 핵심 판단

파일 분리는 수단이지 목표가 아니다. 좋은 분리는 읽는 사람이 먼저 봐야 하는 흐름과,
나중에 봐도 되는 세부 규칙을 분리한다. 반대로 얕은 wrapper, 단일 사용처용 facade,
불필요한 DTO는 파일 수만 늘리고 읽기 비용을 올린다.

`shared` contract, cross-boundary payload, runtime port, 다중 구현 registry처럼 안정성이
중요한 경계는 강한 타입을 유지한다. 실험 runner, catalog 선언, diagnostics, report writer,
일회성 변환 helper는 평범한 함수와 직관적인 dict/list table을 우선한다.

## 코드 표현 규칙

- 먼저 plain function과 명확한 local helper로 쓴다. 패턴 객체는 바뀌는 축이 실제로
  둘 이상이거나 boundary contract가 필요할 때만 도입한다.
- `@dataclass(frozen=True, slots=True)`, `Protocol`, `TypedDict`, 긴 `Mapping[...]`
  generic은 contract, stable value object, port/interface에 제한한다.
- `Any`는 Hydra/OmegaConf, ML framework, tokenizer/model 같은 untyped 외부 경계에만
  허용하고, 내부로 들어오기 전에 작은 helper나 typed payload로 정규화한다.
- runner/controller/service의 첫 화면은 orchestration 흐름이 보여야 한다. 학습 loop,
  artifact IO, plotting, report row assembly, validation table이 한 함수에 섞이면
  의미 단위 helper로 내린다.
- data declaration과 실행 로직을 섞지 않는다. 긴 catalog/rule table은 helper factory나
  plain table로 낮추고, 실행 함수에는 lookup 흐름만 남긴다.
- 한 줄짜리 pass-through facade나 package-level export는 기본적으로 만들지 않는다.
  compatibility가 필요하면 제거 조건이나 유지 이유가 이름/문서에서 보여야 한다.
- type hint는 읽기를 도와야 한다. return type이 긴 union/generic으로 흐름을 가리면
  domain alias나 작은 payload object로 이름을 주고, 단일 사용처라면 과한 타입을 줄인다.
- 주석은 코드가 말하지 못하는 이유와 계약 의미만 설명한다. 필드 설명과 계약 해석은
  코드 가까이에 짧은 한국어로 둔다.
- 새 helper를 만들 때는 삭제 테스트를 생각한다. 지워도 caller가 한 단계 직접 호출만 하면
  되는 helper는 얕은 추상화일 가능성이 높다.

## 계층별 적용

- `shared`: 타입 엄격성을 낮추지 않는다. 대신 의미 단위 파일 분리, 짧은 필드 설명,
  compatibility 격리로 읽기 비용을 줄인다.
- `methods`: algorithm core는 원본 수식과 단계 이름을 보존한다. descriptor/config/fallback
  선언부는 반복 generic과 boilerplate를 줄인다.
- `agent`, `main_server`: 큰 흐름 orchestration을 먼저 보이게 한다. method-specific 의미는
  runtime adapter로 흡수하지 않고 `methods`에 남긴다.
- `scripts`: Hydra config load, core 호출, artifact dump 흐름을 얇게 유지한다. 새 method
  추가를 위해 script-local framework나 registry를 만들지 않는다.
- `docs`: 현재 규칙은 active docs나 code-adjacent README로 승격하고, notes archive를
  source of truth처럼 쓰지 않는다.

## 2026-05-08 점검 결과

최근 정리로 `training_contracts.py`는 926줄에서 512줄로 줄었고, objective/selection
config와 secure aggregation 계약은 별도 contract 파일로 분리됐다. 이 분리는 단순 LOC
축소가 아니라 envelope, objective config, secure aggregation이 서로 다른 이유로
바뀌는 축이라는 판단에 따른 것이다.

남은 읽기 난이도 후보는 아래처럼 본다. 즉시 리팩터링 대상이라기보다, 해당 파일을
다음에 만질 때 함께 낮출 후보다.

- `scripts/datasets/run_dataset_pipeline.py`: stage helper는 나뉘었지만 dataset pipeline
  entrypoint가 여러 stage orchestration을 모두 가진다.
- `methods/adaptation/lora_classifier/training.py`: ML framework 경계 때문에 `Any`가 많다.
  내부 경로에서는 batch/report alias로 의미를 좁힐 여지가 있다.
- `scripts/experiments/query_lora_ssl/harness/common.py`: Hydra config, model, tokenizer,
  dataloader가 한 run context에 모여 있어 외부 경계와 내부 실행값이 섞여 보인다.
- `agent/src/services/wellbeing/child_support_service.py`: prompt, suggestion, response
  orchestration이 긴 service 안에 남아 있다.
정리됨:

- `scripts/experiments/prototype_analysis/prototype_strategy/projection.py`: projection
  artifact orchestration만 남기고, projected row/visual-center 계산은 `projection_rows.py`,
  figure drawing은 `projection_figure.py`로 분리했다.
- research web catalog/backend는 더 이상 활성 런타임이 아니어서 제거했다.
