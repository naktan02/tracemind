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

## 읽기 경로와 파일 분리 예산

사람이 새 기능이나 방법론을 추가할 때 첫 화면에서 봐야 하는 파일 수가 많으면 구조가
깊어진 것이 아니라 얕은 seam이 흩어진 것이다. 기본 읽기 경로는 README/index,
descriptor/contract, core 구현, 필요할 때 adapter/config 순서다. 일반 feature family는
3-5개 주요 파일 안에서 현재 의미와 확장 위치가 보여야 한다.

새 파일은 아래 중 하나를 만족할 때만 만든다.

- 서로 다른 이유로 바뀌는 구현을 분리한다.
- caller가 알아야 할 interface를 줄이고, 구현 세부를 충분히 숨긴다.
- 두 개 이상 caller나 implementation이 같은 의미를 공유한다.
- contract, runtime port, artifact writer처럼 독립 테스트 표면이 생긴다.
- 한 파일 안에 두면 orchestration 흐름이 가려질 만큼 구현이 길거나 순서 의존적이다.

아래는 기본적으로 합치는 쪽을 우선한다.

- 이름/상수만 가진 단일 축 파일이 같은 capability plan 주변에 여러 개 생긴다.
- `descriptor.recipe`를 다시 노출하는 수준의 pass-through module이다.
- 한 줄 wrapper나 package-level re-export라서 삭제해도 caller가 거의 단순해진다.
- config leaf YAML과 Python default catalog가 같은 기본값 의미를 동시에 소유한다.

예외도 있다. FastAPI route leaf, Hydra leaf config, test fixture, marker `__init__.py`,
짧은 value object는 얇아도 정상일 수 있다. 다만 얇은 파일이 reader path를 늘리면
같은 owner 문서에 "먼저 볼 파일"을 적거나 가까운 axis 파일로 합친다.

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
- 단순히 호출 이름만 바꾸는 helper는 만들지 않는다. 예를 들어
  `hook.compute_loss(...)`, `hook.generate_targets(...)`, `compute_prob(x.detach())`,
  `tensor.new_zeros(())` 같은 한 줄짜리 wrapper는 여러 곳에서 반복돼도 기본적으로
  caller에 직접 둔다. helper가 유지되려면 batch/view contract, 외부 framework 경계,
  validation 전제, 여러 줄의 순서 의존 로직처럼 숨길 만한 구현을 가져야 한다.

## 계층별 적용

- `shared`: 타입 엄격성을 낮추지 않는다. 대신 의미 단위 파일 분리, 짧은 필드 설명,
  compatibility 격리로 읽기 비용을 줄인다.
- `methods`: algorithm core는 원본 수식과 단계 이름을 보존한다. descriptor/config/fallback
  선언부는 반복 generic과 boilerplate를 줄인다. 논문/USB method를 이식할 때는
  원본 framework를 그대로 흉내 내기보다 TraceMind seam에 맞추되, 수식 조립이 보이는
  편을 우선한다. 공통 helper는 view/loader/forward contract처럼 의미가 있을 때만 둔다.
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
- `methods/adaptation/text_classifier/peft_encoder/training/`: ML framework 경계 때문에
  `Any`가 많다. 내부 경로에서는 batch/report alias로 의미를 좁힐 여지가 있다.
- `scripts/experiments/query_lora_ssl/harness/common.py`: Hydra config, model, tokenizer,
  dataloader가 한 run context에 모여 있어 외부 경계와 내부 실행값이 섞여 보인다.
- `agent/src/services/wellbeing/child_support_service.py`: prompt, suggestion, response
  orchestration이 긴 service 안에 남아 있다.
정리됨:

- `scripts/experiments/prototype_analysis/prototype_strategy/projection.py`: projection
  artifact orchestration만 남기고, projected row/visual-center 계산은 `projection_rows.py`,
  figure drawing은 `projection_figure.py`로 분리했다.
- research web catalog/backend는 더 이상 활성 런타임이 아니어서 제거했다.

## 2026-05-23 repo-wide 점검 결과

tracked file 1041개 중 Python 741개, Markdown 148개, YAML 79개를 기준으로 읽기 표면을
점검했다. `__init__.py`의 import/re-export와 `__all__` 사용은 발견되지 않았다.
작은 Python 파일은 route leaf, value object, config/adapter leaf가 대부분이라 파일 길이만
기준으로 위반으로 보지 않는다.

이번 점검에서 바로 정리한 후보는 아래다.

- `methods/federated_ssl/local_ssl_policy.py`와 `server_update_policy.py`: 이름과 작은
  payload 정규화만 가진 sibling 축 파일이라 `capability_axes.py`로 합쳤다.
- `methods/federated_ssl/fedmatch/recipe.py`: `descriptor.recipe`를 다시 노출하는
  pass-through라 제거했다. recipe metadata는 `descriptor.py`에서 바로 읽는다.

남은 후보는 즉시 리팩터링 대상이 아니라, 해당 영역을 다음에 실제로 수정할 때 함께
검토한다.

- `scripts/experiments/fl_ssl/federated_simulation/io/`: direct file 20개로 넓다. IO
  leaf 자체는 정상이나, 새 writer/builder를 추가할 때 기존 builder/writer 경계를 먼저
  재사용한다.
- `scripts/experiments/query_lora_ssl/harness/common.py`: Hydra config, model,
  tokenizer, dataloader가 한 context에 모여 있어 외부 경계와 내부 실행값 분리 여지가 있다.
- `agent/src/services/wellbeing/child_support_service.py`: prompt/suggestion/response
  orchestration이 길어 다음 변경 시 읽기 경로를 낮춘다.
- `methods/adaptation/text_classifier/peft_encoder/training_backend.py`,
  `agent/src/services/training/examples/service.py`: 이름상 facade/service라서 다음 변경 시
  삭제 테스트를 적용한다. 현재는 동작 보존 후보로만 둔다.
