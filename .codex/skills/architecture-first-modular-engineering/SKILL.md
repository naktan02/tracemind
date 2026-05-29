---
name: architecture-first-modular-engineering
description: Use when a project needs a reusable engineering style that prioritizes contract-first design, clear change-axis separation, canonical representations, source-adjacent documentation, explicit separation between common and context-specific concerns, and pattern choice based on problem structure rather than pattern preference.
---

# Architecture-First Modular Engineering

이 스킬은 "패턴 이름을 고르는 것"보다 "문제의 변화 구조를 먼저 읽고, 그에 맞는 경계와 계약을 세우는 것"을 우선하는 작업 스타일이다.

## 언제 쓸지

- 장기 유지보수 가능한 리팩터링
- 계약/도메인 경계 설계
- 공통 계층과 문맥별 계층 분리
- 함께 바뀌는 것과 독립적으로 바뀌어야 하는 것 분리
- repo-wide 지침을 재사용 가능한 형태로 만들기

## 핵심 원칙

1. 구현보다 먼저 문제 구조를 본다.
   - 무엇이 자주 바뀌는지
   - 무엇이 안정적으로 유지돼야 하는지
   - 무엇이 함께 바뀌는지
   - 무엇이 서로 독립적으로 바뀌어야 하는지
   - 무엇이 공통 관심사이고 무엇이 문맥별/사용자별/환경별 관심사인지 먼저 적는다.
   - 요구가 모호하면 구현 전에 가정, 선택지, 성공 기준을 먼저 드러낸다.
2. 계약과 경계를 먼저 세운다.
   - 도메인 객체
   - 입력/출력 계약
   - 책임 경계
   - 상태 소유권
   - 의존 방향
   을 구현보다 먼저 정의하거나 검증한다.
3. 책임은 의미 기준으로 분리한다.
   - 파일 수보다 역할과 변화 축을 기준으로 나눈다.
   - 서로 다른 이유로 바뀌는 것은 같은 곳에 두지 않는다.
   - 요청과 직접 연결된 파일만 고치고, 주변 리팩터링은 별도 concern으로 분리한다.
4. 공통성과 특수성을 구분한다.
   - 여러 문맥에서 안정적으로 공유되는 것은 공통 계층에 둔다.
   - 문맥, 사용자, 환경, 조직, 기능별로 달라지는 것은 지역 계층으로 분리한다.
   - 두 곳 이상에서 같은 의미, 같은 규칙, 같은 lifecycle이 반복되면 복사하지 말고
     가장 가까운 공통 owner로 승격한다.
   - 공통화는 무조건 최상위로 올리는 것이 아니라 반복되는 의미가 살아 있는 가장
     가까운 공통 경계에 모으는 것이다.
   - 모양만 비슷하고 의미나 변경 이유가 다르면 공통화하지 않는다.
5. 패턴 이름을 고집하지 않는다.
   - 패턴은 문제 구조에 맞춰 선택한다.
   - 특정 패턴을 미리 정해 두고 억지로 끼워 맞추지 않는다.
6. 하드코딩된 강한 결합을 줄인다.
   - 하나를 바꾸기 위해 여러 계층을 동시에 뜯어야 하면 구조 문제로 본다.
7. source of truth는 구현 가까이에 둔다.
   - 중요한 필드 의미, 계약 의미, 해석 규칙은 코드 또는 코드 가까운 문서에서 바로 보여야 한다.
8. 문서는 짧고 역할이 분명해야 한다.
   - 같은 배경 설명과 계획을 여러 문서에 반복하지 않는다.
9. 미래 교체 비용을 계속 점검한다.
   - 핵심 도메인 모델, 실행 방식, 저장 방식, 외부 연동, 규칙/정책, 보안/프라이버시 계층, 학습/추론 방식을 바꿀 때 나머지를 얼마나 건드려야 하는지 본다.
10. 임시 해결보다 확장 가능한 해결을 우선한다.
   - 단, 과설계는 피한다.
11. 경계에서는 canonical representation을 우선한다.
   - 같은 의미의 데이터가 경로마다 다른 shape로 흐르지 않게 한다.
   - 정규화가 필요하면 한 군데에서만 수행하고, 나머지는 canonical shape를 사용한다.
12. 호환성 계층은 명시적으로 격리한다.
   - legacy format, 임시 변환, 하위 호환 로직은 핵심 경로에 섞지 않는다.
   - compatibility는 별도 계층 또는 명시적 adapter로 두고, 제거 조건도 함께 남긴다.
13. policy와 mechanism을 분리한다.
   - 무엇을 할지 결정하는 규칙과, 그것을 어떻게 실행하는지는 분리한다.
14. 튜닝보다 먼저 관측 가능성을 만든다.
   - 파라미터를 조정하기 전에 어떤 단계에서 실패하는지 dump, metric, trace, summary로 먼저 보이게 만든다.
15. producer와 consumer를 함께 설계한다.
   - 데이터를 만드는 쪽과 읽는 쪽이 서로 다른 가정을 갖지 않게 한다.
16. 공통 계층의 변경은 drift와 blast radius를 함께 본다.
   - 문맥별 편향이 공통 계층으로 새어 들어가면 지역 계층 분리를 우선 검토한다.
17. 설정도 계약처럼 다룬다.
   - 중요한 config는 typed structure와 명확한 source of truth를 가진다.
18. 리팩터링은 끝점까지 맞춘다.
   - contract, producer, consumer, test, 문서까지 한 흐름으로 닫는다.
19. 검증 가능한 구조를 선호한다.
   - 좋은 구조는 테스트, dump, compose, lint, 실행 결과로 검증 가능해야 한다.
   - 반복해서 지켜야 하는 ownership, dependency, facade 금지, reference 해석 규칙은
     architecture guard test나 smoke test로 고정한다.
20. 무분별한 수평 폴더 누적을 피하고 변화 축 중심의 깊은 구조를 선호한다.
   - 새 기능, method, backend, runtime, integration마다 같은 레벨의 sibling 폴더를
     자동으로 만들지 않는다.
   - 먼저 기존 feature family, domain, capability, runtime, storage, job,
     UI surface 같은 변화 축 중 어디에 속하는지 정한다.
   - 새 top-level 폴더는 새로운 ownership boundary, 배포 단위, 보안 경계,
     bounded context, runtime surface가 실제로 생겼을 때 연다.
   - 단, monorepo의 `apps/*`, `packages/*`, plugin family처럼 각 sibling이 독립
     product나 bounded context라면 수평 확장은 정상 구조일 수 있다.
   - 비슷한 역할의 구현 폴더가 parent/family 없이 수평으로 늘어나면 공통 parent,
     family contract, profile, adapter seam이 빠진 신호로 본다.
21. 경계 밖 reference는 owner가 해석하기 전까지 opaque 값으로 다룬다.
   - artifact ref, payload ref, external id, storage key 같은 값은 caller가 임의로
     파일 경로나 내부 구조로 해석하지 않는다.
   - 실제 materialization은 repository, adapter, gateway처럼 그 ref의 저장소와
     lifecycle을 소유한 경계 안에 둔다.

## 선호하는 설계 스타일

패턴 이름을 적용했다는 사실은 설계 품질의 증거가 아니다. 좋은 구조는 변화 축마다
맞는 패턴을 제한된 책임으로 조합해서, 새 implementation 추가 시 수정 위치를
예측 가능하게 만든다.

- Registry primitive 파일은 저장, 정규화, 조회, 중복 방지, catalog 노출만 맡긴다.
  concrete implementation import와 중앙 등록 목록을 직접 소유하지 않는다.
  구현 옆 decorator를 쓰더라도 import trigger는 name-to-module convention,
  config-declared module path, package manifest 중 하나로 작게 유지한다.
- Descriptor는 identity, capability, required surface 같은 안정 metadata를 맡긴다.
- Hook은 알고리즘 내부에서 한 지점만 교체되는 objective/policy 조각에 쓴다.
- Profile은 여러 strategy 축의 실행 조합을 typed structure로 해석한다.
- Compatibility validator는 실행 중간이 아니라 resolve/bootstrap 전에 실패하게 한다.
- Runtime adapter는 IO, framework, orchestration 차이를 core 의미에서 분리한다.
- 새 구현 위치는 수평 sibling 추가가 아니라 기존 ownership boundary 아래의 깊은 Module로
  먼저 검토한다. 독립적인 product, bounded context, 배포 단위, runtime surface가 있을 때는
  sibling 폴더가 정당할 수 있다.
- 같은 family 안에서 여러 runner, flow, adapter가 공유할 state, config, result, lifecycle
  언어가 보이면 각 파일에 복사하지 말고 가장 가까운 공통 module로 둔다.

예를 들어 “새 method 추가 비용을 낮춘다”는 목표는 `Registry` 하나로 해결하지 않는다.
`Descriptor + Registry + Hook bundle + Profile + Compatibility Validator + Runtime
Adapter`처럼 여러 작은 패턴이 각자의 책임을 지도록 조합한다. 공통 hook이나 helper는
처음부터 끌어올리지 말고, 두 개 이상 implementation에서 의미가 안정될 때 승격한다.

## 코드 표현 규칙

좋은 구조는 읽는 표면도 가벼워야 한다. 파일을 나누는 것만으로는 충분하지 않고,
reader가 먼저 봐야 하는 흐름과 나중에 봐도 되는 세부 규칙이 코드에서 분리되어야 한다.

- plain function, local helper, direct import를 우선한다. 패턴 객체는 실제 변화 축이나
  boundary contract가 있을 때만 도입한다.
- `dataclass`, `Protocol`, `TypedDict`, 긴 generic type은 contract, stable value object,
  port/interface에 제한한다.
- `Any`는 untyped 외부 경계에만 허용하고, 내부 흐름으로 들어오기 전에 이름 있는 payload,
  alias, helper로 의미를 좁힌다.
- runner/controller/service의 첫 화면은 orchestration 흐름이 보여야 한다. 긴 rule table,
  artifact IO, report row assembly, plotting, validation boilerplate가 섞이면 분리한다.
- `Request`/`Result` payload는 boundary use case의 입력/출력 의미가 안정적일 때만 둔다.
  단순히 인자가 많다는 이유로 만들지 않는다.
- 같은 값 묶음이 여러 caller/test에서 반복되거나 검증, 불변식, serialization, trace의
  대상이 될 때 이름을 준다.
- package 공개 표면은 direct-file import로 드러낸다. `__init__.py`는 marker/docstring을
  기본으로 두고, re-export facade가 필요하면 별도 `.py` module로 이유를 드러낸다.
- 한 줄짜리 pass-through facade, 단일 사용처용 DTO, 삭제해도 caller가 거의 단순해지는
  helper는 만들지 않는다.
- type hint는 읽기를 도와야 한다. 긴 union/generic이 핵심 흐름을 가리면 domain alias나
  작은 payload object로 이름을 주고, 단일 사용처라면 과한 타입을 줄인다.

## 패턴 무결성 Guard

패턴을 도입하거나 리뷰할 때 아래를 먼저 확인한다.

- `registry.py`가 concrete class를 import하고 파일 하단에서 여러 `register_*()`를 직접
  호출하면 smell로 본다. 그 파일이 builtin loader라면 이름과 문서가 그렇게 보여야 한다.
- `builtin_loader.py`에 concrete module 목록만 옮기는 것은 transitional step이다. 최종
  구조에서는 convention/config/package manifest 기반 import trigger를 우선한다.
- Decorator 등록은 구현 옆에서 registration source of truth를 표현할 때만 쓴다. import
  순서를 통제하는 explicit builtin loader 없이 자동 discovery처럼 쓰지 않는다.
- Facade는 compatibility를 유지하는 얇은 표면이다. business rule, factory 조합,
  validation이 facade에 들어가면 core 경계가 흐려진다.
- Builder는 canonical object/payload 생성까지만 맡기고, Writer/Exporter는 저장과
  serialization만 맡긴다.
- Adapter는 runtime/framework/외부 시스템 차이를 core Interface로 변환한다. algorithm
  identity, policy 판단, method 의미를 adapter가 흡수하면 잘못된 seam이다.
- Runtime adapter는 method 이름이 아니라 capability 이름으로 확장한다. 새
  method/algorithm 때문에 runtime 계층에 method-specific 파일이 생기면 algorithm core
  경계가 얕다는 신호로 본다.
- Hook은 objective나 policy의 한 교체 지점이다. diagnostics, evidence selection,
  artifact IO 같은 다른 변화 축과 합치지 않는다.
- Profile은 실행 조합이며 실행값의 source of truth가 아니다. source of truth는 config나
  contract에 남긴다.
- Catalog는 display/discovery snapshot이고 implementation registration의 source of
  truth가 아니다. 공통 계층 중앙 catalog에 구현 목록을 계속 누적하면 registry smell과
  같은 문제로 본다.

## 패턴 선택 기준

패턴 자체를 목적처럼 쓰지 않는다. 아래 기준은 출발점일 뿐이며, 필요하면 조합해서 쓴다.

- 알고리즘만 바뀌면 `Strategy`
- 생성 조합이 바뀌면 `Factory` 계열
- 상태 단계가 중요하면 `State`
- 규칙 판단이 핵심이면 `Policy` 또는 `Specification`
- 처리 흐름이 핵심이면 `Pipeline`
- 외부 시스템 교체가 핵심이면 `Port/Adapter`
- 횡단 기능 추가면 `Decorator`
- 단순 구현체 조회면 `Registry`

## 공통 계층 vs 문맥별 계층 판단 기준

### 공통 계층에 둘 것

- 여러 문맥에서 안정적으로 유지된다
- 합쳐도 의미 왜곡이 작다
- 잘못돼도 영향 반경이 상대적으로 작다

### 문맥별 계층에 둘 것

- 사용자, 환경, 기능별로 해석 차이가 크다
- 초기 편향이 공통 계층으로 퍼지면 위험하다
- 민감한 상태나 개인화된 해석을 가진다

## 문서 규칙

- 필드 의미는 계약 파일에 직접 드러나야 한다.
- 별도 문서는 설계 이유, 배경, 예외 규칙만 둔다.
- 같은 배경 설명과 계획을 여러 긴 문서에 반복하지 않는다.

## instruction 계층화 규칙

repo instruction은 계층적으로 둔다.

1. repo-wide 공통 지침 1장
2. 필요한 경로에만 path-specific 지침
3. 개인 취향은 repo 밖 전역 레이어

## 결과물 체크리스트

- 바뀔 축이 코드에서 독립적으로 보이는가
- 공통 계층과 문맥별 계층이 섞이지 않는가
- 새 sibling 폴더를 늘리기 전에 기존 family/capability/runtime 경계 아래에 둘 수 있는지
  확인했는가
- 반복되는 의미, 규칙, state, config, result, lifecycle을 가장 가까운 공통 owner로
  모았는가
- 요청 밖 주변 리팩터링을 섞지 않았는가
- 모호한 요구의 가정, 선택지, 성공 기준을 드러냈는가
- 계약 파일만 읽어도 필드 의미가 이해되는가
- repo-wide 지침과 path-specific 지침이 충돌하지 않는가
- canonical representation이 경계에서 유지되는가
- policy와 mechanism이 분리돼 있는가
- producer와 consumer가 같은 계약을 보고 있는가
- compatibility가 핵심 경로에 새지 않는가
- 구조 변경의 검증 흔적이 남아 있는가
- 반복해서 깨질 수 있는 구조 규칙을 architecture guard나 smoke test로 고정했는가
- 새 implementation 추가 시 기존 구현을 뜯지 않고 확장 가능한가
- test-only implementation을 production tree에 남기지 않았는가
- runtime adapter가 method identity나 policy 의미를 흡수하지 않는가
- catalog snapshot과 implementation-local registration이 분리돼 있는가
- boundary ref/id를 owner repository/adapter 밖에서 경로나 내부 구조로 해석하지 않는가

필요하면 [instruction_layers.md](references/instruction_layers.md)를 읽고,
도구별 적용 형식을 고른다.
