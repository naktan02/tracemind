---
trigger: always_on
---

# Architecture-First Modular Engineering

이 규칙은 특정 프로젝트가 아니라 모든 코드베이스에 적용하는 설계 스타일이다.
"패턴 이름을 고르는 것"보다 문제의 변화 구조를 먼저 읽고, 그에 맞는 경계,
계약, Module, Interface를 세우는 것을 우선한다.

목표는 코드 길이를 줄이는 것이 아니라 새 기능, 새 알고리즘, 새 backend,
새 runtime 조합을 추가할 때 수정 위치가 예측 가능하고 기존 core를 덜 건드리는
구조를 만드는 것이다.

## 언제 쓸지

- 장기 유지보수 가능한 리팩터링
- 계약/도메인 경계 설계
- 공통 계층과 문맥별 계층 분리
- 함께 바뀌는 것과 독립적으로 바뀌어야 하는 것 분리
- codebase-wide 지침을 재사용 가능한 형태로 만들기
- 새 strategy, method, backend, adapter, hook, profile 추가
- 외부 알고리즘이나 논문 구현을 내부 framework에 port

## 핵심 원칙

1. 구현보다 먼저 문제 구조를 본다.
   - 무엇이 자주 바뀌는지 본다.
   - 무엇이 안정적으로 유지돼야 하는지 본다.
   - 무엇이 함께 바뀌는지 본다.
   - 무엇이 서로 독립적으로 바뀌어야 하는지 본다.
   - 무엇이 공통 관심사이고 무엇이 문맥별/사용자별/환경별 관심사인지 본다.
2. 계약과 경계를 먼저 세운다.
   - 도메인 객체, 입력/출력 계약, 책임 경계, 상태 소유권, 의존 방향을
     구현보다 먼저 정의하거나 검증한다.
3. 책임은 의미 기준으로 분리한다.
   - 파일 수보다 역할과 변화 축을 기준으로 나눈다.
   - 서로 다른 이유로 바뀌는 것은 같은 곳에 두지 않는다.
4. 공통성과 특수성을 구분한다.
   - 여러 문맥에서 안정적으로 공유되는 것은 공통 계층에 둔다.
   - 문맥, 사용자, 환경, 조직, 기능별로 달라지는 것은 지역 계층으로 분리한다.
5. 패턴 이름을 고집하지 않는다.
   - 패턴은 문제 구조에 맞춰 선택한다.
   - 특정 패턴을 미리 정해 두고 억지로 끼워 맞추지 않는다.
6. 패턴은 단독 장식이 아니라 조합 가능한 도구로 쓴다.
   - Registry, Strategy, Factory, Hook, Policy, Adapter, Descriptor, Profile,
     Validator를 변화 축에 맞게 조합한다.
   - 좋은 구조는 "Registry를 썼다"가 아니라 "새 implementation 추가 시 수정
     위치가 예측 가능하다"로 증명된다.
7. 하드코딩된 강한 결합을 줄인다.
   - 하나를 바꾸기 위해 여러 계층을 동시에 뜯어야 하면 구조 문제로 본다.
8. source of truth는 구현 가까이에 둔다.
   - 중요한 필드 의미, 계약 의미, 해석 규칙은 코드 또는 코드 가까운 문서에서
     바로 보여야 한다.
9. 문서는 짧고 역할이 분명해야 한다.
   - 같은 배경 설명과 계획을 여러 문서에 반복하지 않는다.
   - 문서는 source of truth를 대체하지 않고 code-adjacent guide 역할을 한다.
10. 미래 교체 비용을 계속 점검한다.
    - 핵심 도메인 모델, 실행 방식, 저장 방식, 외부 연동, 규칙/정책,
      보안/프라이버시 계층, 학습/추론 방식을 바꿀 때 나머지를 얼마나
      건드려야 하는지 본다.
11. 임시 해결보다 확장 가능한 해결을 우선한다.
    - 단, 과설계는 피한다.
12. 경계에서는 canonical representation을 우선한다.
    - 같은 의미의 데이터가 경로마다 다른 shape로 흐르지 않게 한다.
    - 정규화가 필요하면 한 군데에서만 수행하고, 나머지는 canonical shape를 사용한다.
13. 호환성 계층은 명시적으로 격리한다.
    - legacy format, 임시 변환, 하위 호환 로직은 핵심 경로에 섞지 않는다.
    - compatibility는 별도 계층 또는 명시적 adapter로 두고, 제거 조건도 함께 남긴다.
14. policy와 mechanism을 분리한다.
    - 무엇을 할지 결정하는 규칙과, 그것을 어떻게 실행하는지는 분리한다.
15. 튜닝보다 먼저 관측 가능성을 만든다.
    - 파라미터를 조정하기 전에 어떤 단계에서 실패하는지 dump, metric, trace,
      summary로 먼저 보이게 만든다.
16. producer와 consumer를 함께 설계한다.
    - 데이터를 만드는 쪽과 읽는 쪽이 서로 다른 가정을 갖지 않게 한다.
17. 공통 계층의 변경은 drift와 blast radius를 함께 본다.
    - 문맥별 편향이 공통 계층으로 새어 들어가면 지역 계층 분리를 우선 검토한다.
18. 설정도 계약처럼 다룬다.
    - 중요한 config는 typed structure와 명확한 source of truth를 가진다.
    - 실행값의 source of truth와 Python metadata hint를 섞지 않는다.
19. 리팩터링은 끝점까지 맞춘다.
    - contract, producer, consumer, test, 문서까지 한 흐름으로 닫는다.
20. 검증 가능한 구조를 선호한다.
    - 좋은 구조는 테스트, dump, compose, lint, 실행 결과로 검증 가능해야 한다.
21. 임시 단일 구현 후 재리팩터링을 기본값으로 두지 않는다.
    - 같은 family의 후속 알고리즘이나 구현이 예상되면 첫 구현부터 family-level
      extension seam을 만든다.
    - 단, 후속 구현이 가설뿐이면 test-only extension으로 seam을 검증한 뒤 승격한다.
22. 외부 알고리즘 port 시 source traceability를 유지한다.
    - 논문/외부 repo 알고리즘을 가져올 때는 핵심 수식, 하이퍼파라미터 이름,
      단계 구조를 원본과 최대한 1:1로 유지한다.
    - 달라진 부분은 명시적으로 기록한다.
23. 무의식적 근사를 금지한다.
    - 원본 알고리즘의 전제조건이 빠져 있으면 조용히 proxy 동작으로 대체하지 않는다.
    - 빠진 전제와 현재 근사 수준을 먼저 드러낸다.
24. family runner를 우선한다.
    - runner/helper는 알고리즘별 일회성 파일보다 family 공통 runner +
      algorithm-specific adapter 구조를 우선한다.
    - 단, 후속 알고리즘이 없을 때만 one-off를 허용한다.
25. 실험 의미와 코드 완료를 분리한다.
    - "코드가 돌아간다"와 "실험 비교선으로 충분히 닫혔다"를 같은 완료로 취급하지 않는다.
    - 입력 생성, provenance, diagnostics까지 있어야 실험 완료로 본다.
26. 깊은 Module을 선호한다.
    - 좋은 Module은 작은 Interface 뒤에 의미 있는 구현과 검증 가능한 규칙을 숨긴다.
    - 얕은 pass-through 추상화는 삭제 테스트로 확인한다.
27. runtime orchestration과 algorithm 의미를 분리한다.
    - entrypoint, runner, UI, runtime adapter가 algorithm/method 의미를 흡수하지 않게 한다.
    - algorithm/method identity와 계산 의미는 core Module 가까이에 둔다.
28. runtime adapter는 capability 이름으로 확장한다.
    - 새 method/algorithm 때문에 runtime 계층에 method 이름을 가진 파일이 늘어나면
      잘못된 seam으로 본다.
    - runtime 계층은 storage, private state, artifact materialization, transport,
      lifecycle 같은 실행 capability를 소유하고, method identity와 policy 의미는
      algorithm core 계층에 둔다.
29. catalog snapshot과 implementation registration을 구분한다.
    - cross-layer UI/server catalog가 필요하다는 이유로 implementation metadata를
      공통 계층 중앙 파일에 계속 누적하지 않는다.
    - 공통 계층은 catalog entry schema와 안정 snapshot contract를 소유하고,
      implementation-local module은 자기 registration metadata를 소유한다.

## 설계 순서

1. 도메인 언어와 실행 경계를 먼저 맞춘다.
2. source of truth가 어디인지 정한다.
3. producer와 consumer가 공유할 canonical representation을 정한다.
4. extension point가 필요한 변화 축과 그렇지 않은 고정 축을 분리한다.
5. 필요한 패턴을 조합해 Interface를 만든다.
6. 첫 implementation을 그 Interface 위에 얹는다.
7. 두 번째 implementation 또는 test-only extension으로 seam이 실제인지 검증한다.
8. contract, caller, tests, docs를 같은 흐름으로 닫는다.

## 코드 표현 규칙

이 규칙은 특정 프로젝트나 폴더에 묶이지 않는다. 좋은 구조는 파일 수나 패턴 이름이
아니라, reader가 핵심 흐름을 빠르게 찾고 세부 규칙을 필요할 때 내려가 볼 수 있는지로
평가한다.

- plain function, local helper, direct import를 우선한다. 패턴 객체, registry, facade,
  DTO는 실제 변화 축이나 boundary contract가 있을 때만 만든다.
- `dataclass`, `Protocol`, `TypedDict`, 긴 generic type은 contract, stable value object,
  port/interface처럼 안정성이 필요한 경계에 제한한다.
- `Any`는 untyped 외부 경계에만 허용하고, 내부 흐름으로 들어오기 전에 이름 있는 payload,
  alias, helper로 의미를 좁힌다.
- runner/controller/service의 첫 화면은 orchestration 흐름이 보여야 한다. 긴 rule table,
  artifact IO, report row assembly, plotting, validation boilerplate가 섞이면 분리한다.
- data declaration과 실행 로직을 섞지 않는다. 긴 catalog/rule table은 helper factory나
  plain table로 낮추고, 실행 함수에는 lookup 흐름만 남긴다.
- 한 줄짜리 pass-through facade, 단일 사용처용 DTO, 삭제해도 caller가 거의 단순해지는
  helper는 만들지 않는다.
- type hint는 읽기를 도와야 한다. 긴 union/generic이 핵심 흐름을 가리면 domain alias나
  작은 payload object로 이름을 주고, 단일 사용처라면 과한 타입을 줄인다.
- 주석은 코드가 말하지 못하는 이유와 계약 의미만 설명한다. 코드의 동작을 반복하는
  설명이나 장황한 boilerplate 주석은 피한다.

## 패턴 사용 원칙

패턴 자체를 목적처럼 쓰지 않는다. 아래 기준은 출발점일 뿐이며 필요하면 조합해서 쓴다.

- 알고리즘만 바뀌면 `Strategy`
- 생성 조합이 바뀌면 `Factory` 계열
- 상태 단계가 중요하면 `State`
- 규칙 판단이 핵심이면 `Policy` 또는 `Specification`
- 처리 흐름이 핵심이면 `Pipeline`
- 외부 시스템 교체가 핵심이면 `Port/Adapter`
- 횡단 기능 추가면 `Decorator`
- 단순 구현체 조회면 `Registry`

## 패턴 무결성 규칙

패턴을 도입할 때는 이름보다 역할 경계를 먼저 검증한다. "Registry를 썼다",
"Adapter를 만들었다", "Builder로 뺐다"는 설계 품질의 증거가 아니다. 패턴마다
소유해야 할 책임, 허용 dependency, side effect, 확장 방식이 코드에서 드러나야 한다.

- `Registry` primitive 파일은 저장, 정규화, 조회, 중복 방지, catalog 노출만 맡긴다.
  concrete implementation import와 중앙 등록 목록을 직접 소유하지 않는다.
  구현 옆 decorator를 쓰더라도 import trigger는 name-to-module convention,
  config-declared module path, package manifest 중 하나로 작게 유지한다.
- `registry.py` 맨 아래에 concrete class를 import해서 여러 `register_*()`를 직접 호출하는
  구조는 기본적으로 smell이다. `builtin_loader.py`에 concrete module 목록만 옮기는 것도
  최종 구조가 아니라 transitional smell로 본다.
- `Decorator` 등록은 implementation 옆의 local source of truth가 필요하고 import 순서를
  명시적으로 통제할 수 있을 때만 쓴다. canonical contract나 shared payload를 자동
  discovery하는 용도로 남용하지 않는다.
- `Facade`는 compatibility와 public surface 유지용이다. domain 의미, factory 조합,
  validation을 facade에 숨기지 않는다. 프로젝트가 `__init__.py`를 marker로 쓰는 규칙이면
  package-level export 대신 명시적 `.py` facade를 둔다.
- `Builder`는 canonical object/payload를 만들고, `Writer`/`Exporter`는 저장 위치와
  serialization을 맡긴다. builder가 파일 경로 정책까지 소유하거나 writer가 schema 의미를
  만들면 경계가 깨진 것이다.
- `Adapter`는 외부/runtime/framework 차이를 core Interface로 변환한다. 알고리즘 선택,
  method 의미, policy 판단을 adapter에 넣지 않는다.
- runtime adapter 파일명과 Interface는 method 이름이 아니라 capability 이름을 사용한다.
  예를 들어 `fedmatch_server_policy_adapter`보다 `round_state_exchange_adapter`가
  낫다. FedMatch라는 의미는 method-local module에 남긴다.
- `Policy`는 판단을 맡고 mechanism은 실행을 맡는다. policy 안에서 IO, 학습, aggregation 같은
  실행 절차가 커지면 분리한다.
- `Hook`은 한 알고리즘 내부의 교체 가능한 objective/policy 조각이다. runtime diagnostics,
  evidence selection, artifact IO 같은 다른 변화 축과 합치지 않는다.
- `Profile`은 실행 조합을 표현하고 검증 가능한 typed structure로 해석한다. 실행값의
  source of truth를 Python default hint나 descriptor metadata로 옮기지 않는다.
- `Validator`는 조합 불일치를 bootstrap/resolve 시점에 실패하게 한다. runtime 중간에서
  우연히 터지는 검증은 framework seam이 아니다.

패턴을 적용한 뒤에는 삭제 테스트와 두 번째 구현 테스트를 함께 본다. 패턴 파일을 지웠을 때
복잡도가 caller 여러 곳으로 다시 퍼지면 깊은 Module이고, 그냥 한 단계 호출만 줄면 얕은
pass-through다. 두 번째 실제 구현이 없으면 test-only extension으로 seam을 검증한다.

패턴별 책임은 제한한다.

- `Registry`는 lookup과 duplicate guard를 담당한다. 도메인 의미, 실행 조합, builtin 목록을
  registry에 숨기지 않는다.
- `Catalog`는 discovery와 display를 위한 snapshot/metadata 표면이다. implementation
  registration source of truth가 되면 중앙 registry smell과 같은 문제가 된다.
- `Descriptor`는 identity, capability, required surface, ownership hint처럼 안정적인
  metadata를 소유한다.
- `Profile`은 여러 strategy 축의 실행 조합을 typed structure로 해석하고 drift를 막는다.
- `Hook`은 알고리즘 내부에서 한 지점만 교체되는 objective/policy 조각에 쓴다.
- `Policy`와 `Specification`은 선택, 수락, compatibility 같은 판단 규칙에 쓴다.
- `Adapter`는 runtime, IO, framework, 외부 시스템 차이를 core로부터 격리한다.
- `Factory`는 config나 descriptor에서 concrete implementation을 만들 때 쓴다.
- `Validator`는 실행 중간이 아니라 resolve/bootstrap 전에 실패하게 한다.

패턴은 단독으로 쓰기보다 함께 쓴다. 예를 들어 좋은 method framework는
`Descriptor + Registry + Hook bundle + Runtime Adapter + Compatibility Validator`
처럼 여러 작은 패턴이 각자 제한된 책임을 맡는 구조에 가깝다.

## 공통 계층 vs 문맥별 계층

### 공통 계층에 둘 것

- 여러 문맥에서 안정적으로 유지된다.
- 합쳐도 의미 왜곡이 작다.
- 잘못돼도 영향 반경이 상대적으로 작다.
- 두 개 이상 implementation에서 반복되고 의미가 안정됐다.

### 문맥별 계층에 둘 것

- 사용자, 환경, 기능별로 해석 차이가 크다.
- 초기 편향이 공통 계층으로 퍼지면 위험하다.
- 민감한 상태나 개인화된 해석을 가진다.
- 한 implementation의 상태, 편향, 알고리즘 특수 규칙이다.

공통 계층은 특정 구현의 편의를 먼저 흡수하면 안 된다. 한 implementation 전용 helper는
처음에는 local Module에 두고, 두 개 이상 implementation에서 안정적으로 공유될 때만
공통 계층으로 승격한다.

## 외부 알고리즘 Port

- 원본 알고리즘의 수식, 단계 이름, hyperparameter 의미를 가능한 한 유지한다.
- 원본과 다르게 단순화하거나 proxy로 대체한 부분은 명시한다.
- 원본의 hook, state, memory bank, policy처럼 바뀌는 지점은 가능한 한 local Module로
  보존한다.
- 무리하게 거대한 base class로 합치지 않는다.

## 문서 규칙

- 필드 의미는 계약 파일에 직접 드러나야 한다.
- 별도 문서는 설계 이유, 배경, 예외 규칙만 둔다.
- 같은 배경 설명과 계획을 여러 긴 문서에 반복하지 않는다.
- source of truth 문서와 참고/아카이브 문서를 섞지 않는다.

## Instruction 계층화 규칙

1. codebase-wide 공통 지침 1장
2. 필요한 경로에만 path-specific 지침
3. 개인 취향은 codebase 밖 전역 레이어

## 결과물 체크리스트

- 바뀔 축이 코드에서 독립적으로 보이는가
- 공통 계층과 문맥별 계층이 섞이지 않는가
- 계약 파일만 읽어도 필드 의미가 이해되는가
- codebase-wide 지침과 path-specific 지침이 충돌하지 않는가
- canonical representation이 경계에서 유지되는가
- policy와 mechanism이 분리돼 있는가
- producer와 consumer가 같은 계약을 보고 있는가
- compatibility가 핵심 경로에 새지 않는가
- 구조 변경의 검증 흔적이 남아 있는가
- 새 implementation 추가 시 기존 구현을 뜯지 않고 확장 가능한가
- registry가 도메인 의미를 과하게 흡수하지 않는가
- runtime adapter가 algorithm/method 의미를 흡수하지 않는가
- "돌아간다"와 "framework seam이 검증됐다"를 구분했는가
