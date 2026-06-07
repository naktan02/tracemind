# TraceMind Execution Plan

이 문서는 현재 실행 판단만 짧게 유지한다. 구조와 용어는
`docs/architecture/target-method-runtime-structure.md`, 실제 기본값은 `conf/`,
method 의미는 `methods/`와 code-adjacent README를 우선한다. 완료된 migration 이력과
과거 후보 비교는 active guidance로 쓰지 않는다.

## Active Focus

```text
central SSL은 pooled/offline control
FL SSL은 existing manual baseline + existing FedMatch method-owned surface 검증
새 FL SSL 방법론은 현재 추가하지 않음
```

핵심 경계:

- 원문 텍스트와 개인 해석 상태는 agent-local boundary에 남긴다.
- 서버는 shared artifact, round lifecycle, aggregation, publication을 맡는다.
- 재사용 가능한 계산 core와 method policy는 `methods/`가 소유한다.
- 실행 조합과 숫자 기본값은 `conf/`가 소유한다.
- `scripts`는 Hydra entrypoint, simulation orchestration, report/artifact writer에 머문다.

## Current Execution Surface

- FL SSL 기본 entrypoint는 `conf/entrypoints/fl_ssl/run_federated_simulation.yaml`이다.
- 기본 실행은 `fl_method.composition_mode=manual`이며
  `FixMatch + peft_text_encoder + FedAvg` 조합이다.
- 현재 public method-owned FSSL leaf는 `conf/strategy_axes/fssl_method/fedmatch.yaml`
  하나다.
- FedMatch 의미와 policy는 `methods/federated_ssl/fedmatch/`가 소유한다.
- FedMatch PEFT text encoder 실행 bridge와 partitioned optimizer loop는
  `methods/adaptation/peft_text_encoder/federated_ssl/`가 소유한다.
- `lora_classifier`, `adapter_family_name`, `diagonal_scale`은 active 실행 축이 아니다.

## Current Priorities

1. 기존 manual baseline과 기존 FedMatch method-owned 실행 surface의 report/artifact
   metadata를 검증한다.
2. `gpu_local + mxbai` 조건의 reduced/main 산출물이 현재 report protocol을 만족하는지
   확인한다.
3. active docs는 현재 코드와 config만 설명하게 유지하고, 새 방법론 후보나 오래된
   phase map은 active guidance에서 제거한다.

## Validation Criteria

- Central SSL: pooled/offline control table과 output metadata가 같은 고정 조건에서 남는다.
- FL SSL: split, labeled exposure, client count, round budget, method/manual role,
  update family, payload adapter kind, aggregation, metric이 report에 남는다.
- Runtime: update base revision, aggregation, publication, artifact rebuild가 일관된다.
- Privacy: raw text는 서버로 올라가지 않고 privacy layer는 training logic과 분리된다.
