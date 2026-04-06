# 알고리즘 확장 가이드

## 목적

이 문서는 TraceMind에서 **교체 가능하게 설계된 모든 전략/알고리즘 지점**을
정리한 전환 가이드다.

새 알고리즘을 추가하거나 기존 것을 교체할 때 이 문서를 출발점으로 삼는다.

---

## 교체 가능한 전략 지점 전체 맵

| 이름 | 계층 | 현재 구현체 | 파일 |
|---|---|---|---|
| Training Backend | agent | DiagonalScaleHeuristicTrainingBackend | `agent/src/services/training/training_backends.py` |
| Privacy Guard | agent | DiagonalScaleClipOnlyPrivacyGuard | `agent/src/services/training/privacy_guard_service.py` |
| Pseudo-label Selection Policy | agent | Top1MarginSelectionPolicy | `agent/src/services/training/pseudo_label_service.py` |
| Scoring Policy | agent | MaxCosineScoringPolicy | `agent/src/services/inference/scoring_policies.py` |
| Aggregation Backend | main_server | DiagonalScaleAggregationService | `main_server/src/services/rounds/aggregation_service.py` |
| Update Acceptance Policy | main_server | Strict / Idempotent | `main_server/src/services/rounds/update_acceptance_policy.py` |
| Adapter Family | shared | diagonal_scale | `shared/src/contracts/adapter_contracts.py` |

---

## 전략 지점별 상세

---

### 1. Training Backend (agent)

**역할:** 채택된 pseudo-label 예시로 adapter delta를 실제로 계산하는 방식.

**Protocol:**
```python
# agent/src/services/training/training_backends.py
class SharedAdapterTrainingBackend(Protocol):
    backend_name: str
    payload_format: str

    def build_update(
        self,
        *,
        training_task: TrainingTask,
        model_manifest: ModelManifest,
        accepted_examples: tuple[EmbeddedTrainingExample, ...],
        created_at: datetime,
    ) -> SharedAdapterUpdate: ...

    def to_payload(self, update: SharedAdapterUpdate) -> SharedAdapterUpdatePayload: ...
```

**현재:** `DiagonalScaleHeuristicTrainingBackend` — gradient 없이 confidence 가중 방향으로 delta 계산.

**교체 시나리오:**
- Gradient 기반 backend 추가: `DiagonalScaleGradientTrainingBackend`
- 다른 adapter family용 backend 추가 (LoRA 등)

**교체 절차:**
1. 이 파일에 새 클래스 추가 (`SharedAdapterTrainingBackend` Protocol 구현)
2. `build_shared_adapter_training_backend()` factory에 등록
3. `TrainingObjectiveConfigPayload.loss` 값과 backend_name을 맞춤

**건드리지 않는 것:** `LocalTrainingService`는 backend를 주입받으므로 수정 불필요.

---

### 2. Privacy Guard (agent)

**역할:** local training 후 delta를 서버에 보내기 전 privacy 보호 적용.

**Protocol:**
```python
# agent/src/services/training/privacy_guard_service.py
class SharedAdapterPrivacyGuard(Protocol):
    guard_name: str

    def protect(
        self,
        *,
        update: SharedAdapterUpdate,
        training_task: TrainingTask,
    ) -> ProtectedAdapterUpdate: ...
```

**현재:** `DiagonalScaleClipOnlyPrivacyGuard` — L2 norm clipping만 적용.

**교체 시나리오:**
- DP noise 추가: `DiagonalScaleDPPrivacyGuard`
- Secure Aggregation 연동 guard (Phase 6)

**교체 절차:**
1. 이 파일에 새 클래스 추가
2. `build_shared_adapter_privacy_guard()` factory에 등록
3. `TrainingObjectiveConfigPayload.privacy_guard_name`으로 선택

**Envelope 영향:** `clipped`, `dp_applied` 필드가 각 guard의 결과를 반영함.
새 guard 추가 시 해당 필드를 올바르게 채울 것.

---

### 3. Pseudo-label Selection Policy (agent)

**역할:** scored event 중 어떤 예시를 학습에 채택할지 결정.

**Protocol:**
```python
# agent/src/services/training/pseudo_label_service.py
class PseudoLabelAcceptancePolicy(Protocol):
    policy_name: str

    def accept(
        self,
        *,
        candidates: tuple[PseudoLabelCandidate, ...],
        training_task: TrainingTask,
    ) -> tuple[PseudoLabelCandidate, ...]: ...
```

**현재:** `Top1MarginAcceptancePolicy` — confidence ≥ threshold AND margin ≥ threshold.

**교체 시나리오:**
- TopK 기반 선별
- Calibrated confidence 기반 policy

**교체 절차:**
1. 이 파일에 새 Policy 클래스 추가
2. `build_acceptance_policy()` factory에 등록
3. `TrainingSelectionPolicyPayload.policy_name`으로 선택

---

### 4. Scoring Policy (agent)

**역할:** 임베딩과 prototype 간 유사도 계산 방식.

**Protocol:**
```python
# agent/src/services/inference/scoring_policies.py
class CategoryScoringPolicy(Protocol):
    policy_name: str

    def score(
        self,
        *,
        embedding: list[float],
        prototypes: list[list[float]],
    ) -> float: ...
```

**현재:** `MaxCosineScoringPolicy` (최대 cosine), `TopKMeanScoringPolicy` (상위 K 평균).

**교체 시나리오:**
- Mahalanobis distance 기반 policy
- Learnable similarity (학습 가능한 metric)

**교체 절차:**
1. `scoring_policies.py`에 새 클래스 추가
2. `ScoringService.build_from_objective_config()`에 등록
3. `TrainingObjectiveConfigPayload` 또는 별도 config로 선택

---

### 5. Aggregation Backend (main_server)

**역할:** 여러 agent update를 합쳐 새 전역 adapter state를 만드는 방식.

**Protocol:**
```python
# main_server/src/services/rounds/aggregation_service.py
class SharedAdapterAggregationBackend(Protocol):
    adapter_kind: str

    def aggregate(
        self,
        *,
        base_state: SharedAdapterState,
        update_payloads: Sequence[SharedAdapterUpdate],
        next_model_revision: str,
        aggregated_at: datetime,
    ) -> AggregationResult: ...
```

**현재:** `DiagonalScaleAggregationService` — example_count 가중 평균 (FedAvg 변형).

**교체 시나리오:**
- q-합의: 최소 q개 update가 합의할 때만 반영하는 방식. `agent_id`로 per-agent 신뢰도 추적.
- Byzantine-robust aggregation: 이상 update를 제거하는 방식 (Krum, Bulyan 등)
- 가중 FedAvg: example_count 대신 trust score로 가중

**교체 절차:**
1. `aggregation_service.py`에 새 클래스 추가 (Protocol 구현)
2. `adapter_family_service.py`에 등록
3. q-합의 등 trust 기반이라면 `TrainingUpdateEnvelope.agent_id`를 사전 단계에서 활용

**참고 필드:** `client_metrics`의 `ClientMetricKeys` 상수들 (`mean_confidence`, `delta_l2_norm` 등)이
aggregation 품질 판단에 활용 가능.

---

### 6. Update Acceptance Policy (main_server)

**역할:** agent가 보낸 update를 round에 받아들일지, 중복으로 볼지 결정.

**Protocol:**
```python
# main_server/src/services/rounds/update_acceptance_policy.py
class RoundUpdateAcceptancePolicy(Protocol):
    def evaluate(
        self,
        *,
        record: RoundRecord,
        update: TrainingUpdateEnvelope,
        accepted_at: datetime,
    ) -> UpdateAcceptanceDecision: ...
```

**현재:**
- `StrictRoundUpdateAcceptancePolicy` — 같은 update_id 재전송 거부
- `IdempotentRoundUpdateAcceptancePolicy` — 동일 내용 재전송 허용

**교체 시나리오:**
- `agent_id` 기반 중복 차단: 같은 round에 같은 agent가 두 번 제출 불가
- Trust score 기반 사전 필터: 신뢰도 임계값 미달 agent update 거부

**교체 절차:**
1. `update_acceptance_policy.py`에 새 클래스 추가
2. `round_lifecycle_service.py`에서 주입 방식 선택
3. `agent_id` 기반이라면 `_idempotency_fingerprint()`에 `agent_id` 포함 검토

---

### 7. Adapter Family 추가 (shared)

**역할:** diagonal_scale 외 다른 adapter 구조 도입.

**변경 범위:** 가장 넓다. 여러 계층이 함께 바뀐다.

| 파일 | 변경 내용 |
|---|---|
| `shared/src/contracts/adapter_contracts.py` | 새 State/Update payload 클래스 추가 |
| `shared/src/domain/entities/training/` | 새 도메인 객체 추가 |
| `agent/training_backends.py` | 새 family용 backend 추가 |
| `main_server/aggregation_service.py` | 새 family용 aggregation backend 추가 |
| `main_server/adapter_family_service.py` | 라우팅 등록 |
| `main_server/mappers.py` | 새 payload 변환 추가 |

**현재 concrete:** `diagonal_scale` (임베딩 차원별 scale 보정).

**미래 후보:** LoRA adapter, projection head, prefix tuning.

---

## 전략 교체 시 공통 체크리스트

```
[ ] 바뀌는 축이 하나의 Protocol 구현 교체로 끝나는지 확인
    → 그렇지 않으면 설계 경계를 먼저 검토

[ ] shared/src/contracts/ 중 영향 받는 payload 필드 확인
    → 새 메타데이터가 필요하면 해당 contract에 선택 필드로 추가

[ ] agent가 보내고 server가 읽는 것이 같은 계약인지 확인
    → ClientMetricKeys, TrainingUpdateEnvelopePayload

[ ] factory 또는 등록 함수에 새 구현체 추가
    → build_*() 함수 패턴 사용

[ ] 기존 구현체를 제거하지 않고 새 것을 추가 (교체는 나중에)
    → 스위치 전에 병행 운용 가능한 구조 확인

[ ] 단위 테스트
    → 기존 테스트가 깨지지 않는지 확인 (회귀)
    → 새 구현체에 대한 최소 케이스 추가

[ ] 실험은 scripts/ 에서 먼저
    → 운영 후보가 확인되면 agent/main_server로 옮김
    → scripts에 implementation을 두지 않음
```

---

## 안 건드려도 되는 것

| 파일 | 이유 |
|---|---|
| `shared/src/contracts/fl_round_contracts.py` | agent↔server 통신 계약. 알고리즘과 무관 |
| `agent/src/services/federation/round_client.py` | HTTP 통신 레이어. 알고리즘 교체와 무관 |
| `agent/src/services/federation/runtime_service.py` | orchestration 흐름. backend 교체와 무관 |
| `agent/src/api/training.py` | API endpoint. 알고리즘 교체와 무관 |
| `shared/src/contracts/prototype_contracts.py` | prototype 동기화 계약. adapter 교체와 독립 |

---

## 관련 문서

- [shared_adapter_contracts_v1.md](shared_adapter_contracts_v1.md) — adapter payload 구조와 수학적 의미
- [training_update_envelope_v1.md](training_update_envelope_v1.md) — envelope 필드 설계 이유
- AGENTS.md (repo root) — 의존 방향 원칙 및 소유 경계
