# 알고리즘 확장 가이드

## 목적

이 문서는 TraceMind에서 **교체 가능하게 설계된 모든 전략/알고리즘 지점**을
정리한 전환 가이드다.

새 알고리즘을 추가하거나 기존 것을 교체할 때 이 문서를 출발점으로 삼는다.

실제 구현 순서와 테스트 순서를 따라가려면
[strategy_addition_playbook.md](strategy_addition_playbook.md)를
함께 본다.

지금 어떤 축이 실제 runtime에서 선택 가능하고,
어떤 축이 typed metadata만 있는지는
[docs/strategy_surface_map.md](../strategy_surface_map.md)에서
먼저 확인한다.

중요:

- 이 가이드는 현재 시스템/FL runtime에 wiring된 전략 축을 중심으로 설명한다.
- query-domain 적응 단계의 `central LoRA classifier` 비교선은 별도 중앙 trainer를
  사용하며, 그 구현이 자동으로 이 registry 축 안에 들어와야 한다는 뜻은 아니다.
- 후속 알고리즘이 같은 family로 이어질 가능성이 높다면, 첫 구현부터 family-level
  공통 surface를 먼저 둔다.
- 단일 알고리즘 전용 임시 파일을 먼저 만들고 다음 패치에서 다시 일반화하는 흐름은
  가능한 한 피한다.

---

## 교체 가능한 전략 지점 전체 맵

| 이름 | 계층 | 현재 구현체 | 파일 |
|---|---|---|---|
| Algorithm Profile | shared/scripts | `prototype_pseudo_label_v1`, `prototype_top1_confidence_v1` | `shared/src/config/training_algorithm_profiles.py`, `conf/training_algorithm_profile/` |
| Training Backend | methods/agent | DiagonalScaleHeuristicTrainingBackend | `methods/adaptation/diagonal_scale/`, `agent/src/services/training/backends/training/` |
| Example Generation Backend | agent | PrototypeRescoringTrainingExampleBackend, WeakStrongPairTrainingExampleBackend | `agent/src/services/training/backends/inputs/`, `agent/src/services/training/examples/service.py` |
| Evidence Backend | methods/agent | PrototypeSimilarityEvidenceBackend | `methods/prototype/evidence/`, `agent/src/services/training/backends/evidence/` |
| Scorer Backend | methods/agent/scripts | PrototypeSimilarityScoringBackend, ClassifierHeadLogitsScoringBackend | `methods/prototype/scoring/`, `agent/src/services/inference/scoring_backends.py` |
| Privacy Guard | agent | DiagonalScaleClipOnlyPrivacyGuard, ClassifierHeadClipOnlyPrivacyGuard | `agent/src/services/training/execution/privacy_guard_service.py` |
| Pseudo-label Acceptance Policy | agent | Top1MarginThresholdAcceptancePolicy, Top1ConfidenceOnlyAcceptancePolicy | `agent/src/services/training/acceptance_policies/` |
| Pseudo-label Selection Hook | methods/scripts | `top1_margin_threshold`, `top1_confidence_only` | `methods/ssl/hooks/`, `conf/pseudo_label_algorithm/` |
| Query SSL Algorithm | methods/scripts | `fixmatch` | `methods/ssl/`, `scripts/experiments/lora_classifier/query_ssl/`, `conf/query_ssl_method/`, `conf/query_source/` |
| Query SSL Augmenter | agent/scripts | `nllb_backtranslation`, `precomputed_usb_candidates` | `agent/src/services/backtranslation_service.py`, `scripts/experiments/lora_classifier/query_ssl/augmentation.py`, `conf/query_ssl_augmenter/` |
| PEFT Adapter Builder | methods/scripts | `lora`, `rslora` | `methods/adaptation/`, `conf/lora/` |
| Federated Shard Policy | methods/scripts | `label_dominant`, `dirichlet_label_skew` | `methods/federated/shard_policy/`, `scripts/experiments/federated_simulation/sharding.py`, `conf/federated_shard_policy/` |
| FL SSL Method Descriptor | methods/scripts | `fedavg_pseudo_label` | `methods/federated_ssl/`, `scripts/experiments/federated_simulation/method_runtime.py`, `conf/federated_ssl_method/` |
| Scoring Policy | methods | MaxCosineScorePolicy, TopKMeanCosineScorePolicy | `methods/prototype/scoring/policies.py` |
| Aggregation Backend | methods/main_server | DiagonalScaleAggregationService (`fedavg`), ClassifierHeadFedAvgAggregationService (`fedavg`) | `methods/federated/aggregation/fedavg/`, `main_server/src/services/federation/rounds/aggregation/` |
| Update Acceptance Policy | main_server | CompositeRoundUpdateAcceptancePolicy | `main_server/src/services/federation/rounds/acceptance/` |
| Secure Update Codec | shared/agent/main_server | `noop` | `shared/src/services/secure_update_codec.py` |
| Adapter Family | main_server/shared | `diagonal_scale`, `classifier_head` | `main_server/src/services/federation/rounds/families/`, `shared/src/contracts/adapter_contracts.py` |

---

## 전략 지점별 상세

---

### 1. Algorithm Profile (shared/scripts)

**역할:** 논문/알고리즘 단위로 여러 전략 축을 묶어 기본 조합을 제공한다.

**현재:**
- `prototype_pseudo_label_v1`
- `prototype_top1_confidence_v1`

주의:
- query-domain 적응 단계의 핵심 비교선은 별도 중앙 trainer 레일에서 먼저 닫고,
  그 뒤에 system translation 여부를 판단한다.

**구성 위치:**
- shared canonical registry:
  - `shared/src/config/training_algorithm_profiles.py`
- scripts 실험 선택:
  - `conf/training_algorithm_profile/*.yaml`

**권장 사용법:**
- 새 논문 방법을 넣을 때 먼저 공통 backend 축을 구현한다.
- 그 다음 해당 논문 이름의 algorithm profile을 추가해 기본 조합을 묶는다.
- 실험에서는 profile을 고르고, 필요할 때 개별 축만 override한다.
- 단, 중앙 query-domain 적응 실험처럼 runtime broad preset보다 더 좁은 비교축이 필요하면
  `conf/pseudo_label_algorithm/`처럼 별도 Hydra group을 두는 편이 낫다.

---

### 2. Training Backend (methods/agent)

**역할:** 채택된 pseudo-label 예시로 adapter delta를 실제로 계산하는 방식.

**Protocol:**
```python
# agent/src/services/training/backends/training/base.py
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

**현재:** `DiagonalScaleHeuristicTrainingBackend`
- `DiagonalScaleHeuristicTrainingBackend`는 gradient 없이 confidence 가중 방향으로 delta를 계산한다.
- 실제 `VectorAdapterDelta` 산술 core는 `methods/adaptation/diagonal_scale/`가
  소유하고, agent backend는 `SharedAdapterTrainingBackend` protocol과 runtime
  registry에 연결한다.

주의:
- 현재 runtime의 `SharedAdapterTrainingBackend`는 family별 shared state delta를 반환한다.
- `classifier_head` family는 classifier weight/bias delta를, `diagonal_scale` family는
  차원별 scale delta를 만든다.

**교체 시나리오:**
- Gradient 기반 backend 추가: `DiagonalScaleGradientTrainingBackend`
- 다른 adapter family용 backend 추가 (LoRA 등)

**교체 절차:**
1. 재사용 가능한 update 산술이면 `methods/adaptation/<adapter_family>/`에 먼저 둔다
2. `training_backends/` 아래에 agent backend adapter를 추가한다
3. `register_shared_adapter_training_backend()`로 thin registry wiring에 등록
4. `TrainingObjectiveConfigPayload.training_backend_name` 값과 backend_name을 맞춤

**건드리지 않는 것:** `LocalTrainingService`는 backend를 주입받으므로 수정 불필요.

---

### 3. Example Generation Backend (agent)

**역할:** raw row 또는 stored scored event를 `EmbeddedTrainingExample`으로
재구성하는 방식. 역할상 `training input backend`에 가깝지만,
현재 코드 식별자는 `Example Generation Backend`를 유지한다.

**Protocol:**
```python
# agent/src/services/training/backends/inputs/base.py
class TrainingExampleBackend(Protocol):
    backend_name: str

    def build_examples(
        self,
        request: TrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]: ...

    def build_examples_from_stored_events(
        self,
        request: StoredEventTrainingExampleBuildRequest,
    ) -> tuple[EmbeddedTrainingExample, ...]: ...
```

**현재:** `PrototypeRescoringTrainingExampleBackend`
- 현재 adapter state를 적용하고
- 현재 prototype pack 기준으로 다시 점수 계산한 뒤
- 학습용 example로 변환한다.

**교체 시나리오:**
- cached score 재사용 backend
- feedback-only example backend
- generic multiview augmentation example backend

**현재 추가 구현:**
- `weak_strong_pair`
  - source row에 weak/strong view가 미리 준비돼 있으면 multiview example을 만든다.
  - 현재는 stored scored event 재구성 경로를 아직 지원하지 않는다.

**교체 절차:**
1. `training/input_backends/` 아래에 새 backend 구현 파일을 추가한다
2. `register_training_example_backend()`로 thin registry wiring에 등록
3. `TrainingObjectiveConfigPayload.example_generation_backend_name`으로 선택

---

### 4. Evidence Backend (agent)

**역할:** `ScoredEvent`나 다른 방법별 신호를 공통 `PseudoLabelEvidence`로
정규화하는 방식.

**Protocol:**
```python
# agent/src/services/training/backends/evidence/base.py
class PseudoLabelEvidenceBackend(Protocol):
    backend_name: str

    def build_evidence(
        self,
        *,
        scored_event: ScoredEvent,
    ) -> PseudoLabelEvidence: ...
```

**현재:** `PrototypeSimilarityEvidenceBackend`
- `ScoredEvent.category_scores`를 top1/top2/margin 기반 evidence로 정규화한다.
- 현재 baseline에서는 `confidence_kind=prototype_similarity`를 사용한다.

**교체 시나리오:**
- classifier posterior evidence backend
- teacher-student agreement evidence backend

**교체 절차:**
1. `training/evidence_backends/` 아래에 새 backend 구현 파일을 추가한다
2. `register_pseudo_label_evidence_backend()`로 thin registry wiring에 등록
3. `TrainingObjectiveConfigPayload.evidence_backend_name`으로 선택

---

### 5. Scorer Backend (methods/agent/scripts)

**역할:** 카테고리 score를 만드는 전체 방식 자체를 고른다.
`score_policy`보다 한 단계 위 축이다.

**예시 구분:**
- scorer backend: prototype similarity, learned scorer g, calibrated scorer
- scoring policy: 같은 scorer backend 안에서 `max`, `top-k mean` 같은 집계 방식

**교체 절차:**
1. 재사용 가능한 prototype score 계산이면 `methods/prototype/scoring/`에 먼저 둔다
2. `scoring_backends.py`에 새 backend 추가
3. `register_scoring_backend()`로 등록
4. query buffer/projection에 남길 `confidence_kind`를 backend가 직접 선언
5. `TrainingObjectiveConfigPayload.scorer_backend_name`으로 선택
6. 필요하면 scripts의 threshold/prototype 전략도 같은 축을 타게 맞춤

---

### Query SSL Augmenter (agent/scripts)

**역할:** consistency family가 소비할 unlabeled text view를 canonical shape로 준비한다.

현재 strict USB NLP baseline에서는 아래를 고정한다.

- weak view: `text`
- strong view: `aug_0`, `aug_1` 중 랜덤 1개
- canonical unlabeled row shape: `text + aug_0 + aug_1`

현재 구현:
- `nllb_backtranslation`
  - scripts preparation/cache가 agent의 일반 backtranslation service를 호출해 `aug_0`, `aug_1`를 생성한다
- `precomputed_usb_candidates`
  - 이미 `aug_0`, `aug_1`가 있는 JSONL을 그대로 소비한다

source of truth:
- Hydra group: `conf/query_ssl_augmenter/*.yaml`
- reusable core: `agent/src/services/backtranslation_service.py`
- scripts preparation/cache: `scripts/experiments/lora_classifier/query_ssl/augmentation.py`

주의:
- `query_ssl_method`와 `query_ssl_augmenter`는 분리한다.
- objective가 같아도 input generation/caching 정책은 별도 축으로 본다.

---

### 6. Privacy Guard (agent)

**역할:** local training 후 delta를 서버에 보내기 전 privacy 보호 적용.

**Protocol:**
```python
# agent/src/services/training/execution/privacy_guard_service.py
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
2. `register_shared_adapter_privacy_guard()`로 thin registry wiring에 등록
3. `TrainingObjectiveConfigPayload.privacy_guard_name`으로 선택

**Envelope 영향:** `clipped`, `dp_applied` 필드가 각 guard의 결과를 반영함.
새 guard 추가 시 해당 필드를 올바르게 채울 것.

---

### 7. Pseudo-label Acceptance Policy (agent)

**역할:** `PseudoLabelEvidence`를 pseudo-label acceptance decision으로 해석한다.

**Protocol:**
```python
# agent/src/services/training/acceptance_policies/base.py
class PseudoLabelAcceptancePolicy(Protocol):
    policy_name: str

    def evaluate(
        self,
        *,
        evidence: PseudoLabelEvidence,
        confidence_threshold: float,
        margin_threshold: float,
    ) -> AcceptanceDecision: ...
```

**현재:** `Top1MarginThresholdAcceptancePolicy` — confidence ≥ threshold AND margin ≥ threshold.

**교체 시나리오:**
- TopK 기반 선별
- Calibrated confidence 기반 policy

**교체 절차:**
1. `training/acceptance_policies/` 아래에 새 Policy 구현 파일을 추가한다
2. `register_pseudo_label_acceptance_policy()`로 thin registry wiring에 등록
3. `TrainingObjectiveConfigPayload.acceptance_policy_name`으로 선택한다

주의:
- central query-domain 실험에서는 compatibility field로만 남을 수 있다.
- 현재 bootstrap / pseudo-label self-training 실험의 selection hook은
  `methods/ssl/hooks/`가 소유한다.

---

### 7-1. Pseudo-label Selection Hook (methods/scripts)

**역할:** 중앙/FL SSL에서 pseudo-label evidence를 어떤 selection hook으로
해석할지 결정한다.

**현재 구현:**
- `top1_margin_threshold`
- `top1_confidence_only`

**구성 위치:**
- hook 코어:
  - `methods/ssl/hooks/selection.py`
  - `methods/ssl/hooks/registry.py`
- agent adapter:
  - `agent/src/services/training/selection/pseudo_label_service.py`
  - `agent/src/services/training/acceptance_policies/`
- scripts preset:
  - `conf/pseudo_label_algorithm/*.yaml`

**교체 절차:**
1. `methods/ssl/hooks/selection.py`에 selection hook 구현을 추가한다
2. `methods/ssl/hooks/registry.py`에 얇게 등록한다
3. `TrainingObjectiveConfigPayload.pseudo_label_algorithm_name`으로 선택한다
4. scripts에서는 `pseudo_label_algorithm=<preset>`으로 preset을 고른다
5. acceptance runtime 검증이 필요하면 별도 `acceptance_policy_name`을 함께 둔다

---

### 7-2. Query SSL Algorithm (agent/scripts)

**역할:** 중앙 query adaptation trainer에서 `LoRA + classifier` SSL algorithm 자체를 바꾼다.

**현재 구현:**
- `fixmatch`

**구성 위치:**
- method 코어:
  - `methods/ssl/base.py`
  - `methods/ssl/registry.py`
  - `methods/ssl/fixmatch/fixmatch.py`
- agent 학습 loop:
  - `agent/src/services/training/query_classifier_adaptation/training.py`
- scripts family runner:
  - `scripts/experiments/lora_classifier/query_ssl/common.py`
  - `scripts/experiments/lora_classifier/query_ssl/consistency_runner.py`
- scripts preset:
  - `conf/query_ssl_method/*.yaml`
  - `conf/query_source/*.yaml`

**교체 절차:**
1. `methods/ssl/<algorithm_name>/` 아래에 algorithm core를 구현한다
2. algorithm adapter가 `QuerySslAlgorithm`를 만족하게 만든다
3. `methods/ssl/registry.py`에 `algorithm_name`으로 등록한다
4. `train_query_ssl_classifier(...)` 공통 loop에는 새 objective만 주입한다
5. scripts에서는 `query_ssl_method=<preset>`과 `query_source=<preset>`으로 선택한다
6. scripts runner의 `QuerySslAlgorithmAdapter` registry에 loader/preparation wiring을 추가한다
7. weak/strong view가 필요한 objective면 source row contract와 export helper를 함께 맞춘다
8. 같은 family runner 안에서는 algorithm-specific wiring만 추가하고, eval/artifact 껍데기는 재사용한다

주의:
- `FixMatch`는 selection rule이 아니라 adaptation objective다.
- 즉 기존 `methods/ssl/hooks/selection.py` selection hook과 같은 축으로 넣지 않는다.
- USB `fixmatch.py::train_step`의 수식 코어는 `algorithms/fixmatch/algorithm.py`에 두고,
  USB `AlgorithmBase`가 맡던 loop/iterator/hook orchestration만 TraceMind trainer adapter로 둔다.
- `query_ssl_method` manifest는 `parameters`를 canonical method parameter map으로 남기고,
  기존 report consumer 호환을 위해 현재 parameter를 top-level에도 함께 남긴다.

---

### 8. Scoring Policy (methods/prototype)

**역할:** 임베딩과 prototype 간 유사도 계산 방식.

**Protocol:**
```python
# methods/prototype/scoring/base.py
class PrototypeScorePolicy(Protocol):
    def score_category(
        self,
        *,
        embedding_vector: Sequence[float],
        prototype_vectors: tuple[tuple[float, ...], ...],
        similarity_name: str,
        category: str,
    ) -> float: ...
```

**현재:** `MaxCosineScorePolicy` (최대 cosine), `TopKMeanCosineScorePolicy` (상위 K 평균).

**교체 시나리오:**
- Mahalanobis distance 기반 policy
- Learnable similarity (학습 가능한 metric)

**교체 절차:**
1. `methods/prototype/scoring/policies.py`에 새 클래스 추가
2. `register_prototype_score_policy()`로 thin registry wiring에 등록
3. `ScoringService.from_objective_config()`가 읽는 `TrainingObjectiveConfigPayload.score_policy_name`과 맞춤

---

### 7. Aggregation Backend (main_server)

**역할:** 여러 agent update를 합쳐 새 전역 adapter state를 만드는 방식.

**Protocol:**
```python
# main_server/src/services/federation/rounds/aggregation/registry.py
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

**현재:** `fedavg`
- 순수 계산 core: `methods/federated/aggregation/fedavg/`
- server boundary adapter: `DiagonalScaleAggregationService`,
  `ClassifierHeadFedAvgAggregationService`
- example_count 가중 평균 기반

**교체 시나리오:**
- q-합의: 최소 q개 update가 합의할 때만 반영하는 방식. `agent_id`로 per-agent 신뢰도 추적.
- Byzantine-robust aggregation: 이상 update를 제거하는 방식 (Krum, Bulyan 등)
- 가중 FedAvg: example_count 대신 trust score로 가중

**교체 절차:**
1. 순수 계산은 `methods/federated/aggregation/` 아래 method core로 추가한다.
2. `main_server`의 `aggregation/`에는 `SharedAdapterAggregationBackend` adapter와
   registry wiring만 추가한다.
3. `register_shared_adapter_aggregation_backend()` 또는 family 조합 지점에 등록한다.
4. q-합의 등 trust 기반이라면 `TrainingUpdateEnvelope.agent_id`를 사전 단계에서 활용한다.

**참고 필드:** `client_metrics`의 `ClientMetricKeys` 상수들 (`mean_confidence`, `delta_l2_norm` 등)이
aggregation 품질 판단에 활용 가능.

---

### 8. Update Acceptance Policy (main_server)

**역할:** agent가 보낸 update를 round에 받아들일지, 중복으로 볼지 결정.

**Protocol:**
```python
# main_server/src/services/federation/rounds/acceptance/policies.py
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
- `CompositeRoundUpdateAcceptancePolicy` — network policy와 trust policy를 조합
- `StrictRoundNetworkPolicy` / `IdempotentRoundNetworkPolicy` — update_id 재전송 처리
- `AllowAllRoundTrustPolicy` / `SingleSubmissionPerAgentTrustPolicy` — 제출 주체 규칙

**교체 시나리오:**
- `agent_id` 기반 중복 차단: 같은 round에 같은 agent가 두 번 제출 불가
- Trust score 기반 사전 필터: 신뢰도 임계값 미달 agent update 거부

**교체 절차:**
1. `acceptance/network_policies.py` 또는 `acceptance/trust_policies.py`에 정책을 추가
2. `CompositeRoundUpdateAcceptancePolicy`에 조합해 `round_lifecycle_service.py`에서 주입
3. trust 판단과 idempotency 판단을 같은 클래스에 섞지 않음

---

### 9. Adapter Family 추가 (shared/main_server)

**역할:** diagonal_scale 외 다른 adapter 구조 도입.

**변경 범위:** 가장 넓다. 여러 계층이 함께 바뀐다.

| 파일 | 변경 내용 |
|---|---|
| `shared/src/contracts/adapter_contracts.py` | 새 State/Update payload 클래스 추가 |
| `shared/src/domain/entities/training/` | 새 도메인 객체 추가 |
| `agent/src/services/training/backends/training/` | 새 family용 backend 추가 |
| `methods/federated/aggregation/` | 새 family용 aggregation 계산 core 추가 |
| `main_server/src/services/federation/rounds/aggregation/` | 새 family용 server aggregation adapter 추가 |
| `main_server/src/services/federation/rounds/families/` | 라우팅 등록 |
| `main_server/src/services/federation/rounds/boundary/mappers.py` | 새 payload 변환 추가 |

**현재 concrete:** `diagonal_scale` (임베딩 차원별 scale 보정).

**미래 후보:** LoRA adapter, projection head, prefix tuning.

구조 원칙:
- aggregation backend, privacy guard, secure aggregation runtime도 같은 철학으로 본다.
- 즉 `계약 -> protocol/base -> 구현체 파일 -> 얇은 wiring -> singular config source`
  순서를 유지하고, trust policy / encryption scheme / aggregation rule을 한 클래스에
  섞지 않는다.

### PEFT Adapter Builder (methods/scripts)

**역할:** 중앙 query adaptation rail에서 frozen backbone 위에 얹는 PEFT adapter
생성 방식을 바꾼다.

**현재 구현:**
- `lora`
- `rslora`

**구성 위치:**
- `methods/adaptation/peft/base.py`
- `methods/adaptation/peft/registry.py`
- `methods/adaptation/lora/lora_adapter.py`
- `conf/lora/*.yaml`

**교체 절차:**
1. `methods/adaptation/<method_name>/` 아래에 builder 구현을 추가한다
2. `PeftAdapterBuilder`를 만족하게 만든다
3. `methods/adaptation/peft/registry.py`에 `peft_adapter_name`으로 등록한다
4. `conf/lora/<preset>.yaml`에서 `peft_adapter_name`과 method별
   하이퍼파라미터를 둔다

주의:
- DoRA/IA3는 이름만 미리 열지 않는다. 실제 PEFT config 생성과 manifest summary가
  준비될 때 builder로 추가한다.

### Secure Update Codec (shared/agent/main_server)

**역할:** agent가 만든 update envelope을 server acceptance/aggregation 전에
secure aggregation 또는 encryption 제출 형태로 변환한다.

**현재 구현:**
- `noop`

**구성 위치:**
- `shared/src/services/secure_update_codec.py`
- agent encode 지점: `LocalTrainingService.secure_update_codec`
- server decode 지점: `RoundLifecycleService.secure_update_codec`

**교체 절차:**
1. codec 구현이 `SecureUpdateCodec`을 만족하게 만든다
2. agent encode와 server decode 양쪽에서 같은 envelope metadata 계약을 따른다
3. `TrainingTask.secure_aggregation`과 `TrainingUpdateEnvelope.secure_aggregation`
   필드 의미를 함께 검증한다
4. aggregation backend가 plaintext를 요구하는지 secure aggregate를 요구하는지
   명시한다

---

## 현재 구조에서 바로 쉬운 교체와 아닌 교체

### 바로 쉬운 교체

- 같은 계약 안의 새 `training_backend`
- 새 `example_generation_backend`
- 새 `scorer_backend`
- 새 `score_policy`
- 새 `acceptance_policy`
- 새 `privacy_guard`
- 같은 family 안의 새 `aggregation_backend`

### 아직 계약 확장이 필요한 교체

- `LoRA`처럼 state/update payload가 달라지는 family
- multiview objective처럼 example generation과 training backend가 함께 바뀌는 방식
- `PabLO`처럼 learned scorer artifact를 fit/store/load해야 하는 방식
- 실제 동형암호/secure aggregation runtime

즉 registry는 "같은 계약 안의 교체"를 쉽게 만드는 장치이고,
계약 자체가 달라지는 경우까지 자동으로 해결하지는 않는다.

---

## 전략 교체 시 공통 체크리스트

```
[ ] 바뀌는 축이 하나의 Protocol 구현 교체로 끝나는지 확인
    → 그렇지 않으면 설계 경계를 먼저 검토

[ ] shared/src/contracts/ 중 영향 받는 payload 필드 확인
    → 새 메타데이터가 필요하면 해당 contract에 선택 필드로 추가

[ ] agent가 보내고 server가 읽는 것이 같은 계약인지 확인
    → ClientMetricKeys, TrainingUpdateEnvelopePayload

[ ] thin registry wiring 또는 family registration에 새 구현체 추가
    → register_*() 또는 family 조합 지점을 사용

[ ] 기존 구현체를 제거하지 않고 새 것을 추가 (교체는 나중에)
    → 스위치 전에 병행 운용 가능한 구조 확인

[ ] 단위 테스트
    → 기존 테스트가 깨지지 않는지 확인 (회귀)
    → 새 구현체에 대한 최소 케이스 추가

[ ] 운영 후보 구현체는 shared/agent/main_server의 소유 경계에 바로 둔다
    → scripts는 config, sweep, report, visualization만 담당
```

---

## 안 건드려도 되는 것

| 파일 | 이유 |
|---|---|
| `shared/src/contracts/fl_round_contracts.py` | agent↔server 통신 계약. 알고리즘과 무관 |
| `agent/src/services/federation/round_client.py` | HTTP 통신 레이어. 알고리즘 교체와 무관 |
| `agent/src/services/federation/runtime_service.py` | orchestration 흐름. 같은 계약 안의 backend 교체와 무관 |
| `agent/src/api/training.py` | API endpoint. 교체 축 선택은 task/objective config가 담당 |
| `shared/src/contracts/prototype_contracts.py` | prototype 동기화 계약. adapter 교체와 독립 |

---

## 관련 문서

- [shared_adapter_contracts_v1.md](shared_adapter_contracts_v1.md) — adapter payload 구조와 수학적 의미
- [training_update_envelope_v1.md](training_update_envelope_v1.md) — envelope 필드 설계 이유
- AGENTS.md (repo root) — 의존 방향 원칙 및 소유 경계
