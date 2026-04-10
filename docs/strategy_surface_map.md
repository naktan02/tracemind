# 전략 표면 맵

## 목적

이 문서는 TraceMind에서 **지금 바꿀 수 있는 전략 축**과
**아직 계약만 있고 runtime은 없는 축**을 한눈에 보여주는 참조 문서다.

질문을 아래처럼 나눠서 답하게 만드는 것이 목표다.

1. 이 전략 축은 어디서 고르는가
2. 기본값 source of truth는 어디인가
3. 실험에서 바로 override 가능한가
4. 새 구현을 추가하는 것과 기본값을 바꾸는 것은 어디서 나뉘는가
5. 현재 runtime이 실제로 구현된 축인가, 아니면 metadata만 있는 축인가

전략을 **추가**하는 절차는
[docs/contracts/strategy_addition_playbook.md](/home/jmgjmg102/tracemind_server/docs/contracts/strategy_addition_playbook.md)를
보고, 교체 가능한 전략 지점 목록은
[docs/contracts/algorithm_extension_guide.md](/home/jmgjmg102/tracemind_server/docs/contracts/algorithm_extension_guide.md)를
본다.

## 읽는 법

- `선택 위치`
  - 어떤 config field나 runtime config가 해당 전략 이름을 고르는지
- `기본값`
  - 사용자가 값을 안 줬을 때 어디에서 fallback 되는지
- `실험 override`
  - Hydra leaf override나 config 값만으로 바로 바꿔볼 수 있는지
- `상태`
  - `활성 runtime`, `코드 주입 지점`, `typed metadata only` 중 무엇인지

## 0. 모델/전처리 축

| 축 | 현재 값 | 선택 위치 | 기본값 | 실험 override | 상태 |
|---|---|---|---|---|---|
| Scripts/Prototype Embedding Adapter | `mxbai`, `hash_debug` | Hydra `embedding` group -> `EmbeddingAdapterSpec` -> `EmbeddingAdapterFactory.create(...)` | 각 script의 `defaults: - /embedding: ...` | `embedding=mxbai`, `embedding=hash_debug`, `embedding.model_id=...` | 활성 runtime |
| Agent Embedding Model | 현재 기본 config는 `mxbai_large` | `agent/conf/embedding/*.yaml` | `agent/conf/config.yaml` | agent Hydra config override | 활성 runtime |
| Agent Translation Model | `nllb_200_distilled_600m`, `opus_mt_tc_big_ko_en` | `agent/conf/translation/*.yaml` + `TranslationService` | `agent/conf/config.yaml` | agent Hydra config override | 활성 runtime |
| Prototype Translation Provenance | `translation_model_id`, `translation_model_revision`, `translation_direction` | `scripts/conf/prototypes/seed_prototypes.yaml`, `scripts/conf/experiments/run_federated_simulation.yaml` | 대부분 `null` | leaf override 가능 | typed metadata only |

중요:

- 임베딩 모델 교체는 scripts와 agent 양쪽 모두 실제 runtime knob가 있다.
- 번역 모델 교체는 현재 **agent inference runtime**에서는 실제 모델 교체 knob다.
- 반면 prototype seeding/rebuild의 `translation_model_*` 필드는 현재
  번역 adapter를 실제로 실행하는 설정이라기보다
  `이 pack/build state가 어떤 translation provenance를 가정했는지`를 남기는
  metadata에 가깝다.
- `InferencePipelineService.translation_locales`는 현재 코드 기본값
  (`ko`, `ja`, `zh`)으로 관리되며 Hydra leaf로 노출돼 있지는 않다.

관련 파일:

- [agent/src/infrastructure/model_adapters/embedding/factory.py](/home/jmgjmg102/tracemind_server/agent/src/infrastructure/model_adapters/embedding/factory.py)
- [shared/src/domain/value_objects/embedding_adapter_spec.py](/home/jmgjmg102/tracemind_server/shared/src/domain/value_objects/embedding_adapter_spec.py)
- [agent/conf/embedding/mxbai_large.yaml](/home/jmgjmg102/tracemind_server/agent/conf/embedding/mxbai_large.yaml)
- [agent/conf/translation/nllb_200_distilled_600m.yaml](/home/jmgjmg102/tracemind_server/agent/conf/translation/nllb_200_distilled_600m.yaml)
- [agent/conf/translation/opus_mt_tc_big_ko_en.yaml](/home/jmgjmg102/tracemind_server/agent/conf/translation/opus_mt_tc_big_ko_en.yaml)
- [agent/src/services/inference/pipeline_service.py](/home/jmgjmg102/tracemind_server/agent/src/services/inference/pipeline_service.py)
- [scripts/prototypes/seed_prototypes.py](/home/jmgjmg102/tracemind_server/scripts/prototypes/seed_prototypes.py)
- [scripts/conf/prototypes/seed_prototypes.yaml](/home/jmgjmg102/tracemind_server/scripts/conf/prototypes/seed_prototypes.yaml)

## 1. agent 로컬 학습/추론 전략 축

| 축 | 현재 값 | 선택 위치 | 기본값 | 실험 override | 상태 |
|---|---|---|---|---|---|
| Training Backend | `diagonal_scale_heuristic`, `classifier_head_fixmatch_consistency` | `TrainingObjectiveConfig.training_backend_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.training_backend_name` | 활성 runtime |
| Algorithm Profile | `prototype_pseudo_label_v1`, `prototype_top1_confidence_v1`, `fixmatch_v1` | `TrainingObjectiveConfig.algorithm_profile_name`, Hydra `training_algorithm_profile` group | `shared/src/config/training_default_values.py`, `scripts/conf/training_algorithm_profile/` | `training_algorithm_profile=...`, `training_task.objective.algorithm_profile_name=...` | 활성 runtime |
| Example Generation Backend | `prototype_rescore`, `weak_strong_pair` | `TrainingObjectiveConfig.example_generation_backend_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.example_generation_backend_name` | 활성 runtime |
| Evidence Backend | `prototype_similarity_evidence`, `fixmatch_weak_view_evidence` | `TrainingObjectiveConfig.evidence_backend_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.evidence_backend_name` | 활성 runtime |
| Scorer Backend | `prototype_similarity`, `classifier_head_logits` | `TrainingObjectiveConfig.scorer_backend_name` | `shared/src/config/training_defaults.py` | `prototype_strategy`의 `runner.scorer_backend_name`, `run_federated_simulation`의 `training_task.objective.scorer_backend_name` | 활성 runtime |
| Score Policy | `max_cosine`, `topk_mean_cosine` | `TrainingObjectiveConfig.score_policy_name` + `score_top_k` | `shared/src/config/training_defaults.py` | `prototype_strategy`의 `runner.score_policy_name`, `runner.score_top_k`, `run_federated_simulation`의 `training_task.objective.score_policy_name`, `score_top_k` | 활성 runtime |
| Pseudo-label Acceptance Policy | `top1_margin_threshold`, `top1_confidence_only` | `TrainingObjectiveConfig.acceptance_policy_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.acceptance_policy_name` | 활성 runtime |
| Acceptance Threshold | `confidence_threshold`, `margin_threshold` | `TrainingObjectiveConfig` 필드 | `shared/src/config/training_defaults.py` | `prototype_strategy`의 `runner.confidence_threshold`, `runner.margin_threshold`, `run_federated_simulation`의 `confidence_threshold`, `margin_threshold` | 활성 runtime |
| Privacy Guard | `diagonal_scale_clip_only`, `classifier_head_clip_only`, `noop` | `TrainingObjectiveConfig.privacy_guard_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.privacy_guard_name` | 활성 runtime |

추가 설명:

- scoring은 `scorer backend`와 `score policy` 두 축으로 나뉜다.
- pseudo-label selection은 `example generation -> evidence backend -> acceptance policy`로 분리돼 있다.
- `weak_strong_pair` backend는 official FixMatch류 multiview input을 위한 활성 runtime이다.
- `fixmatch_weak_view_evidence`는 weak-view confidence gating backend다.
  `classifier_head_logits` scorer와 결합되면 classifier posterior를 직접 사용한다.
- 다만 현재 agent의 stored event 재구성 경로는 weak/strong view를 저장하지 않으므로 `prototype_rescore`만 안전하게 지원한다.
- `run-current-task` live agent 경로는 stored-event 재구성을 지원하지 않는 backend나
  active shared_state가 필요한 scorer를 받으면 `unsupported_runtime`으로 조기 종료한다.
- 실험에서는 위 축을 하나씩 직접 override할 수도 있고, `algorithm profile`로 묶어 한 번에 바꿀 수도 있다.
- acceptance는 `policy`와 `threshold`가 분리돼 있다.
- privacy는 현재 `clip only`와 `noop`만 runtime 구현이 있다.

관련 파일:

- [agent/src/services/training/training_backends/__init__.py](/home/jmgjmg102/tracemind_server/agent/src/services/training/training_backends/__init__.py)
- [agent/src/services/training/input_backends/__init__.py](/home/jmgjmg102/tracemind_server/agent/src/services/training/input_backends/__init__.py)
- [agent/src/services/federation/training_example_service.py](/home/jmgjmg102/tracemind_server/agent/src/services/federation/training_example_service.py)
- [agent/src/services/training/evidence_backends/__init__.py](/home/jmgjmg102/tracemind_server/agent/src/services/training/evidence_backends/__init__.py)
- [shared/src/config/training_algorithm_profiles.py](/home/jmgjmg102/tracemind_server/shared/src/config/training_algorithm_profiles.py)
- [scripts/conf/training_algorithm_profile/prototype_pseudo_label_v1.yaml](/home/jmgjmg102/tracemind_server/scripts/conf/training_algorithm_profile/prototype_pseudo_label_v1.yaml)
- [agent/src/services/inference/scoring_backends.py](/home/jmgjmg102/tracemind_server/agent/src/services/inference/scoring_backends.py)
- [agent/src/services/inference/scoring_policies.py](/home/jmgjmg102/tracemind_server/agent/src/services/inference/scoring_policies.py)
- [agent/src/services/training/acceptance_policies/__init__.py](/home/jmgjmg102/tracemind_server/agent/src/services/training/acceptance_policies/__init__.py)
- [agent/src/services/training/privacy_guard_service.py](/home/jmgjmg102/tracemind_server/agent/src/services/training/privacy_guard_service.py)

## 2. main_server round/runtime 전략 축

| 축 | 현재 값 | 선택 위치 | 기본값 | 실험 override | 상태 |
|---|---|---|---|---|---|
| Adapter Family | `diagonal_scale`, `classifier_head` | `ServerRoundRuntimeConfig.adapter_family_name` | `main_server/src/services/rounds/runtime_config.py` | `run_federated_simulation`의 `round_runtime.adapter_family_name` | 활성 runtime |
| Aggregation Backend | `fedavg` | `ServerRoundRuntimeConfig.aggregation_backend_name` | `main_server/src/services/rounds/runtime_config.py` | `run_federated_simulation`의 `round_runtime.aggregation_backend_name` | 활성 runtime |
| Classifier Head Bootstrap Scale | `8.0` 기본값 | `FederatedRoundRuntimeConfig.classifier_head_bootstrap_logit_scale` | `scripts/conf/experiments/run_federated_simulation.yaml` | `run_federated_simulation`의 `round_runtime.classifier_head_bootstrap_logit_scale` | simulation 전용 |
| Update Acceptance Network Policy | `StrictRoundNetworkPolicy`, `IdempotentRoundNetworkPolicy` | `StrictRoundUpdateAcceptancePolicy` / `IdempotentRoundUpdateAcceptancePolicy` 구성 | runtime factory / DI | public Hydra leaf 없음 | 코드 주입 지점 |
| Update Trust Policy | `AllowAllRoundTrustPolicy`, `SingleSubmissionPerAgentTrustPolicy` | acceptance policy 구성 객체 내부 | runtime factory / DI | public Hydra leaf 없음 | 코드 주입 지점 |

중요:

- `aggregation backend`는 문자열로 고를 수 있지만, 현재 `run_federated_simulation` Hydra config에는 top-level knob로 노출돼 있지 않다.
- `update acceptance`는 교체 지점은 있지만 registry나 public config key보다 코드 주입 형태에 가깝다.
- `classifier_head` family bootstrap은 현재 category별 centroid 하나를 classifier weight로 바꾸므로
  `prototype_builder=single`일 때만 안전하다.

관련 파일:

- [main_server/src/services/rounds/runtime_config.py](/home/jmgjmg102/tracemind_server/main_server/src/services/rounds/runtime_config.py)
- [main_server/src/services/rounds/aggregation_service.py](/home/jmgjmg102/tracemind_server/main_server/src/services/rounds/aggregation_service.py)
- [main_server/src/services/rounds/adapter_family_service.py](/home/jmgjmg102/tracemind_server/main_server/src/services/rounds/adapter_family_service.py)
- [main_server/src/services/rounds/update_acceptance_policy.py](/home/jmgjmg102/tracemind_server/main_server/src/services/rounds/update_acceptance_policy.py)

## 3. 보호/암호화 관련 축

### 3-1. 현재 실제 runtime이 있는 것

| 축 | 현재 값 | 선택 위치 | 상태 |
|---|---|---|---|
| Local clipping guard | `diagonal_scale_clip_only`, `classifier_head_clip_only` | `TrainingObjectiveConfig.privacy_guard_name` + `gradient_clip_norm` | 활성 runtime |
| No-op guard | `noop` | `TrainingObjectiveConfig.privacy_guard_name` | 활성 runtime |

반영 위치:

- envelope의 `clipped`
- envelope의 `dp_applied`

관련 파일:

- [agent/src/services/training/privacy_guard_service.py](/home/jmgjmg102/tracemind_server/agent/src/services/training/privacy_guard_service.py)
- [docs/contracts/training_update_envelope_v1.md](/home/jmgjmg102/tracemind_server/docs/contracts/training_update_envelope_v1.md)

### 3-2. typed contract는 있으나 runtime은 아직 없는 것

| 축 | 계약 필드 | 현재 상태 |
|---|---|---|
| Secure Aggregation Backend | `TrainingTask.secure_aggregation.aggregation_backend_name` | typed metadata only |
| Encryption Scheme | `TrainingTask.secure_aggregation.encryption_scheme_name` | typed metadata only |
| Key Reference | `TrainingTask.secure_aggregation.key_ref` | typed metadata only |
| Ciphertext Format | `TrainingTask.secure_aggregation.ciphertext_format` | typed metadata only |
| Secure submission metadata | `TrainingUpdateEnvelope.secure_aggregation` | typed metadata only |
| DP noise runtime | `dp_applied` 필드만 계약에 존재 | 실제 guard 미구현 |

중요:

- 지금은 `secure_aggregation` 관련 field를 task/envelope에 실을 수는 있다.
- 하지만 실제 secure aggregation, HE, DP runtime은 아직 붙지 않았다.
- 따라서 이 축은 “지금 당장 전략 비교가 가능한 active runtime knob”가 아니라
  “future runtime을 위한 typed contract”로 읽어야 한다.

관련 파일:

- [shared/src/contracts/training_contracts.py](/home/jmgjmg102/tracemind_server/shared/src/contracts/training_contracts.py)
- [docs/contracts/training_task_v1.md](/home/jmgjmg102/tracemind_server/docs/contracts/training_task_v1.md)
- [docs/project_execution_plan.md](/home/jmgjmg102/tracemind_server/docs/project_execution_plan.md)

## 4. 실험에서 바로 돌려볼 수 있는 조정 축

### 4-1. Prototype strategy experiment

entrypoint:

- [scripts/experiments/prototype_strategy_experiment.py](/home/jmgjmg102/tracemind_server/scripts/experiments/prototype_strategy_experiment.py)
- config: [scripts/conf/experiments/prototype_strategy.yaml](/home/jmgjmg102/tracemind_server/scripts/conf/experiments/prototype_strategy.yaml)

바로 조절 가능한 값:

- `embedding`
- `runner.confidence_threshold`
- `runner.margin_threshold`
- `runner.scorer_backend_name`
- `runner.score_policy_name`
- `runner.score_top_k`
- `strategy.name`
- `strategy.kmeans_candidate_ks`
- `strategy.dbscan_eps_values`
- `strategy.dbscan_min_samples_values`

예시:

```bash
python -m scripts.experiments.prototype_strategy_experiment \
  embedding=hash_debug \
  runner.confidence_threshold=0.7 \
  runner.margin_threshold=0.05 \
  runner.score_policy_name=topk_mean_cosine \
  runner.score_top_k=2 \
  strategy.name=kmeans \
  strategy.kmeans_candidate_ks=[2,3,4]
```

### 4-2. Prototype threshold sweep

entrypoint:

- [scripts/experiments/prototype_threshold_sweep.py](/home/jmgjmg102/tracemind_server/scripts/experiments/prototype_threshold_sweep.py)
- config: [scripts/conf/experiments/prototype_threshold_sweep.yaml](/home/jmgjmg102/tracemind_server/scripts/conf/experiments/prototype_threshold_sweep.yaml)

바로 조절 가능한 값:

- `embedding`
- `runner.scorer_backend_name`
- `runner.score_policy_name`
- `runner.score_top_k`
- `threshold_policies[0].thresholds`
- `threshold_policies[1].target_errors`
- `threshold_policies[2].target_errors`
- `threshold_policies[2].default_confidence_threshold`

예시:

```bash
python -m scripts.experiments.prototype_threshold_sweep \
  embedding=hash_debug \
  runner.score_policy_name=topk_mean_cosine \
  runner.score_top_k=2 \
  threshold_policies[0].thresholds=[0.7,0.8,0.9] \
  threshold_policies[1].target_errors=[0.03,0.05,0.1]
```

### 4-3. Federated simulation

entrypoint:

- [scripts/experiments/run_federated_simulation.py](/home/jmgjmg102/tracemind_server/scripts/experiments/run_federated_simulation.py)
- config: [scripts/conf/experiments/run_federated_simulation.yaml](/home/jmgjmg102/tracemind_server/scripts/conf/experiments/run_federated_simulation.yaml)

바로 조절 가능한 값:

- `embedding`
- `confidence_threshold`
- `margin_threshold`
- `training_task.objective.training_backend_name`
- `training_task.objective.example_generation_backend_name`
- `training_task.objective.scorer_backend_name`
- `training_task.objective.score_policy_name`
- `training_task.objective.score_top_k`
- `training_task.objective.acceptance_policy_name`
- `training_task.objective.privacy_guard_name`
- `training_task.selection_policy.max_examples`
- `validation.scorer_backend_name`
- `validation.score_policy_name`
- `validation.score_top_k`

예시:

```bash
python -m scripts.experiments.run_federated_simulation \
  embedding=hash_debug \
  confidence_threshold=0.7 \
  margin_threshold=0.05 \
  training_task.objective.score_policy_name=topk_mean_cosine \
  training_task.objective.score_top_k=2 \
  training_task.objective.privacy_guard_name=noop \
  validation.score_policy_name=topk_mean_cosine \
  validation.score_top_k=2
```

주의:

- `aggregation_backend_name`과 `adapter_family_name`은 현재 simulation Hydra top-level knob로는 직접 노출돼 있지 않다.
- `secure_aggregation.*` 값을 config에 실을 수는 있지만 실제 secure aggregation runtime을 실험하는 것은 아니다.
- `prototype_rebuild.translation_model_*` 필드는 provenance metadata를 남기는 축이지,
  simulation 중 실제 translation adapter를 바꾸는 축은 아니다.

## 5. 빠른 판별 규칙

처음 보는 사람이 전략 변경 가능 여부를 볼 때는 아래 순서로 보면 된다.

1. 이 문서에서 축이 `활성 runtime`인지 `typed metadata only`인지 먼저 본다.
2. `선택 위치`가 `TrainingObjectiveConfig`면 task override로 바꿔볼 수 있다.
3. `기본값`이 `shared/src/config/training_defaults.py`면 local training 공용 fallback이다.
4. `기본값`이 `main_server/src/services/rounds/runtime_config.py`면 server-owned round runtime fallback이다.
5. `public Hydra leaf 없음`이라고 적혀 있으면, 코드에는 교체 지점이 있어도 CLI에서 바로 안 바뀐다.
