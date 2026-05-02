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
[docs/contracts/strategy_addition_playbook.md](contracts/strategy_addition_playbook.md)를
보고, 교체 가능한 전략 지점 목록은
[docs/contracts/algorithm_extension_guide.md](contracts/algorithm_extension_guide.md)를
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

## 현재 권장 기본선

현재 v1에서 권장하는 기본 실험선은 아래다.

주의:

- 이 문서는 현재 시스템/FL runtime에서 바로 고를 수 있는 축을 중심으로 정리한다.
- query-domain 적응 단계의 `central LoRA classifier` trainer는 별도 중앙 실험 레일이며, 이 표의 active runtime knob와 1:1 대응한다고 보지 않는다.

```text
Embedding
-> Global Classifier
-> Local Interpretation
-> Final Decision
```

중요:

- 전역 classifier는 공통 class evidence를 만드는 역할로 본다.
- 최종 판단은 로컬 `PersonalizationState`, 시계열 누적, decision policy가 수행한다.
- shared adapter와 prototype scoring은 제거 대상이 아니라 비교 실험/확장 축으로 유지한다.
- single prototype baseline은 허용하지만, multi-prototype runtime은 필요성이 확인될 때 다시 연다.

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

- [agent/src/infrastructure/model_adapters/embedding/factory.py](../agent/src/infrastructure/model_adapters/embedding/factory.py)
- [shared/src/domain/value_objects/embedding_adapter_spec.py](../shared/src/domain/value_objects/embedding_adapter_spec.py)
- [agent/conf/embedding/mxbai_large.yaml](../agent/conf/embedding/mxbai_large.yaml)
- [agent/conf/translation/nllb_200_distilled_600m.yaml](../agent/conf/translation/nllb_200_distilled_600m.yaml)
- [agent/conf/translation/opus_mt_tc_big_ko_en.yaml](../agent/conf/translation/opus_mt_tc_big_ko_en.yaml)
- [agent/src/services/inference/pipeline_service.py](../agent/src/services/inference/pipeline_service.py)
- [scripts/prototypes/seed_prototypes.py](../scripts/prototypes/seed_prototypes.py)
- [scripts/conf/prototypes/seed_prototypes.yaml](../scripts/conf/prototypes/seed_prototypes.yaml)

## 1. agent 로컬 학습/추론 전략 축

| 축 | 현재 값 | 선택 위치 | 기본값 | 실험 override | 상태 |
|---|---|---|---|---|---|
| Training Backend | `diagonal_scale_heuristic` | `TrainingObjectiveConfig.training_backend_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.training_backend_name` | 활성 runtime |
| Algorithm Profile | `prototype_pseudo_label_v1`, `prototype_top1_confidence_v1` | `TrainingObjectiveConfig.algorithm_profile_name`, Hydra `training_algorithm_profile` group | `shared/src/config/training_default_values.py`, `scripts/conf/training_algorithm_profile/` | `training_algorithm_profile=...`, `training_task.objective.algorithm_profile_name=...` | 활성 runtime |
| Example Generation Backend | `prototype_rescore`, `weak_strong_pair` | `TrainingObjectiveConfig.example_generation_backend_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.example_generation_backend_name` | 활성 runtime |
| Evidence Backend | `prototype_similarity_evidence` | `TrainingObjectiveConfig.evidence_backend_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.evidence_backend_name` | 활성 runtime |
| Scorer Backend | `prototype_similarity`, `classifier_head_logits` | `TrainingObjectiveConfig.scorer_backend_name` | `shared/src/config/training_defaults.py` | `prototype_strategy`의 `runner.scorer_backend_name`, `run_federated_simulation`의 `training_task.objective.scorer_backend_name` | 활성 runtime |
| Score Policy | `max_cosine`, `topk_mean_cosine` | `TrainingObjectiveConfig.score_policy_name` + `score_top_k` | `shared/src/config/training_defaults.py` | `prototype_strategy`의 `runner.score_policy_name`, `runner.score_top_k`, `run_federated_simulation`의 `training_task.objective.score_policy_name`, `score_top_k` | 활성 runtime |
| Pseudo-label Selection Algorithm | `top1_margin_threshold`, `top1_confidence_only` | `TrainingObjectiveConfig.pseudo_label_algorithm_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.pseudo_label_algorithm_name`, central 실험의 `pseudo_label_algorithm=<preset>` | 활성 runtime/central selection |
| Pseudo-label Acceptance Policy | `top1_margin_threshold`, `top1_confidence_only` | `TrainingObjectiveConfig.acceptance_policy_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.acceptance_policy_name` | 활성 runtime |
| Acceptance Threshold | `confidence_threshold`, `margin_threshold` | `TrainingObjectiveConfig` 필드 | `shared/src/config/training_defaults.py` | `prototype_strategy`의 `runner.confidence_threshold`, `runner.margin_threshold`, `run_federated_simulation`의 `confidence_threshold`, `margin_threshold` | 활성 runtime |
| Privacy Guard | `diagonal_scale_clip_only`, `classifier_head_clip_only`, `noop` | `TrainingObjectiveConfig.privacy_guard_name` | `shared/src/config/training_defaults.py` | `run_federated_simulation`의 `training_task.objective.privacy_guard_name` | 활성 runtime |

추가 설명:

- 현재 v1에서 권장 baseline은 `global classifier + local interpretation`이다.
- scoring은 `scorer backend`와 `score policy` 두 축으로 나뉜다.
- pseudo-label selection은 `example generation -> evidence backend -> selection algorithm -> acceptance policy`로 분리돼 있다.
- `weak_strong_pair` backend는 generic multiview input backend로 유지한다.
- classifier posterior는 공통 evidence로 읽고, final decision은 local interpretation이 계속 맡는다.
- 다만 현재 agent의 stored event 재구성 경로는 weak/strong view를 저장하지 않으므로 `prototype_rescore`만 안전하게 지원한다.
- `run-current-task` live agent 경로는 stored-event 재구성을 지원하지 않는 backend나
  active shared_state가 필요한 scorer를 받으면 `unsupported_runtime`으로 조기 종료한다.
- 실험에서는 위 축을 하나씩 직접 override할 수도 있고, `algorithm profile`로 묶어 한 번에 바꿀 수도 있다.
- acceptance는 `policy`와 `threshold`가 분리돼 있다.
- privacy는 현재 `clip only`와 `noop`만 runtime 구현이 있다.
- query-domain 중앙 `LoRA + classifier` 비교 레일은 위 active runtime knob와 별도다.
  bootstrap / pseudo-label self-training 실험의 selection rule source of truth는
  `scripts/conf/pseudo_label_algorithm/`이고, 구현 코어는
  `agent/src/services/training/query_adaptation/ssl/`이 소유한다.
- 중앙 query-domain consistency objective는 또 다른 별도 축이다.
  현재 `FixMatch`의 method/source source of truth는
  `scripts/conf/query_ssl_method/`, `scripts/conf/query_ssl_train_source/`,
  `scripts/conf/query_ssl_augmenter/`이고,
  USB core mapping은 `agent/src/services/training/query_adaptation/algorithms/fixmatch.py`가 소유한다.
  scripts family runner는 `scripts/experiments/lora_classifier/query_ssl/` 아래에서 공통화하고,
  strict USB NLP input preparation/cache는 `query_ssl/augmentation.py`가 담당한다.
  실제 backtranslation 메커니즘은 `agent/src/services/backtranslation_service.py`를 재사용한다.
- 현재 central 실험에서 selection source of truth는
  `pseudo_label_algorithm_name` / `scripts/conf/pseudo_label_algorithm/`이고,
  `acceptance_policy_name`은 runtime compatibility용 compatibility field로 유지된다.

관련 파일:

- [agent/src/services/training/backends/training/__init__.py](../agent/src/services/training/backends/training/__init__.py)
- [agent/src/services/training/backends/inputs/__init__.py](../agent/src/services/training/backends/inputs/__init__.py)
- [agent/src/services/training/examples/service.py](../agent/src/services/training/examples/service.py)
- [agent/src/services/training/backends/evidence/__init__.py](../agent/src/services/training/backends/evidence/__init__.py)
- [shared/src/config/training_algorithm_profiles.py](../shared/src/config/training_algorithm_profiles.py)
- [scripts/conf/training_algorithm_profile/prototype_pseudo_label_v1.yaml](../scripts/conf/training_algorithm_profile/prototype_pseudo_label_v1.yaml)
- [agent/src/services/inference/scoring_backends.py](../agent/src/services/inference/scoring_backends.py)
- [agent/src/services/inference/scoring_policies.py](../agent/src/services/inference/scoring_policies.py)
- [agent/src/services/training/acceptance_policies/__init__.py](../agent/src/services/training/acceptance_policies/__init__.py)
- [agent/src/services/training/query_adaptation/ssl/registry.py](../agent/src/services/training/query_adaptation/ssl/registry.py)
- [scripts/conf/pseudo_label_algorithm/margin_threshold_v1.yaml](../scripts/conf/pseudo_label_algorithm/margin_threshold_v1.yaml)
- [agent/src/services/training/query_adaptation/algorithms/fixmatch.py](../agent/src/services/training/query_adaptation/algorithms/fixmatch.py)
- [scripts/conf/query_ssl_method/fixmatch_usb_v1.yaml](../scripts/conf/query_ssl_method/fixmatch_usb_v1.yaml)
- [scripts/conf/query_ssl_train_source/dataset_default.yaml](../scripts/conf/query_ssl_train_source/dataset_default.yaml)
- [agent/src/services/training/execution/privacy_guard_service.py](../agent/src/services/training/execution/privacy_guard_service.py)

### 1-1. 아이용 지원 대화 전략 축

| 축 | 현재 값 | 선택 위치 | 기본값 | 실험 override | 상태 |
|---|---|---|---|---|---|
| Child Support Reply Provider | `local_guarded`, `ollama` | `TRACEMIND_CHILD_SUPPORT_LLM_PROVIDER` | unset -> `local_guarded` | 환경변수 `ollama`, `TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL` | 활성 runtime |
| Child Support Safety Policy | `supportive`, `check_in`, `parent_handoff`, `urgent` | `ChildSupportSafetyPolicy` | code default | public config 없음 | 활성 runtime |
| Child Support Safety Intent | `self_harm_signal`, `other_harm_ideation`, `other_harm_method_request`, `post_urgent_deescalation`, `peer_response_planning`, `post_handoff_emotional_followup`, etc. | `ChildSupportSafetyIntent` | code default | public config 없음 | agent-local runtime |
| Child Support Scope Policy | `in_scope`, `redirected` | `ChildSupportSafetyPolicy` | code default | public config 없음 | 활성 runtime |
| Child Support Response Policy | `scope_redirect`, `supportive_reflection`, `check_in`, `post_urgent_deescalation`, `post_incident_emotional_followup`, `peer_response_planning`, `safety_check`, `harm_to_others_safety`, `urgent_safety` | `ChildSupportResponsePolicy` | code default | public config 없음 | 활성 runtime |

중요:

- 이 축은 분류/FL 학습 family가 아니라 child UI 응답 생성 provider 축이다.
- raw child message, conversation history, query context는 agent-local 저장소와 prompt
  context에만 남기며 main_server로 올리지 않는다.
- 같은 `conversation_id`의 최근 `parent_handoff` 기록은 후속 메시지의 감정 정리와
  친구 대응 계획 strategy를 고르는 데 사용한다.
- 화면 노출용 `safety_level`은 shared contract에 남기고, agent 내부 분기에는
  typed `SafetyIntent`와 conversation state를 사용한다.
- 타인을 해치려는 의도나 방법 요청은 peer planning보다 먼저 `urgent`로 라우팅하고
  LLM rewrite를 우회한 guarded response를 쓴다.
- 타인 위해 intent 직후의 `너무 힘든데` 같은 follow-up은 일반 감정 선택 질문으로
  리셋하지 않고, 감정 수용과 위해 행동 경계를 함께 담은 de-escalation으로 처리한다.
- LLM provider는 응답 결정을 소유하지 않는다. agent가 먼저 response plan과
  required move를 만들고, LLM은 그 plan을 실행한 뒤 plan validation을 통과해야
  한다.
- cloud LLM provider를 열 경우에도 기본값으로 승격하지 말고 명시적 opt-in과
  prompt context 축소 정책을 별도로 둔다.

관련 파일:

- [agent/src/services/wellbeing/child_support_service.py](../agent/src/services/wellbeing/child_support_service.py)
- [agent/src/services/wellbeing/child_support_conversation_state.py](../agent/src/services/wellbeing/child_support_conversation_state.py)
- [agent/src/services/wellbeing/child_support_safety_intent.py](../agent/src/services/wellbeing/child_support_safety_intent.py)
- [agent/src/services/wellbeing/child_support_response_policy.py](../agent/src/services/wellbeing/child_support_response_policy.py)
- [agent/src/services/wellbeing/child_support_llm_provider.py](../agent/src/services/wellbeing/child_support_llm_provider.py)
- [agent/src/services/wellbeing/child_support_safety_policy.py](../agent/src/services/wellbeing/child_support_safety_policy.py)
- [agent/src/infrastructure/repositories/child_support_repository.py](../agent/src/infrastructure/repositories/child_support_repository.py)
- [shared/src/contracts/child_support_contracts.py](../shared/src/contracts/child_support_contracts.py)

## 2. main_server round/runtime 전략 축

| 축 | 현재 값 | 선택 위치 | 기본값 | 실험 override | 상태 |
|---|---|---|---|---|---|
| Adapter Family | `diagonal_scale`, `classifier_head` | `ServerRoundRuntimeConfig.adapter_family_name` | `main_server/src/services/federation/rounds/runtime/config.py` | `run_federated_simulation`의 `round_runtime.adapter_family_name` | 활성 runtime |
| Aggregation Backend | `fedavg` | `ServerRoundRuntimeConfig.aggregation_backend_name` | `main_server/src/services/federation/rounds/runtime/config.py` | `run_federated_simulation`의 `round_runtime.aggregation_backend_name` | 활성 runtime |
| Classifier Head Bootstrap Scale | `8.0` 기본값 | `FederatedRoundRuntimeConfig.classifier_head_bootstrap_logit_scale` | `scripts/conf/experiments/run_federated_simulation.yaml` | `run_federated_simulation`의 `round_runtime.classifier_head_bootstrap_logit_scale` | simulation 전용 |
| Update Acceptance Network Policy | `StrictRoundNetworkPolicy`, `IdempotentRoundNetworkPolicy` | `StrictRoundUpdateAcceptancePolicy` / `IdempotentRoundUpdateAcceptancePolicy` 구성 | runtime factory / DI | public Hydra leaf 없음 | 코드 주입 지점 |
| Update Trust Policy | `AllowAllRoundTrustPolicy`, `SingleSubmissionPerAgentTrustPolicy` | acceptance policy 구성 객체 내부 | runtime factory / DI | public Hydra leaf 없음 | 코드 주입 지점 |

중요:

- `aggregation backend`는 문자열로 고를 수 있지만, 현재 `run_federated_simulation` Hydra config에는 top-level knob로 노출돼 있지 않다.
- `update acceptance`는 교체 지점은 있지만 registry나 public config key보다 코드 주입 형태에 가깝다.
- `classifier_head` family bootstrap은 현재 category별 centroid 하나를 classifier weight로 바꾸므로
  `prototype_builder=single`일 때만 안전하다.
- aggregation도 SSL selection과 같은 구조 철학으로 본다.
  즉 `protocol/base -> 구현체 파일 -> 얇은 wiring -> config source of truth`를 유지하되,
  소유 경계는 `main_server`가 가진다.

관련 파일:

- [main_server/src/services/federation/rounds/runtime/config.py](../main_server/src/services/federation/rounds/runtime/config.py)
- [main_server/src/services/federation/rounds/aggregation/registry.py](../main_server/src/services/federation/rounds/aggregation/registry.py)
- [main_server/src/services/federation/rounds/families/registry.py](../main_server/src/services/federation/rounds/families/registry.py)
- [main_server/src/services/federation/rounds/acceptance/policies.py](../main_server/src/services/federation/rounds/acceptance/policies.py)

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

- [agent/src/services/training/execution/privacy_guard_service.py](../agent/src/services/training/execution/privacy_guard_service.py)
- [docs/contracts/training_update_envelope_v1.md](contracts/training_update_envelope_v1.md)

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
- 나중에 secure aggregation / encryption runtime을 붙일 때도 같은 구조 철학을 권장한다.
  즉 계약은 `shared`, runtime 구현은 `agent`/`main_server`, 조합은 얇은 config 또는
  DI에서 선택하고, key 관리/암호화 포맷/네트워크 전송 판단은 한 클래스에 섞지 않는다.

관련 파일:

- [shared/src/contracts/training_contracts.py](../shared/src/contracts/training_contracts.py)
- [docs/contracts/training_task_v1.md](contracts/training_task_v1.md)
- [docs/project_execution_plan.md](project_execution_plan.md)

## 4. 실험에서 바로 돌려볼 수 있는 조정 축

### 4-1. Prototype strategy experiment

entrypoint:

- [scripts/experiments/prototype_strategy_experiment.py](../scripts/experiments/prototype_strategy_experiment.py)
- config: [scripts/conf/experiments/prototype_strategy.yaml](../scripts/conf/experiments/prototype_strategy.yaml)

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

- [scripts/experiments/prototype_threshold_sweep.py](../scripts/experiments/prototype_threshold_sweep.py)
- config: [scripts/conf/experiments/prototype_threshold_sweep.yaml](../scripts/conf/experiments/prototype_threshold_sweep.yaml)

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

- [scripts/experiments/run_federated_simulation.py](../scripts/experiments/run_federated_simulation.py)
- config: [scripts/conf/experiments/run_federated_simulation.yaml](../scripts/conf/experiments/run_federated_simulation.yaml)

바로 조절 가능한 값:

- `embedding`
- `federated_run_preset`
- `federated_shard_policy`
- `federated_ssl_method`
- `federated_report`
- `round_runtime.adapter_family_name`
- `round_runtime.aggregation_backend_name`
- `confidence_threshold`
- `margin_threshold`
- `training_task.objective.training_backend_name`
- `training_task.objective.example_generation_backend_name`
- `training_task.objective.scorer_backend_name`
- `training_task.objective.score_policy_name`
- `training_task.objective.score_top_k`
- `training_task.objective.pseudo_label_algorithm_name`
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
  federated_run_preset=standard \
  federated_shard_policy=dirichlet_alpha03 \
  federated_ssl_method=fedavg_pseudo_label \
  federated_report=fl_ssl_main_comparison \
  confidence_threshold=0.7 \
  margin_threshold=0.05 \
  training_task.objective.score_policy_name=topk_mean_cosine \
  training_task.objective.score_top_k=2 \
  training_task.objective.privacy_guard_name=noop \
  validation.score_policy_name=topk_mean_cosine \
  validation.score_top_k=2
```

주의:

- `aggregation_backend_name`과 `adapter_family_name`은 `round_runtime.*` leaf override로 노출된다.
- `federated_run_preset=standard`는 `10 clients`, `50 rounds`를 기본 main budget으로 둔다.
- 기본 smoke preset은 실행 확인용으로 `4 clients`, `3 rounds`를 쓴다.
- `federated_shard_policy=dirichlet_alpha03`는 FL SSL main split,
  `dirichlet_alpha01`은 stress split이다.
- `federated_ssl_method=fedavg_pseudo_label`는 현재 active runtime baseline이다.
- 후보 논문 method는 확정 전까지 config/파일을 미리 추가하지 않는다.
- `federated_report=fl_ssl_main_comparison`은 중앙 SSL control과 섞이지 않는
  FL main comparison report schema를 고른다.
- `secure_aggregation.*` 값을 config에 실을 수는 있지만 실제 secure aggregation runtime을 실험하는 것은 아니다.
- `prototype_rebuild.translation_model_*` 필드는 provenance metadata를 남기는 축이지,
  simulation 중 실제 translation adapter를 바꾸는 축은 아니다.

## 5. 빠른 판별 규칙

처음 보는 사람이 전략 변경 가능 여부를 볼 때는 아래 순서로 보면 된다.

1. 이 문서에서 축이 `활성 runtime`인지 `typed metadata only`인지 먼저 본다.
2. `선택 위치`가 `TrainingObjectiveConfig`면 task override로 바꿔볼 수 있다.
3. `기본값`이 `shared/src/config/training_defaults.py`면 local training 공용 fallback이다.
4. `기본값`이 `main_server/src/services/federation/rounds/runtime/config.py`면 server-owned round runtime fallback이다.
5. `public Hydra leaf 없음`이라고 적혀 있으면, 코드에는 교체 지점이 있어도 CLI에서 바로 안 바뀐다.
