# 전략 표면 맵

이 문서는 현재 TraceMind에서 바꿀 수 있는 전략 축과 아직 metadata만 있는 축을
짧게 보여준다. 추가 절차는 `docs/contracts/algorithm_extension_guide.md`와
`docs/contracts/strategy_addition_playbook.md`를 따른다.

## 현재 기본 방향

```text
central fixed embedding + classifier seed
-> central SSL pooled/offline control
-> FL SSL non-IID main comparison
-> FL/runtime translation
```

- 최종 method/runtime 구조 판단은
  `docs/architecture/target-method-runtime-structure.md`를 우선한다. 이 문서는 현재
  실행 표면과 legacy compatibility 이름을 함께 보여준다.
- 현재 표의 historical `adapter_family_name`, `lora_classifier`, `fedmatch_agreement`는
  migration 기록에서만 보일 수 있다. target 구조에서는 각각
  `payload_adapter_kind`/`update_family_name`, `peft_text_encoder`,
  method-local FedMatch objective로 정리한다. `lora_classifier` shared parser/factory와
  report/result reader fallback은 제거됐으므로 새 실험 산출물은 `peft_classifier`
  payload format을 쓴다.
- 중앙 SSL은 pooled/offline control이다.
- 논문 메인 비교는 `FL SSL under non-IID`에서 수행한다.
- runtime v1은 `embedding -> global classifier -> local interpretation` baseline을 우선한다.
- prototype은 bootstrap/comparison/reference artifact와 일부 local mechanism으로 유지한다.
- prototype 기반 pseudo-label/SSL은 중앙 SSL/FL SSL 비교군에 포함한다.

## 읽는 법

- `선택 위치`: 어떤 config나 runtime field가 전략 이름을 고르는가.
- `기본값`: 값이 없을 때 fallback 되는 source of truth.
- `상태`: `활성 runtime`, `중앙 control`, `simulation`, `metadata only` 중 하나.

## 실행/모델 축

| 축 | 현재 값 | 선택 위치 | 기본값 | 상태 |
|---|---|---|---|---|
| Dataset asset | `ourafla`, `cssrs`, `mental_health_kaggle` | `execution_context/dataset_asset` | entrypoint defaults | 활성 |
| Embedding adapter | `mxbai`, `hash_debug` | `execution_context/embedding_adapter` | entrypoint defaults | 활성 |
| Query split artifact | `dataset_default`, `ourafla_ssl_*`, `szegeelim_general4_*` | `execution_context/query_split` | entrypoint defaults | 활성 |
| Query data source selection | `ourafla_reddit`, `szegeelim_general4` for labeled/unlabeled/validation/test | `execution_context/query_data_source` + `query_data_selection.*` | all `ourafla_reddit` | 활성 |
| Query text view | `szegeelim_general4_*_nllb_v1` | `execution_context/query_view` | data pipeline entrypoint defaults | 활성 |
| FL client split manifest | `runtime_split_from_train`, `materialized_client_split` | `fl_data.source_mode`, `fl_data.split_manifest` | runtime split for debug, manifest for paper comparison | simulation |
| Runtime env | `cpu_local`, `gpu_local`, `gpu_online`, `auto_local`, `auto_online` | `execution_context/runtime_env` | entrypoint defaults | 활성 |
| Transformer backbone | `mxbai_encoder` | `strategy_axes/adaptation/transformer_backbone` | central SSL defaults | 중앙 control |
| Initial checkpoint | `canonical_fixed_classifier_seed`, `none`, `required` | `strategy_axes/adaptation/initial_checkpoint` | Query SSL method comparison defaults to `none` | 중앙 control |

## SSL/adaptation 축

| 축 | 현재 값 | 선택 위치 | core | 상태 |
|---|---|---|---|---|
| Pseudo-label selection | `fixed_confidence_095`, `margin_threshold_v1` | `strategy_axes/ssl/pseudo_label_selection` | `methods/ssl/hooks/*` | 활성/중앙 control |
| Consistency method | `pseudolabel_usb_v1`, `fixmatch_usb_v1`, `flexmatch_usb_v1`, `freematch_usb_v1`, `adamatch_usb_v1` | `strategy_axes/ssl/consistency_method` | `methods/ssl/algorithms/*` | 중앙 control / FL manual baseline |
| SSL augmentation source | `precomputed_usb_candidates_v1` | `strategy_axes/ssl/augmentation_source` | scripts runner | 중앙 control |
| SSL strong view policy | `first_aug`, `second_aug`, `row_parity_aug`, `query_id_hash_aug` | `query_ssl_strong_view_policy` scalar | `methods/adaptation/query_text_views/data.py` | 중앙 control |
| Query text view glue | supervised, bootstrap, pseudo-label, prototype SSL, FixMatch entrypoints | `conf/entrypoints/central_ssl_control/*` | `methods/adaptation/peft_text_encoder/*` + `methods/adaptation/query_text_views/*` | 중앙 control |
| PEFT adapter | `lora`, `rslora` | `strategy_axes/adaptation/peft_adapter` | `methods/adaptation/peft_adapters/*` | 중앙 control seam |
| Central SSL run budget | `smoke`, `main` | `run_controls/central_ssl/budget` | smoke는 `runs/_smoke`, main은 `runs`; batch/epoch/step budget과 output root를 함께 고른다 | 중앙 control |

## Agent local runtime 축

| 축 | 현재 값 | 선택 위치 | core/runtime | 상태 |
|---|---|---|---|---|
| Training backend | `peft_classifier_trainer` | `TrainingObjectiveConfig.training_backend_name` | `methods/adaptation/*` core + agent runtime adapter | FL simulation scaffold |
| Example generation | `prototype_rescore`, `weak_strong_pair` | `TrainingObjectiveConfig.example_generation_backend_name` | `methods/prototype/training_inputs/*` core + agent runtime adapter | 활성 runtime |
| Evidence backend | `prototype_similarity_evidence` | `TrainingObjectiveConfig.evidence_backend_name` | `methods/prototype/evidence/*` core + agent runtime adapter | 활성 runtime |
| Scorer backend | `prototype_similarity`, `classifier_head_logits` | `TrainingObjectiveConfig.scorer_backend_name` | `methods/prototype/scoring/*`, `methods/classification/linear_head/scoring.py` core + agent runtime adapter | 활성 runtime |
| Score policy | `max_cosine`, `topk_mean_cosine` | `TrainingObjectiveConfig.score_policy_name` | `methods/prototype/scoring/policy_registry.py` + `score_policies/*` | 활성 runtime |
| Acceptance policy | `top1_margin_threshold`, `top1_confidence_only` | `TrainingObjectiveConfig.acceptance_policy_name` | `methods/ssl/hooks/selection.py` + agent compatibility metadata | 활성 runtime |
| Privacy guard | `classifier_head_clip_only`, `noop` | `TrainingObjectiveConfig.privacy_guard_name` | `methods/adaptation/privacy_guards/*` core + agent executor | 활성 runtime |

## FL/서버 축

| 축 | 현재 값 | 선택 위치 | core/runtime | 상태 |
|---|---|---|---|---|
| Shard policy | `label_dominant`, `dirichlet_alpha03`, `dirichlet_alpha01` | `strategy_axes/fl/shard_policy` | `methods/federated/shard_policy/*` | simulation |
| Client labeled/unlabeled split | materialized manifest or runtime split fallback | `materialize_fl_client_split.py`, `fl_client_split_materialization.labeled_policy`, `fl_data.*`, `client_pool_split.*` | manifest preserves source selection, labeled pool selection policy, `weak=text`, `strong=[aug_0, aug_1]` | simulation |
| Labeled exposure policy | `shared_client_seed` 기본, `client_local_split` legacy/ablation, `server_only_seed` artifact/request metadata | `strategy_axes/fl/labeled_exposure_policy` + materialized manifest metadata | separates how many labeled rows are selected from where selected labeled rows are visible | simulation capability |
| Validation evaluator | `peft_classifier_eval` | `local_update_profile.validation_*` -> `validation.*` | `methods/adaptation/peft_text_encoder/evaluation.py` | FL SSL simulation |
| Client participation policy | `all_clients`, `fraction_random`, `fixed_count_random` | `strategy_axes/fl/client_participation_policy` | `methods/federated/participation.py`, round loop selection | simulation capability |
| Local supervision regime | `client_labeled_and_unlabeled`, `client_unlabeled_only`, `server_labeled_only` | `strategy_axes/fl/local_supervision_regime` | `methods/federated_ssl/capability_plan.py`, compatibility validator | metadata/validator |
| Server step policy | `none`, `supervised_seed_step` | `strategy_axes/fl/server_step_policy` | `methods/federated_ssl/capability_plan.py`, config-declared runtime adapter executor | simulation active; `supervised_seed_step` implementation is a PEFT encoder runtime adapter, not script-owned policy logic |
| Server update policy | `fedavg_merged_delta`, `fedmatch_partitioned` | `strategy_axes/fl/server_update_policy` | `methods/federated_ssl/capability_axes.py`, compatibility validator, adapter-family server update resolver | merged FedAvg active, partitioned PEFT-classifier update simulation active |
| Peer context policy | `none`, `fixed_probe_output_knn` (`prediction_similarity_topk` legacy alias) | `strategy_axes/fl/peer_context_policy` | `methods/federated_ssl/capability_plan.py`, simulation peer-context adapter, `methods/federated_ssl/peer_context.py` | `none` active, KDTree 우선 fixed-probe nearest-neighbor helper context selection and PEFT classifier helper weak-probability provider active |
| Peer probe surface | `label_balanced`, max 128 rows | `peer_probe.*` in FL simulation entrypoint | simulation fixed text probe selector + report protocol metadata | active for `fixed_probe_output_knn` helper selection |
| Update partition policy | `unified`, `partitioned` | `strategy_axes/fl/update_partition_policy` | common capability + method/adaptation partition helpers | `unified` active, `partitioned` method-gated |
| Local SSL policy | `profile_pseudo_label`, `fixmatch`, `flexmatch`, `freematch`, `adamatch`, `pseudolabel`, `fedmatch_agreement` | `strategy_axes/fl/local_ssl_policy` + `strategy_axes/ssl/consistency_method` | `methods/federated_ssl/capability_axes.py`, `methods/ssl/algorithms/*`, method-local objective | Query SSL-backed policies active in manual mode, FedMatch agreement active in method-owned slice |
| Aggregation weight policy | `example_count`, `uniform`, `accepted_count` | `strategy_axes/fl/aggregation_weight_policy` | `methods/federated/aggregation_weighting.py` + family FedAvg cores | simulation capability |
| Query multiview source | `materialized_rows`, `agent_generated_or_cached` | `strategy_axes/fl/query_multiview_source` | materialized JSONL rows now, live agent source later | materialized active, live planned |
| FL SSL method descriptor | `fedmatch` original core/config snapshot, future FedLGMatch/(FL)^2 | `strategy_axes/fl/method_descriptor` | `methods/federated_ssl/*`, simulation adapter | method-owned 전용 |
| FL method execution plan | `method_owned`, `manual` | `fl_method.composition_mode` | `methods/federated_ssl/execution_plan.py` | simulation validator |
| FL local-update profile | `peft_pseudo_label_v1` | `strategy_axes/fl/local_update_profile` -> `cfg.local_update_profile` | PEFT-classifier Query SSL training/evidence/scoring/privacy runtime | FL SSL simulation profile |
| Aggregation backend | `fedavg`, effective `partitioned_delta_average` for partitioned server update | `round_runtime.aggregation_backend_name` + update-family/server policy resolver | reusable backend는 `methods/federated/aggregation/*` + update family aggregation projection, method-only 변형은 `methods/federated_ssl/<method>/` + main_server generic aggregation executor | 활성 runtime |
| Update family | `linear_head`, `peft_text_encoder`, future `prototype_pack`; v1 adapter family는 compatibility field | `strategy_axes/trainable_state/update_family`, `round_runtime.update_family_name`, model/update manifest | `methods/classification/*`, `methods/adaptation/*`, `methods/prototype/*`, shared payload contract | 활성 runtime / server aggregation scaffold |
| FL local train budget | `local_epochs`, `batch_size`, `max_steps`, `query_ssl_method.unlabeled_batch_size` | `training_task.*`, `query_ssl_method.unlabeled_batch_size` | `scripts/runtime_adapters/federated_agent/peft_encoder_local_training.py`, `scripts/runtime_adapters/federated_agent/artifact_store.py` + `methods/adaptation/peft_text_encoder/training/` | simulation |
| Runtime resource cache | run-scoped in-memory cache | simulation bootstrap context | `methods.common` cache protocol + simulation implementation; adapter family consumes optional cache | simulation optimization, not method identity |
| FL run budget | `smoke`, `reduced`, `main` | `run_controls/fl_ssl/budget` | smoke는 `runs/_smoke/fl_ssl`, reduced/main은 `runs/fl_ssl`; reduced는 5 rounds, main은 30 rounds | simulation |
| Update acceptance | composite round policy | main_server round service | main_server acceptance service | 활성 runtime |
| Security policy | `plaintext` | `security_policy.name` | `methods/federated_ssl/execution_plan.py`, future secure-update runtime capability | simulation validator |
| Secure update codec | `noop` | shared service/runtime wiring | `shared/src/services/secure_update_codec.py` | 활성 placeholder |
| Evaluation report metric | classification report payload | FL/central evaluation adapter | `methods/evaluation/*` | 활성 |

주의:

- `local_update_profile`은 local update 실행 조합 preset이고,
  `strategy_axes/trainable_state/update_family`와
  `round_runtime.aggregation_backend_name`은 server round 실행 조합을 고르는 leaf다.
  v1 adapter-family 이름은 payload compatibility field로만 남아 있고,
  method-specific local objective와 server policy 의미는 profile이 아니라
  `methods/`의 descriptor/policy module이 소유한다.
- method identity와 local/server policy 의미는 `methods/`가 소유한다. `agent`와
  `main_server`의 backend/adapter는 선택된 core를 runtime data, artifact, contract
  payload에 연결하는 capability다.
- `fl_method.composition_mode=method_owned`에서는 FedMatch 같은 상위 method가
  client objective와 server policy를 소유한다. `manual`은 논문 method가 아니라
  `query_ssl_method/round_runtime.*`를 직접 조합하는 baseline, ablation용
  모드다. report에 남기는 `client_ssl_objective/server_aggregation/update_family`는
  `query_ssl_method.algorithm_name`과 최종 `round_runtime.*` leaf에서 파생한다.
  manual baseline은 method descriptor를 참조하지 않으며, report/index에는
  `execution_role=manual_baseline`과 `descriptor_name=null`로 남긴다.
- `security_policy`는 method identity가 아니라 runtime capability 축이다. 현재는
  `plaintext`만 지원하며, secure aggregation/DP/암호화 artifact ref는 이후 capability
  adapter와 validator를 붙일 자리로 남긴다.
- FL SSL capability plan은 method 전용 코드가 아니다. client participation,
  labeled exposure, server step, server update, peer context, update partition,
  local SSL policy, aggregation weight, query multiview source를 공통 축으로 고정하고,
  FedMatch 같은 method descriptor가
  필요한 값을 요구한다.
- `server_step_policy`는 server-side supervised seed step 여부이고,
  `server_update_policy`는 merged delta/FedMatch-style partitioned delta 같은 update
  해석 방식이다. 같은 이름 공간에 섞지 않는다.
- `local_ssl_policy=query_ssl_method`는 기존 `query_ssl_method.algorithm_name`을
  canonical 이름으로 가져온다. FixMatch/FlexMatch/FreeMatch 파라미터를 FL config에
  복제하지 않는다.
- 공통 update partition 축은 `partitioned`까지만 표현한다. FedMatch의
  `sigma/psi`처럼 partition 이름과 loss routing 의미는 method package가 소유하고,
  두 개 이상 method에서 같은 scheme이 반복될 때만 공통 scheme으로 승격한다.
- 공통 peer context 축은 `fixed_probe_output_knn` 같은 exchange mechanism까지만
  표현한다. nearest-neighbor 실행은 `methods/federated_ssl/peer_context.py`의
  KDTree 우선 index가 소유하고, FedMatch의 `num_helpers`, `h_interval` 같은 원본
  helper 기본값은 `methods/federated_ssl/fedmatch/original_spec.py`와 method package가
  소유한다. `prediction_similarity_topk`는 기존 config override 호환용 alias다.
  TraceMind fixed text probe surface는 `peer_probe.*` 실행 metadata가 소유하며,
  report protocol에 source, row count, label distribution, query id hash를 남긴다.
- 논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다.
  FedMatch는 원본 repository/commit과 config snapshot, confidence filter,
  agreement pseudo-label vote, helper selection, sigma/psi partition 의미,
  supervised/unsupervised tensor local objective를 method package에 고정했다.
  PEFT text-classifier logical partition 실행 loop와 method-owned local simulation bridge는
  `methods/adaptation/peft_text_encoder/federated_ssl/`의 method-neutral
  update-family runtime primitive가 소유한다. FedMatch의 `sigma/psi` 의미는
  `methods/federated_ssl/fedmatch/`에서 읽는다.
  현재 server path는 원본 sparse sigma/psi sync가 아니라 PEFT-classifier merged
  delta/FedAvg 또는 `server_update_policy=fedmatch_partitioned`에서
  PEFT-classifier `partitioned_delta_average` simulation backend다. 이 backend는
  원본 sparse sigma/psi sync 전체가 아니라 logical partition delta 평균 slice다.
  이전 round client-local PEFT snapshot 기반 helper weak-probability provider와
  labels-at-server supervised seed server step은 simulation에서 활성화했다. sparse
  S2C/C2S는 client-local previous partition snapshot과 partitioned global state 기준
  simulation slice로 활성화했다.
  method-only 변형은 이 폴더에 남기고, 두 개 이상
  방법론에서 공유되는 aggregation, adapter projection, SSL hook은 축별 methods
  패키지로 승격한다.
- method descriptor YAML은 원본 상세값을 복제하지 않는다. `scenario`와
  `parameter_overrides`만 실행 표면으로 열고, `original_parameters`와
  `effective_parameters`는 method-local `original_spec.py`에서 runner/report로 주입한다.
- Runtime resource cache는 method identity가 아니라 simulation/live runtime별 lifecycle
  concern이다. Protocol은 `methods.common`에 두고, simulation은 run-scoped in-memory
  cache를 소유한다. Live agent translation 때는 같은 protocol을 agent process/round
  lifecycle에 맞는 implementation으로 다시 연결한다.

## Prototype 축

| 축 | 현재 값 | 선택 위치 | core | 상태 |
|---|---|---|---|---|
| Build strategy | `single`, `kmeans` | `strategy_axes/prototype/build_strategy` | `methods/prototype/building/*` | 활성 |
| Scoring policy | `max_cosine`, `topk_mean_cosine` | training/scoring config | `methods/prototype/scoring/*` | 활성 |
| Evidence normalization | `prototype_similarity_evidence` | training objective config | `methods/prototype/evidence/*` | 활성 |
| Training input view | `prototype_rescore`, multiview helpers | training objective config | `methods/prototype/training_inputs/*` | 활성 |

Prototype SSL 비교군은 위 prototype mechanism을 SSL objective/selection과 조합한
method다. `PrototypePack` 자체가 최종 판정기라는 뜻은 아니다.

Prototype-only 평가 파일이 필요하면 `scripts/experiments/prototype_analysis/`나
FL simulation 아래 thin wrapper로 먼저 둔다. 여러 track에서 같은 metric 의미가
안정되면 계산 helper만 `methods/evaluation/`으로 승격한다.

## Metadata-only 또는 주의 축

- Prototype translation provenance는 pack/build state metadata이며 번역 실행 knob가 아니다.
- Central SSL control의 `acceptance_policy_name`은 일부 runtime compatibility field로 남아 있다.
- Multi-prototype runtime은 실험 분석에는 가능하지만 v1 active runtime 기본값은 아니다.
- `peft_text_encoder` update family는 FL simulation research path를 1차 범위로
  열고, live `agent`/`main_server` runtime translation은 capability adapter가 준비된
  뒤 연다.
- `lora_classifier`는 historical v1 family 이름이고, active 실행 config와
  report/result reader의 source-of-truth는 `peft_text_encoder` update family와
  `peft_classifier` payload kind다.
- `peft_text_encoder` 기본 scaffold는 `mxbai_encoder`, LoRA
  `rank=8/alpha=16/dropout=0.1/target_modules=all-linear`, canonical seed
  checkpoint, label schema, split, seed, metric을 고정한다.
- Agent `peft_classifier_trainer`는 fixed embedding-only example을 받지 않는다.
  source row metadata 또는 translated text에서 agent-local raw text를 읽고,
  shared update payload에는 raw text 없이 LoRA/classifier artifact ref와 통계만 남긴다.
  live stored-event runtime translation은 raw text 저장 경계가 정리될 때까지 2차 범위다.
- PEFT-classifier FedAvg 경로는 inline PEFT adapter delta와 classifier-head
  delta의 FedAvg shape/version을 methods-owned core로 검증한다. Server-owned
  `aggregation_artifact::` JSON artifact-ref update는 main_server가 제공한 loader로
  materialize한다. client는 base revision 기준 delta를 보내고, 서버는
  `base global state + aggregated delta`를 다음 LoRA/head state artifact snapshot으로
  저장한다. `agent-local://` ref는 upload 경로가 붙기 전까지 거부한다.
- FL simulation 기본 조합은 `composition_mode=manual`,
  `strategy_axes/ssl/consistency_method=fixmatch_usb_v1`,
  `local_update_profile=peft_pseudo_label_v1`,
  `strategy_axes/trainable_state/update_family=peft_text_encoder`,
  `round_runtime.aggregation_backend_name=fedavg`다. 기존 `diagonal_scale` shared
  payload family와 prototype scorer fallback은 FL SSL simulation에서 제거했으며, 실제 method로
  다시 필요해질 때 method-owned capability로 추가한다. 이 조합은
  `strategy_axes/fl/method_descriptor`를 compose하지 않는다.
- 기본 manual PEFT-classifier simulation 경로는 `query_ssl_method.algorithm_name`으로
  `methods/ssl/algorithms/*`를 resolve하고, client별 `labeled_rows`와
  `unlabeled_rows`로 실제 PEFT adapter/classifier local optimizer를 실행한다.
  update는 server-owned uploaded artifact ref로 제출된다. 실제 step 수는
  `min(training_task.max_steps, training_task.local_epochs * full_epoch_steps)`이며,
  `training_task.batch_size`와 `query_ssl_method.unlabeled_batch_size`가
  `full_epoch_steps`를 바꾼다.
- `inline_delta`는 debug/compatibility 경로로만 유지한다. FL simulation의
  agent-local artifact upload, server-owned materialization, manifest/version
  compatibility는 닫혔다. live `agent`/`main_server` runtime translation은 winner
  method 확정 뒤 별도 범위로 진행한다.
- live `main_server`의 no-config round runtime fallback은 server runtime config의
  named `default_peft_classifier.v1` profile이다. `RoundManagerService`는 이
  기본값을 소유하지 않는다. 논문 비교와 simulation entrypoint는 `conf/`의 명시
  `round_runtime.*` leaf를 source of truth로 사용한다.
