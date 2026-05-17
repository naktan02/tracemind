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
| SSL strong view policy | `first_aug`, `second_aug`, `row_parity_aug`, `query_id_hash_aug` | `query_ssl_strong_view_policy` scalar | `methods/adaptation/query_classifier_adaptation/data.py` | 중앙 control |
| Query classifier adaptation | supervised, bootstrap, pseudo-label, prototype SSL, FixMatch entrypoints | `conf/entrypoints/central_ssl_control/*` | `methods/adaptation/lora_classifier/*` + query data glue | 중앙 control |
| PEFT adapter | `lora`, `rslora` | `strategy_axes/adaptation/peft_adapter` | `methods/adaptation/*` | 중앙 control seam |

## Agent local runtime 축

| 축 | 현재 값 | 선택 위치 | core/runtime | 상태 |
|---|---|---|---|---|
| Training backend | `diagonal_scale_heuristic`, `lora_classifier_trainer` | `TrainingObjectiveConfig.training_backend_name` | `methods/adaptation/*` core + agent runtime adapter | 활성 runtime / FL simulation scaffold |
| Example generation | `prototype_rescore`, `weak_strong_pair` | `TrainingObjectiveConfig.example_generation_backend_name` | `methods/prototype/training_inputs/*` core + agent runtime adapter | 활성 runtime |
| Evidence backend | `prototype_similarity_evidence` | `TrainingObjectiveConfig.evidence_backend_name` | `methods/prototype/evidence/*` core + agent runtime adapter | 활성 runtime |
| Scorer backend | `prototype_similarity`, `classifier_head_logits` | `TrainingObjectiveConfig.scorer_backend_name` | `methods/prototype/scoring/*` core + agent runtime adapter | 활성 runtime |
| Score policy | `max_cosine`, `topk_mean_cosine` | `TrainingObjectiveConfig.score_policy_name` | `methods/prototype/scoring/policy_registry.py` + `score_policies/*` | 활성 runtime |
| Acceptance policy | `top1_margin_threshold`, `top1_confidence_only` | `TrainingObjectiveConfig.acceptance_policy_name` | `methods/ssl/hooks/selection.py` + agent compatibility metadata | 활성 runtime |
| Privacy guard | `diagonal_scale_clip_only`, `classifier_head_clip_only`, `noop` | `TrainingObjectiveConfig.privacy_guard_name` | agent privacy service | 활성 runtime |

## FL/서버 축

| 축 | 현재 값 | 선택 위치 | core/runtime | 상태 |
|---|---|---|---|---|
| Shard policy | `label_dominant`, `dirichlet_alpha03`, `dirichlet_alpha01` | `strategy_axes/fl/shard_policy` | `methods/federated/shard_policy/*` | simulation |
| Client labeled/unlabeled split | materialized manifest or runtime split fallback | `materialize_fl_client_split.py`, `fl_client_split_materialization.labeled_policy`, `fl_data.*`, `client_pool_split.*` | manifest preserves source selection, labeled policy, `weak=text`, `strong=[aug_0, aug_1]` | simulation |
| FL SSL method spec | `fedavg_pseudo_label` | `strategy_axes/fl/method_descriptor` | `methods/federated_ssl/*`, simulation adapter | simulation baseline |
| FL method execution plan | `method_owned`, `manual` | `fl_method.composition_mode` | `methods/federated_ssl/execution_plan.py` | simulation validator |
| FL local-update profile | `prototype_pseudo_label_v1`, `prototype_top1_confidence_v1`, `lora_pseudo_label_v1` | `strategy_axes/fl/local_update_profile` -> `cfg.local_update_profile` | agent training/evidence/scoring/privacy runtime | simulation/runtime profile |
| Aggregation backend | `fedavg` | `round_runtime.aggregation_backend_name` | reusable backend는 `methods/federated/aggregation/fedavg/*` + `methods/adaptation/<family>/aggregation/fedavg.py`, method-only 변형은 `methods/federated_ssl/<method>/` + main_server generic aggregation executor | 활성 runtime |
| Adapter family | `diagonal_scale`, `classifier_head`, `lora_classifier` | `round_runtime.adapter_family_name`, model/update manifest | shared contracts, main_server generic family runtime | 활성 runtime / server aggregation scaffold |
| FL local train budget | `local_epochs`, `batch_size`, `max_steps`, `query_ssl_method.unlabeled_batch_size` | `training_task.*`, `query_ssl_method.unlabeled_batch_size` | `scripts/runtime_adapters/federated_agent/query_ssl_lora_classifier_trainer.py` + `methods/adaptation/lora_classifier/training.py` | simulation |
| Update acceptance | composite round policy | main_server round service | main_server acceptance service | 활성 runtime |
| Security policy | `plaintext` | `security_policy.name` | `methods/federated_ssl/execution_plan.py`, future secure-update runtime capability | simulation validator |
| Secure update codec | `noop` | shared service/runtime wiring | `shared/src/services/secure_update_codec.py` | 활성 placeholder |
| Evaluation report metric | classification report payload | FL/central evaluation adapter | `methods/evaluation/*` | 활성 |

주의:

- `local_update_profile`은 local update 실행 조합 preset이고,
  `round_runtime.adapter_family_name`과 `round_runtime.aggregation_backend_name`은
  server round 실행 조합을 직접 고르는 leaf다. method-specific local objective와
  server policy 의미는 profile이 아니라 `methods/`의 descriptor/policy module이
  소유한다.
- method identity와 local/server policy 의미는 `methods/`가 소유한다. `agent`와
  `main_server`의 backend/adapter는 선택된 core를 runtime data, artifact, contract
  payload에 연결하는 capability다.
- `fl_method.composition_mode=method_owned`에서는 FedMatch 같은 상위 method가
  client objective와 server policy를 소유한다. `manual`은 논문 method가 아니라
  `query_ssl_method/round_runtime.*`를 직접 조합하는 baseline, ablation용
  모드다. report에 남기는 `client_ssl_objective/server_aggregation/update_family`는
  `query_ssl_method.algorithm_name`과 최종 `round_runtime.*` leaf에서 파생한다.
- `security_policy`는 method identity가 아니라 runtime capability 축이다. 현재는
  `plaintext`만 지원하며, secure aggregation/DP/암호화 artifact ref는 이후 capability
  adapter와 validator를 붙일 자리로 남긴다.
- 논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다.
  method-only 변형은 이 폴더에 남기고, 두 개 이상 방법론에서 공유되는 aggregation,
  adapter projection, SSL hook은 축별 methods 패키지로 승격한다.

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
- `lora_classifier`는 FL simulation research path를 1차 범위로 열고, live
  `agent`/`main_server` runtime translation은 2차 범위로 둔다.
- `lora_classifier`는 기존 `classifier_head` family의 옵션이 아니다. LoRA
  adapter state와 classifier head state를 함께 표현하는 별도 family로 설계한다.
- `lora_classifier` 기본 scaffold는 `mxbai_encoder`, LoRA
  `rank=8/alpha=16/dropout=0.1/target_modules=all-linear`, canonical seed
  checkpoint, label schema, split, seed, metric을 고정한다.
- Agent `lora_classifier_trainer`는 fixed embedding-only example을 받지 않는다.
  source row metadata 또는 translated text에서 agent-local raw text를 읽고,
  shared update payload에는 raw text 없이 LoRA/classifier artifact ref와 통계만 남긴다.
  live stored-event runtime translation은 raw text 저장 경계가 정리될 때까지 2차 범위다.
- LoRA-classifier FedAvg 경로는 inline LoRA parameter delta와 classifier-head
  delta의 FedAvg shape/version을 methods-owned core로 검증한다. Server-owned
  `aggregation_artifact::` JSON artifact-ref update는 main_server가 제공한 loader로
  materialize한다. client는 base revision 기준 delta를 보내고, 서버는
  `base global state + aggregated delta`를 다음 LoRA/head state artifact snapshot으로
  저장한다. `agent-local://` ref는 upload 경로가 붙기 전까지 거부한다.
- FL simulation 기본 조합은 `composition_mode=manual`,
  `strategy_axes/ssl/consistency_method=fixmatch_usb_v1`,
  `local_update_profile=lora_pseudo_label_v1`,
  `round_runtime.adapter_family_name=lora_classifier`,
  `round_runtime.aggregation_backend_name=fedavg`다. `diagonal_scale` baseline은
  `local_update_profile=prototype_pseudo_label_v1`와
  `round_runtime.adapter_family_name=diagonal_scale`를 함께 override한다.
  이 조합은 기존 `fedavg_pseudo_label` method descriptor를 유지한다.
- 기본 manual LoRA-classifier simulation 경로는 `query_ssl_method.algorithm_name`으로
  `methods/ssl/algorithms/*`를 resolve하고, client별 `labeled_rows`와
  `unlabeled_rows`로 실제 PEFT LoRA/classifier local optimizer를 실행한다.
  update는 server-owned uploaded artifact ref로 제출된다. 실제 step 수는
  `min(training_task.max_steps, training_task.local_epochs * full_epoch_steps)`이며,
  `training_task.batch_size`와 `query_ssl_method.unlabeled_batch_size`가
  `full_epoch_steps`를 바꾼다.
- `inline_delta`는 debug/compatibility 경로로만 유지한다. FL simulation의
  agent-local artifact upload, server-owned materialization, manifest/version
  compatibility는 닫혔다. live `agent`/`main_server` runtime translation은 winner
  method 확정 뒤 별도 범위로 진행한다.
