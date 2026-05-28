# Hydra Config Layout

`conf/`는 TraceMind 실험 조합과 method/runtime 파라미터의 루트 Hydra config
공간이다. 폴더명은 사람이 탐색할 때의 의미를 우선하고, YAML `# @package`는
기존 코드가 읽는 canonical config shape를 유지한다.

예를 들어 `execution_context/dataset_asset=ourafla`는 compose 후에도
`cfg.dataset`으로 들어가고, `strategy_axes/ssl/pseudo_label_selection=...`은
`cfg.pseudo_label_algorithm`으로 들어간다.

## 현재 분류

- `entrypoints/`: `@hydra.main(config_name=...)`이 읽는 실행 시작점별 root config.
- `execution_context/`: 여러 실험이 공유하는 데이터 자산, embedding adapter, runtime 환경.
- `strategy_axes/`: 중앙 SSL과 FL SSL이 공유하거나 각 track에서 교체하는 전략 축.
- `run_controls/`: 논문 비교 track의 실행 budget과 runner bridge.

```text
conf/
├── entrypoints/
│   ├── central_classifier_seed/
│   ├── central_ssl_control/
│   ├── dataset_pipeline/
│   ├── fl_ssl/
│   ├── prototype_analysis/
│   └── prototype_pack/
├── execution_context/
│   ├── dataset_asset/
│   ├── embedding_adapter/
│   ├── query_data_source/
│   ├── query_split/
│   ├── query_view/
│   └── runtime_env/
├── strategy_axes/
│   ├── adaptation/
│   │   ├── initial_checkpoint/
│   │   ├── peft_adapter/
│   │   └── transformer_backbone/
│   ├── fl/
│   │   ├── materialized_split/
│   │   ├── local_update_profile/
│   │   ├── method_descriptor/
│   │   └── shard_policy/
│   ├── prototype/
│   │   └── build_strategy/
│   └── ssl/
│       ├── augmentation_source/
│       ├── consistency_method/
│       └── pseudo_label_selection/
└── run_controls/
    ├── central_ssl/
    │   └── budget/
    └── fl_ssl/
        └── budget/
```

## 이름 기준

- entrypoint config는 실행 스크립트의 시작점이다.
- execution context는 방법론 비교가 아니라 실행 재료다.
- strategy axis는 실제로 교체 가능한 계산/정책 축이다.
- run control은 central SSL control, FL SSL처럼 비교표의 맥락 안에서 쓰는 실행 조건 묶음이다.

최종 method/runtime vocabulary는
`docs/architecture/target-method-runtime-structure.md`를 기준으로 한다. 현재
`round_runtime.adapter_family_name`은 v1 실행 field이고, prototype 기반 update까지
포괄하는 target 구조에서는 `update_family_name` 또는
`trainable_state_family_name`으로 전환한다.

`execution_context/query_data_source`는 query-domain 데이터 주소록과 선택값을
소유한다. `query_data_sources`에 source별 labeled/unlabeled/validation/test JSONL을
등록하고, 실행 시 `query_data_selection.labeled`, `unlabeled`, `validation`,
`test`만 override한다. 이 context는 기존 runner가 읽는 `train_jsonl`,
`unlabeled_jsonl`, `eval_sets.*`로 선택값을 변환한다.

## FL SSL config contract

FL SSL simulation은 config 의미가 겹치기 쉬우므로 아래처럼 읽는다.

- `strategy_axes/fl/method_descriptor`
  - `cfg.ssl_method`로 compose된다.
  - FedMatch/FedLGMatch/(FL)^2 같은 method-owned 논문 method identity, report
    role, custom runtime 필요 여부를 설명한다.
  - 논문 원본 기본값은 YAML에 복제하지 않고
    `methods/federated_ssl/<method>/original_spec.py`가 소유한다. YAML은
    `scenario`, `use_original_parameters`, `parameter_overrides` 같은 실행 표면만
    둔다.
  - 실제 local update 계산 조합을 단독으로 소유하지 않는다.
  - 기본 manual baseline은 이 그룹을 compose하지 않고,
    `query_ssl_method/local_update_profile/round_runtime.*` 조합으로 실행한다.
  - 사람이 읽는 method recipe metadata와 method-only 변형은
    `methods/federated_ssl/<method>/`가 소유한다.
- `strategy_axes/fl/local_update_profile`
  - `cfg.local_update_profile`로 compose된다.
  - agent local update를 만드는 training/evidence/scoring/privacy 조합을 소유한다.
  - adapter family나 aggregation backend를 소유하지 않는다.
- `strategy_axes/ssl/consistency_method`
  - `cfg.query_ssl_method`로 compose된다.
  - manual FL 조합에서 client SSL objective의 source of truth다.
  - 기본은 `fixmatch_usb_v1`이며, `flexmatch_usb_v1`,
    `freematch_usb_v1`, `pseudolabel_usb_v1` 같은 이름 변경은
    `methods/ssl/algorithms/*` descriptor를 resolve해야 한다.
  - 이 축은 report label이 아니라 실제 local optimizer loop의 algorithm 선택으로
    연결되어야 하며, 연결되지 않은 조합은 bootstrap 전에 실패해야 한다.
- `strategy_axes/trainable_state/update_family`, `round_runtime.aggregation_backend_name`
  - FL entrypoint의 update family와 aggregation backend 실행 leaf다.
  - `strategy_axes/trainable_state/update_family/*`가 `update_family_name`과
    `runtime_payload_key`, `composition_slug_builder`, `initial_state_builder`, 필요한 경우
    `validation_evaluator`를 함께 제공한다.
    scripts는 family 이름으로 분기하지 않고 이 callable들을 실행한다.
  - high-level compose preset은 중복 source-of-truth를 피하기 위해 두지 않는다.
  - PEFT text classifier backbone/adapter 세부 값은
    `strategy_axes/adaptation/transformer_backbone`과
    `strategy_axes/adaptation/peft_adapter`에서 온다.
  - v1 payload compatibility 때문에 active config에는
    `adapter_family_name=peft_classifier`도 남아 있지만, 새 실행 조합과 산출물
    slug는 `update_family_name=peft_text_classifier`를 기준으로 해석한다.
    `lora_classifier`는 old-run artifact/report compatibility reader에서만
    해석하고, prototype은 adapter가 아니라 `prototype_pack` update family로
    표현한다.
- `strategy_axes/fl/shard_policy`
  - `cfg.shard_policy`로 compose된다.
  - non-IID client split 방식만 소유한다.
- `strategy_axes/fl/materialized_split`
  - `cfg.fl_data.source_mode`, `cfg.fl_data.split_manifest`,
    `cfg.query_data_selection.*`를 함께 고르는 실행 selector다.
  - 논문 비교에서 긴 manifest path를 직접 override하지 않기 위한 축이며,
    method semantics나 labeled exposure policy 의미를 소유하지 않는다.
  - 현재 열린 selector는 `shared_client_seed` exposure의 10-client,
    Dirichlet alpha=0.3 split만 대상으로 한다.
  - source pair는 `shared_reddit_reddit_*`와 `shared_general_reddit_*`로
    분리한다. 여기서 `general_reddit`은 labeled source가
    `szegeelim_general4`, unlabeled/validation/test source가
    `ourafla_reddit`인 조합이다.
  - labeled budget은 `pc25`, `pc100`, `pc400`, `pc1024`처럼 class당 labeled
    row 수로 표현한다.
- `run_controls/fl_ssl/budget`
  - client 수, round budget, output dir 같은 FL SSL 실행 budget을 소유한다.
  - method semantics나 local update policy를 소유하지 않는다.
  - `smoke`는 wiring 검증 산출물을 `runs/_smoke/fl_ssl` 아래에 둬서
    논문/웹용 run과 섞이지 않게 한다.
  - `reduced`는 10 clients, 5 rounds 검증용 preset이고 `runs/fl_ssl`에 둔다.
  - `main`은 10 clients, 30 rounds full-budget preset이고 `runs/fl_ssl`에 둔다.
  - `output_dir`는 run root만 지정하고, runner가
    `<method_family>/<method_composition>/<split>/<clients_rounds>/<run_id>`를
    뒤에 붙인다.
- `run_controls/central_ssl/budget`
  - 중앙 SSL의 epoch/step/batch 크기 같은 반복 실행 budget을 소유한다.
  - `main.output_root=runs`, `smoke.output_root=runs/_smoke`로 나눠 중앙 SSL
    smoke/test 산출물이 논문/웹용 run과 섞이지 않게 한다.
- `training_task.local_epochs`, `training_task.batch_size`,
  `training_task.max_steps`
  - FL round에서 각 client가 수행하는 local optimizer 반복을 소유한다.
  - manual `Query SSL + PEFT-classifier`에서는 실제 step 수가
    `min(max_steps, local_epochs * full_epoch_steps)`로 계산된다.
  - `batch_size`는 labeled loader step 수를 바꾸고, 기본 설정에서는
    `query_ssl_method.unlabeled_batch_size=${training_task.batch_size}`라서
    unlabeled loader step 수도 함께 바꾼다. 필요하면
    `query_ssl_method.unlabeled_batch_size=<N>`으로 따로 override한다.
- `training_task.objective`
  - Hydra YAML에서는 `query_ssl`, `peft_classifier`, `evidence_backend`처럼
    owner scope별 nested mapping으로 적을 수 있다.
  - `shared` contract 경계에서는 기존 report/runtime compatibility를 위해
    `query_ssl.algorithm_name`, `peft_classifier.delta_format` 같은 dotted flat key로
    정규화된다.
  - 새 adapter family 값을 추가할 때도 entrypoint에 긴 quoted dotted key를 늘리기보다
    owner scope 아래에 묶고, canonical 해석은 `TrainingObjectiveConfig`가 맡긴다.
- `seed_sweep`
  - FL SSL seed sweep runner가 순회할 seed 목록과 sweep output root를 소유한다.
  - `seed_sweep.seeds` 길이는 `report.seed_count`와 같아야 한다.
  - sweep root도 단일 실행과 같은 split/method-composition 계층을 따른다.
- `client_count_sweep`
  - FL SSL client-count sweep runner가 순회할 client 수 목록과 sweep output root를
    소유한다.
  - method semantics, shard policy, local update profile은 기존 FL config 축을
    그대로 쓰고 client 수만 바꾼다.
  - 각 child run은 sweep root 아래 `clients_<NN>`로 분리한다.
- `run_safety`
  - FL SSL runner가 시작 전에 확인하는 accidental long-run guard다.
  - 단일 run은 `rounds`, seed/client-count sweep은 `rounds * sweep 항목 수`로 총
    예정 communication round를 계산한다.
  - 장시간 실행 승인 여부만 소유하며, method semantics나 budget 값 자체는
    소유하지 않는다.
  - `budget=main`처럼 긴 실행은 명시적으로 선택하고, guard를 넘는 경우
    `run_safety.allow_long_run=true`와 ack를 함께 지정한다.
- `client_pool_split`
  - 각 client shard 안에서 local labeled/unlabeled pool 비율을 deterministic하게
    나눈다.
  - manual `FedAvg + FixMatch + LoRA-classifier` 조합은 client별
    `labeled_rows`와 `unlabeled_rows`를 함께 local SSL 학습 입력으로 사용한다.
    legacy inline-delta fallback은 `unlabeled` partition만 pseudo-label training
    후보로 쓴다.
  - 이 값은 `fl_data.source_mode=runtime_split_from_train` fallback에서만 pool을
    다시 나누는 실행값이다. `materialized_client_split`에서는 이미 분리된
    `query_source.train_jsonl` 전체와 `query_source.unlabeled_jsonl` 전체를 각각
    client에 분배하고, ratio는 report의 실제 count로 확인한다.
- `fl_client_split_materialization.labeled_policy`
  - materialized split 생성 시 선택된 labeled source 중 얼마를 FL split에 포함할지
    결정한다.
  - 기본 `mode=all`은 `query_data_selection.labeled`가 가리키는 labeled source
    전체를 bootstrap/client labeled artifact에 분배한다.
  - 일부 라벨만 쓰는 ablation은 실행 시
    `mode=count_per_class,count_per_class=<N>` 또는 `mode=fraction,fraction=<R>`로
    명시한다. 이 선택은 manifest와 report metadata에 남긴다.
- `strategy_axes/fl/labeled_exposure_policy`
  - 선택된 labeled rows를 server/client 어디에 노출할지 결정한다.
  - `shared_client_seed`가 현재 entrypoint 기본값이며, 같은 selected labeled seed를
    모든 client에 공유하고,
    server bootstrap subset은 기존 `bootstrap_ratio`로 유지한다.
  - `client_local_split`은 legacy/ablation 값으로, server bootstrap subset과
    client-local labeled pool을 분리한다.
  - `server_only_seed`는 artifact/request metadata를 보존하고,
    `server_step_policy=supervised_seed_step` 조합에서 round open 전 supervised server
    seed step을 실행한다.
    실행 구현은 `server_step_policy` leaf의 `executor`가 가리키는 runtime adapter가
    맡고, FL simulation script는 executor를 import/dispatch하는 얇은 wrapper만
    소유한다.
  - materialized split artifact는 실행자 표면을 단순하게 유지하기 위해
    `data/datasets/fl_client_splits/<exposure_group>/<split_id>/manifest.json`
    아래에 둔다. manifest 내부 policy name은 canonical 값
    `client_local_split`, `shared_client_seed`, `server_only_seed`를 유지한다.
    `<exposure_group>`은 `client_local_labeled`, `shared_client_labeled`,
    `server_only_labeled` 중 하나다.
- `strategy_axes/trainable_state/update_family`
  - update family 이름과 실행에 필요한 callable path를 선언한다.
  - `local_objective_executors`, `initial_state_builder`, `validation_evaluator`,
    `final_projection_builder`, `transient_resource_cleaner`는 scripts가 family별
    구현을 직접 import하지 않도록 하는 runtime adapter 표면이다.
  - 현재 target leaf는 `peft_text_classifier`, `linear_head`, `prototype_pack`이다.
    `diagonal_scale`는 target update-family 축이 아니며, shared v1 contract
    compatibility 표면에만 남긴다. 이 config group에 leaf를 다시 만들지 않는다.
- `fl_data.source_mode`
  - 기본 `runtime_split_from_train`은 기존 smoke/debug용으로 `train_jsonl`을 즉석
    sharding한다.
  - 논문 비교는 `materialized_client_split`과 `fl_data.split_manifest`를 써서
    `materialize_fl_client_split` 산출물을 고정 입력으로 소비한다.
  - 이 mode에서는 `shard_policy`, `client_count`, `bootstrap_ratio`,
    `query_data_selection`이 manifest metadata와 맞는지 검증한다.
- `report.labeled_ratio`, `report.unlabeled_ratio`, `report.seed_count`
  - runtime split fallback의 ratio 값은 `client_pool_split`에서 파생된다.
  - materialized split의 실제 ratio는 client별 labeled/unlabeled count 합계로 읽는다.
  - `seed_count`는 `seed_sweep` runner에서 실행 seed 수와 일치하는지 검증한다.
- `report.primary_metrics`, `report.secondary_metrics`
  - primary는 `macro_f1`, `worst_client_macro_f1`이다.
  - secondary는 loss, weighted/balanced metric, worst-category metric,
    ECE/max-ECE, communication proxy, per-client variance를 포함한다.
  - 실제 metric 계산 shape는 `methods/evaluation`과 FL report builder가 소유한다.

## 정리 기준

- 새 실행 시작점은 `entrypoints/<track_or_stage>/` 아래에 둔다.
- 새 reusable group은 위치만 보고도 역할이 드러나야 한다.
- 실행 조건 묶음이 1개뿐이고 해당 entrypoint 전용이면 group으로 만들지 않는다.
- YAML은 실행 조합표와 파라미터를 담고, Python 구현이나 복잡한 계산 로직은 두지 않는다.
- namespace 이동은 Hydra config test와 active docs 갱신을 같이 닫는다.

## Dataset Output Convention

기존 `ourafla` 자산은 stage 중심 legacy 경로를 유지한다. 새 dataset asset은
가능하면 dataset 중심 root를 선언한다.

```text
data/datasets/<dataset_id>/
├── raw/
├── mapped/
├── splits/
├── query_ssl/
├── views/
└── pipeline_runs/
```

`execution_context/dataset_asset/<name>.yaml`의 `output_paths`는 dataset pipeline의
`raw`, `mapped`, `splits`, `pipeline_runs` 위치만 override한다. Query SSL
labeled/unlabeled split은
`entrypoints/dataset_pipeline/materialize_query_ssl_split`로
`data/datasets/<dataset_id>/query_ssl` 아래에 만들고, NLLB view materialization은
`execution_context/query_view` preset과
`entrypoints/dataset_pipeline/materialize_query_ssl_views`로
`data/datasets/<dataset_id>/views` 아래에 만든다. 모델 cache처럼 dataset artifact가
아닌 파일은 공유 cache 경로를 유지할 수 있다.
새 model/prototype/adapter 산출물은 `data/artifacts/`, cache성 파일은 `data/cache/`
아래에 두고, 기존 `data/processed/` 산출물은 legacy로 유지한다.

## Query Split And View Context

`execution_context/query_split`은 `train_jsonl`, `unlabeled_jsonl`, `validation_jsonl`,
`test_jsonl`로 구성된 단일 query-domain split artifact를 소유한다. 한 source 안의
고정 split artifact를 참조할 때 쓰며, 여러 source의 labeled/unlabeled/eval을
교차 조합하는 실행 선택 축은 아니다.

중앙 Query SSL처럼 labeled, unlabeled, validation, test source를 독립적으로 바꾸는
실험은 `execution_context/query_data_source`의 `query_data_sources` 주소록과
`query_data_selection.*` override를 사용한다.
이렇게 해야 source 조합마다 `query_split` YAML을 새로 만들지 않는다.

`execution_context/query_view`는 미리 저장할 weak/strong text view artifact의 생성
파라미터를 소유한다. NLLB 역번역은 method가 아니라 데이터 view를 만드는 실행 재료다.

- `split_name`, `split_dir`: 어떤 query SSL split을 증강할지 결정한다.
- `augmenter_name`, `source_lang`, `pivot_languages`, `model_id`: NLLB 역번역 view
  생성 정책을 결정한다.
- `batch_size`, `chunk_size`, `cache_dir`, `resume`, `overwrite`: 긴 materialization
  작업의 실행/재시작 방식을 결정한다.

Hydra package shape는 기존 runner compatibility를 위해 각각 `cfg.query_source`와
`cfg.query_view_materialization`을 유지한다.

중앙 SSL 학습에서 materialized view를 소비하는 축은 둘로 나눈다.

- `strategy_axes/ssl/augmentation_source`
  - `cfg.query_ssl_augmenter`로 compose된다.
  - 학습 입력 strong candidate를 어디서 확보할지 결정한다.
  - 기본 `precomputed_usb_candidates_v1`는 이미 JSONL에 저장된
    `text + aug_0 + aug_1`을 읽고, 학습 중 NLLB 역번역을 다시 수행하지 않는다.
- `query_ssl_strong_view_policy`
  - 중앙 SSL entrypoint의 단순 scalar 값이다.
  - 저장된 후보 중 어떤 strong view를 학습 batch에 노출할지 결정한다.
  - 기본 `first_aug`는 기존 동작처럼 `aug_0`만 strong view로 사용한다.
  - 선택지가 복잡한 parameter set이 아니므로 별도 Hydra strategy group으로
    승격하지 않는다.
