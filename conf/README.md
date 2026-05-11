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
- `track_presets/`: 논문 비교 track 안에서만 의미가 생기는 preset.

```text
conf/
├── entrypoints/
│   ├── central_classifier_seed/
│   ├── central_ssl_control/
│   ├── data_pipeline/
│   ├── fl_ssl/
│   ├── prototype_analysis/
│   └── prototype_pack/
├── execution_context/
│   ├── dataset_asset/
│   ├── embedding_adapter/
│   ├── query_split/
│   ├── query_view/
│   └── runtime_env/
├── strategy_axes/
│   ├── adaptation/
│   │   ├── initial_checkpoint/
│   │   ├── peft_adapter/
│   │   └── transformer_backbone/
│   ├── fl/
│   │   ├── local_update_profile/
│   │   ├── method_descriptor/
│   │   ├── round_runtime_profile/
│   │   └── shard_policy/
│   ├── prototype/
│   │   └── build_strategy/
│   └── ssl/
│       ├── augmentation/
│       ├── consistency_method/
│       └── pseudo_label_selection/
└── track_presets/
    ├── central_ssl_control/
    │   ├── lora_classifier_defaults/
    │   └── training_preset/
    └── fl_ssl/
        └── simulation_preset/
```

## 이름 기준

- entrypoint config는 실행 스크립트의 시작점이다.
- execution context는 방법론 비교가 아니라 실행 재료다.
- strategy axis는 실제로 교체 가능한 계산/정책 축이다.
- track preset은 central SSL control, FL SSL처럼 비교표의 맥락 안에서 쓰는 옵션 묶음이다.

## FL SSL config contract

FL SSL simulation은 config 의미가 겹치기 쉬우므로 아래처럼 읽는다.

- `strategy_axes/fl/method_descriptor`
  - `cfg.ssl_method`로 compose된다.
  - 논문 method identity, report role, custom runtime 필요 여부를 설명한다.
  - 실제 local update 계산 조합을 단독으로 소유하지 않는다.
  - 사람이 읽는 method recipe metadata와 method-only 변형은
    `methods/federated_ssl/<method>/`가 소유한다.
- `strategy_axes/fl/local_update_profile`
  - `cfg.local_update_profile`로 compose된다.
  - agent local update를 만드는 training/evidence/scoring/privacy 조합을 소유한다.
  - adapter family나 aggregation backend를 소유하지 않는다.
- `strategy_axes/fl/round_runtime_profile`
  - `cfg.round_runtime_profile`로 compose된다.
  - server round runtime의 adapter family, aggregation backend, bootstrap runtime
    knob를 소유한다.
  - `fedavg_lora_classifier`는 기존 `fedavg_pseudo_label` method descriptor와
    조합 가능한 LoRA-classifier server family profile이다. backbone/LoRA 세부
    값은 `strategy_axes/adaptation/transformer_backbone`과
    `strategy_axes/adaptation/peft_adapter`에서 온다.
- `strategy_axes/fl/shard_policy`
  - `cfg.shard_policy`로 compose된다.
  - non-IID client split 방식만 소유한다.
- `track_presets/fl_ssl/simulation_preset`
  - client 수, round budget, output dir 같은 track 실행 budget을 소유한다.
  - method semantics나 local update policy를 소유하지 않는다.
- `seed_sweep`
  - FL SSL seed sweep runner가 순회할 seed 목록과 sweep output root를 소유한다.
  - `seed_sweep.seeds` 길이는 `report.seed_count`와 같아야 한다.
- `client_pool_split`
  - 각 client shard 안에서 local labeled/unlabeled pool 비율을 deterministic하게
    나눈다.
  - 현재 `fedavg_pseudo_label` baseline은 `unlabeled` partition만 pseudo-label
    training 후보로 사용한다.
- `report.labeled_ratio`, `report.unlabeled_ratio`, `report.seed_count`
  - ratio 값은 `client_pool_split`에서 파생된 report protocol field다.
  - `seed_count`는 `seed_sweep` runner에서 실행 seed 수와 일치하는지 검증한다.

## 정리 기준

- 새 실행 시작점은 `entrypoints/<track_or_stage>/` 아래에 둔다.
- 새 reusable group은 위치만 보고도 역할이 드러나야 한다.
- preset이 1개뿐이고 해당 entrypoint 전용이면 group으로 만들지 않는다.
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
labeled/unlabeled split은 `materialize_query_ssl_split.py --output-root`로
`data/datasets/<dataset_id>/query_ssl` 아래에 만들고, NLLB view materialization은
`execution_context/query_view` preset의 `output_root`로 `data/datasets/<dataset_id>/views` 아래에
만든다. 모델 cache처럼 dataset artifact가 아닌 파일은 공유 cache 경로를 유지할 수 있다.
새 model/prototype/adapter 산출물은 `data/artifacts/`, cache성 파일은 `data/cache/`
아래에 두고, 기존 `data/processed/` 산출물은 legacy로 유지한다.

## Query Split And View Context

`execution_context/query_split`은 `train_jsonl`, `unlabeled_jsonl`, `validation_jsonl`,
`test_jsonl`로 구성된 query-domain split artifact를 소유한다. 중앙 SSL뿐 아니라
FL simulation도 같은 artifact를 참조할 수 있으므로 track preset 아래에 두지 않는다.

`execution_context/query_view`는 미리 저장할 weak/strong text view artifact의 생성
파라미터를 소유한다. NLLB 역번역은 method가 아니라 데이터 view를 만드는 실행 재료다.

- `split_name`, `split_dir`: 어떤 query SSL split을 증강할지 결정한다.
- `augmenter_name`, `source_lang`, `pivot_languages`, `model_id`: NLLB 역번역 view
  생성 정책을 결정한다.
- `batch_size`, `chunk_size`, `cache_dir`, `resume`, `overwrite`: 긴 materialization
  작업의 실행/재시작 방식을 결정한다.

Hydra package shape는 기존 runner compatibility를 위해 각각 `cfg.query_source`와
`cfg.query_view_materialization`을 유지한다.
