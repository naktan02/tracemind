# Scripts Guide

`scripts/`는 Hydra 실험 entrypoint, sweep, report, visualization thin wrapper만
소유한다. 알고리즘 core는 `methods/`, 공통 contract/domain은 `shared/`,
runtime/service adapter는 `agent/`와 `main_server/`가 소유한다.

## 구조

- `scripts/datasets/`: dataset asset 생성용 CLI.
- `scripts/prototypes/`: prototype pack 생성, 평가, publication 보조 CLI.
- `scripts/experiments/`: track별 실험 entrypoint와 실험 전용 harness.
- `scripts/experiments/fl_ssl/federated_simulation/`: FL SSL synthetic harness와 artifact dump.
- `scripts/experiments/query_lora_ssl/`: 중앙/FL에서 공유 가능한 query-domain LoRA SSL harness와 adaptation IO.
- `scripts/experiments/result_index/`: `runs` report를 SQLite/dashboard JSON으로 정규화하는 비교/시각화 index.
- `scripts/runtime_adapters/`: scripts가 불가피하게 agent/main_server runtime을 재사용할 때 쓰는 명시 bridge.
- `scripts/reporting/`: report/diagnostics helper.
- `scripts/artifacts/`: run output 경로 helper.
- `scripts/codegen/`: app type 생성 entrypoint.

## 불변 규칙

- 운영 후보 알고리즘을 `scripts`에 먼저 만들고 나중에 복사하지 않는다.
- `scripts`는 `agent.src`, `main_server.src`를 직접 import하지 않는다.
- runtime 재사용이 필요하면 `scripts/runtime_adapters/`에 역할이 드러나는 bridge를 둔다.
- `conf/`가 Hydra 실행 조합과 파라미터의 source of truth다.
- `scripts` 하위 helper는 해당 entrypoint가 직접 쓰는 범위까지만 둔다.

## Config 지도

- `conf/entrypoints/`: 사람이 실행하는 top-level Hydra config.
- `conf/execution_context/`: dataset asset, embedding adapter, runtime env.
- `conf/strategy_axes/ssl/`: pseudo-label selection, consistency method, augmentation.
- `conf/strategy_axes/adaptation/`: PEFT adapter, transformer backbone, initial checkpoint.
- `conf/strategy_axes/fl/`: shard policy, method descriptor, client training profile.
- `conf/strategy_axes/prototype/`: prototype build strategy.
- `conf/run_controls/`: central SSL control과 FL SSL track별 실행 budget.

기본 실행 방식:

```bash
uv run python <entrypoint>.py execution_context/runtime_env=auto_local
```

최종 조합 확인:

```bash
uv run python <entrypoint>.py --cfg job
```

## 자주 쓰는 명령

설치:

```bash
uv sync --extra dev --extra experiments
```

Dataset pipeline:

```bash
uv run python scripts/datasets/run_dataset_pipeline.py
```

새 dataset asset은 가능하면 `data/datasets/<dataset_id>/` 아래에 `raw`, `mapped`,
`splits`, `query_ssl`, `views`를 모은다. 기존 stage 중심 `data/processed/*` 자산은
이동하지 않고 유지한다.

중앙 Query SSL labeled/unlabeled split 생성:

```bash
uv run python scripts/datasets/materialize_query_ssl_split.py \
  execution_context/dataset_asset=mental_health_kaggle \
  query_ssl_split_materialization.name=labeled1024_per_class_seed42_v1
```

중앙 Query SSL NLLB weak/strong view 생성:

```bash
uv run python scripts/datasets/materialize_query_ssl_views.py \
  execution_context/query_view=szegeelim_general4_ssl_labeled1024_per_class_seed42_nllb_v1
```

Prototype pack 생성:

```bash
uv run python scripts/prototypes/seed_prototypes.py
```

Prototype pack 평가:

```bash
uv run python scripts/prototypes/evaluate_prototype_pack.py
```

중앙 classifier seed:

```bash
uv run python scripts/experiments/central_classifier_seed/train_softmax_classifier.py
```

중앙 LoRA supervised control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_supervised_classifier.py
```

중앙 LoRA USB PseudoLabel control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_ssl_classifier.py \
  strategy_axes/ssl/consistency_method=pseudolabel_usb_v1 \
  output_dir=runs/train_lora_ssl_classifier_pseudolabel
```

중앙 LoRA FixMatch control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_ssl_classifier.py
```

중앙 LoRA FreeMatch control:

```bash
uv run python scripts/experiments/central_ssl_control/train_lora_ssl_classifier.py \
  strategy_axes/ssl/consistency_method=freematch_usb_v1
```

FL client split manifest 생성:

```bash
uv run python scripts/experiments/fl_ssl/materialize_fl_client_split.py \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  run_controls/fl_ssl/budget=smoke \
  federated_run_budget.client_count=10 \
  strategy_axes/fl/shard_policy=dirichlet_alpha03
```

기본값은 선택된 labeled source 전체와 unlabeled source 전체를 client에 분배한다.
위 명령은 client split manifest를 만드는 단계라 round loop를 실행하지 않는다.
라벨 데이터를 일부만 쓰는 ablation은 split 생성 시 정책을 명시한다.

```bash
uv run python scripts/experiments/fl_ssl/materialize_fl_client_split.py \
  query_data_selection.labeled=szegeelim_general4 \
  run_controls/fl_ssl/budget=smoke \
  federated_run_budget.client_count=10 \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  fl_client_split_materialization.labeled_policy.mode=count_per_class \
  fl_client_split_materialization.labeled_policy.count_per_class=256
```

FL SSL simulation smoke:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_simulation.py \
  run_controls/fl_ssl/budget=smoke
```

고정 FL split으로 simulation 실행:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_simulation.py \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<split_id>/manifest.json \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

FL SSL seed sweep smoke:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_seed_sweep.py \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

FL SSL client-count sweep smoke:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_client_count_sweep.py \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.rounds=1
```

FL SSL runner는 `run_safety.max_total_rounds_without_ack`보다 큰 총 예정
communication round를 기본 차단한다. 총 예정 round는 단일 simulation은
`rounds`, seed/client-count sweep은 `rounds * sweep 항목 수`다. 장시간 실행을
명시 승인받은 경우에만 아래 override를 함께 붙인다.

신규 FL SSL 산출물은 root에 수평으로 쌓지 않고 method-first 계층으로 저장한다.
기존 `runs/federated_simulation*` 산출물도 같은 구조로 마이그레이션했고 원본
수평 root는 제거했다.

```text
runs/fl_ssl/<method_family>/<method_composition>/<split_slug>/<clients_rounds_slug>/<run_id>/
runs/fl_ssl/<method_family>/<method_composition>/<split_slug>/sweeps/<sweep_kind_rounds>/<sweep_run_id>/<member_slug>/
```

```bash
run_safety.allow_long_run=true \
run_safety.long_run_ack=ALLOW_FL_SSL_LONG_RUN
```

현재 FL SSL 비교에서는 새 `50-round`/full-budget 실행을 하지 않는다. 기존
alpha=0.3 `50-round` report는 read-only artifact로 검증하고, 새 method/wiring
확인은 `1-round` smoke 또는 `5-round` reduced run으로 제한한다. 위 실행 예시도
이 정책에 맞춰 short 검증용 override만 보여준다.

기존 FL SSL 산출물 metadata 검증:

```bash
uv run python scripts/experiments/fl_ssl/verify_federated_report_artifacts.py \
  --client-count-sweep-summary runs/<run_id>/reports/fl_ssl_client_count_sweep.summary.json \
  --expected-client-counts 1,2,3,4,5,6,7,8,9,10 \
  --expected-completed-rounds 1 \
  --expected-round-budget 1 \
  --expected-round-record-count 1 \
  --expect-round-update-count-matches-client-count \
  --expected-seed 42 \
  --expected-shard-policy-name dirichlet_label_skew \
  --expected-shard-alpha 0.3 \
  --expected-split-id-contains alpha0.3 \
  --expected-ssl-algorithm fixmatch \
  --expected-ssl-method fixmatch_usb_v1 \
  --expected-adapter-family lora_classifier \
  --expected-aggregation fedavg \
  --expected-delta-format server_uploaded_artifact_ref \
  --expect-shared-update-count-matches-round-updates \
  --expect-server-owned-update-artifacts \
  --expect-no-agent-local-update-refs \
  --expect-lora-classifier-aggregate-snapshot \
  --expected-embedding-backend transformers_mxbai \
  --expected-embedding-device cuda \
  --expected-embedding-local-files-only true \
  --expected-local-trainer-device cuda \
  --expected-local-trainer-local-files-only true
```

여러 report와 sweep summary를 한 번에 검증해야 하면 같은 기대값을 CLI에
반복하지 말고 JSON manifest로 묶는다.

```bash
uv run python scripts/experiments/fl_ssl/verify_federated_report_artifacts.py \
  --manifest path/to/fl_ssl_artifact_verification_manifest.json
```

manifest 구조 예시는
`docs/operations/fl_ssl_artifact_verification_manifest.example.json`를 본다.

FL SSL 실험 기본값은 `execution_context/runtime_env=gpu_local`과
`execution_context/embedding_adapter=mxbai`다. `gpu_online`은 cache warm-up/최초
다운로드용이고, `cpu_local + hash_debug`는 entrypoint wiring smoke나 빠른 디버그에만 쓴다.

Prototype strategy analysis:

```bash
uv run python scripts/experiments/prototype_analysis/prototype_strategy_experiment.py
```

Type generation:

```bash
uv run python scripts/codegen/generate_family_extension_types.py
```

Experiment result index:

```bash
uv run python -m scripts.experiments.result_index.ingest \
  --runs-root runs \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json
```

기본 ingest는 중앙 SSL `reports/report.json`과 FL SSL
`reports/fl_ssl_main_comparison.report.json`를 읽고, 같은 `run_id`를 upsert하면서 새
run을 누적한다. cache를 전체 재빌드해야 할 때만 `--reset`을 추가한다.

정적 dashboard preview:

```bash
python -m http.server 5175 -d apps/experiment_dashboard
```

## Runtime Adapters

`scripts/runtime_adapters/`는 허용된 예외 지점이다. scripts가 runtime을 실행하거나
runtime payload를 읽어야 할 때 이 폴더를 통해서만 연결한다.

- `embedding_runtime.py`: agent embedding adapter factory/device resolver bridge.
- `backtranslation_runtime.py`: agent backtranslation service bridge.
- `federated_agent/`: FL simulation에서 agent local runtime을 역할별로 호출.
  - `scoring_runtime.py`: agent scoring service 조립.
  - `training_example_mapper.py`: simulation row -> agent training example request 변환.
  - `row_validator.py`: selected example backend가 요구하는 row shape 검증.
  - `backend_resolver.py`: objective config -> runtime backend 이름/adapter kind resolve.
  - `lora_classifier_artifacts.py`: simulation LoRA-classifier client delta artifact
    저장과 agent-local ref -> server-owned ref materialization bridge.
  - `training_runtime.py`: local training service/request bridge.
  - `selection_runtime.py`: pseudo-label selection service bridge.
- `federated_server/`: FL simulation에서 main_server round/aggregation 호출을 책임별로
  나눈 실제 runtime adapter package.
  - `runtime.py`: `SimulationServerRuntime` orchestration.
  - `repositories.py`: simulation output root 기준 main_server repository wiring.
  - `prototype_rebuild_bridge.py`: prototype rebuild runtime/service bridge.
  - `initial_state_factory.py`: adapter family별 initial shared state 생성.
  - `round_request_mapper.py`: experiment task config -> round open request 변환.
- `prototype_publication_runtime.py`: prototype pack publication/sync bridge.

새 bridge를 추가하기 전에 먼저 확인한다.

- 해당 로직이 `methods` core인가.
- 해당 계약이 `shared`로 올라가야 하는 cross-boundary payload인가.
- scripts 전용 harness 조합인지, production runtime 동작인지.

## 산출물

- `data/datasets/`: 새 dataset별 raw/mapped/split/query_ssl/view artifact.
- `data/artifacts/`: 새 model/prototype/adapter artifact.
- `data/cache/`: 새 model/translation/query cache.
- `data/processed/`: legacy dataset/model/prototype artifact.
- `runs/fl_ssl/...`: 신규 FL SSL 실행 결과, report, dump.
- `runs/<job>/<run_id>/`: 중앙 control과 legacy 실행 결과.
- `agent/state/`: agent runtime state.
- `main_server/state/`: server runtime state와 publication artifact.

GPU/모델 캐시가 필요한 실험 전에는 실제 실행 환경에서 `nvidia-smi`와
`torch.cuda.is_available()`를 먼저 확인한다.
