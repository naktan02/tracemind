# FL SSL 실행

이 폴더는 FL SSL 실험 entrypoint만 둔다. FL method identity와 method-only
정책은 `methods/federated_ssl/`, SSL objective core는 `methods/ssl`, LoRA
classifier 계산 core는 `methods/adaptation/lora_classifier`, 실행 조합과
파라미터는 `conf/` Hydra config가 소유한다.

현재 기본 실행은 논문 method가 아니라 manual baseline이다.

```text
FixMatch USB objective + LoRA-classifier local update + FedAvg aggregation
```

FedMatch/FedLGMatch 같은 method-owned FL SSL method는 선택/구현 전까지
descriptor/config placeholder를 만들지 않는다.

## 먼저 확인

실제 실행 전에 compose 결과를 확인한다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation --cfg job
```

기본값:

```text
runtime_env=gpu_local
embedding_adapter=mxbai
budget=smoke
client_count=4
rounds=3
shard_policy=label_dominant
composition_mode=manual
query_ssl_method=fixmatch_usb_v1
local_update_profile=lora_pseudo_label_v1
adapter_family=lora_classifier
aggregation_backend=fedavg
output_dir=runs/_smoke/fl_ssl
```

`run_controls/fl_ssl/budget=smoke`는 wiring 검증용 산출물을
`runs/_smoke/fl_ssl` 아래에 둔다. `budget=reduced`는 `10 clients`, `5 rounds`
확인용 preset이고, `budget=main`은 `10 clients`, `50 rounds` full-budget
preset이다. 성능 방향 확인은 `budget=reduced`, full-budget 실행이 필요할 때만
`budget=main`을 명시한다.

## 데이터 선택

FL도 중앙 SSL과 같은 `execution_context/query_data_source` catalog를 사용한다.
라벨/논라벨/validation/test는 `query_data_selection.*`으로 고른다.

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10
```

중앙 SSL과의 차이는 소비 방식이다. 중앙 SSL은 선택된 JSONL을 바로 학습에
사용하지만, FL 논문 비교는 선택된 labeled/unlabeled/validation/test JSONL을 먼저
client split manifest로 materialize한다.

```text
query_data_selection
-> query_source.train_jsonl / unlabeled_jsonl / validation_jsonl / test_jsonl
-> materialize_fl_client_split
-> data/datasets/fl_client_splits/<split_id>/manifest.json
-> run_federated_simulation fl_data.source_mode=materialized_client_split
```

`query_data_selection.labeled`는 client labeled pool과 bootstrap labeled pool의
원본이다. `query_data_selection.unlabeled`는 client unlabeled pool의 원본이다.
`query_data_selection.validation`과 `query_data_selection.test`는 global/client
평가용 row 원본으로 manifest에 고정된다.

## 고정 Client Split 생성

논문 비교용 FL 실행은 먼저 client별 labeled/unlabeled split을 materialize한다.
기본 비교 split은 Dirichlet alpha 0.3, seed 42를 기준으로 둔다.

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  run_controls/fl_ssl/budget=smoke \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10 \
  fl_client_split_materialization.labeled_policy.mode=all
```

출력되는 `manifest_json=...` 값을 다음 실행의 `fl_data.split_manifest`로 넘긴다.

모든 client가 같은 public labeled seed를 보고, unlabeled만 client별 non-IID로
나누는 `shared_client_seed` split은 labeled exposure policy만 바꿔 별도
manifest로 materialize한다. split id에는 `shared_client_seed_`가 들어가 기존
`client_local_split` manifest를 덮어쓰지 않는다.
`server_only_seed`는 runtime capability가 열리기 전까지 예약된 축이다. 현재
materialization과 `fl_data.source_mode=materialized_client_split` 실행 request는
`server_only_seed` manifest를 모두 실행 전에 거부한다.

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  run_controls/fl_ssl/budget=main \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  strategy_axes/fl/labeled_exposure_policy=shared_client_seed
```

## 실행 명령 구조

FL SSL 실행 명령은 보통 다섯 축을 동시에 고른다.

```text
1. method/composition: manual lower-axis 조합인지, FedMatch 같은 method-owned인지
2. data source: labeled/unlabeled/validation/test source 선택
3. data split: runtime split인지, materialized client split인지
4. run condition: client_count, rounds, seed, shard policy
5. runtime family: adapter_family, aggregation_backend, local_update_profile
```

현재 기본값은 `fl_method.composition_mode=manual`이다. 즉 별도 override가 없으면
상위 FL method를 고르는 것이 아니라 아래 lower axes를 조합한다.

```text
query_ssl_method + round_runtime.adapter_family_name + round_runtime.aggregation_backend_name
```

기본 조합은 `fixmatch_usb_v1 + lora_classifier + fedavg`다.

## 단일 Simulation 실행

고정 split을 쓰는 1-round smoke 예시:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<split_id>/manifest.json \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

이 명령은 방법론을 새로 고르는 예시가 아니라, 기본 manual 조합
`FixMatch + LoRA-classifier + FedAvg`를 고정 client split에서 `10 clients`,
`1 round`로 실행하는 예시다. `federated_run_budget.client_count`와
`federated_run_budget.rounds`는 임의 override 가능하다. 단, long-run guard를
넘는 실행은 현재 정책상 피한다.

짧은 reduced run 예시:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<split_id>/manifest.json \
  training_task.max_steps=20
```

산출 경로는 아래 구조로 쌓인다.

```text
runs/fl_ssl/
  manual_baselines/
    fixmatch_usb_v1__lora_classifier__fedavg/
      alpha03_seed42/
        clients10_rounds1/
        clients10_rounds5/
```

`budget=smoke` 산출물은 같은 하위 구조를 `runs/_smoke/fl_ssl/` 아래에 만든다.

각 run의 canonical report는
`reports/fl_ssl_main_comparison.report.json`이다.

## Sweep

Seed sweep:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_seed_sweep \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

Client-count sweep:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_client_count_sweep \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  federated_run_budget.rounds=1
```

`materialized_client_split` 기반 client-count sweep은 client count별 manifest가
필요하다. `client_count_sweep.split_manifest_by_client_count`에 count별
manifest를 지정한다.

## 다른 Manual Baseline 조합

manual baseline에서는 lower axes를 직접 고른다. `composition_mode=manual`은
기본값이지만, 문서/로그 가독성을 위해 명시해도 된다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  fl_method.composition_mode=manual \
  strategy_axes/ssl/consistency_method=flexmatch_usb_v1 \
  round_runtime.adapter_family_name=lora_classifier \
  round_runtime.aggregation_backend_name=fedavg \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<split_id>/manifest.json \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1 \
  training_task.max_steps=50
```

위 명령은 `FlexMatch + LoRA-classifier + FedAvg` manual 조합이다.
client 수와 round 수는 `federated_run_budget.client_count`,
`federated_run_budget.rounds`로 바꾼다. client가 중앙으로 update를 보내기 전
local optimizer step 상한은 `training_task.max_steps`로 조절한다. 고정 split을
쓰려면 예시처럼 `fl_data.source_mode=materialized_client_split`과
`fl_data.split_manifest=...`를 함께 넘긴다.

`fl_method.composition_mode=method_owned`는 FedMatch처럼 상위 FL SSL method가
client/server 정책을 함께 소유할 때 사용한다. 이 경우
`strategy_axes/fl/method_descriptor=<method>`와 실제
`methods/federated_ssl/<method>/` 구현이 함께 있어야 한다.

예시 형태:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  fl_method.composition_mode=method_owned \
  strategy_axes/fl/method_descriptor=fedmatch \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

현재 FedMatch descriptor/구현은 아직 열지 않았으므로 위 명령은 구현 이후의
형태를 보여주는 예시다.

## Report Index 갱신

실험 웹/대시보드용 index는 중앙 SSL과 FL SSL report를 같은 ingest로 읽는다.
대시보드 cache 생성 명령과 `--reset` 의미는
[experiment dashboard README](../../../apps/experiment_dashboard/README.md)를 기준으로
본다. 이 폴더의 README는 FL 실행 방법만 소유하고, 웹 데이터 생성 절차는
`apps/experiment_dashboard`가 설명한다.

## 주의

- 현재 기존 FL runs 중 prototype scorer로 LoRA-classifier validation을 평가한
  결과는 성능 근거로 쓰지 않는다. scorer 수정 이후 run을 재실행해야 한다.
- `report.protocol.embedding_adapter`, `local_trainer_runtime`이 논문용 실행
  환경인지 확인한다. CPU/hash debug smoke 결과를 성능 비교에 섞지 않는다.
- scripts는 Hydra entrypoint, sweep, report/index wrapper만 소유한다. 새 method
  core는 scripts에 추가하지 않는다.

## 예시: FlexMatch Shared Labeled 5라운드 Reduced Run

이미 materialize된 `shared_client_seed` 10-client Dirichlet alpha=0.3 split을
고정 입력으로 사용해 `FlexMatch + LoRA-classifier + FedAvg` manual 조합을
5라운드 실행한다. 모든 client는 같은 labeled seed를 보고, unlabeled shard만
client별 non-IID로 나뉜다. 각 client의 round당 local optimizer step 상한은
`training_task.max_steps=20`이고, central SSL main과 맞춰 labeled/unlabeled
batch size는 `12`로 둔다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  fl_method.composition_mode=manual \
  strategy_axes/fl/shard_policy=dirichlet_alpha03 \
  strategy_axes/ssl/consistency_method=flexmatch_usb_v1 \
  round_runtime.adapter_family_name=lora_classifier \
  round_runtime.aggregation_backend_name=fedavg \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/labeled-ourafla_reddit_unlabeled-ourafla_reddit_validation-ourafla_reddit_test-ourafla_reddit_shared_client_seed_dirichlet_label_skew_dominantNone_alpha0.3_clients10_seed42/manifest.json \
  training_task.batch_size=12 \
  training_task.max_steps=20
```
