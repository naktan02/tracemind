# FL SSL 실행

이 폴더는 FL SSL 실험 entrypoint와 support helper만 둔다. FL method identity와
method-only 정책은 `methods/federated_ssl/`, SSL objective core는 `methods/ssl`,
update-family 계산 core는 `methods/adaptation/*`, 실행 조합과 파라미터는 `conf/`
Hydra config가 소유한다.

긴 과거 runbook과 특정 시점 실행 예시는
`docs/notes/decisions/2026-05-28-archived-fl-ssl-runbook.md`에 보관했다. 현재 실행
구조 판단은 `docs/architecture/target-method-runtime-structure.md`와 이 README를
우선한다.

## 기본 실행

실행 전 compose 결과를 먼저 확인한다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation --cfg job
```

기본 smoke는 manual baseline이다.

```text
composition_mode=manual
query_ssl_method=fixmatch_usb_v1
update_family=peft_text_encoder
aggregation_backend=fedavg
budget=smoke
output_dir=runs/_smoke/fl_ssl
```

`budget=reduced`는 `10 clients x 5 rounds`, `budget=main`은 `10 clients x 30
rounds`다. 새 wiring은 smoke 또는 reduced로 먼저 확인한다.

## 읽기 경로

```text
conf/entrypoints/fl_ssl/run_federated_simulation.yaml
-> run_federated_simulation.py
-> federated_simulation/config_request.py
-> federated_simulation/simulation.py
-> federated_simulation/flow/bootstrap.py
-> federated_simulation/flow/round_loop.py
-> federated_simulation/flow/result_builder.py
```

`run_federated_simulation.py`는 sweep 처리, output dir 결정, request 생성, runner
호출, 결과 출력만 맡는다. `flow/round_loop.py`는 server step부터 summary assembly까지
round lifecycle phase를 보여준다.

## Output Layout

새 FL SSL run은 같은 split과 숫자 조건 아래에서 방법론을 빠르게 비교하도록 아래
구조로 저장한다.

```text
runs/fl_ssl/{split}/{condition}/{surface}/{method}/{run_id}/
```

예:

```text
runs/fl_ssl/sz4_ourafla_shared_s42/c10_r30_e1_b8_s50/peft_text_encoder_lora/fixmatch_fedavg/20260605T101139Z/
```

과거 run의 기존 경로는 이동하지 않는다. Report protocol에는 긴 source 이름과 실행
metadata가 그대로 남으므로 folder slug는 비교용으로 짧게 유지한다.

## Client Split

논문 비교용 FL 실행은 materialized client split을 우선한다.

```bash
uv run python -m scripts.experiments.fl_ssl.materialize_fl_client_split \
  run_controls/fl_ssl/budget=smoke \
  query_data_selection.labeled=ourafla_reddit \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit \
  strategy_axes/fl_topology/shard_policy=dirichlet_alpha03 \
  federated_run_budget.client_count=10
```

생성된 `manifest_json=...` 값을 `fl_data.split_manifest`로 넘긴다.
`shared_client_seed`, `client_local_split`, `server_only_seed`는 labeled exposure
policy이며 shard policy가 아니다.

## Simulation

고정 split smoke:

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  fl_data.source_mode=materialized_client_split \
  fl_data.split_manifest=data/datasets/fl_client_splits/<exposure_group>/<split_id>/manifest.json \
  federated_run_budget.client_count=10 \
  federated_run_budget.rounds=1
```

Method-owned 실행은 method descriptor와 capability leaf를 함께 고른다. 원본 상세값은
YAML에 복제하지 않고 method package에서 report protocol로 주입한다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=reduced \
  fl_method.composition_mode=method_owned \
  strategy_axes/fssl_method=fedmatch \
  ssl_method.scenario=labels-at-client
```

manual 조합은 selector 축을 직접 고르는 baseline/debug 경로다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  strategy_axes/fl_topology/shard_policy=dirichlet_alpha03
```

중앙 supervised checkpoint에서 시작하는 실행은 initial checkpoint preset을 고른다.
긴 manifest path를 CLI에 직접 넣지 않는다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=main \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1 \
  strategy_axes/model_architecture/initial_checkpoint=supervised_20260612_step2000
```

FedMatch도 같은 checkpoint preset을 사용한다.

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=main \
  fl_method.composition_mode=method_owned \
  strategy_axes/fssl_method=fedmatch \
  strategy_axes/model_architecture/initial_checkpoint=supervised_20260612_step2000
```

## Sweep

```bash
uv run python -m scripts.experiments.fl_ssl.run_federated_simulation \
  run_controls/fl_ssl/budget=smoke \
  sweep.axis=seed \
  strategy_axes/fl_topology/shard_policy=dirichlet_alpha03
```

`materialized_client_split` 기반 client-count sweep은
`sweep.client_count.split_manifest_by_client_count`에 count별 manifest를 명시한다.

## Report

각 run의 canonical report는 `reports/fl_ssl_main_comparison.report.json`이다. report
verifier는 `scripts/experiments/fl_ssl/verify_federated_report_artifacts.py`를
사용한다. 대시보드 cache 생성과 `--reset` 의미는
`apps/experiment_dashboard/README.md`가 소유한다.

## Final Projection Artifacts

FL SSL는 run 종료 시 final projection이 `projections/`에 자동 생성된다.

- `runs/<...>/projections/validation.projection.jsonl`
- `runs/<...>/projections/validation.projection.png`
- `runs/<...>/projections/projection_manifest.json`

시험/논문 실행에서 같은 위치의 figure를 그대로 사용할 수 있다.

## 주의

- scripts runner에 새 method core를 추가하지 않는다.
- 논문용 산출물은 `report.protocol.embedding_adapter`와
  `local_trainer_runtime`으로 `gpu_local + mxbai` 여부를 확인한다.
- smoke/test artifact는 `runs/_smoke` 아래에 둔다.
