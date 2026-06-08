# Central SSL Control

중앙집중형 query-domain text encoder control entrypoint다. SSL objective는
`methods/ssl`, PEFT adaptation core는 `methods/adaptation/peft_text_encoder`,
full fine-tuning core는 `methods/adaptation/full_text_encoder`, 실행 조합과
파라미터는 루트 `conf/`가 소유한다.

긴 과거 cookbook은
`docs/notes/decisions/2026-05-28-archived-central-ssl-control-readme.md`에
보관했다. 현재 판단은 이 README와 active architecture 문서를 우선한다.

## 실행 순서

1. `run_peft_supervised_control.py`로 같은 PEFT scaffold의 supervised baseline을 만든다.
2. 필요하면 `run_full_text_encoder_supervised_control.py`로 full-model supervised-only
   transfer baseline을 별도 scaffold에서 만든다.
3. `run_peft_ssl_control.py`에서 SSL objective만 바꿔 pooled/offline control을 비교한다.

중앙 SSL method 비교의 기본 initial checkpoint는 `none`이다. teacher는 별도 public
entrypoint가 아니라 방법론이 필요할 때 method hook으로 소비한다.

## 읽기 경로

```text
conf/entrypoints/central/ssl_control/*.yaml
-> run_peft_supervised_control.py 또는 run_peft_ssl_control.py
-> scripts/support/query_ssl_text_encoder/runners/supervised_text_encoder.py
   또는 scripts/support/query_ssl_text_encoder/runners/consistency.py
-> scripts/support/query_ssl_text_encoder/{text_encoder_run_context.py,query_ssl/run_context.py}
-> methods/adaptation/peft_text_encoder/training/query_ssl_training_session.py
-> scripts/support/query_ssl_text_encoder/io/artifacts.py
```

## 기본 실행

```bash
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py --cfg job
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_full_text_encoder_supervised_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1
```

smoke 산출물은 main run과 섞지 않는다. method는
`strategy_axes/ssl_objective/consistency_method`로 선택한다. PEFT supervised/SSL
entrypoint의 학습 표면은 `trainable_surface=peft_text_encoder`이고,
full-model supervised-only control은 `trainable_surface=full_text_encoder`를 쓴다.

## Method Projection Figure

중앙 method 비교용 figure는 run 종료 후 아래 CLI로 직접 생성한다.

```bash
uv run python scripts/experiments/central/ssl_control/build_method_projection_figure.py \
  --run supervised=<supervised_report_json> \
  --run fixmatch=<fixmatch_report_json> \
  --run consistency=<consistency_report_json> \
  --split test \
  --max-rows-per-run 2000 \
  --output-root runs/figures/central_ssl/method_projection
```

여기서 `<*_report_json>`은 각 run에서 출력된 `report_json` 경로를 넣는다.
예: `runs/central/.../reports/report.json`

실행 결과는 `output_dir`가 stdout에 찍히고, 기본적으로
`method_projection_manifest.json`, `test/*.method_projection.png`가 생성된다.

폴더 단위로 지정하면 내부의 모든 `reports/report.json`을 읽어 method 라벨을
자동으로 붙여 한 번에 실행할 수 있다.

```bash
uv run python scripts/experiments/central/ssl_control/build_method_projection_figure.py \
  --run-dir runs/central/ssl/peft_classifier/labeled-szegeelim_general4_unlabeled-ourafla_reddit_test-ourafla_reddit \
  --split test \
  --max-rows-per-run 2000 \
  --output-root runs/figures/central_ssl/method_projection
```

## 데이터 source 변경

source 주소록은 `conf/execution_context/query_data_source/default.yaml`이 소유한다.
실행 시에는 `query_data_selection.labeled`, `unlabeled`, `validation`, `test`만
override한다.

```bash
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  query_data_selection.labeled=szegeelim_general4 \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit
```

같은 selector override는 supervised baseline에도 적용된다.

## Public Surface

사용자가 고르는 public surface는 아래로 제한한다.

- `strategy_axes/ssl_objective/consistency_method`
- `strategy_axes/model_architecture/{backbone,trainable_surface,peft,initial_checkpoint}`
- `execution_context/{dataset_asset,query_data_source,query_view,runtime_env}`
- `run_controls/central_ssl/budget`

`input_mode`, `teacher_provider`, `pseudo_label_selection`은 central SSL public
Hydra group이 아니다. 기본 중앙 supervised/SSL 실행은 `selection_set=test`이고
`eval_sets`에는 단일 `test`만 포함한다.

## Artifacts

중앙 supervised/SSL run의 학습된 모델 산출물은 각 run 폴더 아래에 묶는다. PEFT run은
`artifacts/adapter/`와 `artifacts/classifier_head.pt`, full text encoder run은
`artifacts/model/`과 `artifacts/classifier_head.pt`를 저장한다. run id와
`trainer_version` timestamp는 한국시간(`Asia/Seoul`) 기준이다.

## 경계

이 폴더는 dataset, method, adapter family 기본값을 새로 정의하지 않는다.
`conf/entrypoints/central/ssl_control/*.yaml`에서 시작해
`conf/strategy_axes/ssl_objective/**`, `conf/strategy_axes/model_architecture/**`,
`methods/ssl/**`, `methods/adaptation/{peft_text_encoder,full_text_encoder}/**`
순서로 본다.
중앙 Query SSL runner는 pooled/offline orchestration만 맡고, PEFT local SSL 학습은
FL과 같은 `methods/adaptation/peft_text_encoder/training/query_ssl_training_session.py`
session을 호출한다.
