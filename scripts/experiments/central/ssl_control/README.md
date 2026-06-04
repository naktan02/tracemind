# Central SSL Control

이 폴더는 중앙집중형 query-domain text encoder control entrypoint를 둔다.
SSL objective는 `methods/ssl`, PEFT adaptation core는
`methods/adaptation/peft_text_encoder`, full fine-tuning 실험 core는
`methods/adaptation/full_text_encoder`, 실행 조합과 파라미터는 루트 `conf/`가
소유한다.

긴 과거 cookbook은
`docs/notes/decisions/2026-05-28-archived-central-ssl-control-readme.md`에 보관했다.

## 실행 순서

1. `run_peft_supervised_control.py`로 같은 PEFT scaffold의 supervised baseline을 만든다.
2. `run_full_text_encoder_supervised_control.py`로 full-model supervised-only
   transfer baseline을 필요할 때 별도 scaffold로 만든다.
3. `run_peft_ssl_control.py`에서 SSL objective만 바꿔 pooled/offline control을 비교한다.

중앙 SSL method 비교의 기본 initial checkpoint는 `none`이다. teacher는 SSL 실행의
숨은 sub-step이 아니라 방법론이 필요할 때 teacher hook으로 소비한다.

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
  strategy_axes/ssl_objective/consistency_method=freematch_usb_v1
```

smoke 산출물은 main run과 섞지 않는다. method는
`strategy_axes/ssl_objective/consistency_method`로 선택한다. PEFT supervised/SSL
entrypoint의 학습 표면은 `trainable_surface=peft_text_encoder`이고, full-model
supervised-only control은 `trainable_surface=full_text_encoder`를 쓴다.

## 데이터 source 변경

source 주소록은 `conf/execution_context/query_data_source/default.yaml`이 소유한다.
실행 시에는 `query_data_selection.labeled`, `unlabeled`, `validation`, `test`만
override한다.

```bash
# general labeled + Reddit unlabeled + Reddit validation/test
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  query_data_selection.labeled=szegeelim_general4 \
  query_data_selection.unlabeled=ourafla_reddit \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit

# 같은 override는 supervised baseline에도 적용된다.
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  query_data_selection.labeled=szegeelim_general4 \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit

uv run python scripts/experiments/central/ssl_control/run_full_text_encoder_supervised_control.py \
  query_data_selection.labeled=szegeelim_general4 \
  query_data_selection.validation=ourafla_reddit \
  query_data_selection.test=ourafla_reddit
```

## Method Surface

`run_peft_ssl_control.py`는 `unlabeled_jsonl`과 precomputed weak/strong view를 읽는
consistency-family central SSL entrypoint다. pseudo-label replay나 teacher bootstrap은
독립 public 실험 entrypoint가 아니다. 기존 scripts teacher bootstrap helper는
제거했고, 새 teacher source가 필요하면 method hook/recipe로 정의한다.

`run_full_text_encoder_supervised_control.py`는 중앙 supervised-only ablation이다.
FL update family, shared payload, agent/main_server runtime을 열지 않는다.

중앙 supervised/SSL run의 학습된 모델 산출물은 각 run 폴더 아래에 묶는다.
PEFT run은 `artifacts/adapter/`와 `artifacts/classifier_head.pt`를 저장하고,
full text encoder run은 `artifacts/model/`과 `artifacts/classifier_head.pt`를
저장한다. `data/artifacts/**`는 canonical seed나 별도 migration 전 기존 산출물
같은 run 외부 공유 artifact에만 남긴다.

사용자가 고르는 public surface는 아래로 제한한다.

- `strategy_axes/ssl_objective/consistency_method`
- `strategy_axes/model_architecture/{backbone,trainable_surface,peft,initial_checkpoint}`
- `execution_context/{dataset_asset,query_data_source,query_view,runtime_env}`
- `run_controls/central_ssl/budget`

`input_mode`, `teacher_provider`, `pseudo_label_selection`은 central SSL public
Hydra group이 아니다.

```bash
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1
```

기본 중앙 supervised/SSL 실행은 `selection_set=test`이고 `eval_sets`에는 단일
`test`만 포함한다. 이 `test`는 기존 validation/test pool을 합쳐 class별 최소 수로
맞춘 `test_balanced_validation_test_seed42.jsonl`이며, epoch별 best checkpoint
선택과 final report 모두 같은 test eval set을 사용한다.

중앙 text encoder run id와 `trainer_version`의 자동 timestamp는 한국시간
(`Asia/Seoul`) 기준이다. PEFT/SSL뿐 아니라 full text encoder supervised도 최종
`projections/test.projection.{png,jsonl}`와 `projection_manifest.json`을 저장한다.

## Method Projection Figure

각 run이 종료될 때 저장되는 `projections/*.png`는 단일 run 진단용이다. 논문용으로
여러 method representation을 같은 좌표계에서 비교할 때는 후처리 script가 split별
feature를 모아 reducer를 한 번 fit한다.

```bash
uv run python scripts/experiments/central/ssl_control/build_method_projection_figure.py \
  --run supervised=runs/central/supervised/peft_classifier/<run_id>/reports/report.json \
  --run fixmatch=runs/central/ssl/peft_classifier/<selection>/<method>/<run_id>/reports/report.json \
  --split test
```

기본 저장 위치는
`runs/figures/central_ssl/method_projection/<YYYY_MM_DD_HHMMSS>/`이다. 산출물은
`method_projection_manifest.json`과 split 폴더 아래 method별 PNG/`jsonl`/`npz`
feature dump다. 예를 들어 `test/mixmatch.method_projection.png`처럼 저장한다.
기본 split은 `test`이며,
예전 report처럼 validation 산출물이 남아 있는 경우 `--split validation`을 명시할 수
있다. reducer는 split별 전체 method
feature에 한 번 fit하고, 저장만 method별로 나눠 같은 좌표계 비교를 유지한다.
재현용 이름을 붙이려면 `--figure-version <name>`을 사용한다. 이 경우에도 폴더명은
`<YYYY_MM_DD_HHMMSS>_<name>`처럼 시간 prefix를 유지한다. 오분류 marker를 보고
싶을 때만 `--mark-incorrect`를 추가한다. `--output-dir`를 주면 날짜 폴더를 만들지
않고 해당 경로에 바로 쓴다.

## 경계와 Read Path

이 폴더는 dataset, method, adapter family 기본값을 새로 정의하지 않는다.
`conf/entrypoints/central/ssl_control/*.yaml`에서 시작해
`conf/strategy_axes/ssl_objective/**`, `conf/strategy_axes/model_architecture/**`,
`methods/ssl/**`, `methods/adaptation/{peft_text_encoder,full_text_encoder}/**`
순서로 본다.
