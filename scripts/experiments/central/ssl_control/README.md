# Central SSL Control

이 폴더는 중앙집중형 query-domain PEFT text encoder control entrypoint만 둔다.
SSL objective는 `methods/ssl`, adaptation core는 `methods/adaptation/peft_text_encoder`,
실행 조합과 파라미터는 루트 `conf/`가 소유한다.

긴 과거 cookbook은
`docs/notes/decisions/2026-05-28-archived-central-ssl-control-readme.md`에 보관했다.

## 실행 순서

1. `run_peft_supervised_control.py`로 같은 PEFT scaffold의 supervised baseline을 만든다.
2. `run_peft_ssl_control.py`에서 SSL objective만 바꿔 pooled/offline control을 비교한다.

중앙 SSL method 비교의 기본 initial checkpoint는 `none`이다. teacher는 SSL 실행의
숨은 sub-step이 아니라 방법론이 필요할 때 teacher hook으로 소비한다.

## 기본 실행

```bash
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py --cfg job
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_peft_supervised_control.py \
  run_controls/central_ssl/budget=smoke
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  strategy_axes/ssl_objective/consistency_method=freematch_usb_v1
```

smoke 산출물은 main run과 섞지 않는다. method는
`strategy_axes/ssl_objective/consistency_method`로 선택한다. 현재 중앙 trainer의
학습 표면은 `strategy_axes/model_architecture/trainable_surface=peft_text_encoder`다.

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
```

## Method Surface

`run_peft_ssl_control.py`는 `unlabeled_jsonl`과 precomputed weak/strong view를 읽는
consistency-family central SSL entrypoint다. pseudo-label replay나 teacher bootstrap은
독립 public 실험 entrypoint가 아니라 내부 helper/workflow와 method hook로만 남긴다.

```bash
uv run python scripts/experiments/central/ssl_control/run_peft_ssl_control.py \
  strategy_axes/ssl_objective/consistency_method=fixmatch_usb_v1
```

## 경계와 Read Path

이 폴더는 dataset, method, adapter family 기본값을 새로 정의하지 않는다.
`conf/entrypoints/central/ssl_control/*.yaml`에서 시작해
`conf/strategy_axes/ssl_objective/**`, `conf/strategy_axes/model_architecture/**`,
`methods/ssl/**`, `methods/adaptation/peft_text_encoder/**` 순서로 본다.
