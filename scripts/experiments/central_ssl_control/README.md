# Central SSL Control

이 폴더는 중앙집중형 query-domain PEFT text encoder control entrypoint만 둔다.
SSL objective는 `methods/ssl`, adaptation core는 `methods/adaptation/peft_text_encoder`,
실행 조합과 파라미터는 루트 `conf/`가 소유한다.

긴 과거 cookbook은
`docs/notes/decisions/2026-05-28-archived-central-ssl-control-readme.md`에 보관했다.

## Entry Points

- `run_peft_supervised_control.py`: supervised seed/control 학습.
- `run_peft_ssl_control.py`: FixMatch, FlexMatch, FreeMatch, PseudoLabel 등 SSL control 실행.

## 기본 실행

실행 전 compose를 먼저 확인한다.

```bash
uv run python scripts/experiments/central_ssl_control/run_peft_ssl_control.py --cfg job
```

smoke 산출물은 main run과 섞지 않는다.

```bash
uv run python scripts/experiments/central_ssl_control/run_peft_ssl_control.py \
  run_controls/central_ssl/budget=smoke
```

method는 `strategy_axes/ssl/consistency_method`로 선택한다.

```bash
uv run python scripts/experiments/central_ssl_control/run_peft_ssl_control.py \
  strategy_axes/ssl/consistency_method=freematch_usb_v1
```

## 경계

- 이 폴더는 dataset, method, adapter family 기본값을 새로 정의하지 않는다.
- query data source와 view 주소록은 `conf/execution_context/**`가 소유한다.
- SSL method별 objective/state 의미는 `methods/ssl/<method>/`가 소유한다.
- PEFT encoder training/materialization 의미는 `methods/adaptation/peft_text_encoder/`가 소유한다.
- `runs/_smoke/**`와 main `runs/**`는 report ingest에서 구분한다.

## Read Path

1. `conf/entrypoints/central_ssl_control/*.yaml`
2. `conf/strategy_axes/ssl/**`
3. `conf/strategy_axes/adaptation/**`
4. `methods/ssl/**`
5. `methods/adaptation/peft_text_encoder/**`
