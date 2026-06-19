# Query SSL Text Encoder Support

`scripts/support/query_ssl_text_encoder/`는 중앙 query-domain text encoder 실험이
공유하는 runner와 artifact IO 계층이다. 직접 실행하는 CLI가 아니라
`scripts/experiments/central/ssl_control/*.py` entrypoint가 호출하는 내부 지원
패키지다.

이 패키지는 중앙 지도학습 control과 중앙 SSL control의 실행 scaffolding을
공유한다. 알고리즘 의미는 `methods/`, 실행 조합은 `conf/`, 계약 의미는
`shared/`가 소유한다.

## Used By

| Entry point | 이 패키지에서 호출하는 runner |
| --- | --- |
| `scripts/experiments/central/ssl_control/run_peft_supervised_control.py` | `runners/supervised.py` |
| `scripts/experiments/central/ssl_control/run_full_text_encoder_supervised_control.py` | `runners/full_text_encoder_supervised.py` |
| `scripts/experiments/central/ssl_control/run_query_ssl_control.py` | `runners/consistency.py` |

실험을 실행할 때는 위 entrypoint를 사용한다. 이 디렉터리의 runner 파일을 직접
실행하지 않는다.

## What It Owns

- Hydra config를 받은 뒤 text encoder 학습에 필요한 run context를 만든다.
- labeled/unlabeled query row를 읽고 evaluation loader를 구성한다.
- PEFT text encoder, full text encoder, SSL consistency runner가 공유하는
  train/eval 흐름을 맞춘다.
- adapter, classifier head, report, manifest, projection artifact를 저장한다.
- `methods`가 제공하는 SSL algorithm descriptor와 trainable surface callable을
  entrypoint 실행 흐름에 연결한다.

## What It Does Not Own

- SSL objective 자체의 의미나 loss 구현
- PEFT/full fine-tuning mechanism의 core 학습 구현
- Hydra strategy axis의 기본값
- shared contract field 의미
- FL round lifecycle 또는 production API behavior

이 중 하나를 바꾸려면 각각 `methods/`, `conf/`, `shared/`, `agent/`,
`main_server/`의 owner 문서를 먼저 본다.

## Layout

| Path | 역할 |
| --- | --- |
| `runners/supervised_text_encoder.py` | supervised text encoder train/eval 공통 흐름 |
| `runners/supervised.py` | PEFT/LoRA text encoder + classifier head 지도학습 runner |
| `runners/full_text_encoder_supervised.py` | full text encoder + classifier head 지도학습 runner |
| `runners/consistency.py` | FixMatch, AdaMatch, FreeMatch, PseudoLabel 계열 중앙 SSL runner |
| `text_encoder_run_context.py` | model, tokenizer, dataset, dataloader, evaluator 공통 context |
| `query_ssl/run_context.py` | Query SSL method manifest, parameter, evaluation context |
| `query_ssl/view_preparation.py` | precomputed weak/strong view와 unlabeled row 준비 |
| `config/initial_checkpoint.py` | Hydra initial checkpoint 설정을 manifest/runtime 입력으로 변환 |
| `io/artifacts.py` | PEFT text encoder run artifact 저장 |
| `io/full_text_encoder_artifacts.py` | full text encoder run artifact 저장 |
| `io/artifact_paths.py` | run directory와 output path 계산 |
| `io/model_artifact_exporter.py` | adapter, tokenizer, classifier file export |
| `io/manifest_builder.py` | manifest와 report payload 조립 |
| `io/artifact_writer.py` | JSON report/manifest 쓰기 |
| `io/embedding_projection.py` | embedding projection artifact 생성 |
| `io/labeled_row_export.py` | labeled row JSONL export |

## Read Path

중앙 SSL 실험을 따라갈 때는 아래 순서가 가장 짧다.

```text
scripts/experiments/central/ssl_control/README.md
-> scripts/experiments/central/ssl_control/run_query_ssl_control.py
-> scripts/support/query_ssl_text_encoder/runners/consistency.py
-> scripts/support/query_ssl_text_encoder/query_ssl/run_context.py
-> scripts/support/query_ssl_text_encoder/query_ssl/view_preparation.py
-> methods/ssl/<method>/
-> scripts/support/query_ssl_text_encoder/io/
```

중앙 지도학습 control은 아래 순서로 보면 된다.

```text
scripts/experiments/central/ssl_control/README.md
-> run_peft_supervised_control.py 또는 run_full_text_encoder_supervised_control.py
-> scripts/support/query_ssl_text_encoder/runners/supervised_text_encoder.py
-> scripts/support/query_ssl_text_encoder/runners/{supervised,full_text_encoder_supervised}.py
-> methods/adaptation/{peft_text_encoder,full_text_encoder}/
-> scripts/support/query_ssl_text_encoder/io/
```

## Extension Rules

새 Query SSL method를 추가할 때는 이 패키지에 새 runner를 만들지 않는 것이
기본이다. `methods/ssl/`에 method descriptor와 core를 추가하고,
`conf/strategy_axes/ssl_objective/consistency_method`에서 Hydra leaf를 연 뒤
`run_query_ssl_control.py`를 재사용한다.

새 trainable surface가 필요하면 먼저 `methods/adaptation/`에서 surface의 학습
mechanism과 callable boundary를 정의한다. 이 패키지는 그 callable을 받아 중앙
실험 artifact shape에 맞춰 실행하는 adapter 역할만 맡는다.

teacher source, pseudo-label replay, offline bootstrap 같은 workflow를 새 public
axis로 만들지는 않는다. 그런 의미가 필요하면 `methods`의 method recipe나 hook으로
먼저 정의하고, 이 패키지는 entrypoint orchestration에 필요한 최소 연결만 둔다.

## Output Shape

PEFT text encoder run은 보통 아래 산출물을 만든다.

```text
artifacts/adapter/
artifacts/classifier_head.safetensors
reports/report.json
projections/
```

Full text encoder run은 보통 아래 산출물을 만든다.

```text
artifacts/model/
artifacts/classifier_head.safetensors
reports/report.json
projections/
```

`report.json`의 compatibility alias와 final-selection metadata 의미는
`scripts/experiments/central/ssl_control/README.md`의 output 설명을 따른다.
