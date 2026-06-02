# Query SSL Text Encoder Support

이 패키지는 중앙 SSL control과 future FL client-local SSL에서 공유 가능한
query-domain SSL text encoder runtime support를 둔다. PEFT text encoder control과
중앙 full text encoder supervised control은 같은 supervised runner/context helper를
재사용하되, PEFT-specific artifact/export 이름은 PEFT leaf runner 안에만 남긴다.
직접 실행하는 Hydra entrypoint는 `scripts/experiments/central/ssl_control/*.py`가
소유하고, 이 패키지는 그 entrypoint가 호출하는 runner/helper 구현을 소유한다.

## 구조

- `runners/`
  - `supervised_text_encoder.py`: 중앙 supervised text encoder train/eval 공통 흐름.
  - `supervised.py`: frozen backbone + PEFT text encoder + linear head supervised baseline.
  - `full_text_encoder_supervised.py`: full encoder + linear head supervised-only baseline.
  - `consistency.py`: USB PseudoLabel, FixMatch 등 Query SSL runner.
- `runtime_context.py`
  - text encoder model/data/eval 공통 scaffolding. 기본 model builder는 현재
    PEFT text encoder control을 가리키고, full encoder runner는 별도 builder를 주입한다.
- `io/`
  - `artifacts.py`: run artifact 저장 public orchestration entrypoint.
  - `full_text_encoder_artifacts.py`: full-model supervised control artifact 저장.
  - `artifact_paths.py`: run directory와 output path 계산.
  - `model_artifact_exporter.py`: adapter/tokenizer/classifier 파일 export.
  - `manifest_builder.py`: manifest/report payload 조립.
  - `artifact_writer.py`: manifest/report JSON 쓰기.
- `config/`
  - Hydra initial checkpoint metadata helper.
- `query_ssl/`
  - Query SSL method 공통 모델, augmentation preparation, 중앙 unlabeled view
    preparation resolver.

## 실행 경계

실험을 직접 실행할 때는 이 패키지의 runner 파일을 직접 실행하지 않고 아래
Hydra entrypoint를 실행한다.

- `scripts/experiments/central/ssl_control/run_peft_supervised_control.py`
- `scripts/experiments/central/ssl_control/run_full_text_encoder_supervised_control.py`
- `scripts/experiments/central/ssl_control/run_peft_ssl_control.py`

공통으로 재사용될 알고리즘 core는 이 패키지 안에서 키우지 않고 `methods/`로
올린다. cross-boundary contract/domain은 `shared/`, runtime bridge는
`scripts/runtime_adapters/`에 둔다.

`scripts`는 teacher를 별도 실험 stage나 public strategy axis로 소유하지 않는다.
중앙 entrypoint는 consistency-family workflow를 직접 실행한다. 기존 teacher bootstrap
compatibility workflow와 fixed-classifier fallback은 scripts surface에서 제거했다.
새 teacher source가 필요하면 `methods`의 teacher hook이나 method recipe로 먼저
정의한다.

새 PEFT-backed Query SSL 알고리즘은 원칙적으로 `run_peft_ssl_control.py`를 재사용하고
`strategy_axes/ssl_objective/consistency_method` Hydra config만 추가/교체한다.

## Owner Refactor Queue

아래 분류는 2026-05-29 기준의 다음 정리 대상이다.

| 분류 | 대상 | 판단 |
|---|---|---|
| `삭제 완료` | `runners/pseudo_label.py`, `methods/ssl/teacher_pseudo_label.py`, `methods/ssl/pseudo_label_replay.py` | offline pseudo-label replay/self-training workflow는 중앙 online SSL canonical surface가 아니므로 제거했다. |
| `methods 이동 검토` | `query_ssl/common.py`, `query_ssl/view_preparation.py` | Query SSL method 공통 준비 로직이 scripts helper를 넘어 method scaffolding이면 `methods/ssl` 또는 adaptation helper로 승격한다. |
| `scripts 유지` | `runners/consistency.py` | central SSL canonical experiment runner다. 다만 method 분기 없이 thin orchestration만 유지한다. |
| `scripts 유지` | `runners/supervised_text_encoder.py`, `runners/supervised.py`, `runners/full_text_encoder_supervised.py` | central supervised control entrypoint가 호출하는 runner다. 공통 train/eval 흐름은 `supervised_text_encoder.py`가 소유한다. |
| `scripts 유지` | `runtime_context.py`, `runtime_metrics.py` | 실험 실행 context/metric bridge다. |
| `scripts 유지` | `io/artifacts.py`, `io/full_text_encoder_artifacts.py`, `artifact_paths.py`, `artifact_writer.py`, `manifest_builder.py`, `model_artifact_exporter.py` | artifact write/export/orchestration IO다. |
| `scripts 유지` | `io/labeled_row_export.py` | labeled JSONL export helper다. |
| `scripts 유지` | `config/initial_checkpoint.py` | Hydra initial checkpoint surface 해석과 manifest bridge다. |
| `삭제 완료` | teacher bootstrap compatibility subtree | active method family가 아니고 scripts owner가 아니므로 제거했다. |
| `삭제 완료` | `runners/query_adaptation.py`, `io/query_adaptation.py`, `io/query_adaptation_multiview.py` | agent-local query adaptation export bridge는 중앙 지도/중앙 SSL/FSSL canonical experiment surface가 아니므로 제거했다. |

권장 순서:

1. central canonical path에 남길 최소 scripts surface를 `consistency.py`, `supervised.py`, artifact IO로 한정한다.
