# Query SSL PEFT Support

이 패키지는 중앙 SSL control과 future FL client-local SSL에서 공유 가능한
query-domain SSL PEFT runtime support를 둔다. 직접 실행하는 Hydra entrypoint는
`scripts/experiments/central/ssl_control/*.py`가 소유하고, 이 패키지는 그
entrypoint가 호출하는 runner/helper 구현을 소유한다.

## 구조

- `runners/`
  - `supervised.py`: frozen backbone + PEFT text encoder + linear head supervised baseline.
  - `pseudo_label.py`: pseudo-label replay/self-training helper.
  - `consistency.py`: USB PseudoLabel, FixMatch 등 Query SSL runner.
  - `query_adaptation.py`: agent-local query adaptation dataset runner wrapper.
- `runtime_context.py`
  - PEFT encoder model/data/eval 공통 scaffolding.
- `io/`
  - `artifacts.py`: run artifact 저장 public orchestration entrypoint.
  - `artifact_paths.py`: run directory와 output path 계산.
  - `model_artifact_exporter.py`: adapter/tokenizer/classifier 파일 export.
  - `manifest_builder.py`: manifest/report payload 조립.
  - `artifact_writer.py`: manifest/report JSON 쓰기.
  - `teacher_pseudo_label_artifact_writer.py`: teacher prediction trace/summary JSON artifact 저장.
  - query adaptation dataset export.
- `config/`
  - Hydra initial checkpoint metadata helper.
- `query_ssl/`
  - Query SSL method 공통 모델, augmentation preparation, 중앙 unlabeled view
    preparation resolver.

## 실행 경계

실험을 직접 실행할 때는 이 패키지의 runner 파일을 직접 실행하지 않고 아래
Hydra entrypoint를 실행한다.

- `scripts/experiments/central/ssl_control/run_peft_supervised_control.py`
- `scripts/experiments/central/ssl_control/run_peft_ssl_control.py`

공통으로 재사용될 알고리즘 core는 이 패키지 안에서 키우지 않고 `methods/`로
올린다. cross-boundary contract/domain은 `shared/`, runtime bridge는
`scripts/runtime_adapters/`에 둔다.

`scripts`는 teacher를 별도 실험 stage나 public strategy axis로 소유하지 않는다.
중앙 entrypoint는 consistency-family workflow를 직접 실행한다. 기존 teacher bootstrap
compatibility workflow와 fixed-classifier fallback은 scripts surface에서 제거했다.
새 teacher source가 필요하면 `methods`의 teacher hook이나 method recipe로 먼저
정의한다.

새 Query SSL 알고리즘은 원칙적으로 `run_peft_ssl_control.py`를 재사용하고
`strategy_axes/ssl_objective/consistency_method` Hydra config만 추가/교체한다.

## Owner Refactor Queue

아래 분류는 2026-05-29 기준의 다음 정리 대상이다.

| 분류 | 대상 | 판단 |
|---|---|---|
| `methods 이동 완료` | `methods/ssl/teacher_pseudo_label.py` | pseudo-label acceptance preset 해석과 teacher prediction -> pseudo-label export 의미를 `methods` owner로 이동했다. |
| `methods 이동` | `runners/pseudo_label_inputs.py` | pseudo-label replay 입력 정규화가 단순 파일 입출력보다 학습 row 의미에 가깝다. |
| `methods 이동 검토` | `query_ssl/common.py`, `query_ssl/view_preparation.py` | Query SSL method 공통 준비 로직이 scripts helper를 넘어 method scaffolding이면 `methods/ssl` 또는 adaptation helper로 승격한다. |
| `scripts 유지` | `runners/consistency.py` | central SSL canonical experiment runner다. 다만 method 분기 없이 thin orchestration만 유지한다. |
| `scripts 유지` | `runners/supervised.py` | central supervised control entrypoint가 호출하는 thin runner다. |
| `scripts 유지` | `runtime_context.py`, `runtime_metrics.py` | 실험 실행 context/metric bridge다. |
| `scripts 유지` | `io/artifacts.py`, `artifact_paths.py`, `artifact_writer.py`, `manifest_builder.py`, `model_artifact_exporter.py` | artifact write/export/orchestration IO다. |
| `scripts 유지` | `io/query_adaptation.py`, `io/query_adaptation_multiview.py`, `io/labeled_row_export.py` | dataset export와 run artifact write에 가깝다. |
| `scripts 유지` | `config/initial_checkpoint.py` | Hydra initial checkpoint surface 해석과 manifest bridge다. |
| `삭제 완료` | teacher bootstrap compatibility subtree | active method family가 아니고 scripts owner가 아니므로 제거했다. |
| `보류` | `runners/pseudo_label.py` | pseudo-label self-training이 central control family에 남는지, 아니면 method recipe/helper로 내려갈지 먼저 결정이 필요하다. |
| `보류` | `runners/query_adaptation.py` | agent-local query adaptation bridge라 central SSL cleanup과 분리해서 판단해야 한다. |

권장 순서:

1. `pseudo_label_inputs.py`를 `methods` owner로 옮길지 작은 contract로 먼저 자른다.
2. central canonical path에 남길 최소 scripts surface를 `consistency.py`, `supervised.py`, artifact IO로 한정한다.
