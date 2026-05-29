# Query SSL PEFT Support

이 패키지는 중앙 SSL control과 future FL client-local SSL에서 공유 가능한
query-domain SSL PEFT runtime support를 둔다. 직접 실행하는 Hydra entrypoint는
`scripts/experiments/central/ssl_control/*.py`가 소유하고, 이 패키지는 그
entrypoint가 호출하는 runner/helper 구현을 소유한다.

## 구조

- `runners/`
  - `supervised.py`: frozen backbone + PEFT text encoder + linear head supervised baseline.
  - `pseudo_label.py`: pseudo-label replay/self-training helper.
  - `bootstrap_teacher.py`: teacher bootstrap source 기반 pseudo-label 생성/재생 helper.
  - `teacher_source.py`: teacher bootstrap workflow가 쓰는 checkpoint artifact source adapter.
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
  - `teacher_pseudo_label_builder.py`: teacher prediction을 pseudo-label row와 diagnostics payload로 변환.
  - `teacher_pseudo_label_artifact_writer.py`: teacher prediction trace/summary JSON artifact 저장.
  - query adaptation dataset export.
- `teacher_providers/`
  - `fixed_embedding_classifier/`: 기존 fixed embedding classifier teacher 구현을
    임시로 감싼 migration 위치다. 새 provider family는 여기 늘리지 않고,
    재사용 가능한 provider 의미와 prediction core는 `methods/` owner 아래로 옮긴다.
- `config/`
  - Hydra initial checkpoint와 pseudo-label algorithm preset 해석.
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
중앙 entrypoint는 consistency-family workflow를 직접 실행하고, teacher bootstrap은
내부 helper/workflow와 compatibility source 설정으로만 드러난다. 이 패키지 안에
남은 provider 구현은 fixed classifier seed 제거 과정의 compatibility 표면으로만
취급한다.

새 Query SSL 알고리즘은 원칙적으로 `run_peft_ssl_control.py`를 재사용하고
`strategy_axes/ssl_objective/consistency_method` Hydra config만 추가/교체한다.
