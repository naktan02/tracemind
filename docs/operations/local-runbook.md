# TraceMind Local Runbook

이 문서는 현재 로컬 개발, smoke, GPU preflight, stale process 확인 절차를 정리한다.

현재 저장소에는 Docker Compose 또는 `infra/` manifest가 없다. 로컬 실행 기준은 Python `uv`, FastAPI/Uvicorn, Vite app이다.

## 0. Codex 실행 권한 모델

프로젝트 기본 Codex 설정은 아래를 기준으로 한다.

```toml
approval_policy = "on-request"
sandbox_mode = "danger-full-access"
```

목적은 GPU, model cache, `uv` cache, local server처럼 sandbox 안에서 실패하거나
환경이 다르게 보이는 검증을 실제 실행 환경에서 먼저 수행하기 위함이다.

### 기본 실행 가능

작업 검증에 직접 필요하면 추가 확인 없이 실행한다.

- 읽기/탐색: `rg`, `find`, `sed`, `git diff`, `git status`
- 검증: `uv run pytest`, `uv run ruff check`, `uv run ruff format --check`
- GPU preflight: `nvidia-smi`, `uv run python -c "import torch; ..."`
- 로컬 smoke: `uv run python ...`, `uv run uvicorn ...`, app `npm run build`
- 로컬 실험 smoke: 관련 Hydra config와 entrypoint가 명확한 `uv run python scripts/...`

### 사전 확인 필요

아래는 full access 환경에서도 사용자 확인을 먼저 받는다.

- repo 밖 파일 쓰기, 삭제, 이동, 권한 변경
- `rm -rf`, `git reset --hard`, `git clean`, checkout으로 작업물 되돌리기
- commit, push, remote 변경, branch 삭제
- `.env`, credential, system config, shell profile 변경
- 장시간/대용량 다운로드, paid API 호출, 외부 서비스에 데이터 전송
- 관련 없는 프로세스 종료 또는 포트 점유 프로세스 강제 종료

### 하지 말 것

- sandbox 우회나 권한 상승을 목적으로 임시 경로에 source-of-truth를 복제하지 않는다.
- GPU가 안 보인다는 이유만으로 CPU fallback 결과를 최종 검증으로 간주하지 않는다.
- 실패한 명령을 숨기고 다른 명령으로 우회하지 않는다. 실패 원인과 재실행 환경을 남긴다.

## 1. 사전 조건

| 항목 | 기준 |
|---|---|
| Python | `>=3.11,<3.13` |
| Python dependency source | `pyproject.toml`, `uv.lock` |
| Package manager | `uv` 권장 |
| Frontend package manager | 각 app의 `package-lock.json` 기준 npm |
| GPU 실험 | 실제 실행 환경에서 CUDA 확인 후 실행 |

## 2. Python 환경 준비

권장 경로:

```bash
uv sync --extra dev --extra experiments
```

가상환경을 직접 만들 때:

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,experiments]"
```

주의:

- `requirements.txt`는 현재 source of truth가 아니다.
- PyTorch는 `pyproject.toml`의 `pytorch-cu124` index 설정을 따른다.
- transformer/model 다운로드가 필요한 실행은 network와 cache 상태에 영향을 받는다.

## 3. 기본 검증

```bash
uv run pytest
uv run ruff check main_server/src agent/src shared/src scripts tests
uv run ruff format --check main_server/src agent/src shared/src scripts tests
```

빠른 부분 검증:

```bash
uv run pytest shared/tests agent/tests main_server/tests
uv run pytest tests/unit
uv run pytest tests/integration
uv run pytest tests/architecture
```

## 4. API 서버 실행

Main server:

```bash
uv run uvicorn main_server.src.api.main:app --reload --port 8000
```

Agent:

```bash
uv run uvicorn agent.src.api.main:app --reload --port 8001
```

Health check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/api/v1/system/health
```

## 5. Frontend 실행

Family extension UI:

```bash
cd apps/family_extension
npm install
npm run dev
```

기본 dev origin은 `http://localhost:5174` 또는 `http://127.0.0.1:5174`다.

아이용 AI 마음 도움을 로컬 Ollama로 확인하려면 `agent/.env.example`을
`agent/.env`로 복사해서 로컬 주소와 모델을 설정한다. `agent/.env`는 커밋하지
않는다.

```bash
cp agent/.env.example agent/.env
```

agent API는 `agent/.env`를 먼저 읽고, repo root `.env`는 fallback으로 읽는다.
이미 shell에 같은 환경변수가 있으면 shell 값이 `.env`보다 우선한다.

Ollama 자체는 별도 터미널에서 실행한다.

```bash
ollama serve
ollama pull exaone3.5:2.4b
```

App build:

```bash
cd apps/family_extension && npm run build
```

Experiment dashboard:

```bash
uv run python -m scripts.experiments.result_index.ingest \
  --runs-root runs \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json

python -m http.server 5175 -d apps/experiment_dashboard
```

정적 dashboard JS만 바꾼 경우 빠른 문법 검증은 아래로 닫는다.

```bash
node --check apps/experiment_dashboard/src/app.js
```

## 6. 자주 쓰는 실험 실행

Dataset pipeline:

```bash
uv run python scripts/datasets/run_dataset_pipeline.py
```

Query SSL split/view materialization:

```bash
uv run python scripts/datasets/materialize_query_ssl_split.py \
  execution_context/dataset_asset=mental_health_kaggle

uv run python scripts/datasets/materialize_query_ssl_views.py \
  execution_context/query_view=szegeelim_general4_ssl_labeled1024_per_class_seed42_nllb_v1
```

Fixed classifier seed:

```bash
uv run python scripts/experiments/central_classifier_seed/train_softmax_classifier.py
```

PEFT supervised baseline:

```bash
uv run python scripts/experiments/central_ssl_control/train_peft_supervised_classifier.py
```

USB PseudoLabel baseline:

```bash
uv run python scripts/experiments/central_ssl_control/train_peft_ssl_classifier.py \
  strategy_axes/ssl/consistency_method=pseudolabel_usb_v1 \
  output_dir=runs/train_peft_ssl_classifier_pseudolabel
```

FixMatch baseline:

```bash
uv run python scripts/experiments/central_ssl_control/train_peft_ssl_classifier.py
```

중앙 SSL smoke/test 실행은 `run_controls/central_ssl/budget=smoke`를 사용한다.
이 경우 산출물은 `runs/_smoke/train_peft_ssl_classifier/...` 또는
`runs/_smoke/train_peft_supervised_classifier` 아래에 저장되어 main run과 섞이지
않는다.
기본 dashboard/index ingest(`--runs-root runs`)는 `runs/_smoke/**` report를
제외한다.

FL simulation:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_simulation.py \
  run_controls/fl_ssl/budget=smoke \
  federated_run_budget.rounds=1
```

FL SSL seed sweep smoke:

```bash
uv run python scripts/experiments/fl_ssl/run_federated_seed_sweep.py \
  run_controls/fl_ssl/budget=smoke \
  federated_run_budget.rounds=1
```

FL SSL runner는 accidental long run을 막기 위해 총 예정 communication round가
`run_safety.max_total_rounds_without_ack`를 넘으면 시작 전에 실패한다. 단일 run은
`rounds`, sweep은 `rounds * sweep 항목 수`로 계산한다. 장시간 실행을 별도 승인받은
경우에만 `run_safety.allow_long_run=true`와
`run_safety.long_run_ack=ALLOW_FL_SSL_LONG_RUN`을 같이 override한다.

새 wiring이나 method 검증은 먼저 `1-round` smoke 또는 `5-round` reduced run으로
확인한다. 현재 FL SSL reduced preset은 `run_controls/fl_ssl/budget=reduced`이며
`10 clients`, `5 rounds`, `runs/fl_ssl` root를 쓴다. smoke preset 산출물은
`runs/_smoke/fl_ssl` 아래에 쌓아 웹/논문용 run과 섞지 않는다. full-budget 실행은
후보와 비교 조건을 명시한 뒤 `budget=main`과
필요한 long-run ack를 함께 지정한다.

기존 FL SSL 산출물 metadata 검증:

```bash
uv run python scripts/experiments/fl_ssl/verify_federated_report_artifacts.py \
  --client-count-sweep-summary runs/<run_id>/reports/fl_ssl_client_count_sweep.summary.json \
  --expected-client-counts 1,2,3,4,5,6,7,8,9,10 \
  --expected-completed-rounds 1 \
  --expected-round-budget 1 \
  --expected-round-record-count 1 \
  --expect-round-update-count-matches-client-count \
  --expected-seed 42 \
  --expected-shard-policy-name dirichlet_label_skew \
  --expected-shard-alpha 0.3 \
  --expected-split-id-contains alpha0.3 \
  --expected-labeled-exposure-policy client_local_split \
  --expected-run-control-budget-name smoke \
  --expected-run-control-output-dir runs/_smoke/fl_ssl \
  --expected-ssl-algorithm fixmatch \
  --expected-ssl-method fixmatch_usb_v1 \
  --expected-adapter-family lora_classifier \
  --expected-aggregation fedavg \
  --expected-delta-format server_uploaded_artifact_ref \
  --expect-shared-update-count-matches-round-updates \
  --expect-server-owned-update-artifacts \
  --expect-no-agent-local-update-refs \
  --expect-lora-classifier-aggregate-snapshot \
  --expected-embedding-backend transformers_mxbai \
  --expected-embedding-device cuda \
  --expected-embedding-local-files-only true \
  --expected-local-trainer-device cuda \
  --expected-local-trainer-local-files-only true
```

여러 report와 sweep summary를 한 번에 검증해야 하면 artifact별 기대값을 JSON
manifest에 담아 실행한다.

```bash
uv run python scripts/experiments/fl_ssl/verify_federated_report_artifacts.py \
  --manifest path/to/fl_ssl_artifact_verification_manifest.json
```

manifest 구조 예시는
`docs/operations/fl_ssl_artifact_verification_manifest.example.json`를 본다.
현재 로컬 artifact 감사에 쓰는 실제 manifest는
`docs/operations/fl_ssl_artifact_verification_manifest.current.json`다.
현재 FL SSL 목표 대비 artifact 감사표는
`docs/operations/fl_ssl_execution_audit.md`를 본다.

FL SSL은 별도 override가 없으면 `gpu_local + mxbai` 기준으로 실행한다.
`gpu_online`은 cache warm-up/최초 다운로드용이고, `cpu_local + hash_debug` 조합은
entrypoint wiring smoke나 빠른 디버그 용도다. 성능 숫자, report 비교, 논문 판단에는
GPU/mxbai 실행 결과만 사용한다.

Hydra 설정 preview:

```bash
uv run python scripts/experiments/central_ssl_control/train_peft_ssl_classifier.py --cfg job
```

## 7. Runtime Profiles

`conf/execution_context/runtime_env/*.yaml`이 script runtime source of truth다.

| Profile | 의미 |
|---|---|
| `gpu_online` | CUDA 사용, cache miss 시 online download 허용 |
| `gpu_local` | CUDA 사용, local cache만 사용 |
| `cpu_local` | CPU 사용, local cache만 사용 |
| `auto_local` | GPU가 있으면 GPU, 없으면 CPU, local cache만 사용 |
| `auto_online` | 자동 device 선택, online download 허용 |

반복 실험과 논문 비교는 `execution_context/dataset_asset=ourafla`,
`execution_context/embedding_adapter=mxbai`, `execution_context/runtime_env=gpu_local`을
기준으로 본다. cache가 없을 때만 `gpu_online`으로 먼저 준비한다.

## 8. GPU Preflight

GPU 의존 실험 전 실제 실행 환경에서 확인한다.

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

주의:

- sandbox나 제한 환경에서 GPU가 안 보여도 실제 머신에는 GPU가 있을 수 있다.
- GPU 확인 실패를 곧바로 GPU 부재로 단정하지 말고, `danger-full-access` 실행 환경에서 다시 확인한다.
- GPU profile 실행 전 `torch.cuda.is_available()`가 `False`면 CPU fallback으로 계속하지 말고, 사용자가 CPU smoke를 원한 경우에만 runtime을 `cpu_local` 또는 `auto_local`로 낮춘다.
- `local_files_only=false` profile은 모델 다운로드를 시도할 수 있다.

## 9. Stale Process 확인

테스트나 디버그 실행이 중단됐거나 터미널 상태가 이상하면 먼저 stale process를 본다.

```bash
ps aux | rg "pytest|uv run python|uvicorn|train_lora|run_federated"
```

종료가 필요하면 대상 PID와 명령을 확인한 뒤 최소 범위로 종료한다.

## 10. 산출물과 상태 위치

| 위치 | 용도 |
|---|---|
| `data/datasets/` | 새 dataset별 raw/mapped/split/query_ssl/view 산출물 |
| `data/artifacts/` | 새 classifier head, LoRA adapter, prototype pack 등 재사용 산출물 |
| `data/cache/` | 새 모델/cache/translation/query view cache |
| `data/processed/` | legacy dataset/model/prototype 산출물 |
| `runs/` | 실험 실행별 report/log/artifact |
| `agent/state/` | local prototype pack, query/scored event 등 agent state |
| `main_server/state/` | server prototype/model/round state |
| `hf_cache/` | legacy model cache |
| `tmp/` | 임시 비교/외부 reference checkout |

위 경로 대부분은 Git ignore 대상이다. 재현 가능한 값은 코드, Hydra config, manifest, report summary로 남긴다.

## 11. Local Smoke Checklist

1. `uv run pytest tests/architecture`가 통과한다.
2. `uv run pytest shared/tests agent/tests main_server/tests`가 통과한다.
3. main server `/health`가 응답한다.
4. agent `/health`와 `/api/v1/system/health`가 응답한다.
5. family extension 변경이면 해당 app `npm run build`, experiment dashboard 변경이면
   `node --check apps/experiment_dashboard/src/app.js`가 통과한다.
6. script/config 변경이면 관련 entrypoint import 또는 Hydra config test를 함께 실행한다.
