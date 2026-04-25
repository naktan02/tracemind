# TraceMind Local Runbook

이 문서는 2026-04-25 기준 로컬 개발, smoke, GPU preflight, stale process 확인 절차를 정리한다.

현재 저장소에는 Docker Compose 또는 `infra/` manifest가 없다. 로컬 실행 기준은 Python `uv`, FastAPI/Uvicorn, Vite app이다.

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

Developer experiment web:

```bash
cd apps/experiment_web
npm install
npm run dev
```

기본 dev origin은 `http://localhost:5173` 또는 `http://127.0.0.1:5173`이다.

Family extension UI:

```bash
cd apps/family_extension
npm install
npm run dev
```

기본 dev origin은 `http://localhost:5174` 또는 `http://127.0.0.1:5174`다.

아이용 AI 마음 도움을 로컬 Ollama로 확인하려면 agent API 실행 전에 아래 환경변수를
설정한다. 설정하지 않으면 deterministic `local_guarded` 응답을 쓴다.

```bash
export TRACEMIND_CHILD_SUPPORT_LLM_PROVIDER=ollama
export TRACEMIND_CHILD_SUPPORT_OLLAMA_MODEL=llama3.1:8b
export TRACEMIND_CHILD_SUPPORT_OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Ollama 자체는 별도 터미널에서 실행한다.

```bash
ollama serve
ollama pull llama3.1:8b
```

각 app build:

```bash
cd apps/experiment_web && npm run build
cd apps/family_extension && npm run build
```

## 6. 자주 쓰는 실험 실행

Dataset pipeline:

```bash
uv run python scripts/datasets/run_dataset_pipeline.py
```

Fixed classifier seed:

```bash
uv run python scripts/experiments/train_softmax_classifier.py
```

LoRA supervised baseline:

```bash
uv run python scripts/experiments/train_lora_classifier.py
```

Pseudo-label bootstrap:

```bash
uv run python scripts/experiments/train_lora_bootstrap_classifier_teacher.py
```

FixMatch baseline:

```bash
uv run python scripts/experiments/train_lora_fixmatch.py
```

FL simulation:

```bash
uv run python scripts/experiments/run_federated_simulation.py
```

Hydra 설정 preview:

```bash
uv run python scripts/experiments/train_lora_classifier.py --cfg job
```

## 7. Runtime Profiles

`scripts/conf/runtime/*.yaml`이 script runtime source of truth다.

| Profile | 의미 |
|---|---|
| `gpu_online` | CUDA 사용, cache miss 시 online download 허용 |
| `gpu_local` | CUDA 사용, local cache만 사용 |
| `cpu_local` | CPU 사용, local cache만 사용 |
| `auto_local` | GPU가 있으면 GPU, 없으면 CPU, local cache만 사용 |
| `auto_online` | 자동 device 선택, online download 허용 |

기본 실험 문서는 `dataset=ourafla`, `embedding=mxbai`, `runtime=gpu_online`을 기준으로 본다.

## 8. GPU Preflight

GPU 의존 실험 전 실제 실행 환경에서 확인한다.

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

주의:

- sandbox에서 GPU가 안 보여도 실제 머신에는 GPU가 있을 수 있다.
- GPU 확인 실패를 곧바로 GPU 부재로 단정하지 말고, 필요하면 sandbox 밖에서 다시 확인한다.
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
| `data/processed/` | classifier head, processed split 등 재사용 산출물 |
| `runs/` | 실험 실행별 report/log/artifact |
| `agent/state/` | local prototype pack, query/scored event 등 agent state |
| `main_server/state/` | server prototype/model/round state |
| `hf_cache/` | model cache |
| `tmp/` | 임시 비교/외부 reference checkout |

위 경로 대부분은 Git ignore 대상이다. 재현 가능한 값은 코드, Hydra config, manifest, report summary로 남긴다.

## 11. Local Smoke Checklist

1. `uv run pytest tests/architecture`가 통과한다.
2. `uv run pytest shared/tests agent/tests main_server/tests`가 통과한다.
3. main server `/health`가 응답한다.
4. agent `/health`와 `/api/v1/system/health`가 응답한다.
5. family extension 또는 experiment web을 다루는 변경이면 해당 app `npm run build`가 통과한다.
6. script/config 변경이면 관련 entrypoint import 또는 Hydra config test를 함께 실행한다.
