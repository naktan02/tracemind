# TraceMind

TraceMind is a research prototype for privacy-preserving wellbeing signal
processing and semi-supervised learning experiments. The repository brings
together a local agent, FastAPI services, reusable learning methods, experiment
entrypoints, result dashboards, and a family-facing extension UI in one
monorepo.

The project is organized around two goals:

- Keep raw personal text and interpretation state inside the local agent
  boundary.
- Make shared model improvement and experiment results reproducible through
  contracts, methods, artifacts, and dashboards.

## What You Can Do

- Run the local agent API for captured text, inference, and wellbeing summaries.
- Explore child and parent flows through the family extension UI.
- Run central SSL, fixed-feature baseline, and federated SSL simulation
  experiments.
- Convert reports under `runs/` into a SQLite index and static dashboard JSON.
- Compare central and federated experiment results, metrics, per-class results,
  and projection artifacts in the experiment dashboard.

## Research Snapshot

The current dashboard export is generated from run reports under `runs/`.
Central SSL remains a pooled/offline control, while FL SSL is the main
non-IID comparison rail. The FL SSL rows below use 10 clients, Dirichlet
label-skew alpha `0.3`, seed `42`, PEFT text encoder LoRA updates, and 30
rounds.

Selected central PEFT SSL results on the `szegeelim_general4` labeled source
and `ourafla_reddit` unlabeled/test source:

| Method | Run | Macro-F1 | Accuracy | ECE | Worst-class F1 |
|---|---|---:|---:|---:|---:|
| AdaMatch | `peft_adamatch_2026_06_16_034721` | 0.788 | 0.788 | 0.101 | 0.690 |
| FlexMatch | `peft_flexmatch_2026_06_04_130155` | 0.784 | 0.783 | 0.132 | 0.694 |
| SimMatch | `peft_simmatch_2026_06_04_143519` | 0.781 | 0.782 | 0.126 | 0.662 |
| ReMixMatch | `peft_remixmatch_2026_06_10_073507` | 0.780 | 0.779 | 0.081 | 0.664 |
| FreeMatch | `peft_freematch_2026_06_04_134451` | 0.780 | 0.779 | 0.112 | 0.678 |
| FixMatch | `peft_fixmatch_2026_06_04_120231` | 0.777 | 0.772 | 0.178 | 0.672 |

Selected FL SSL simulation results:

| Label budget | Method | Role | Macro-F1 | Accuracy | ECE |
|---|---|---|---:|---:|---:|
| 100/class | FedMatch | method-owned | 0.714 | 0.733 | 0.253 |
| 100/class | FreeMatch | manual baseline | 0.712 | 0.729 | 0.223 |
| 100/class | FlexMatch | manual baseline | 0.711 | 0.725 | 0.229 |
| 100/class | AdaMatch | manual baseline | 0.692 | 0.710 | 0.253 |
| 100/class | FixMatch | manual baseline | 0.680 | 0.694 | 0.272 |
| 1024/class | FixMatch | manual baseline | 0.746 | 0.753 | 0.233 |
| 1024/class | FreeMatch | manual baseline | 0.731 | 0.726 | 0.251 |
| 1024/class | FedMatch | method-owned | 0.731 | 0.726 | 0.265 |

These are selected orientation numbers, not a replacement for the full result
index. Detailed metrics, per-client validation, calibration, communication
cost, and projection artifacts come from the run reports under `runs/` and the
static dashboard data generated at
[apps/experiment_dashboard/data/experiment_dashboard.json](apps/experiment_dashboard/data/experiment_dashboard.json).

## Repository Layout

| Path | Purpose |
|---|---|
| `shared/` | Contracts, domain entities, and canonical payload interpretation shared across runtimes |
| `methods/` | Reusable computation cores for SSL, adaptation, classification, and FL aggregation |
| `conf/` | Hydra execution compositions and dataset/model/runtime/strategy parameters |
| `agent/` | Local inference, captured text, local state, local training, and wellbeing API |
| `main_server/` | FL round lifecycle, aggregation, and artifact publication |
| `scripts/` | Thin entrypoints for datasets, central SSL, FL SSL, result indexing, and reports |
| `apps/family_extension/` | Family-facing Chrome extension UI and collector shell |
| `apps/experiment_dashboard/` | Static dashboard for comparing experiment results |
| `tests/` | Unit, integration, architecture guard, and cross-boundary verification tests |
| `docs/` | Architecture, operations, contracts, quality, and governance documentation |

## Quick Start

Python dependencies are defined by `pyproject.toml` and `uv.lock`.

```bash
uv sync --extra dev --extra experiments
uv run pytest
```

Check lint and formatting:

```bash
uv run ruff check main_server/src agent/src shared/src scripts tests
uv run ruff format --check main_server/src agent/src shared/src scripts tests
```

The repository currently does not include Docker Compose or a separate `infra/`
manifest. Local execution is based on Python `uv`, FastAPI/Uvicorn, and Vite
apps.

## Run The APIs

Main server:

```bash
uv run uvicorn main_server.src.api.main:app --reload --port 8000
```

Agent:

```bash
uv run uvicorn agent.src.api.main:app --reload --port 8001
```

If local agent settings are needed, copy `agent/.env.example` to `agent/.env`.
Do not commit `agent/.env`.

```bash
cp agent/.env.example agent/.env
```

## Run The Family Extension UI

Start the agent API first, then run the frontend.

```bash
cd apps/family_extension
npm install
npm run dev
```

The default agent API target is `http://127.0.0.1:8001`. To use another address:

```bash
cd apps/family_extension
VITE_AGENT_API_BASE_URL=http://127.0.0.1:9001 npm run dev
```

Chrome extension build:

```bash
cd apps/family_extension
npm run build
```

## Run The Experiment Dashboard

First convert reports under `runs/` into the dashboard index, then open the
static file server.

```bash
uv run python -m scripts.workflows.result_index.ingest \
  --runs-root runs \
  --db data/processed/experiment_index/experiment_results.sqlite \
  --dashboard-json apps/experiment_dashboard/data/experiment_dashboard.json
```

Serve the dashboard as static files.

```bash
python -m http.server 5175 -d apps/experiment_dashboard
```

Open `http://127.0.0.1:5175` in a browser. Cache regeneration rules, including
`--reset`, are documented in
[apps/experiment_dashboard/README.md](apps/experiment_dashboard/README.md).

## Main Workflows

### Local Wellbeing Flow

```text
captured text
-> agent-local scoring and evidence summary
-> wellbeing signal projection
-> child / parent UI payload
-> family extension UI
```

Relevant entrypoints:

- [agent/README.md](agent/README.md)
- [agent/src/features/wellbeing/README.md](agent/src/features/wellbeing/README.md)
- [apps/family_extension/README.md](apps/family_extension/README.md)

### Experiment Flow

```text
Hydra config
-> experiment script
-> methods core
-> run artifacts and reports
-> result index
-> experiment dashboard
```

Relevant entrypoints:

- [conf/README.md](conf/README.md)
- [methods/README.md](methods/README.md)
- [scripts/README.md](scripts/README.md)
- [apps/experiment_dashboard/README.md](apps/experiment_dashboard/README.md)

### Experiment Guides

| Experiment | Guide |
|---|---|
| Central SSL / PEFT text encoder SSL | [scripts/experiments/central/ssl_control/README.md](scripts/experiments/central/ssl_control/README.md) |
| Central PEFT / full text encoder supervised controls | [scripts/experiments/central/ssl_control/README.md](scripts/experiments/central/ssl_control/README.md#supervised-controls) |
| Central fixed-feature supervision and classical self-training | [scripts/experiments/central/fixed_feature_control/README.md](scripts/experiments/central/fixed_feature_control/README.md) |
| FL SSL simulation / FedMatch comparison | [scripts/experiments/fl_ssl/README.md](scripts/experiments/fl_ssl/README.md) |

## Documentation

For a first pass, start with [docs/README.md](docs/README.md) and choose only the
documents relevant to the task.

| Document | Description |
|---|---|
| [docs/architecture/system-overview.md](docs/architecture/system-overview.md) | Current runtime composition and code ownership boundaries |
| [docs/operations/local-runbook.md](docs/operations/local-runbook.md) | Local execution, GPU preflight, and smoke procedures |
| [docs/quality/test-strategy.md](docs/quality/test-strategy.md) | Test layers and protected behavior |
| [docs/api/api-surface.md](docs/api/api-surface.md) | Agent and main server FastAPI endpoint map |
| [shared/src/contracts/README.md](shared/src/contracts/README.md) | Shared payload contract interpretation |

Detailed work routing for internal contributors and AI agents lives in
[docs/execution_index.md](docs/execution_index.md) and [AGENTS.md](AGENTS.md).

## Safety And Scope

TraceMind is a research/prototype codebase. Wellbeing signals and child-support
UI surfaces are intended as assistive guidance based on local state. They do
not replace medical diagnosis or emergency response systems. The default
architecture keeps raw text and personal interpretation state inside the
agent-local boundary.
