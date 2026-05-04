"""실험 catalog runtime path와 공통 blocker 상수."""

SEED_RUNTIME_PATH = "scripts.seed"
CENTRAL_ADAPTATION_RUNTIME_PATH = "scripts.central_adaptation"
FEDERATED_SIMULATION_RUNTIME_PATH = "scripts.federated_simulation"
AGENT_LIVE_STORED_EVENT_RUNTIME_PATH = "agent.live_stored_event"
MAIN_SERVER_ROUND_RUNTIME_PATH = "main_server.round_runtime"

PHASE2_METADATA_ONLY_BLOCKER = (
    "Phase 2 compiler는 entrypoint와 Hydra preset selection만 지원한다. "
    "이 항목은 metadata-only catalog surface이며, 후속 phase에서 전용 compile "
    "규칙이 추가돼야 한다."
)
