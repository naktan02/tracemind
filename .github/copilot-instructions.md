# Repository-Wide AI Instructions

Prefer contract-first and architecture-first changes over quick local hacks.

Separate change axes explicitly. Do not mix concerns that change for different reasons in one class or one payload unless there is a strong reason.

Keep shared/common responsibilities distinct from context-specific responsibilities. Do not push user-, environment-, or feature-specific interpretation into a common layer unless that tradeoff is explicit.

Treat files in `shared/src/contracts/` and `shared/src/domain/entities/` as source-of-truth contracts. Put field meaning close to those files instead of relying only on distant docs.

Prefer canonical representations at boundaries. Do not let the same meaning flow through different shapes across producers and consumers unless compatibility is explicitly isolated.

Treat compatibility logic as an explicit layer. Keep legacy format conversion, temporary translation, and backward-compatibility code out of the core path, and document the condition for removing it.

Separate policy from mechanism. Thresholds, selection rules, build choices, aggregation rules, and similar policies should not be buried as accidental implementation details.

Before tuning parameters, add observability. Prefer dumps, traces, summaries, and metrics that show where failure occurs before changing thresholds or heuristics.

Design producers and consumers together. Do not generalize one side while leaving the other side on incompatible assumptions.

Choose patterns based on variation structure. Strategy, Factory, State, Policy, Specification, Pipeline, Port/Adapter, Decorator, and Registry are all valid when they match the problem. Use raw registry dictionaries only as thin composition-root wiring, not as the core domain abstraction.

When adding docs, keep one short source-adjacent explanation near the code and one optional design note elsewhere only if architectural background matters.
