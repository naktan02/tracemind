# Repository-Wide AI Instructions

Prefer contract-first and architecture-first changes over quick local hacks.

Separate change axes explicitly. In this repository, do not mix adapter family, local training algorithm, server aggregation, privacy layer, and runtime scoring concerns in one class or one payload unless there is a strong reason.

Keep global/shared and local/private responsibilities distinct. Shared representation, shared adapter state, and server aggregation are different concerns from personal thresholds, personal prototypes, persistence, and other user-specific interpretation layers.

Treat files in `shared/src/contracts/` and `shared/src/domain/entities/` as source-of-truth contracts. Put field meaning close to those files instead of relying only on distant docs.

Prefer composition, strategy interfaces, and family/factory objects when one choice changes multiple behaviors together. Use raw registry dictionaries only as thin composition-root wiring, not as the core domain abstraction.

When adding docs, keep one short source-adjacent explanation near the code and one optional design note elsewhere only if architectural background matters.
