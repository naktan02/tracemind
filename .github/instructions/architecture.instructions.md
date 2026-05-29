---
applyTo: "shared/src/contracts/**/*.py,shared/src/domain/entities/**/*.py,agent/src/services/**/*.py,main_server/src/services/**/*.py,scripts/**/*.py,docs/**/*.md"
---

Use contract-first design.

When refactoring, identify the real variation axes first. Ask what changes often, what must remain stable, what changes together, and what must remain independent.

Do not hide field meaning in distant design docs only. If a contract field is important for runtime behavior, add a short explanation in the contract file or in a source-adjacent contract guide.

Prefer canonical representations at boundaries. If producer and consumer handle the same meaning, they should converge on one canonical shape. Isolate normalization in one explicit place instead of spreading it across the flow.

Keep compatibility logic explicit and separate. Legacy formats, temporary transforms, and backward-compatibility shims should not leak into the core path. Leave a clear removal condition.

Separate policy from mechanism. If rules such as thresholds, ranking, build choices, aggregation choices, or fallback conditions are important, keep them explicit instead of burying them inside infrastructure code.

Before tuning, make failures observable. Add dumps, traces, summaries, or metrics that show which stage failed before changing parameters.

Design producers and consumers together. Avoid one-sided generalization where writers and readers silently depend on different assumptions.

Choose patterns based on what changes together:

- Use `Strategy` when algorithms vary under one stable contract.
- Use `Factory` or family objects when one selection changes multiple behaviors together.
- Use `State` when lifecycle stage matters.
- Use `Policy` or `Specification` when rules are the core variation.
- Use `Pipeline` when processing flow is the core abstraction.
- Use `Port/Adapter` when external dependency replacement is the core concern.
- Use `Decorator` for cross-cutting behavior.
- Use `Registry` only as thin wiring at the composition root.

Preserve the distinction between:

- shared/common concerns
- context-specific concerns
- domain logic and cross-cutting infrastructure

If a design pushes context-specific interpretation or sensitive state into a common component, call that out explicitly and propose a split.

If a change affects a common layer, assess drift risk and blast radius before accepting it.
