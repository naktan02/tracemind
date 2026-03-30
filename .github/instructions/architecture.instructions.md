---
applyTo: "shared/src/contracts/**/*.py,shared/src/domain/entities/**/*.py,agent/src/services/**/*.py,main-server/src/services/**/*.py,scripts/**/*.py,docs/**/*.md"
---

Use contract-first design.

When refactoring, identify the real variation axes first. Common examples in this repository are adapter family, training backend, aggregation backend, privacy guard, scoring strategy, and projection/experiment configuration.

Do not hide field meaning in distant design docs only. If a contract field is important for runtime behavior, add a short explanation in the contract file or in a source-adjacent contract guide.

Choose patterns based on what changes together:

- Use `Strategy` for interchangeable algorithms under one stable contract.
- Use a family object or abstract factory when one selection changes payload conversion, accepted formats, and backend composition together.
- Use a registry only as thin wiring at the composition root.

Preserve the distinction between:

- global/shared representation and server-aggregated state
- local/private interpretation and personalization
- privacy protection layers and training logic

If a design pushes user-specific drift or privacy-sensitive interpretation into a global shared component, call that out explicitly and propose a split.
