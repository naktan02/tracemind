---
applyTo: "shared/src/contracts/**/*.py,shared/src/domain/entities/**/*.py,agent/src/services/**/*.py,main-server/src/services/**/*.py,scripts/**/*.py,docs/**/*.md"
---

Use contract-first design.

When refactoring, identify the real variation axes first. Ask what changes often, what must remain stable, what changes together, and what must remain independent.

Do not hide field meaning in distant design docs only. If a contract field is important for runtime behavior, add a short explanation in the contract file or in a source-adjacent contract guide.

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
