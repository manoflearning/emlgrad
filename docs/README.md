# Documentation Structure

This repo is intentionally tiny. The document set should stay tiny too.

Project name: `emlgrad`

Current source of truth:

- `docs/architecture.md`: project shape, invariants, object model, and milestone scope.

Next docs to add only if needed:

- `docs/api.md`: public `Value` API and supported derived ops.
- `docs/lowering.md`: how each helper op expands into `eml`.
- `docs/roadmap.md`: implementation checklist once the architecture is accepted.

Documentation rules:

1. Prefer concrete invariants over broad vision.
2. Keep the project scalar-first until there is a strong reason not to.
3. Every new math helper should explain its `eml` lowering.
