# AGENTS.md

## Project Intent

Build `emlgrad`, the smallest useful EML autograd project:

- Python only
- scalar autograd engine first
- `micrograd`-style DAG and reverse-mode backprop
- one primitive op: `eml(x, y) = exp(x) - ln(y)`

This repo is no longer a Rust backend or a tensor runtime. It is a tiny autograd system whose higher ops are built by composing `eml`.

## Non-Negotiable Invariants

1. The only primitive math op is `eml`.
2. Higher ops are Python-level compositions of `eml`.
3. Internal values are complex-domain first, using principal-branch `log`.
4. The built-in distinguished constant is `1`.
5. The core abstraction is a scalar `Value`, not a tensor.
6. Keep the code understandable in one sitting.

## Minimality Rules

1. Prefer the Python standard library.
2. Do not add NumPy, PyTorch, JAX, or symbolic-math dependencies to the core path.
3. Avoid frameworks, registries, codegen, and config-heavy abstractions.
4. If a feature does not help the scalar autograd core, defer it.
5. If an op cannot be explained as an `eml` composition, do not add it yet.

## Intended Repository Shape

```text
/AGENTS.md
/docs/
/emlgrad/__init__.py
/emlgrad/engine.py
/emlgrad/ops.py
/tests/
```

`engine.py` should stay tiny and own the `Value` graph object plus backward pass.

## Implementation Guidance

1. Model the core after `micrograd`: scalar nodes, parents, `_backward`, topological traversal.
2. Put the forward formula and local derivative rule for `eml` in one obvious place.
3. Implement derived ops as thin helpers that return composed `Value` graphs.
4. Keep eager execution only.
5. Start without tensors, broadcasting, devices, dtypes, or optimizers.
6. Add convenience methods only if they preserve the "one primitive op" story.

## Documentation Rules

1. Update `docs/architecture.md` when the object model or invariants change.
2. Document each supported derived op and how it lowers to `eml`.
3. If the repo ever grows beyond scalar `Value`, record that decision explicitly first.

## Validation Bar

1. Check forward values on small hand-computed cases.
2. Check gradients against finite differences on representative complex-safe inputs.
3. Include tests around branch cuts, `log(0)`, and complex intermediates.
