# emlgrad Architecture

## Goal

Build `emlgrad`: the smallest possible autograd engine inspired by `micrograd`, except the only primitive math op is:

`eml(x, y) = exp(x) - ln(y)`

This is a Python-only project. The target is not a tensor library. The target is a tiny scalar `Value` engine that proves we can build a usable differentiable system from one primitive operator.

## Design Reset

The previous Rust-plus-Python tensor direction is intentionally dropped.

The new design principles are:

1. Python only
2. scalar `Value` graph only
3. eager execution only
4. reverse-mode autograd only
5. higher ops are compositions of `eml`, not new primitives

If this core works well, a later layer can wrap lists or tensors around it. That is explicitly not v0.

## Non-Goals For V0

Out of scope:

- tensors
- broadcasting
- GPU
- JIT
- Rust bindings
- PyTorch-like module system
- optimizers
- large neural-network framework features beyond a tiny pedagogical wrapper
- performance work beyond keeping the code small and clear

## Minimal Repository Layout

```text
/AGENTS.md
/docs/
  README.md
  architecture.md
  lowering.md
/emlgrad/
  __init__.py
  engine.py
  nn.py
/tests/
```

Recommended responsibility split:

- `emlgrad/engine.py`: `Value`, graph links, `eml`, topological backward pass
- `emlgrad/nn.py`: tiny `micrograd`-style `Module`/`Neuron`/`Layer`/`MLP` wrappers on top of scalar `Value`
- `tests/`: forward and gradient checks

## Core Object Model

The entire runtime should center on one object:

```text
Value(
  data: complex,
  grad: complex,
  requires_grad: bool,
  _prev: set[Value],
  _op: str,
  _backward: callable,
)
```

Notes:

- `data` is complex from day one
- real inputs are stored as complex numbers with zero imaginary part
- literal constants introduced during EML lowering are cached non-gradient leaves
- `_prev`, `_op`, and `_backward` should look familiar to anyone who has read `micrograd`
- `label` can be added for debugging, but only if it stays lightweight

## Primitive Operation

There should be one primitive constructor on `Value`:

`eml(other)`

Semantics:

```text
out = exp(self.data) - log(other.data)
```

using Python's complex math and the principal branch of `log`.

Local derivatives:

- `d eml(x, y) / dx = exp(x)`
- `d eml(x, y) / dy = -1 / y`

That means the backward rule is simple and local:

```text
self.grad += out.grad * exp(self.data)
other.grad += out.grad * (-1 / other.data)
```

## Derived Ops Strategy

Every user-facing math helper must be defined as a composition of `eml`.

The architecture should support two categories:

### 1. Primitive-adjacent helpers

These are immediate wrappers around obvious `eml` identities.

Examples:

- `exp(x)` from `eml(x, 1)`
- `one()` as the distinguished constant leaf

### 2. Library helpers

These are `Value` methods and operator overloads that assemble bigger graphs using known EML identities from the paper.

Examples we may support early:

- `log`
- `neg`
- `add`
- `sub`
- `mul`
- `div`
- `pow`

Important rule:

- if we do not yet have a clean EML construction for an op, we do not fake it with a new primitive

## Execution Model

Execution is eager:

1. construct `Value` leaves
2. call `eml` or a derived helper
3. compute forward values immediately
4. keep parent links for backprop
5. call `.backward()` on a scalar output

This is exactly the right complexity level for v0. No tapes, interpreters, schedulers, or graph compilers are needed.

## Backward Pass

The backward pass should stay close to `micrograd`:

1. build a topological ordering from the output node
2. seed `out.grad = 1`
3. traverse the graph in reverse topological order
4. call each node's `_backward`

The implementation uses an explicit stack rather than recursive DFS, because EML lowerings create much deeper scalar graphs than the original `micrograd` arithmetic.

Because `eml` is the only primitive op, the whole autodiff engine has one real derivative rule. Everything else inherits correctness from composition.

## Numeric Semantics

The runtime should be honest about complex analysis:

- values are complex by default
- `log` uses the principal branch
- branch-cut behavior is part of the model, not a bug to hide
- singularities such as `log(0)` or division by zero may produce `inf`, `nan`, or exceptions depending on implementation choices

We should document those choices rather than smooth them away.

The reference Python implementation makes two explicit choices:

- `log(0)` is represented with an IEEE754-style extended value `-inf + i * arg(0)` so the compiler-style EML lowerings for `zero()` and `neg()` can execute in plain Python while preserving signed-zero branch information
- backward passes through singularities may therefore accumulate `inf`/`nan` gradients
- the tiny pedagogical `nn.py` wrapper initializes biases away from exact zero, so `micrograd`-style regularization examples stay in the real-safe subset by default

## Public API Recommendation

Keep the first API extremely small.

Recommended v0:

- `Value(x)`
- `Value.eml(other)`
- `Value.backward()`
- a very small set of operator overloads and convenience methods on `Value`

Reasonable early conveniences:

- `Value.exp()`
- `Value.log()`
- `+`, `-`, `*`, `/`, `**` once their EML lowerings are written down clearly

Do not rush operator overloading. It is optional. The important part is correctness and conceptual cleanliness.

## Dependency Policy

Start with no third-party runtime dependencies if possible.

The standard library is likely enough:

- `cmath` for complex math
- `math` only if needed for real-side checks

Avoid for v0:

- `numpy`
- `sympy`
- `torch`
- `jax`

If we need a test dependency later, add it intentionally and keep it out of the core package.

## Milestones

### M0: Docs

- lock the scalar-only architecture
- document invariants

### M1: Engine

- implement `Value`
- implement primitive `eml`
- implement topological `.backward()`

### M2: Derived Ops

- add `one()`
- add `exp()`
- add the first small batch of EML-derived helpers

### M3: Validation

- forward tests
- finite-difference gradient checks
- branch-cut and singularity tests

## First Principle

If a choice makes the project more like `micrograd` and less like a general framework, prefer that choice.

This repo should feel small, inspectable, and mathematically disciplined.
