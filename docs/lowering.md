# EML Lowerings

`emlgrad` keeps one primitive math node:

`eml(x, y) = exp(x) - ln(y)`

Everything else is a graph composition of that node and the distinguished constant `1`, implemented directly inside `Value` methods and operator overloads.

## Supported Derived Ops

- `exp(x) = eml(x, 1)`
- `log(x) = eml(1, exp(eml(1, x)))`
- `zero() = log(1)`
- `sub(x, y) = eml(log(x), exp(y))`
- `neg(x) = sub(zero(), x)`
- `add(x, y) = sub(x, neg(y))`
- `inv(x) = exp(neg(log(x)))`
- `mul(x, y) = exp(add(log(x), log(y)))`
- `div(x, y) = mul(x, inv(y))`
- `pow(x, y) = exp(mul(y, log(x)))`
- `relu(x) = (x + sqrt(x^2)) / 2` on the real axis, with `sqrt(x) = x^(1/2)`

These are the same small identities used by the author's public EML compiler.

## Branch Notes

The implementation works in the complex domain with Python's principal-branch `cmath.log`.

- `exp(log(z))` matches `z` away from the logarithm singularity.
- `log(exp(z))` is branch-sensitive, so arithmetic lowerings such as `sub`, `add`, and anything built on top of them are exact on the usual real-safe inputs and inherit principal-branch behavior elsewhere.
- `log(0)` uses an IEEE754-style extended value with a signed-zero-aware phase: `-inf + i * arg(0)`. This keeps compiler-style lowerings such as `zero()` and `neg()` executable in plain Python and preserves the side of the branch cut when the imaginary part is `-0.0`.
- Backward passes through singularities still produce non-finite gradients.
- The tiny `nn.py` wrapper now keeps parameters away from exact zero at initialization so notebook-style `p*p` regularization stays inside the real-safe subset.
