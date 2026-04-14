# emlgrad

`emlgrad` is a tiny scalar autograd toy with strong `micrograd` vibes.

The twist is that it only has one primitive math op:

`eml(x, y) = exp(x) - ln(y)`

This project is mostly inspired by:

- Andrej Karpathy's [`micrograd`](https://github.com/karpathy/micrograd)
- Andrzej Odrzywolek's paper, ["All elementary functions from a single binary operator"](https://arxiv.org/abs/2603.21852)

So the basic idea is:

What if a `micrograd`-style engine kept the same small DAG + backprop feel, but built its math by composing EML?

## What's here

- [emlgrad/engine.py](emlgrad/engine.py): the scalar `Value` engine and backward pass
- [emlgrad/nn.py](emlgrad/nn.py): a very small `micrograd`-style `MLP` wrapper
- [demo.ipynb](demo.ipynb): a notebook in the spirit of the original `micrograd` demo

Higher ops like `log`, `+`, `-`, `*`, `/`, `**`, and `relu` are all lowered into `eml`.

## Notes

- complex-domain first
- principal-branch `log`
- branch cuts and `log(0)` are not hidden away
- some paths can still produce `inf` or `nan`

This repo is meant to stay small, readable, and a little weird.

## Quickstart

This repo uses `uv`.

```bash
uv sync --group dev
uv run python -m unittest discover -s tests
uv run --group dev ruff check .
```

To open the notebook:

```bash
uv run --with jupyter jupyter notebook demo.ipynb
```
