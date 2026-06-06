# WraithAdam / WraithAdamW

Drop-in PyTorch optimizers that increase the contribution of squared gradients to the second-moment accumulator, producing stronger adaptive damping while retaining standard Adam bias correction.

**WraithAdamW** adds decoupled weight decay (AdamW-style). **WraithAdam** is the no–weight-decay variant.

## What changes vs Adam / AdamW

Both optimizers follow the usual first-moment update. The difference is in the second moment:

| | Second-moment update |
|---|---|
| **Adam / AdamW** | `v = β₂·v + (1 − β₂)·g²` |
| **WraithAdam / WraithAdamW** | `v = β₂·v + (1 − β₁)·g²` |

With the default `β₁ = 0.9`, `β₂ = 0.999`, the squared-gradient term is weighted by `0.1` instead of `0.001` — two orders of magnitude larger. That makes the denominator in the parameter update react faster to recent gradient magnitude, i.e. stronger adaptive damping.

Other defaults:

- `eps = 1e-10` (Adam uses `1e-8`) for slightly more stable division
- `weight_decay = 0.01` on WraithAdamW (decoupled, applied before the Adam step)

## Installation

Requires Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<your-org>/WrathfulAdamW.git
cd WrathfulAdamW
uv sync
```

On Linux, PyTorch is resolved from the ROCm 7.2 wheel index (see `pyproject.toml`). Adjust `[tool.uv.sources]` if you need CUDA or CPU-only builds.

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `lr` | — | Learning rate |
| `betas` | `(0.9, 0.999)` | `(β₁, β₂)` for moment estimates |
| `eps` | `1e-10` | Term added to the denominator for numerical stability |
| `weight_decay` | `0.01` | Decoupled L2 penalty (WraithAdamW only) |

## When to try it

WraithAdam/WraithAdamW can help when training is noisy or loss spikes from large per-parameter gradients. The stronger second-moment accumulation dampens aggressive steps more quickly than Adam.

Start with the same `lr` you would use for AdamW and tune from there — the different `v` dynamics may warrant a slightly higher or lower rate depending on the task.

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run mypy WraithAdam.py WraithAdamW.py
```

## License

MIT — see [LICENSE](LICENSE).
