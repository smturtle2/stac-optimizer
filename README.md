# stac-optimizer

[![PyPI version](https://img.shields.io/pypi/v/stac-optimizer)](https://pypi.org/project/stac-optimizer/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/release/python-3130/)
[![Torch >= 2.10](https://img.shields.io/badge/torch-%3E%3D2.10-ee4c2c)](https://pytorch.org/)
[![CI](https://github.com/smturtle2/stac-optimizer/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/stac-optimizer/actions/workflows/workflow.yml)

[Korean README](README.ko.md)

STAC stands for SignSGD Trunk, AdamW Cap.

It is a PyTorch optimizer for models where you want sign-based updates through
most of the network, but still want AdamW on the last few trainable layers
where optimization is usually more sensitive. The trunk uses
`sign(momentum-smoothed gradient)` by default because momentum-stabilized sign
methods are materially more reliable than plain `sign(grad)` in both theory and
practice.

| Item | Value |
| --- | --- |
| Python | `>=3.13` |
| PyTorch | `>=2.10` |
| Default split | last `1` trainable layer uses AdamW |
| Trunk update | decoupled weight decay + `sign(EMA(grad))` |
| Cap update | AdamW with decoupled weight decay |
| CUDA validation | local tests and benchmark suite |

## Why STAC

- Keeps the bulk of the model on sign-based updates.
- Preserves AdamW on the last `N` trainable layers.
- Uses materially less optimizer state than full AdamW when only the cap keeps
  adaptive moments.
- Uses deterministic partitioning based on `model.named_modules()`.
- Exposes the chosen split through `optimizer.partition`.
- Rejects sparse gradients and dynamic `add_param_group()` explicitly.
- Skips the whole step on non-finite dense gradients unless
  `error_if_nonfinite=True`.
- Validates optimizer checkpoints against layer names, parameter names, and
  saved state shapes.

## Optimizer Layout

```mermaid
flowchart LR
    A[Trainable layers in registration order] --> B[Earlier layers]
    A --> C[Last N layers]
    B --> D[STAC trunk<br/>decoupled weight decay<br/>EMA(grad) -> sign update]
    C --> E[STAC cap<br/>AdamW<br/>decoupled weight decay]
```

## Installation

Install from PyPI:

```bash
python -m pip install stac-optimizer
```

Install the local repository for development:

```bash
python -m pip install -e ".[dev]"
```

## Quickstart

```python
import torch
from torch import nn

from stac_optimizer import STAC


model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)

optimizer = STAC(
    model,
    lr=1e-3,
    last_n_layers=1,
    trunk_momentum=0.9,
    trunk_lr=8e-4,
    cap_lr=1e-3,
    weight_decay=1e-2,
    error_if_nonfinite=True,
)

inputs = torch.randn(8, 128)
targets = torch.randn(8, 10)

loss = torch.nn.functional.mse_loss(model(inputs), targets)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print("trunk layers:", optimizer.partition.trunk_layer_names)
print("trunk params:", optimizer.partition.trunk_parameter_names)
print("cap layers:", optimizer.partition.cap_layer_names)
print("cap params:", optimizer.partition.cap_parameter_names)
```

## Partition Rules

STAC walks `model.named_modules()` in registration order and treats each module
that owns trainable parameters directly (`recurse=False`) as one layer.

- The final `last_n_layers` trainable layers become the AdamW cap.
- Frozen parameters are skipped when counting layers.
- Shared parameters are assigned to the first discovered owner.
- Root-level parameters are exposed as `"<root>"`.
- `last_n_layers=0` keeps the whole model in the sign-based trunk.
- Oversized `last_n_layers` moves the whole model into the AdamW cap.

## Public API

The public package exports:

- `STAC`: the optimizer itself.
- `partition_trainable_layers(model, last_n_layers=1)`: inspect the split
  without constructing the optimizer.
- `LayerGroup`: one trainable module slice with `name`, `parameter_names`, and
  `parameters`.
- `STACPartition`: immutable split metadata with
  `trunk_layer_names`, `cap_layer_names`,
  `trunk_parameter_names`, `cap_parameter_names`,
  `trunk_parameters`, and `cap_parameters`.

## Design Notes

The defaults are intentionally conservative:

- The trunk uses momentum because momentum-smoothed sign methods are more
  stable than plain sign-only updates. See
  [signSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434)
  and
  [Momentum Ensures Convergence of SIGNSGD under Weaker Assumptions](https://proceedings.mlr.press/v202/sun23l.html).
- The cap uses AdamW-style decoupled weight decay rather than mixing weight
  decay into the gradient. See
  [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- Layer selection stays explicit because sign-based and adaptive methods have
  different tradeoffs depending on noise, conditioning, and where adaptation is
  most useful in the network.

Practical tuning guidance:

- If training is unstable, increase `trunk_momentum` before increasing
  `trunk_lr`.
- If the head adapts too slowly, increase `cap_lr`.
- If the model underfits, move more layers into the cap by increasing
  `last_n_layers`.

## CUDA Benchmark Suite

The repository includes [`examples/toy_benchmark.py`](examples/toy_benchmark.py),
which runs synthetic regression and classification tasks across multiple seeds
on CUDA. It compares:

- `STAC default (cap=1)`
- `STAC plain sign trunk`
- `STAC wider cap (cap=2)`
- `AdamW baseline`

Run it with:

```bash
python examples/toy_benchmark.py --device cuda --seeds 5 --steps 150
```

Verified local snapshot from `2026-03-18` on `torch 2.10.0+cu126` and an
`NVIDIA GeForce RTX 3070`:

| Task | STAC default | Plain sign trunk | Wider cap (`last_n_layers=2`) | AdamW |
| --- | ---: | ---: | ---: | ---: |
| Regression mean loss | `0.075852` | `0.140853` | `0.077104` | `0.118262` |
| Classification mean loss | `0.006573` | `0.022765` | `0.011192` | `0.017693` |

Representative optimizer-state snapshot on the same machine from the benchmark's
deeper memory probe:

| Optimizer | Optimizer state MB |
| --- | ---: |
| `STAC` default | `3.637` |
| `STAC` plain sign trunk | `0.004` |
| `STAC` wider cap | `3.762` |
| `AdamW` | `7.270` |

This benchmark is designed as a reproducible sanity check, not a universal
leaderboard. It focuses on optimization quality and optimizer-state memory
rather than claiming a universal wall-clock speedup.

## Verification

Local CUDA verification:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
python examples/toy_benchmark.py --device cuda --seeds 5 --steps 150
```

Most recent local CUDA run on `2026-03-18`:

- `python -m pytest -q`: `28 passed`
- `python examples/toy_benchmark.py --device cuda --seeds 5 --steps 150`:
  produced the tables above

What the test suite covers:

- deterministic partitioning behavior
- optimizer-step parity against AdamW for the cap
- sparse-gradient and non-finite gradient safeguards
- checkpoint round-trips and mismatch rejection
- CUDA comparisons showing the default trunk beating plain signSGD on both
  regression and classification tasks
- CUDA integration checks showing STAC stays competitive with AdamW while
  using materially less optimizer state

GitHub Actions automation:

- On pull requests and pushes to `main`: CPU tests, packaging, and wheel smoke
  checks.
- On `v*` tags: version validation, rebuild, `twine check`, PyPI publishing,
  and GitHub Release creation.

## Release

This project uses `setuptools-scm`, so releases are created from Git tags.
Repository changelog entries live in GitHub Releases rather than a committed
`CHANGELOG.md`.

Typical release flow:

```bash
git push origin main
git tag vX.Y.Z
git push origin vX.Y.Z
```

The tag workflow then:

1. Verifies that `vX.Y.Z` matches the computed package version.
2. Builds fresh distributions and runs `twine check`.
3. Publishes to PyPI via GitHub Actions Trusted Publishing.
4. Creates the matching GitHub Release and attaches the built artifacts.

PyPI Trusted Publishing must be configured for this repository and
`.github/workflows/workflow.yml` before the publish step can succeed.
