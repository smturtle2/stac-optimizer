# stac-optimizer

[![PyPI version](https://img.shields.io/pypi/v/stac-optimizer)](https://pypi.org/project/stac-optimizer/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/release/python-3130/)
[![Torch >= 2.10](https://img.shields.io/badge/torch-%3E%3D2.10-ee4c2c)](https://pytorch.org/)
[![CI](https://github.com/smturtle2/stac-optimizer/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/stac-optimizer/actions/workflows/workflow.yml)

[한국어 README](https://github.com/smturtle2/stac-optimizer/blob/main/README.ko.md)

STAC stands for SignSGD Trunk, AdamW Cap.

It is a PyTorch optimizer that keeps the earlier trainable layers on a
momentum-stabilized sign trunk and the last `N` trainable layers on AdamW.
The goal is simple: keep optimizer-state VRAM lower than full AdamW while
preserving strong optimization behavior where adaptive updates matter most.

| Item | Value |
| --- | --- |
| Python | `>=3.13` |
| PyTorch | `>=2.10` |
| Default split | last `1` trainable layer uses AdamW |
| Trunk | decoupled weight decay + `sign(EMA(grad))` |
| Cap | AdamW with decoupled weight decay |
| Extra VRAM knob | `trunk_state_dtype=torch.bfloat16` |
| Validation | local CUDA test suite + research benchmark |

## Optimizer Layout

```mermaid
flowchart LR
    A[Trainable layers in registration order] --> B[Earlier trainable layers]
    A --> C[Last N trainable layers]
    B --> D[Sign trunk<br/>decoupled weight decay<br/>EMA(grad) -> sign update<br/>optional bf16 state]
    C --> E[AdamW cap<br/>decoupled weight decay]
```

## Installation

```bash
python -m pip install stac-optimizer
```

For local development:

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
    weight_decay=1e-2,
    trunk_state_dtype=torch.bfloat16,
    error_if_nonfinite=True,
)

inputs = torch.randn(8, 128)
targets = torch.randn(8, 10)

loss = torch.nn.functional.mse_loss(model(inputs), targets)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print("trunk layers:", optimizer.partition.trunk_layer_names)
print("cap layers:", optimizer.partition.cap_layer_names)
```

## Why This Design

- [signSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434)
  motivates sign-based updates as a low-state alternative to adaptive methods.
- [Momentum Ensures Convergence of SIGNSGD under Weaker Assumptions](https://proceedings.mlr.press/v202/sun23l.html)
  supports using momentum before taking the sign instead of raw `sign(grad)`.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  supports the AdamW-style decoupled decay used in the cap.
- [Deconstructing What Makes a Good Optimizer for Autoregressive Language Models](https://openreview.net/forum?id=zfeso8ceqr)
  argues that much of the benefit of adaptivity can come from a small subset of
  parameters, which is the main motivation for concentrating AdamW in the cap.

This does not mean one fixed STAC setting is best on every task. Local CUDA
investigation on this repository showed a real tradeoff:

- `trunk_lr=lr` fits small dense toy problems faster.
- The default conservative split (`trunk_lr=0.75 * lr`) was slightly more
  stable on the held-out teacher/student benchmark below.

Treat `trunk_lr` as a tuning knob, not a universal constant.

## CUDA Research Benchmark

Primary benchmark script:
[examples/research_benchmark.py](https://github.com/smturtle2/stac-optimizer/blob/main/examples/research_benchmark.py)

Machine-readable report:
[docs/benchmark/research_benchmark.json](https://github.com/smturtle2/stac-optimizer/blob/main/docs/benchmark/research_benchmark.json)

Methodology:

- CUDA only
- separate train/validation splits
- `5` seeds
- `12` epochs and `20` updates per epoch
- reports epoch-by-epoch validation loss curves
- measures optimizer state plus peak CUDA allocated/reserved memory on first step

Snapshot from `2026-03-19` on `torch 2.10.0+cu126` and `NVIDIA GeForce RTX 3070`:

![STAC CUDA research benchmark](https://raw.githubusercontent.com/smturtle2/stac-optimizer/main/docs/benchmark/research_benchmark.png)

Regression validation loss:

| Optimizer | Final val loss mean | Final val loss range |
| --- | ---: | ---: |
| `STAC` default (`cap=1`) | `0.046044` | `0.044386 - 0.047686` |
| `STAC` matched trunk lr | `0.046207` | `0.044730 - 0.047581` |
| `STAC` plain sign trunk | `0.043162` | `0.041903 - 0.044614` |
| `AdamW` baseline | `0.043753` | `0.042771 - 0.045108` |

Classification validation:

| Optimizer | Final val loss mean | Final val loss range | Final val acc mean |
| --- | ---: | ---: | ---: |
| `STAC` default (`cap=1`) | `0.303325` | `0.252935 - 0.333419` | `0.8926` |
| `STAC` matched trunk lr | `0.323920` | `0.287477 - 0.333865` | `0.8828` |
| `STAC` plain sign trunk | `0.314426` | `0.279694 - 0.330161` | `0.9039` |
| `AdamW` baseline | `0.304733` | `0.275815 - 0.317797` | `0.9074` |

Memory probe:

| Optimizer | Optimizer state MB | Peak allocated MB | Peak reserved MB |
| --- | ---: | ---: | ---: |
| `STAC` default (`cap=1`) | `3.637` | `31.925` | `38.000` |
| `STAC` matched trunk lr | `3.637` | `31.925` | `38.000` |
| `STAC` plain sign trunk | `0.004` | `28.292` | `34.000` |
| `AdamW` baseline | `7.270` | `35.565` | `40.000` |

This benchmark is evidence, not a universal leaderboard. It is meant to answer
two practical questions for this repository:

- Does STAC remain competitive with AdamW on held-out CUDA tasks?
- Does STAC reduce optimizer-state and peak-memory pressure in practice?

## Public API

The package exports:

- `STAC`
- `partition_trainable_layers(model, last_n_layers=1)`
- `LayerGroup`
- `STACPartition`

Useful runtime guarantees:

- deterministic trunk/cap partitioning based on `model.named_modules()`
- explicit rejection of sparse gradients
- whole-step skip on non-finite dense gradients unless
  `error_if_nonfinite=True`
- checkpoint validation against saved layer names, parameter names, and state
  tensor shapes

## Verification

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
python examples/research_benchmark.py --device cuda
```

The repository also keeps the older quick smoke benchmark at
[examples/toy_benchmark.py](https://github.com/smturtle2/stac-optimizer/blob/main/examples/toy_benchmark.py)
for fast sanity checks, but the research benchmark above is the primary CUDA
evidence for README claims.
