# stac-optimizer

[![PyPI version](https://img.shields.io/pypi/v/stac-optimizer)](https://pypi.org/project/stac-optimizer/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/release/python-3130/)
[![Torch >= 2.10](https://img.shields.io/badge/torch-%3E%3D2.10-ee4c2c)](https://pytorch.org/)
[![CI](https://github.com/smturtle2/stac-optimizer/actions/workflows/workflow.yml/badge.svg)](https://github.com/smturtle2/stac-optimizer/actions/workflows/workflow.yml)

[Korean README](https://github.com/smturtle2/stac-optimizer/blob/main/README.ko.md) |
[Optimizer docs](https://github.com/smturtle2/stac-optimizer/blob/main/docs/en/optimizer.md) |
[Korean docs](https://github.com/smturtle2/stac-optimizer/blob/main/docs/ko/optimizer.md) |
[Benchmark JSON](https://github.com/smturtle2/stac-optimizer/blob/main/docs/benchmark/research_benchmark.json)

STAC means "SignSGD Trunk, AdamW Cap". It keeps the sign trunk state-free,
uses AdamW only on the final trainable-module tail, and is tuned to reduce
optimizer-state VRAM without giving up tail stability.

| Item | Value |
| --- | --- |
| Python | `>=3.13` |
| PyTorch | `>=2.10` |
| Default split | `last_n_ratio=0.125` |
| Explicit override | `last_n_modules` |
| Default sign decay in hybrid mode | `0.5 * weight_decay` |
| Default no-decay policy | bias + 1-D parameters |
| Preferred public ratio arg | `last_n_ratio` (`adamw_ratio` remains supported) |

## Flow

```mermaid
flowchart LR
    A["Trainable modules<br/>registration order"]
    B["Resolve AdamW cap<br/>`last_n_modules` or<br/>default `last_n_ratio=12.5%`"]

    subgraph S["State-free sign trunk"]
        C["Earlier modules"]
        D["Decoupled weight decay on weight tensors<br/>bias + 1-D params skip decay by default<br/>parameter -= lr * sign(grad)<br/>no momentum, no sign-side state"]
    end

    subgraph T["AdamW cap"]
        E["Final tail modules"]
        F["Standard AdamW on the tail<br/>bias + 1-D params skip decay by default<br/>exp_avg + exp_avg_sq"]
    end

    A --> B
    B --> C
    B --> E
    C --> D
    E --> F

    classDef neutral fill:#f8fafc,stroke:#475569,color:#0f172a,stroke-width:1px;
    classDef sign fill:#d7f0e8,stroke:#0f766e,color:#134e4a,stroke-width:1.5px;
    classDef adam fill:#dbeafe,stroke:#2563eb,color:#1d4ed8,stroke-width:1.5px;

    class A,B neutral;
    class C,D sign;
    class E,F adam;
```

## Install

```bash
python -m pip install stac-optimizer
```

For local development and benchmark generation:

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
    last_n_ratio=0.125,
    weight_decay=1e-2,
    error_if_nonfinite=True,
)

loss = torch.nn.functional.mse_loss(
    model(torch.randn(8, 128)),
    torch.randn(8, 10),
)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

`last_n_ratio` counts only modules that directly own trainable parameters.
Pure containers such as `nn.Sequential` are skipped unless they own parameters
themselves. Use `last_n_modules` when you want an explicit cap size instead.
Bias tensors and 1-D parameters such as LayerNorm scales skip decoupled weight
decay by default in both sections.

## CUDA Research Snapshot

The repository benchmark is CUDA-only and uses held-out validation splits,
`5` paired seeds, seeded teachers, seeded student initialization, fixed batch
schedules per seed, deep residual models, a transformer-like sequence task
with embeddings and LayerNorm, BF16 autocast when supported, epoch-by-epoch
validation loss curves, and a first-step optimizer-memory probe.
The AdamW baseline uses the same bias/1-D no-decay grouping so the comparison
does not hinge on a different weight-decay policy.

![STAC CUDA research benchmark](https://raw.githubusercontent.com/smturtle2/stac-optimizer/main/docs/benchmark/research_benchmark.png)

Snapshot from `2026-03-19` on `torch 2.10.0+cu126` and
`NVIDIA GeForce RTX 3070`:

| Config | Setup | Deep regression val loss | Deep classification val acc | TailNorm val acc | Sequence val acc | Optimizer state MB | Peak step delta MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `STAC default` | `last_n_ratio=0.125`, hybrid default sign decay, bias/1-D no-decay | `0.015066` | `0.7006` | `0.7984` | `0.6909` | `8.133` | `16.125` |
| `STAC full-decay trunk` | `last_n_ratio=0.125`, `sign_weight_decay=weight_decay`, bias/1-D no-decay | `0.015075` | `0.6994` | `0.8064` | `0.7089` | `8.133` | `16.125` |
| `STAC wider cap` | `last_n_ratio=0.25`, bias/1-D no-decay | `0.014726` | `0.6943` | `0.7996` | `0.6909` | `24.149` | `36.125` |
| `AdamW baseline` | full AdamW with the same no-decay policy in practice | `0.013574` | `0.7129` | `0.8268` | `0.7190` | `98.227` | `147.188` |

Repository takeaway: the default preset cuts optimizer state from
`98.227 MB` to `8.133 MB`, the full-decay variant keeps the same memory profile
while helping the norm-heavy and sequence tasks a bit, and the wider cap spends
more AdamW state to improve regression. Those are repository-local measurements,
not universal guarantees.

## Verify

```bash
python -m pytest -q
python examples/research_benchmark.py --device cuda
rm -rf build dist
python -m build
python -m twine check dist/*
```
