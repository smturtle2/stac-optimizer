# stac-optimizer

STAC stands for SignSGD Trunk, AdamW Cap.

This library provides a PyTorch optimizer that applies `signSGD` to the trunk
of a model and `AdamW` only to the last `N` trainable layers. The default is
`last_n_layers=1`.

It targets `torch>=2.10`.

## Layer rule

STAC discovers layers deterministically from `model.named_modules()` order.

- A module counts as one layer when it directly owns trainable parameters.
- Parameter discovery uses `named_parameters(recurse=False)`.
- Frozen parameters are skipped when counting layers.
- The last `N` discovered layers become the AdamW cap.
- All earlier layers stay in the signSGD trunk.

## Install

Local install:

```bash
python -m pip install .
```

Development install:

```bash
python -m pip install -e ".[dev]"
```

## Usage

```python
import torch
from torch import nn

from stac_optimizer import STAC


model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.Linear(32, 10),
)

optimizer = STAC(
    model,
    lr=1e-3,
    last_n_layers=1,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
)

inputs = torch.randn(8, 128)
targets = torch.randn(8, 10)

loss = torch.nn.functional.mse_loss(model(inputs), targets)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

You can inspect the resolved partition after construction:

```python
print(optimizer.partition.trunk_layer_names)
print(optimizer.partition.cap_layer_names)
```

## Development

```bash
python -m pytest -q
python -m build
```

## Release

`.github/workflows/workflow.yml` is configured to:

- run tests on pushes and pull requests
- build the package on version tags that match `v*`
- verify that the tag matches the package version derived by `setuptools-scm`
- create a GitHub Release and upload the built `sdist` and `wheel`

Example:

```bash
git tag v0.1.0
git push origin v0.1.0
```
