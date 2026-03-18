# stac-optimizer

STAC stands for SignSGD Trunk, AdamW Cap.

It is a PyTorch optimizer for models where you want cheap sign-based updates
through most of the network, but still want AdamW on the last few trainable
layers where optimization is often most sensitive. The default trunk is
`sign(momentum)` rather than plain `sign(grad)` because the momentum-smoothed
variant is materially more stable in both theory and practice.

| Item | Value |
| --- | --- |
| Python | `>=3.13` |
| PyTorch | `>=2.10` |
| Default split | last `1` trainable layer uses AdamW |
| Trunk update | sign-based update with momentum smoothing |
| Cap update | AdamW with decoupled weight decay |

## Why STAC

- Keeps the bulk of the model on sign-based updates.
- Preserves AdamW where late-layer adaptation matters most.
- Partitions layers deterministically from `model.named_modules()`.
- Supports separate learning rates and weight decay for trunk and cap.
- Exposes the chosen partition through `optimizer.partition`.
- Rejects sparse gradients and dynamic `add_param_group()` explicitly.

## Install

```bash
python -m pip install .
```

Development install:

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

print("trunk:", optimizer.partition.trunk_layer_names)
print("cap:", optimizer.partition.cap_layer_names)
```

## Partition Rule

STAC walks trainable layers in module registration order and splits them into
two regions:

```text
[ earlier trainable layers ................. ][ last N trainable layers ]
                  trunk: signSGD-like                        cap: AdamW
```

- Layer discovery uses `named_parameters(recurse=False)`.
- Frozen parameters are skipped when counting layers.
- Shared parameters are assigned to the first discovered owner.
- Root-level parameters are exposed as `"<root>"`.
- `last_n_layers=0` keeps the whole model in the trunk.
- Oversized `last_n_layers` moves the whole model into the cap.

## Hyperparameters

| Argument | Meaning |
| --- | --- |
| `lr` | Shared base learning rate. |
| `trunk_lr`, `cap_lr` | Role-specific learning rates. If `trunk_lr` is omitted in hybrid mode, STAC defaults it to `0.75 * lr`. |
| `last_n_layers` | Number of final trainable layers that become AdamW. |
| `trunk_momentum` | EMA factor for the trunk before taking the sign. |
| `weight_decay` | Shared default decoupled weight decay. |
| `trunk_weight_decay`, `cap_weight_decay` | Role-specific decoupled weight decay. |
| `betas`, `eps`, `amsgrad` | AdamW cap hyperparameters. |
| `maximize` | Maximize instead of minimize. |
| `error_if_nonfinite` | Raise on `NaN` or `Inf` gradients. |

## Stability Notes

The defaults are intentionally conservative:

- The trunk uses momentum because sign-only methods are substantially more
  stable when the sign is taken after smoothing. See
  [signSGD with Majority Vote](https://arxiv.org/abs/1810.05291) and
  [Momentum Ensures Convergence of SIGNSGD under Weaker Assumptions](https://proceedings.mlr.press/v202/sun23l.html).
- The cap uses AdamW-style decoupled weight decay rather than mixing decay
  into the gradient. See
  [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- Recent analysis shows sign-based methods have different optimization
  tradeoffs from SGD and Adam depending on noise and conditioning, which is
  why STAC exposes both `last_n_layers` and separate trunk/cap learning rates.
  See
  [Exact Risk Curves of SignSGD in Modern Overparameterized Linear Regression](https://proceedings.mlr.press/v267/xiao25c.html).

Practical tuning guidance:

- If training is noisy or unstable, raise `trunk_momentum` before increasing
  the trunk learning rate.
- If the model underfits, move more layers into the AdamW cap with a larger
  `last_n_layers`.
- If the head adapts too slowly, raise `cap_lr` without forcing the entire
  network into AdamW.

## Benchmark Snapshot

The repository includes [`examples/toy_benchmark.py`](examples/toy_benchmark.py)
for a quick sanity check. A representative local run on `Python 3.13.12` and
`torch 2.10.0+cu126` produced:

| Optimizer | Mean final loss |
| --- | ---: |
| `STAC` default | `0.033961` |
| `STAC` with plain sign trunk | `0.107899` |
| `torch.optim.AdamW` | `0.074642` |

This is a sanity benchmark, not a universal ranking. The important signal is
that the default STAC trunk is meaningfully better than a plain sign trunk on a
real optimization loop.

## Constraints

- Sparse gradients are unsupported in both trunk and cap.
- `add_param_group()` is intentionally unsupported because STAC derives its
  parameter groups from model structure.
- The split follows module registration order, not dynamic forward order.

## Verification

GitHub Actions automation:

- On pull requests and pushes to `main`: CPU-based tests, packaging, and built
  wheel smoke checks.
- On `v*` tags: version validation, rebuild, `twine check`, PyPI publishing,
  and GitHub Release creation.

Local CUDA verification for maintainers before a release:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
python examples/toy_benchmark.py
```

Most recent local CUDA run:

- `python -m pytest -q`: `17 passed in 6.45s`
- `python -m build` and `python -m twine check dist/*`: passed
- `python examples/toy_benchmark.py`:
  `STAC` default `0.033961`, plain sign trunk `0.107899`, `AdamW` `0.074642`

## Release

This repository uses `setuptools-scm`, so release tags must match the package
version that the workflow computes from the tagged commit.

Typical release flow:

```bash
git push origin main
git tag v0.1.2
git push origin v0.1.2
```

The tag workflow then:

1. Verifies that `vX.Y.Z` matches the computed package version.
2. Builds fresh distributions and runs `twine check`.
3. Publishes to PyPI via GitHub Actions Trusted Publishing.
4. Creates the matching GitHub Release and attaches the built artifacts.

Project maintainers must register this repository and
`.github/workflows/workflow.yml` as a Trusted Publisher on PyPI for the publish
step to succeed.

See [CHANGELOG.md](CHANGELOG.md) for released versions only.
