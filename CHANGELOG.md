# Changelog

All notable changes to this project will be documented in this file.

## 0.1.2 - 2026-03-18

- Added partition-aware `load_state_dict()` validation so STAC checkpoints fail
  fast when loaded into a mismatched trunk/cap split.
- Fixed the AdamW cap so `amsgrad=True` now matches PyTorch's AMSGrad behavior.
- Moved optimizer step regression tests and the effectiveness benchmark to CUDA
  coverage on supported machines.
- Rewrote the README as Markdown-only documentation and removed tracked SVG
  assets from the repository.
- Added a GitHub Actions path for PyPI Trusted Publishing on release tags.

## 0.1.1 - 2026-03-18

- Added a momentum-accumulating sign trunk that keeps the STAC update sign-based
  while improving convergence stability.
- Added role-specific hyperparameters for trunk vs. cap learning rates and
  weight decay.
- Added opt-in non-finite gradient checks and broader optimizer regression
  tests.
- Tightened package metadata to Python 3.13 and added packaging validation to
  CI.
- Refreshed the README with diagrams, benchmark notes, and release guidance.

## 0.1.0 - 2026-03-18

- Initial public release of the STAC optimizer.
