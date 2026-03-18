# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

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
