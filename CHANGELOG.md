# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-08-16

First public release. Beta: the API is stable and the test suite is thorough.
Speed is measured on CPU and one GPU (NVIDIA T4); see the README's "Where this
has actually run" for the exact scope.

### Added
- `qste.convert` / `qste.plan` / `qste.export_packed` — in-place model surgery
  with glob selection and a human-readable plan.
- `QSTEOptimizer` — owns the int8 coordinates; all-reduces evidence by default
  under a process group.
- `qste.nn` packed activations that retain bits or an int8 derivative instead
  of a full-precision tensor, plus `packed_activations()` / `elementwise()` /
  `packed()` for functionally-called activations.
- Three numerically-identical kernel backends (C++, Triton, pure torch),
  dispatched by device and tested against each other bit-for-bit.
- Per-device runtime tuning in `kernels/device.py`; no compute-capability
  branches or hard-coded constants.
- Distributed support (DDP evidence reduction, FSDP ignored-states helper).
- `bench/bench.py` and `tools/bundle_cell.py`.

[Unreleased]: https://github.com/BleedingXiko/QSTE/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BleedingXiko/QSTE/releases/tag/v0.1.0
