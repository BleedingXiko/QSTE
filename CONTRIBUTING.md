# Contributing to QSTE

Thanks for your interest. This is a small project with a strict correctness
bar, so the guidelines below are short but load-bearing.

## Development setup

```bash
git clone https://github.com/BleedingXiko/QSTE
cd QSTE
pip install torch            # any build >= 2.1; a CPU wheel is fine
pip install -e ".[test]"
pytest
```

The native C++ kernels build on first use and cache to `~/.cache/qste`. Ninja
is not required — a setuptools fallback runs when it is missing. Set
`QSTE_KERNELS=torch` to skip native kernels entirely, or `QSTE_KERNELS=native`
to make a build failure raise instead of silently degrading.

## The correctness bar

The three kernel backends (C++, Triton, pure torch) are tested against each
other **bit-for-bit**. Nothing in the suite is mocked. If you touch numerics:

- Add or extend a test that compares your change against the pure-torch
  reference in `kernels/fallback.py` — that file *is* the specification.
- Keep the reference path correct and complete; `import qste` must never fail
  because a native kernel is unavailable.
- GPU-only tests live in `tests/test_gpu.py` and skip cleanly without a device.
  If you cannot run them, say so in the PR and CI will cover the CPU tests.

## Pull requests

- Run `pytest` locally and note what you ran on (CPU only, or a specific GPU).
- Keep the README honest: it only claims speeds that have actually been
  measured. If you add a measured number, say on what hardware.
- One focused change per PR is easier to review than several bundled together.
- Update `CHANGELOG.md` under `## [Unreleased]`.

## Reporting bugs

Open an issue with the smallest model or tensor shape that reproduces it, the
output of `qste.kernel_status()` and `qste.device_profile()`, and your torch
and Python versions.

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0, the same license that covers the project.
