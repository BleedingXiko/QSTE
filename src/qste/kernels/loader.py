"""Kernel dispatch: native C++ when it builds, pure torch when it does not.

Import of :mod:`qste` never triggers a compile. The first call that would
benefit from a native kernel attempts the build once, caches the result to
disk keyed by source hash, and falls back permanently if the toolchain is
missing. Set ``QSTE_KERNELS=torch`` to skip the native path entirely, or
``QSTE_KERNELS=native`` to make a build failure raise instead of degrade.

Two build routes are tried. ``cpp_extension.load`` is fast but hard-requires
ninja, which plenty of environments do not have and which is not this
library's business to install. When it is missing, a plain setuptools
``build_ext`` runs in a subprocess instead -- slower to build once, identical
object afterwards.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import subprocess
import sys
import sysconfig
import threading
import warnings
from pathlib import Path

from . import fallback

_SOURCE = Path(__file__).with_name("cpu.cpp")
_lock = threading.Lock()
_extension: object | None = None
_attempted = False
_failure: str | None = None


def _mode() -> str:
    value = os.environ.get("QSTE_KERNELS", "auto").strip().lower()
    if value not in ("auto", "native", "torch"):
        raise ValueError("QSTE_KERNELS must be 'auto', 'native', or 'torch'")
    return value


def _flags() -> list[str]:
    flags = ["-O3", "-std=c++17"]
    if os.environ.get("QSTE_NATIVE_ARCH", "1") != "0":
        machine = platform.machine().lower()
        flags.append("-mcpu=native" if machine in ("arm64", "aarch64") else "-march=native")
    return flags


def _cache_dir() -> Path:
    import torch

    digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:12]
    root = os.environ.get("QSTE_CACHE_DIR") or (Path.home() / ".cache" / "qste")
    tag = (
        f"{platform.system().lower()}-{platform.machine()}"
        f"-py{sys.version_info.major}{sys.version_info.minor}"
        f"-torch{torch.__version__.split('+')[0]}-{digest}"
    )
    return Path(root) / tag


def _import_from(path: Path):
    spec = importlib.util.spec_from_file_location("qste_cpu", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load compiled kernels from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SETUP_TEMPLATE = """\
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="qste_cpu",
    ext_modules=[CppExtension("qste_cpu", [{source!r}], extra_compile_args={flags!r})],
    cmdclass={{"build_ext": BuildExtension.with_options(use_ninja=False)}},
    script_args=["build_ext", "--inplace", "--build-temp", {temp!r}],
)
"""


def _build_with_setuptools(cache: Path):
    """Compile without ninja, via the stock setuptools C++ toolchain."""

    cache.mkdir(parents=True, exist_ok=True)
    script = cache / "_build_qste_cpu.py"
    script.write_text(
        _SETUP_TEMPLATE.format(source=str(_SOURCE), flags=_flags(), temp=str(cache / "obj"))
    )
    verbose = os.environ.get("QSTE_KERNELS_VERBOSE", "0") == "1"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(cache),
        capture_output=not verbose,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()[-12:]
        raise RuntimeError("setuptools build failed:\n" + "\n".join(detail))
    built = _find_built(cache)
    if built is None:
        raise RuntimeError(f"setuptools build produced no extension in {cache}")
    return _import_from(built)


def _find_built(cache: Path) -> Path | None:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    for candidate in (cache.glob(f"qste_cpu*{suffix}"), cache.glob("qste_cpu*.so")):
        for path in candidate:
            return path
    return None


def _build():
    cache = _cache_dir()
    prebuilt = _find_built(cache)
    if prebuilt is not None:
        return _import_from(prebuilt)
    try:
        from torch.utils.cpp_extension import load

        return load(
            name="qste_cpu",
            sources=[str(_SOURCE)],
            extra_cflags=_flags(),
            build_directory=str(_ensure(cache / "ninja")),
            verbose=os.environ.get("QSTE_KERNELS_VERBOSE", "0") == "1",
        )
    except Exception:
        # Usually a missing ninja. Fall through to the toolchain everyone has.
        return _build_with_setuptools(cache)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extension():
    """The compiled module, or ``None`` if the native path is unavailable."""

    global _extension, _attempted, _failure
    mode = _mode()
    if mode == "torch":
        return None
    if _attempted:
        if _extension is None and mode == "native":
            raise RuntimeError(f"QSTE native kernels unavailable: {_failure}")
        return _extension
    with _lock:
        if not _attempted:
            try:
                _extension = _build()
            except Exception as error:  # compiler, ninja, or headers missing
                _failure = f"{type(error).__name__}: {error}"
                _extension = None
            _attempted = True
    if _extension is None and mode == "native":
        raise RuntimeError(f"QSTE native kernels unavailable: {_failure}")
    return _extension


def native_available() -> bool:
    return extension() is not None


# ---------------------------------------------------------------------------
# CUDA backend (Triton). Loaded lazily and independently of the C++ build, so
# a machine with a GPU and no host compiler still gets native GPU kernels.
# ---------------------------------------------------------------------------

_cuda_module: object | None = None
_cuda_attempted = False
_cuda_failure: str | None = None


def cuda_backend():
    global _cuda_module, _cuda_attempted, _cuda_failure
    if _mode() == "torch":
        return None
    if _cuda_attempted:
        return _cuda_module
    with _lock:
        if not _cuda_attempted:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("no CUDA device")
                from . import cuda as _cuda

                _cuda_module = _cuda
            except Exception as error:
                _cuda_failure = f"{type(error).__name__}: {error}"
                _cuda_module = None
            _cuda_attempted = True
    if _cuda_module is None and _mode() == "native":
        raise RuntimeError(f"QSTE CUDA kernels unavailable: {_cuda_failure}")
    return _cuda_module


def cuda_available() -> bool:
    return cuda_backend() is not None


def status() -> dict[str, object]:
    """What the dispatcher will actually do, for a startup log line."""

    module = extension()
    cuda = cuda_backend()
    return {
        "mode": _mode(),
        "native": module is not None,
        "cuda": cuda is not None,
        "source": str(_SOURCE),
        "failure": _failure,
        "cuda_failure": _cuda_failure,
    }


def _backend(tensor):
    """Which implementation owns this tensor's device, or ``None``."""

    if tensor.device.type == "cuda":
        return cuda_backend()
    if tensor.device.type == "cpu":
        return extension()
    return None


def _cpu_float32(module, tensor) -> bool:
    return module is not None and tensor.dtype == __import__("torch").float32


# ---------------------------------------------------------------------------
# Dispatch. Every entry point has the same signature as its fallback.
# ---------------------------------------------------------------------------


def unpack_rows(packed, columns, *, dtype=None):
    import torch

    dtype = dtype or torch.float32
    # Only the CPU extension has a dedicated expansion; on CUDA the reference
    # path is a handful of vectorized ops and is not in any hot loop.
    if packed.device.type == "cpu" and extension() is not None and dtype == torch.float32:
        return extension().unpack_rows(packed, int(columns))
    return fallback.unpack_rows(packed, columns, dtype=dtype)


def pack_bits(mask):
    module = _backend(mask)
    if module is not None:
        return module.pack_bits(mask.contiguous())
    return fallback.pack_bits(mask)


def apply_bits(values, packed, columns):
    module = _backend(values)
    if _cpu_float32(module, values) or (module is not None and values.device.type == "cuda"):
        return module.apply_bits(values.contiguous(), packed, int(columns))
    return fallback.apply_bits(values, packed, columns)


def unpack_bits(packed, columns, *, dtype=None):
    import torch

    return fallback.unpack_bits(packed, columns, dtype=dtype or torch.float32)


def pack_affine_rows(values):
    module = _backend(values)
    if module is not None:
        return module.pack_affine_rows(values.contiguous())
    return fallback.pack_affine_rows(values)


def pack_coordinate(coordinate):
    module = _backend(coordinate)
    if module is not None:
        return module.pack_coordinate(coordinate.contiguous())
    return fallback.pack_coordinate(coordinate)


def packed_linear_affine(inputs, packed, scale, bias, columns):
    module = _backend(inputs)
    if _cpu_float32(module, inputs) or (module is not None and inputs.device.type == "cuda"):
        return module.packed_linear_affine(inputs, packed, scale, bias, int(columns))
    return fallback.packed_linear_affine(inputs, packed, scale, bias, columns)


def packed_transpose(inputs, packed, columns, row_scale=None):
    module = _backend(inputs)
    if _cpu_float32(module, inputs) or (module is not None and inputs.device.type == "cuda"):
        return module.packed_transpose(inputs, packed, int(columns), row_scale)
    return fallback.packed_transpose(inputs, packed, columns, row_scale)


def evidence_from_packed(grad, packed, columns, row_scale=None):
    module = _backend(grad)
    if _cpu_float32(module, grad) or (module is not None and grad.device.type == "cuda"):
        return module.evidence_from_packed(
            grad.contiguous(), packed, int(columns), row_scale
        )
    return fallback.evidence_from_packed(grad, packed, columns, row_scale)


def packed_row_inner(matrix, packed, columns):
    module = _backend(matrix)
    if _cpu_float32(module, matrix) or (module is not None and matrix.device.type == "cuda"):
        return module.packed_row_inner(matrix, packed, int(columns))
    return fallback.packed_row_inner(matrix, packed, columns)


def packed_embedding(ids, packed, scale, columns):
    module = _backend(scale)
    if _cpu_float32(module, scale) or (module is not None and scale.device.type == "cuda"):
        return module.packed_embedding(ids.long(), packed, scale, int(columns))
    return fallback.packed_embedding(ids, packed, scale, columns)


def coordinate_update(
    gradient,
    coordinate,
    packed,
    moment_q,
    moment_scale,
    row_v,
    col_v,
    *,
    beta1,
    beta2,
    update_clip,
    coordinate_lr,
    block_size,
    seed,
    step,
    flips=None,
    update_enabled=None,
):
    module = _backend(gradient)
    if module is not None and gradient.device.type == "cuda":
        return module.coordinate_update(
            gradient, coordinate, packed, moment_q, moment_scale, row_v, col_v,
            beta1=beta1, beta2=beta2, update_clip=update_clip,
            coordinate_lr=coordinate_lr, block_size=block_size, seed=seed,
            step=step, flips=flips, update_enabled=update_enabled,
        )
    if module is not None:
        return int(
            module.coordinate_update(
                gradient.contiguous(), coordinate, packed, moment_q, moment_scale,
                row_v, col_v, float(beta1), float(beta2), float(update_clip),
                float(coordinate_lr), int(block_size), int(seed), int(step),
            )
        )
    return fallback.coordinate_update(
        gradient, coordinate, packed, moment_q, moment_scale, row_v, col_v,
        beta1=beta1, beta2=beta2, update_clip=update_clip,
        coordinate_lr=coordinate_lr, block_size=block_size, seed=seed, step=step,
    )


def warn_if_slow(device_type: str = "cpu") -> None:
    """One-time notice that training is running on the reference kernels."""

    if _mode() != "auto":
        return
    if device_type == "cuda":
        if cuda_backend() is None:
            warnings.warn(
                "QSTE CUDA kernels are unavailable; falling back to the torch "
                f"reference path. Reason: {_cuda_failure}",
                RuntimeWarning,
                stacklevel=2,
            )
        return
    if extension() is None:
        warnings.warn(
            "QSTE native kernels are unavailable; using the pure-torch reference "
            f"path, which is correct but slow. Reason: {_failure}",
            RuntimeWarning,
            stacklevel=2,
        )
