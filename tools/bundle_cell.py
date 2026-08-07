"""Emit one paste-ready cell containing the whole framework, its tests, and its bench.

    python tools/bundle_cell.py > qste_cell.py

The output is a single Python file with every source file embedded. Running it
writes the package to a temp directory, builds the native kernels for whatever
hardware is attached, runs the entire test suite against them, and then measures
speed and memory against the float baseline. No pip install, no upload, no
second cell, no repo.

This exists because the interesting hardware is usually a hosted notebook with
no filesystem you control and no way to get a repo onto it -- and because a
speed table from an unvalidated build is not evidence of anything.
"""

from __future__ import annotations

import base64
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PACKAGE = sorted(
    str(path.relative_to(ROOT))
    for path in (ROOT / "src/qste").rglob("*")
    if path.suffix in {".py", ".cpp"} and "__pycache__" not in path.parts
)
EXTRA = ["bench/bench.py", "pytest.ini"] + [
    f"tests/{path.name}" for path in sorted((ROOT / "tests").glob("*.py"))
]

HEADER = '''\
# ---------------------------------------------------------------------------
# QSTE, self-contained. Paste and run.
#
# True-binary training: training precision is inference precision, with no
# float shadow weight and no post-training quantization step.
#
# This cell unpacks the package to a temp directory, compiles the native
# kernels for whatever hardware is attached, runs the whole test suite against
# them, and then measures speed and memory against the float baseline. Nothing
# is installed and nothing is uploaded.
#
# Nothing in it is tuned for a particular device. Scratch size, reduction
# width, reduction dtype, and the small-batch crossover are all derived at
# startup from what the hardware reports and from timing it, so the numbers
# below describe this machine and the same code describes any other.
#
# The payload is the source, deflated and base64'd so the docstrings inside it
# survive the trip through one string.
# ---------------------------------------------------------------------------

import base64, hashlib, json, os, subprocess, sys, tempfile, time, zlib
from pathlib import Path

_QSTE_PAYLOAD = (
{payload}
)
'''

FOOTER = r'''

# --- materialize ------------------------------------------------------------

_sources = json.loads(zlib.decompress(base64.b64decode(_QSTE_PAYLOAD)))
_root = Path(tempfile.mkdtemp(prefix="qste-"))
for _name, _body in _sources.items():
    _path = _root / _name
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(_body)
(_root / "src/qste/kernels/__pycache__").mkdir(parents=True, exist_ok=True)

# Re-running this cell in a live notebook kernel would otherwise import
# nothing: `qste` is already in sys.modules from the previous run, so the
# freshly written source is ignored and the old code is measured again.
for _stale in [_m for _m in sys.modules if _m == "qste" or _m.startswith("qste.")]:
    del sys.modules[_stale]
sys.path = [str(_root / "src")] + [_p for _p in sys.path if "qste-" not in _p]

_BUILD = hashlib.sha256(_QSTE_PAYLOAD.encode()).hexdigest()[:10]
print(f"QSTE build {_BUILD}")

import torch
import qste
from qste import kernels

# Prove which implementation is loaded, so a stale import cannot masquerade as
# a slow one, and that the two mistakes already made and fixed have not come
# back: a hand-written GEMM, and a reduction contending on one address.
_cuda_source = (_root / "src/qste/kernels/cuda.py").read_text()
assert qste.__file__.startswith(str(_root)), f"stale qste at {qste.__file__}"
# Exactly one hand-written multiply, in the kernel that consumes packed bits
# directly -- cuBLAS has no entry point for a one-bit operand, so there it is
# the only option rather than a competitor. Everywhere cuBLAS *can* take the
# operand it still gets it; a second one appearing here means that rule slipped.
assert _cuda_source.count("tl.dot") == 1, "stale build: hand-written GEMMs are back"
assert not [
    _line for _line in _cuda_source.splitlines()
    if "atomic_add" in _line and not _line.lstrip().startswith("#")
], "stale build: single-address atomics are back"
print("loaded from", qste.__file__)

print(f"torch  {torch.__version__}")
if torch.cuda.is_available():
    _major, _minor = torch.cuda.get_device_capability()
    print(f"gpu    {torch.cuda.get_device_name(0)}  sm_{_major}{_minor}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("gpu    none -- this cell measures CPU only")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_status = kernels.status()
print(f"kernels native_cpu={_status['native']}  gpu={_status['cuda']}")
if not _status["native"]:
    print(f"        native CPU build unavailable: {_status['failure']}")
if DEVICE == "cuda" and not _status["cuda"]:
    print(f"        GPU kernels unavailable: {_status['cuda_failure']}")
print(f"derived {qste.device_profile(DEVICE)}")


# --- 0. does every kernel compile at all ------------------------------------
#
# This runs first and it exists because of how the previous two runs were
# spent. A Triton kernel is only type-checked when it is compiled, and it is
# only compiled when something calls it -- so a frontend error in one kernel
# surfaces as whichever test happens to reach it first, ten identical
# tracebacks, and a benchmark that dies before printing anything. One run, one
# bit of information, and none of it about the other sixteen kernels.
#
# So: compile every kernel up front, on a tiny input, catching each one
# separately. A run now reports the compile status of the whole file at once
# and then carries on to the tests and the benchmark regardless, which turns a
# single broken line into a diagnosis instead of a wasted run.
def _compile_check():
    print(f"\n{'=' * 78}\nKERNEL COMPILE CHECK\n{'=' * 78}", flush=True)

    import torch as _torch

    # Runs on a CPU too, against the CPU backend, and that is deliberate: it
    # means this block is exercised on every machine rather than only on the
    # one it was written for. A check whose own first execution is on the
    # hardware it is meant to diagnose is not a check.
    _cuda = kernels.cuda_backend() if DEVICE == "cuda" else None

    # A GPU present and no GPU backend is a total failure that used to read as
    # a pass. When importing the kernel module raises, the loader catches it and
    # every call silently reverts to the torch reference: the compile check then
    # has fewer paths to run and cheerfully reports all of them ok, the suite
    # reports errors nobody reads past, and the benchmark prints a full table of
    # numbers -- 0.01x inference, a train step three times slower -- ending in
    # the word LEARNED. That is the most dangerous output this cell can produce,
    # so it is now the first thing checked and it is fatal.
    _backend_failure = None
    if DEVICE == "cuda" and _cuda is None:
        _backend_failure = kernels.status().get("cuda_failure") or "unknown"
        print(f"  FATAL  a GPU is present but the CUDA backend did not load")
        print(f"         {_backend_failure}")
        print( "         every number below this line would be the CPU "
               "reference wearing a GPU label")
    _w, _n = 64, 8
    _packed = _torch.randint(0, 256, (_w, _w // 8), dtype=_torch.uint8, device=DEVICE)
    _scale = _torch.rand(_w, device=DEVICE) + 0.5
    _x = _torch.randn(_n, _w, device=DEVICE)
    _coord = _torch.randint(-100, 100, (_w, _w), dtype=_torch.int8, device=DEVICE)
    _cpacked = kernels.pack_coordinate(_coord)
    _blocks = (_w * _w + 255) // 256

    # Each entry names a kernel and exercises the launcher that compiles it.
    # Every kernel in the file has to appear here, which the check below
    # enforces -- an unexercised kernel is one whose first compile happens on
    # someone else's hardware.
    _paths = {
        "_expand_flat + _expand_tiled / _expanded_linear":
            lambda: kernels.packed_linear_affine(_x, _packed, _scale, None, _w),
        "_pack_bit_rows / _apply_bit_mask": lambda: kernels.apply_bits(
            _x, kernels.pack_bits(_x > 0), _w),
        "_pack_affine_rows": lambda: kernels.pack_affine_rows(_x),
        "_pack_coordinate_rows": lambda: kernels.pack_coordinate(_coord),
        "_packed_row_inner": lambda: kernels.packed_row_inner(
            _torch.randn(_w, _w, device=DEVICE), _packed, _w),
        "_packed_embedding": lambda: kernels.packed_embedding(
            _torch.randint(0, _w, (_n,), device=DEVICE), _packed, _scale, _w),
        "the expansion (transpose path)": lambda: kernels.packed_transpose(
            _torch.randn(_n, _w, device=DEVICE), _packed, _w, _scale),
        "evidence_from_packed": lambda: kernels.evidence_from_packed(
            _torch.randn(_n, _w, device=DEVICE), kernels.pack_bits(_x > 0), _w),
        # One launcher, seven kernels: _row_col_squares, _finish_factors,
        # _precondition, _moment_and_scale, _coordinate_and_pack, and the two
        # helpers inlined into it, _low32 and _round_nearest_even (which
        # _hash32 also goes through).
        "the fused optimizer step": lambda: kernels.coordinate_update(
            _torch.randn(_w, _w, device=DEVICE), _coord, _cpacked,
            _torch.zeros_like(_coord),
            _torch.full((_blocks,), 1 / 127, dtype=_torch.float16, device=DEVICE),
            _torch.zeros(_w, dtype=_torch.float16, device=DEVICE),
            _torch.zeros(_w, dtype=_torch.float16, device=DEVICE),
            beta1=0.9, beta2=0.99, update_clip=2.0, coordinate_lr=1.0,
            block_size=256, seed=1, step=0),
    }
    if _cuda is not None:
        # The fused small-batch kernels are only reachable through the GPU
        # backend, and only when the stopwatch picks them -- so they are called
        # directly here rather than left to a measurement that might route
        # around a kernel that does not compile.
        _paths["_packed_small_batch"] = lambda: _cuda._run_small_batch(
            _x[:1], _packed, _scale, None, _w, False)
        _paths["_packed_small_batch (split) / _small_batch_epilogue"] = lambda: (
            _cuda._run_small_batch(_x[:1], _packed, _scale, None, _w, True))
        # Loses on some parts and is kept anyway -- it is timed per device, and
        # a card with different arithmetic is entitled to a different answer.
        # Compiled here so that "it lost" and "it never ran" stay distinct.
        _paths["_packed_tiled"] = lambda: _cuda._run_tiled(
            _torch.randn(32, _w, device=DEVICE), _packed, _scale, None, _w)

    import warnings as _warnings

    _failed = 1 if _backend_failure is not None else 0
    for _name, _call in _paths.items():
        try:
            # Compiler warnings are failures here. Triton reports undefined
            # behaviour -- shifting an int32 by 32, say -- as a warning and
            # then compiles it anyway, so the kernel runs, and trains, and
            # quietly does not learn. That happened. A warning printed next to
            # the word "ok" is worse than no check at all.
            with _warnings.catch_warnings(record=True) as _caught:
                _warnings.simplefilter("always")
                _result = _call()
                if DEVICE == "cuda":
                    _torch.cuda.synchronize()
            # Only warnings that could mean the kernel computes the wrong
            # thing. Triton reports undefined behaviour this way -- shifting an
            # int32 by 32, which is what silently broke the optimizer -- but it
            # also drags in library deprecation notices from its own caching
            # code, and treating those as failures makes the check cry wolf
            # until nobody reads it.
            _messages = sorted({
                str(_w.message) for _w in _caught
                if not issubclass(_w.category, (
                    DeprecationWarning, PendingDeprecationWarning,
                    FutureWarning, ImportWarning, ResourceWarning,
                ))
            })
            if _messages:
                _failed += 1
                print(f"  WARN  {_name}")
                for _message in _messages[:3]:
                    print(f"        {_message.strip().splitlines()[0][:180]}")
                continue
            if _result is None:
                _failed += 1
                _reason = getattr(_cuda, "_FUSED_ERROR", None)
                print(f"  FAIL  {_name}   (declined)")
                if _reason is not None:
                    print(f"        {type(_reason).__name__}: "
                          f"{str(_reason).strip().splitlines()[-1][:200]}")
            else:
                print(f"  ok    {_name}")
        except Exception as _error:
            _failed += 1
            _last = str(_error).strip().splitlines()
            print(f"  FAIL  {_name}")
            print(f"        {type(_error).__name__}: "
                  f"{(_last[-1] if _last else '')[:200]}")

    print(f"\n  {len(_paths) - _failed}/{len(_paths)} kernel paths compiled clean and ran")
    return _failed


def _run(label, arguments, timeout=3600):
    """Run a child process against the unpacked tree and stream its output."""

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_root / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=str(_root),
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = (result.stdout or "") + (result.stderr or "")
    print(output.strip())
    print(f"\n[{label}: exit {result.returncode} in {time.perf_counter() - started:.0f}s]",
          flush=True)
    return result.returncode


# --- 1. the whole test suite, on this hardware ------------------------------
#
# 300-odd tests. Kernel parity against the pure-torch reference on every shape
# including the awkward ones, autograd against torch, the optimizer's GPU path
# against its CPU path bit for bit, graph capture, checkpointing, distributed
# reduction, packed activations against torch's own gradients, and peak memory
# from the allocator rather than from an argument.

_broken = _compile_check()

try:
    import pytest  # noqa: F401
    # --no-header and short tracebacks: a frontend error repeats the same
    # hundred-line traceback once per test that reaches it, which buries every
    # other failure in the run. The compile check above is where the full text
    # of that error lives now.
    _tests = _run("TESTS", ["-m", "pytest", "tests", "-q", "--no-header",
                            "-p", "no:cacheprovider", "-rf", "--tb=line",
                            "--maxfail=60"])
except ImportError:
    print("\npytest is not installed here; skipping the suite")
    _tests = None


# --- 2. speed and memory against the float baseline -------------------------

_bench = _run("BENCHMARK", ["bench/bench.py", "--device", DEVICE,
                            "--width", "2048", "--batch", "4096", "--depth", "6"])


# --- verdict ----------------------------------------------------------------

print(f"\n{'=' * 78}")
print(f"QSTE build {_BUILD} on {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'cpu'}")
print(f"  kernels    {'all compile clean' if _broken == 0 else f'{_broken} FAILED OR WARNED'}")
print(f"  tests      {'PASS' if _tests == 0 else ('SKIPPED' if _tests is None else 'FAIL')}")
print(f"  benchmark  {'ok' if _bench == 0 else 'error'}")
print("=" * 78)
'''


def _every_kernel_is_compile_checked(cuda_source: str) -> None:
    """Refuse to bundle a cell that would leave a kernel uncompiled.

    The compile check only helps if it covers everything. A kernel nobody
    exercises there gets its first real compile on someone else's hardware,
    which is the situation the check exists to end.
    """

    import ast

    kernels = {
        node.name
        for node in ast.walk(ast.parse(cuda_source))
        if isinstance(node, ast.FunctionDef)
        and any("jit" in ast.dump(decorator) for decorator in node.decorator_list)
    }
    missing = sorted(name for name in kernels if name not in FOOTER)
    if missing:
        raise SystemExit(
            "these kernels are not named in the cell's compile check, so nothing "
            f"would compile them before a test happens to: {missing}"
        )


def main() -> None:
    sources = {}
    for relative in PACKAGE + EXTRA:
        sources[relative] = (ROOT / relative).read_text()
    _every_kernel_is_compile_checked(sources["src/qste/kernels/cuda.py"])
    blob = base64.b64encode(zlib.compress(json.dumps(sources).encode(), 9)).decode()
    chunks = [blob[i : i + 76] for i in range(0, len(blob), 76)]
    payload = "\n".join(f'    "{chunk}"' for chunk in chunks)
    sys.stdout.write(HEADER.format(payload=payload) + FOOTER)


if __name__ == "__main__":
    main()
