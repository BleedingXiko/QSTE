"""Execution parameters derived from the device, once, at first use.

No kernel in QSTE branches on a device name, generation, or compute capability.
The constants that matter -- how many resident programs saturate the machine,
how large a scratch tile gets before it costs real memory, whether reduced
precision is actually faster -- vary by more than an order of magnitude across
the hardware people run on, so QSTE derives them from what the device reports
and from timing it.

Four values come out of this module:

``partitions``
    Independent accumulators a reduction splits into, from the reported
    multiprocessor count. Per-program partials plus a small final reduce beat
    an atomic on one address on every parallel machine; the right number of
    partials is enough to fill the device.

``scratch_bytes``
    Ceiling on one expanded operand, from cache size and free memory. A 4 GB
    card tiles and a 192 GB card does not, without either being named.

``fused_batch_limit``
    Largest batch that is a candidate for consuming packed bits in place
    instead of expanding them and calling BLAS. Timing the real shape settles
    which one wins.

``reduction_dtype``
    Dtype for the evidence product, measured rather than assumed. fp16 exists
    on a GTX 1080 and is no faster there, exists on a V100 and is much faster,
    and bf16 exists on newer parts and on ROCm where capability tuples say
    nothing. QSTE times the candidates and keeps the winner; fp32 stands when
    nothing wins.

A device that reports nothing useful gets conservative defaults and correct
results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# Escape hatch for a host where the probe misreads, or a run that must be
# reproducible down to the blocking. It overrides a measurement, never a
# device check.
_ENV_CPU_SCRATCH = "QSTE_CPU_EXPAND_BYTES"

_MIB = 1 << 20

# Minimum speedup for reduced precision to be worth its narrower range.
_SPEEDUP_THRESHOLD = 1.25

# fp16 carries fp32's mantissa poorly and its exponent range not at all, so the
# evidence gradient is rescaled onto a power of two before the cast. bf16 has
# fp32's exponent range and needs none of this.
_FP16_TARGET_MAGNITUDE = 1024.0


@dataclass(frozen=True)
class DeviceProfile:
    """Derived execution parameters for one device. Never device-specific."""

    kind: str
    name: str
    partitions: int
    scratch_bytes: int
    reduction_dtype: torch.dtype
    probe: str
    fused_batch_limit: int = 1

    @property
    def needs_rescale(self) -> bool:
        """Whether the reduction dtype's exponent range is narrower than fp32."""

        return self.reduction_dtype == torch.float16

    def tile_rows(self, rows: int, columns: int, dtype: torch.dtype) -> int:
        """Rows of an operand that fit the scratch ceiling. At least one."""

        per_row = max(1, columns * torch.empty((), dtype=dtype).element_size())
        return max(1, min(int(rows), self.scratch_bytes // per_row))

    def describe(self) -> str:
        return (
            f"{self.name} [{self.kind}]  partitions={self.partitions}  "
            f"scratch={self.scratch_bytes / _MIB:.0f}MiB  "
            f"reduction={str(self.reduction_dtype).replace('torch.', '')}  "
            f"fused<= {self.fused_batch_limit}  ({self.probe})"
        )


_CACHE: dict[str, DeviceProfile] = {}


def _override(name: str, default):
    """Environment escape hatch, for reproducing a run on other hardware."""

    value = os.environ.get(name)
    return default if value is None else value


def _time_matmul(dtype: torch.dtype, device: torch.device) -> float:
    """Milliseconds for one representative product in ``dtype``.

    Shapes are square-ish and large enough to leave the launch-overhead regime
    on a small device without allocating meaningfully on a large one.
    """

    size = 1024
    left = torch.randn(size, size, device=device, dtype=torch.float32).to(dtype)
    right = torch.randn(size, size, device=device, dtype=torch.float32).to(dtype)
    for _ in range(3):
        left @ right
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        left @ right
    stop.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(stop) / 10


def _measure_reduction_dtype(device: torch.device) -> tuple[torch.dtype, str]:
    """Which dtype the evidence product should run in, by measurement.

    The evidence is stochastically rounded into an int8 coordinate a few
    microseconds after it is produced, so its mantissa is not load bearing --
    but its *exponent range* is, and that is why bf16 is preferred over fp16
    at equal speed rather than the other way round.
    """

    forced = os.environ.get("QSTE_REDUCTION_DTYPE")
    if forced:
        table = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if forced not in table:
            raise ValueError("QSTE_REDUCTION_DTYPE must be fp32, fp16, or bf16")
        return table[forced], "forced by environment"

    try:
        baseline = _time_matmul(torch.float32, device)
    except Exception as error:  # a device that cannot time itself keeps fp32
        return torch.float32, f"probe unavailable ({type(error).__name__})"

    best_dtype, best_time, tried = torch.float32, baseline, []
    # bf16 first, so that a tie goes to the dtype that needs no rescaling.
    for dtype, label in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        try:
            elapsed = _time_matmul(dtype, device)
        except Exception:
            continue  # unsupported here; that is an answer, not an error
        tried.append(f"{label} {baseline / elapsed:.1f}x")
        if elapsed * _SPEEDUP_THRESHOLD < best_time:
            best_dtype, best_time = dtype, elapsed
    if not tried:
        return torch.float32, "fp32 only"
    return best_dtype, "measured " + ", ".join(tried)


def _cuda_profile(device: torch.device) -> DeviceProfile:
    properties = torch.cuda.get_device_properties(device)
    processors = int(getattr(properties, "multi_processor_count", 0) or 8)
    # Enough partials to fill the machine several times over, so the final
    # reduce stays small while no single accumulator is contended.
    partitions = max(64, processors * 16)

    cache = int(getattr(properties, "L2_cache_size", 0) or 0)
    total = int(getattr(properties, "total_memory", 0) or 0)
    try:
        free = int(torch.cuda.mem_get_info(device)[0])
    except Exception:
        free = total
    # Large enough that the expanded operand becomes one real GEMM instead of a
    # stack of skinny ones, small enough to stay a minor fraction of what the
    # model has left. Both bounds come from the device's own numbers.
    scratch = max(4 * cache, 8 * _MIB)
    scratch = min(scratch, max(free, total) // 64 or 8 * _MIB, 256 * _MIB)
    scratch = max(scratch, 2 * _MIB)

    capturing = getattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    if capturing():
        # Never allocate or synchronize during graph capture. fp32 is correct
        # everywhere; the profile is rebuilt outside capture.
        return DeviceProfile("cuda", properties.name, partitions, scratch,
                             torch.float32, "deferred during graph capture", 1)
    dtype, probe = _measure_reduction_dtype(device)
    # A ceiling on which batches are *candidates* for a packed path, not the
    # decision itself -- `qste.kernels.cuda` settles that by timing every
    # implementation on the real shape, since the answer varies by device and
    # by layer.
    #
    # The ceiling is generous on purpose. Expansion carries a fixed cost, since
    # it writes the dense weight out before the vendor GEMM reads it, making it
    # cheap once the product is large enough to bury that write and expensive
    # when it is not. A limit of 64 left everything between "small enough for
    # the lane kernel" and "large enough to bury an expansion" with expansion
    # as its only candidate, measuring 0.55x of float in the middle of that
    # range. Timing a few extra shapes once is the cheaper mistake.
    limit = int(os.environ.get("QSTE_FUSED_LIMIT", 2048))
    return DeviceProfile("cuda", properties.name, partitions, scratch, dtype,
                         probe, limit)


def _measure_expansion_budget(native) -> tuple[int, str]:
    """How much of a weight this host expands at once, decided with a stopwatch.

    This number is most of what the CPU forward costs. Too small and BLAS never
    sees a real GEMM, so call overhead swamps the product -- a 16 KiB budget
    measured at a sixth of the float path's speed. Too large and the expansion
    stops being transient. The turn depends on the cache hierarchy and on which
    BLAS the host torch was built against, neither knowable from here.

    Measured rather than derived from a reported cache size. Sizing the tile to
    fit in L2 is the natural argument and it measures wrong: every budget below
    a megabyte was worse, because the skinny-GEMM cost dominated the traffic it
    was meant to save.
    """

    import time

    # Wide enough that the budget actually changes the tiling. A 512-wide probe
    # has a one-megabyte weight, so every budget above a megabyte produces a
    # single tile and the "winner" is whichever ran when the machine was
    # quietest -- the first version of this probe picked the top of its own
    # sweep for exactly that reason.
    width = 2048
    packed = torch.randint(0, 256, (width, width // 8), dtype=torch.uint8)
    scale = torch.rand(width) + 0.5
    inputs = torch.randn(4, width)
    previous = int(native.scratch_bytes())

    def elapsed(budget: int) -> float:
        native.set_scratch_bytes(budget)
        for _ in range(2):
            native.packed_linear_affine(inputs, packed, scale, None, width)
        start = time.perf_counter()
        for _ in range(8):
            native.packed_linear_affine(inputs, packed, scale, None, width)
        return time.perf_counter() - start

    try:
        timings = {1 << bits: elapsed(1 << bits) for bits in range(18, 25)}
    except Exception:  # pragma: no cover - a host that cannot time keeps its default
        native.set_scratch_bytes(previous)
        return previous, "default (probe failed)"
    best = min(timings, key=timings.get)
    native.set_scratch_bytes(best)
    return best, f"expand {best >> 20}MiB" if best >= _MIB else f"expand {best >> 10}KiB"


def _cpu_profile() -> DeviceProfile:
    threads = max(1, torch.get_num_threads())
    # No fused small-batch path here, at any batch size. Consuming packed bits
    # in place beats expanding them when the product is bandwidth bound on the
    # weight -- true on a GPU, measured false on a CPU, where the host sgemv is
    # already vectorized and near the bandwidth bound and a portable scalar
    # loop ran at 0.12x to 0.35x of it. Hence a ceiling of zero.
    budget, probe = 8 * _MIB, "fp32 only"
    if _ENV_CPU_SCRATCH in os.environ:
        budget = int(os.environ[_ENV_CPU_SCRATCH])
        probe = f"expand {budget >> 10}KiB (env)"
    try:
        from . import loader

        native = loader.extension()
        if native is not None and hasattr(native, "set_scratch_bytes"):
            if _ENV_CPU_SCRATCH in os.environ:
                native.set_scratch_bytes(budget)
            else:
                budget, probe = _measure_expansion_budget(native)
            probe = f"fp32 only, {probe}"
    except Exception:  # pragma: no cover - the reference path needs no probe
        pass
    return DeviceProfile("cpu", "cpu", threads, budget, torch.float32, probe, 0)


def profile(device) -> DeviceProfile:
    """Derived parameters for ``device``, computed once and cached."""

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = f"{device.type}:{device.index}"
    cached = _CACHE.get(key)
    if cached is not None and cached.probe != "deferred during graph capture":
        return cached
    if device.type == "cuda":
        built = _cuda_profile(device)
    elif device.type == "cpu":
        built = _cpu_profile()
    else:
        # MPS, XPU, or something that does not exist yet. The reference path is
        # correct on all of them; nothing here needs to know which it is.
        built = DeviceProfile(device.type, device.type, 64, 8 * _MIB,
                              torch.float32, "portable defaults", 1)
    _CACHE[key] = built
    return built


def warm(device) -> DeviceProfile:
    """Build the profile now, so no probe happens inside a captured region."""

    return profile(device)


def reset() -> None:
    """Forget every cached profile. Tests use this; nothing else should."""

    _CACHE.clear()


def rescale_factor(values: torch.Tensor) -> torch.Tensor:
    """A power of two that lands ``values``' peak inside fp16's range.

    Exact: a power-of-two multiply and its reciprocal are lossless in binary
    floating point, so scaling in and out changes nothing but the exponent the
    reduced-precision multiply sees. Stays on device -- no host synchronization,
    so this is safe inside a captured graph.
    """

    peak = values.detach().abs().amax().clamp_min(torch.finfo(values.dtype).tiny)
    return torch.exp2(torch.floor(torch.log2(_FP16_TARGET_MAGNITUDE / peak)))
