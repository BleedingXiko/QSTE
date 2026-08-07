"""Front-loading the measurements, so nothing measures at a bad moment.

QSTE decides several things with a stopwatch rather than a formula: which of
the packed implementations wins for a given shape, what the evidence product's
dtype should be, how much of a weight a CPU should expand at once. Measuring is
the right call -- no formula predicts those correctly across devices -- but it
has to happen somewhere, and the default is "the first time that shape is
seen", which is the first training step.

:func:`warmup` moves it earlier. Call it once with a representative input and
every measurable decision is made and remembered before the run starts.

There are two reasons to bother.

The mild one is timing hygiene: without it the first step of a benchmark is
several times slower than the rest, and someone reads that number.

The one that actually matters is that a host may be recording its training step
into a CUDA graph -- QSTE never creates one, and never needs one, but it has to
be *recordable* inside somebody else's. A stopwatch cannot run inside a capture,
where nothing executes, so an undecided shape falls back to the expansion. That
is always correct and, at small batch, about twice as slow -- and it gets baked
into the graph permanently and silently. Deciding beforehand is the whole fix.
"""

from __future__ import annotations

import torch

from . import kernels
from .convert import surfaces


def _capturing() -> bool:
    return bool(getattr(torch.cuda, "is_current_stream_capturing", lambda: False)())


def warmup(model, *args, **kwargs):
    """Settle every measured decision now. Returns what was decided.

    Run it with the shapes the real workload uses, since the decisions are per
    shape. Cheap, idempotent, and safe to skip -- skipping costs a slow first
    step, and a slow captured one.
    """

    if _capturing():
        raise RuntimeError(
            "qste.warmup() must run before a capture, not inside it: it "
            "measures, and nothing can be measured while a graph is recording"
        )

    for surface in surfaces(model):
        kernels.warm(surface.coordinate.device)

    with torch.no_grad():
        model(*args, **kwargs)

    return decisions()


def decisions() -> dict:
    """Which implementation each measured shape settled on.

    Keyed by ``(device, rows, columns, samples, has_bias, dtype)``. Empty until
    something has run; a shape missing from here is one that would have to be
    decided blind if it came up inside a capture, and would be decided against.
    """

    backend = kernels.cuda_backend()
    if backend is None:
        return {}
    return dict(backend._FUSED_CHOICE)


def undecided(model, *args, **kwargs) -> bool:
    """Whether this call would still have to measure something.

    A cheap assertion for a host about to record a graph: if this is true, the
    graph is about to bake in the fallback for at least one layer.
    """

    before = len(decisions())
    with torch.no_grad():
        model(*args, **kwargs)
    return len(decisions()) > before


def retain(budget_bytes: int) -> None:
    """Keep expanded weights across calls, up to ``budget_bytes``. 0 = off.

    A packed weight has to be expanded before cuBLAS can multiply by it, and
    that expansion costs the same whether the batch is one row or a thousand.
    On a T4 at width 2048 it is 0.088 ms in fp32 and 0.059 ms in fp16, while
    the whole fp16 product at batch 32 is 0.050 ms -- so in the middle of the
    batch range the expansion alone outweighs the multiply it feeds, and the
    packed path cannot win there however well it is written.

    At inference nothing about the weight changes between calls, so every
    expansion after the first recomputes a value that was already correct. This
    turns that off: a weight already resident goes straight to cuBLAS and the
    call costs exactly what the float call costs, at any batch and any
    precision.

    It spends memory to buy speed, which is the opposite of the rest of this
    library, so it is off by default and the budget is yours to set. Storage is
    unaffected -- weights are still one bit per element and the packed tensor
    stays the source of truth; what is held is scratch, and dropping the budget
    to zero frees it immediately.

    Training is unaffected whether it is on or off. The cache keys on the
    version counter torch bumps when the packed bits are written, so an
    optimizer step invalidates every entry it touches, and gradients are not
    served from it at all.
    """

    backend = kernels.cuda_backend()
    if backend is not None:
        backend.retain(budget_bytes)


def retained_stats() -> dict:
    """Hits, misses, resident bytes and budget for :func:`retain`."""

    backend = kernels.cuda_backend()
    if backend is None:
        return {"hits": 0, "misses": 0, "resident_bytes": 0,
                "budget_bytes": 0, "entries": 0}
    return backend.retained_stats()


def invalidate(packed) -> None:
    """Drop anything expanded from these packed bits.

    Only needed if you write packed bits yourself, from your own kernel. QSTE's
    own optimizer already calls this, and torch-level writes are caught by the
    version counter -- but a kernel writing through a pointer bumps nothing,
    and a cache that does not hear about it serves the old weight forever.
    """

    backend = kernels.cuda_backend()
    if backend is not None:
        backend.invalidate(packed)
