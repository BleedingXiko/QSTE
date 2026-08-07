"""Data-parallel and sharded training.

Read this before running QSTE on more than one GPU. There is exactly one way
to get it wrong and it fails silently.

**Coordinate evidence is not an autograd gradient.** It never lands in
``parameter.grad``, because the coordinate is ``requires_grad=False`` -- that
is deliberate, it is what keeps the host's optimizer from touching it. The
consequence is that DDP's gradient all-reduce does not see it. Without an
explicit reduction each rank steps its coordinates from its own local batch,
the ranks' signs drift apart within a few steps, and nothing raises: the loss
still goes down, on a model that is now different on every GPU.

So::

    qste.convert(model, include=[...])
    model = DistributedDataParallel(model, device_ids=[rank])
    coordinates = qste.QSTEOptimizer(
        model, gradient_reducer=qste.distributed.mean_evidence()
    )

``mean_evidence`` averages evidence across ranks, matching what DDP does to
gradients. Because stochastic rounding is seeded by ``(seed, step, index)``
and not by any RNG state, identical evidence gives byte-identical coordinates
on every rank -- no broadcast needed after the first one.

FSDP is different: see :func:`fsdp_ignored_states`.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from .convert import surfaces as _surfaces
from .surface import Surface


def _distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def mean_evidence(group=None) -> Callable[[Tensor], Tensor]:
    """Reducer that averages coordinate evidence across ranks.

    The DDP-equivalent choice: DDP averages gradients, so averaging evidence
    keeps a QSTE run's effective step size independent of world size.
    """

    def reduce(evidence: Tensor) -> Tensor:
        if not _distributed_ready():
            return evidence
        torch.distributed.all_reduce(
            evidence, op=torch.distributed.ReduceOp.SUM, group=group
        )
        return evidence.div_(torch.distributed.get_world_size(group))

    return reduce


def sum_evidence(group=None) -> Callable[[Tensor], Tensor]:
    """Reducer that sums evidence, for hosts that sum gradients instead."""

    def reduce(evidence: Tensor) -> Tensor:
        if not _distributed_ready():
            return evidence
        torch.distributed.all_reduce(
            evidence, op=torch.distributed.ReduceOp.SUM, group=group
        )
        return evidence

    return reduce


@torch.no_grad()
def broadcast_surfaces(model: nn.Module, src: int = 0, group=None) -> int:
    """Make every rank's binary state match ``src``.

    Call once after ``convert`` and before the first step. DDP's own
    ``_sync_module_states`` already broadcasts parameters, so this is only
    needed when surfaces are hidden from DDP (FSDP ignored states, or a
    hand-rolled parallel wrapper) or when resuming a checkpoint on one rank.
    """

    if not _distributed_ready():
        return 0
    count = 0
    for surface in _surfaces(model):
        for tensor in (surface.coordinate.data, surface.packed_sign.data, surface.log_scale.data):
            torch.distributed.broadcast(tensor, src=src, group=group)
            count += 1
    return count


@torch.no_grad()
def surfaces_agree(model: nn.Module, group=None) -> bool:
    """True when every rank holds the same signs. Use it in a debug assert.

    Drifted coordinates are the failure mode this module exists to prevent,
    and they are invisible from the loss curve. This makes them visible.
    """

    if not _distributed_ready():
        return True
    digest = torch.zeros((), dtype=torch.float64, device=_device(model))
    for surface in _surfaces(model):
        digest += surface.packed_sign.data.to(torch.float64).sum()
        digest += surface.coordinate.data.to(torch.float64).sum()
    gathered = digest.clone()
    torch.distributed.all_reduce(gathered, op=torch.distributed.ReduceOp.MIN, group=group)
    highest = digest.clone()
    torch.distributed.all_reduce(highest, op=torch.distributed.ReduceOp.MAX, group=group)
    return bool(torch.equal(gathered, highest))


def _device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cpu")


def fsdp_ignored_states(model: nn.Module) -> list[nn.Parameter]:
    """Parameters FSDP should not flatten, for ``ignored_states=``.

    FSDP flattens parameters into per-dtype shards and assumes they receive
    autograd gradients. A surface satisfies neither: its coordinate is INT8,
    its signs are UINT8, and both are stepped outside autograd. Handing them
    to FSDP produces either a dtype-mixing error or, worse, a shard whose
    optimizer never runs.

    So keep them out, and keep them in sync with an evidence reducer::

        ignored = qste.distributed.fsdp_ignored_states(model)
        model = FullyShardedDataParallel(model, ignored_states=ignored)
        coordinates = qste.QSTEOptimizer(
            model, gradient_reducer=qste.distributed.mean_evidence()
        )

    Surfaces are then replicated rather than sharded. That costs one INT8
    coordinate per rank -- one byte per weight, against the four the float
    model would have sharded -- and everything else in the model shards
    normally.
    """

    ignored: list[nn.Parameter] = []
    for surface in _surfaces(model):
        ignored.extend([surface.coordinate, surface.packed_sign, surface.log_scale])
    return ignored


def fsdp_ignored_modules(model: nn.Module) -> list[nn.Module]:
    """The :class:`~qste.surface.Surface` modules themselves.

    Some FSDP versions take ``ignored_modules`` rather than ``ignored_states``.
    Note this also excludes ``log_scale`` from sharding, which is one float per
    output row and not worth sharding anyway.
    """

    return list(_surfaces(model))


def replicated_bytes(model: nn.Module) -> int:
    """How much per-rank memory the ignored surfaces cost under FSDP."""

    total = 0
    for surface in _surfaces(model):
        stats = surface.memory()
        total += stats["packed_sign"] + stats["coordinate"] + stats["log_scale"]
    return total


__all__ = [
    "broadcast_surfaces",
    "fsdp_ignored_modules",
    "fsdp_ignored_states",
    "mean_evidence",
    "replicated_bytes",
    "sum_evidence",
    "surfaces_agree",
]
