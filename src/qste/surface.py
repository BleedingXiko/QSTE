"""The binary matrix and its INT8 training coordinate.

A :class:`Surface` owns three tensors for one matrix:

    coordinate    int8  [rows, columns]   the training state
    packed_sign   uint8 [rows, ceil(c/8)] the forward weight, one bit each
    log_scale     fp32  [rows]            a learned positive row scale

The forward pass reads ``packed_sign`` only. The coordinate exists so a
gradient has somewhere continuous to accumulate before it is stochastically
rounded back onto the sign lattice; it is never multiplied by anything.

That is what makes training precision equal to inference precision: there is
no float shadow weight to quantize away at the end. The exported model is the
bits that trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from . import kernels
from .config import QSTEConfig, coerce_config

_ACCUM_DTYPE = {"fp16": torch.float16, "fp32": torch.float32}


class Surface(nn.Module):
    """One shareable binary matrix. Tied weights share one instance."""

    def __init__(self, weight: Tensor, cfg: QSTEConfig | None = None):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("a QSTE surface must wrap a rank-2 weight")
        cfg = coerce_config(cfg)
        source = weight.detach()
        rows, columns = (int(size) for size in source.shape)

        row_scale = torch.empty(rows, device=source.device, dtype=torch.float32)
        coordinate = torch.empty(rows, columns, device=source.device, dtype=torch.int8)
        # Chunked so initializing a vocabulary-sized matrix does not allocate a
        # second float copy of it.
        chunk = max(1, (32 << 20) // max(1, columns * 12))
        for start in range(0, rows, chunk):
            stop = min(start + chunk, rows)
            value = source[start:stop].float()
            scale = value.abs().mean(dim=1).clamp_min(1e-5)
            latent = (value / scale[:, None]).clamp(-1.5, 1.5) * 0.25
            quantized = (
                (latent / cfg.coordinate_clip * 127).round().clamp(-127, 127).to(torch.int8)
            )
            # A zero coordinate would decode to +1 regardless of the source
            # weight's sign. Push it one step in the direction it came from.
            quantized = torch.where(
                quantized == 0,
                torch.where(value >= 0, torch.ones_like(quantized), -torch.ones_like(quantized)),
                quantized,
            )
            row_scale[start:stop] = scale
            coordinate[start:stop] = quantized

        self.cfg = cfg
        self.rows = rows
        self.columns = columns
        # Parameters, not buffers, so FSDP and DDP own and shard the coordinate
        # normally. requires_grad=False keeps the QSTE optimizer its only
        # mutator.
        self.coordinate = nn.Parameter(coordinate, requires_grad=False)
        self.packed_sign = nn.Parameter(
            kernels.pack_coordinate(coordinate), requires_grad=False
        )
        self.log_scale = nn.Parameter(row_scale.log())

        self._evidence: Tensor | None = None
        self._call_evidence: Tensor | None = None
        self._pending_calls = 0
        self._immediate = None  # set by QSTEOptimizer

    # -- forward-side state ------------------------------------------------

    @property
    def scale(self) -> Tensor:
        return self.log_scale.exp().clamp(self.cfg.scale_min, self.cfg.scale_max)

    @property
    def sign(self) -> Tensor:
        """Dense +-1 view of the packed matrix. Allocates rows x columns."""

        return kernels.unpack_rows(self.packed_sign.data, self.columns)

    @torch.no_grad()
    def dense_weight(self) -> Tensor:
        """The float matrix this surface is equivalent to. No autograd."""

        return self.sign.mul_(self.scale.detach().unsqueeze(1))

    def extra_repr(self) -> str:
        return f"rows={self.rows}, columns={self.columns}"

    # -- evidence bookkeeping ---------------------------------------------

    def note_forward(self) -> None:
        """Record that a backward for this surface is coming."""

        if torch.is_grad_enabled():
            self._pending_calls += 1

    @torch.no_grad()
    def reset_tracking(self) -> None:
        self._call_evidence = None
        self._pending_calls = 0

    @torch.no_grad()
    def consume_evidence(self, evidence: Tensor) -> None:
        """Accept one use's worth of coordinate evidence.

        A surface used once per backward -- the common case -- hands its
        evidence straight to the optimizer and never allocates a buffer. A
        tied surface accumulates across its uses first, because autograd
        semantics require the sum of contributions, not any one of them.
        """

        self._pending_calls = max(0, self._pending_calls - 1)
        immediate = self._immediate

        if (
            self._pending_calls == 0
            and self._call_evidence is None
            and self._evidence is None
            and immediate is not None
        ):
            immediate(self, evidence.detach().float())
            return

        if (
            self._pending_calls == 0
            and self._call_evidence is None
            and self._evidence is not None
            and immediate is None
        ):
            # A preallocated accumulator already exists (gradient accumulation
            # or graph capture): add in place at a fixed address.
            self._evidence.add_(evidence.detach().to(self._evidence.dtype))
            return

        accum = _ACCUM_DTYPE[self.cfg.evidence_accum_dtype]
        compact = evidence.detach().to(accum)
        if self._call_evidence is None:
            self._call_evidence = compact.clone()
        else:
            self._call_evidence.add_(compact)
        if self._pending_calls:
            return

        total = self._call_evidence.float()
        self._call_evidence = None
        if immediate is not None:
            if self._evidence is not None:
                total.add_(self._evidence.float())
                self._evidence = None
            immediate(self, total)
        elif self._evidence is None:
            self._evidence = total.to(accum)
        else:
            self._evidence.add_(total.to(accum))

    @torch.no_grad()
    def flush_evidence(self, *, keep_buffer: bool = False) -> Tensor | None:
        """Everything accumulated, whether or not the forward count balanced.

        The pending-call count is an optimization: at zero, a single-use
        surface applies its evidence during backward without allocating a
        buffer. Correctness cannot depend on it, because a host may run forward
        more times than backward -- gradient checkpointing's discarded first
        pass increments the count and never produces a gradient. The optimizer
        calls this every step as the backstop.
        """

        total = None
        if self._call_evidence is not None:
            total = self._call_evidence.float()
            self._call_evidence = None
        if self._evidence is not None:
            contribution = self._evidence.float()
            total = contribution if total is None else total.add_(contribution)
            if keep_buffer:
                self._evidence.zero_()
            else:
                self._evidence = None
        self._pending_calls = 0
        return total

    @torch.no_grad()
    def prepare_accumulator(self) -> None:
        """Preallocate a fixed evidence destination for accumulation."""

        self.reset_tracking()
        self._evidence = torch.zeros(
            self.rows,
            self.columns,
            device=self.coordinate.device,
            dtype=_ACCUM_DTYPE[self.cfg.evidence_accum_dtype],
        )

    @torch.no_grad()
    def scale_evidence_(self, divisor: float) -> None:
        """Unscale retained evidence alongside an AMP gradient unscale."""

        if divisor == 0:
            raise ValueError("gradient scale must be nonzero")
        for buffer in (self._evidence, self._call_evidence):
            if buffer is not None:
                buffer.div_(divisor)

    @torch.no_grad()
    def refresh_packed(self) -> None:
        """Rebuild packed signs after an out-of-band coordinate write."""

        self.packed_sign.data.copy_(kernels.pack_coordinate(self.coordinate.data))

    # -- introspection -----------------------------------------------------

    def memory(self) -> dict[str, int]:
        """Bytes this surface costs, split by what the bytes are for."""

        return {
            "packed_sign": self.packed_sign.numel(),
            "coordinate": self.coordinate.numel(),
            "log_scale": self.log_scale.numel() * 4,
            "float_equivalent": self.rows * self.columns * 4,
        }
