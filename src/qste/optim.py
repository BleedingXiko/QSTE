"""The coordinate optimizer.

This owns the INT8 coordinates and nothing else. Row scales and biases are
ordinary float parameters with ``requires_grad=True``; the host's existing
AdamW (or Muon, or whatever it already uses) picks them up from
``model.parameters()`` and steps them normally. That split is deliberate --
adopting QSTE should not mean giving up your optimizer.

Optimizer state per matrix, for a rows x columns surface:

    moment_q      int8   rows x columns    first moment, block-quantized
    moment_scale  fp16   ceil(n / block)   one scale per block
    row_v         fp16   rows              factored second moment
    col_v         fp16   columns           factored second moment

That is 1 byte per weight, against 8 for AdamW's two fp32 moments.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from . import kernels
from .config import QSTEConfig, coerce_config
from .convert import surfaces as _model_surfaces
from .surface import Surface


class QSTEOptimizer:
    """Steps INT8 coordinates from the evidence backward accumulated.

    Args:
        model: A converted model, or an iterable of surfaces.
        config: Numerics. Defaults to the config recorded by ``convert``.
        gradient_reducer: Called on each surface's evidence before it is
            applied. Receives and returns a float32 ``rows x columns`` tensor.

            Leave it as ``None``. When a process group is initialized this
            defaults to :func:`qste.distributed.mean_evidence`, because
            coordinate evidence is not an autograd gradient -- it never lands
            in ``parameter.grad``, so DDP's reducer cannot see it, and without
            an all-reduce every rank steps its coordinates from its own local
            batch and they drift apart while the loss keeps falling. That
            failure is invisible, so it is not left to the caller to remember.

            Pass ``False`` to opt out and reduce evidence yourself. Passing a
            callable uses it as-is.
    """

    def __init__(
        self,
        model: nn.Module | Iterable[Surface],
        config: QSTEConfig | None = None,
        *,
        gradient_reducer: Callable[[Tensor], Tensor] | None | bool = None,
    ):
        if isinstance(model, nn.Module):
            self.surfaces = _model_surfaces(model)
            config = config or getattr(model, "_qste_config", None)
        else:
            self.surfaces = list(model)
        if not self.surfaces:
            raise ValueError(
                "no QSTE surfaces found; call qste.convert(model, ...) first"
            )
        self.cfg = coerce_config(config)
        self.gradient_reducer = self._resolve_reducer(gradient_reducer)
        self.step_number = 0
        self._flips = 0
        self._deferred = False
        self.state: dict[int, dict[str, Tensor]] = {}
        self._index: dict[int, int] = {}

        block = self.cfg.momentum_block_size
        for position, surface in enumerate(self.surfaces):
            self._index[id(surface)] = position
            device = surface.coordinate.device
            blocks = (surface.coordinate.numel() + block - 1) // block
            self.state[id(surface)] = {
                "moment_q": torch.zeros_like(surface.coordinate, dtype=torch.int8),
                "moment_scale": torch.full(
                    (blocks,), 1 / 127, dtype=torch.float16, device=device
                ),
                "row_v": torch.zeros(surface.rows, dtype=torch.float16, device=device),
                "col_v": torch.zeros(surface.columns, dtype=torch.float16, device=device),
            }
            surface._immediate = self._apply

        # Derive the device's execution profile now. It times the hardware
        # once, and the two places that must never do that are inside a
        # captured graph and inside somebody's benchmark loop.
        for surface in self.surfaces:
            kernels.warm(surface.coordinate.device)
            kernels.warn_if_slow(surface.coordinate.device.type)

    @staticmethod
    def _resolve_reducer(requested):
        """Default to all-reducing evidence whenever ranks exist.

        The safe choice is the default because the unsafe one is silent: the
        loss curve of a run whose ranks have diverged looks exactly like the
        loss curve of a healthy one.
        """

        if requested is False:
            return None
        if requested is not None and requested is not True:
            return requested
        from .distributed import _distributed_ready, mean_evidence

        return mean_evidence() if _distributed_ready() else None

    # -- accumulation control ---------------------------------------------

    def defer(self, deferred: bool = True) -> None:
        """Hold evidence until :meth:`step` instead of applying in backward.

        Needed for gradient accumulation, AMP unscaling, and graph capture.
        Costs one evidence buffer per surface while deferred.
        """

        self._deferred = deferred
        for surface in self.surfaces:
            surface._immediate = None if deferred else self._apply

    def prepare_accumulation(self) -> None:
        """Preallocate fixed evidence buffers, then defer into them."""

        self.defer(True)
        for surface in self.surfaces:
            surface.prepare_accumulator()

    def unscale_(self, scale: float | Tensor) -> None:
        """Divide retained evidence by an AMP loss scale."""

        divisor = float(scale)
        for surface in self.surfaces:
            surface.scale_evidence_(divisor)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Discard any evidence retained but not yet applied."""

        del set_to_none
        for surface in self.surfaces:
            surface.flush_evidence()
            surface.reset_tracking()
        self._flips = 0

    # -- the step ----------------------------------------------------------

    @torch.no_grad()
    def _apply(self, surface: Surface, evidence: Tensor) -> None:
        state = self.state[id(surface)]
        evidence = evidence.float()
        if self.gradient_reducer is not None:
            evidence = self.gradient_reducer(evidence)
        self._flips += kernels.coordinate_update(
            evidence,
            surface.coordinate.data,
            surface.packed_sign.data,
            state["moment_q"],
            state["moment_scale"],
            state["row_v"],
            state["col_v"],
            beta1=self.cfg.beta1,
            beta2=self.cfg.beta2,
            update_clip=self.cfg.update_rms_clip,
            coordinate_lr=self.cfg.coordinate_lr,
            block_size=self.cfg.momentum_block_size,
            # Distinct streams per surface, reproducible from (seed, step).
            seed=self.cfg.seed + 1_000_003 * self._index[id(surface)],
            step=self.step_number,
        )

    @torch.no_grad()
    def step(self) -> int:
        """Apply any retained evidence, clamp scales, advance the step.

        Returns the number of sign flips this step -- the only honest measure
        of how much a binary model actually moved.
        """

        for surface in self.surfaces:
            # flush, not take: a host that ran forward more times than backward
            # (gradient checkpointing) leaves evidence the pending count never
            # released, and losing it would silently stop training.
            evidence = surface.flush_evidence(keep_buffer=self._deferred)
            if evidence is not None:
                self._apply(surface, evidence)
            surface.log_scale.data.clamp_(
                math.log(self.cfg.scale_min), math.log(self.cfg.scale_max)
            )
        flips = self._flips
        self._flips = 0
        self.step_number += 1
        return flips

    # -- reporting and checkpointing --------------------------------------

    def state_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for state in self.state.values()
            for tensor in state.values()
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "config": self.cfg.metadata(),
            "step_number": self.step_number,
            "surfaces": [
                {key: tensor.detach().cpu() for key, tensor in self.state[id(surface)].items()}
                for surface in self.surfaces
            ],
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        entries = state_dict["surfaces"]
        if len(entries) != len(self.surfaces):
            raise ValueError(
                f"checkpoint holds {len(entries)} surfaces, model has {len(self.surfaces)}"
            )
        for surface, entry in zip(self.surfaces, entries):
            state = self.state[id(surface)]
            for key, tensor in entry.items():
                state[key].copy_(tensor.to(state[key].device))
        self.step_number = int(state_dict.get("step_number", 0))
