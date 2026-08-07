"""Drop-in replacements for ``nn.Linear`` and ``nn.Embedding``.

Same constructor attributes, same ``weight`` and ``bias`` surface, same call
signature as the modules they replace. Code that introspects a model --
parameter counts, weight tying, initialization sweeps, fusion passes that read
``.weight`` -- keeps working without knowing QSTE exists.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .config import QSTEConfig, coerce_config
from .functional import qste_embedding, qste_linear, qste_weight
from .surface import Surface


class QSTELinear(nn.Module):
    """``y = (x @ sign^T) * scale + bias`` with a binary forward matrix."""

    def __init__(self, surface: Surface, bias: Tensor | None = None):
        super().__init__()
        self.surface = surface
        self.in_features = surface.columns
        self.out_features = surface.rows
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().float().clone())

    @classmethod
    def from_linear(
        cls, module: nn.Linear, cfg: QSTEConfig | None = None, surface: Surface | None = None
    ) -> "QSTELinear":
        return cls(surface or Surface(module.weight, coerce_config(cfg)), module.bias)

    @property
    def scale(self) -> Tensor:
        return self.surface.scale

    @property
    def packed_sign(self) -> Tensor:
        return self.surface.packed_sign

    @property
    def weight(self) -> Tensor:
        """Differentiable dense equivalent, for code that reads ``.weight``.

        Materializes ``out_features x in_features`` floats on every access, and
        the module's own forward does not use it. It exists so a host framework
        that fuses or slices weights by hand still trains the coordinate; see
        :class:`qste.functional.QSTEWeightFn`.
        """

        return qste_weight(self.surface)

    def forward(self, inputs: Tensor) -> Tensor:
        return qste_linear(inputs, self.surface, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class QSTEEmbedding(nn.Module):
    """Binary embedding table: a row lookup into packed signs times a scale."""

    def __init__(self, surface: Surface, padding_idx: int | None = None):
        super().__init__()
        self.surface = surface
        self.num_embeddings = surface.rows
        self.embedding_dim = surface.columns
        self.padding_idx = padding_idx

    @classmethod
    def from_embedding(
        cls, module: nn.Embedding, cfg: QSTEConfig | None = None, surface: Surface | None = None
    ) -> "QSTEEmbedding":
        return cls(
            surface or Surface(module.weight, coerce_config(cfg)),
            getattr(module, "padding_idx", None),
        )

    @property
    def weight(self) -> Tensor:
        return qste_weight(self.surface)

    def forward(self, ids: Tensor) -> Tensor:
        return qste_embedding(ids, self.surface)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class PackedLinear(nn.Module):
    """Inference-only linear. Holds bits and a scale; no training state."""

    def __init__(self, packed: Tensor, scale: Tensor, columns: int, bias: Tensor | None):
        super().__init__()
        self.register_buffer("packed_sign", packed)
        self.register_buffer("scale", scale)
        self.in_features = columns
        self.out_features = packed.shape[0]
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias)

    @torch.no_grad()
    def forward(self, inputs: Tensor) -> Tensor:
        from . import kernels

        return kernels.packed_linear_affine(
            inputs.float(), self.packed_sign, self.scale, self.bias, self.in_features
        ).to(inputs.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class PackedEmbedding(nn.Module):
    """Inference-only embedding over packed signs."""

    def __init__(self, packed: Tensor, scale: Tensor, columns: int):
        super().__init__()
        self.register_buffer("packed_sign", packed)
        self.register_buffer("scale", scale)
        self.num_embeddings = packed.shape[0]
        self.embedding_dim = columns

    @torch.no_grad()
    def forward(self, ids: Tensor) -> Tensor:
        from . import kernels

        return kernels.packed_embedding(ids, self.packed_sign, self.scale, self.embedding_dim)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
