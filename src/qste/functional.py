"""Autograd for binary surfaces.

The interesting line in this file is what forward hands to backward.

The coordinate update needs ``gradᵀ @ x``: which input caused which gradient.
``grad`` exists only in backward and ``x`` only in forward, so the pairing
cannot be formed without something surviving between them -- and reducing
either side before they meet (summing over the batch, say) destroys the
pairing and yields noise, not a smaller gradient.

So the pairing is kept exactly and the *operand* is shrunk instead. Forward
retains one bit per element -- the sign of the row's deviation from its own
mean -- plus an offset and a scale per sample, and backward expands it a tile
at a time inside the same GEMM. The outer product is unchanged in structure;
what crossed the boundary is 30x smaller.

Centering before signing is not cosmetic. Everything out of a ReLU is
non-negative, so plain ``sign(x)`` would be all ones and the outer product
would collapse to rank one -- the exact failure mode this design exists to
avoid. The offset it leaves behind is recovered exactly in backward.

Nothing else in backward needs ``x`` at all: the input gradient is
``(grad * scale) @ sign``, the scale gradient reduces the evidence against the
packed signs, and the bias gradient is a sum over ``grad``.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor
from torch.autograd import Function

from . import kernels
from .surface import Surface


def _autocast(value: Tensor) -> Tensor:
    """Match how autocast would have cast a native GEMM's input."""

    device = value.device.type
    if device not in ("cuda", "cpu"):
        return value
    try:
        if not torch.is_autocast_enabled(device):
            return value
        dtype = torch.get_autocast_dtype(device)
    except (TypeError, AttributeError):  # older torch signatures
        if not torch.is_autocast_enabled():
            return value
        dtype = torch.get_autocast_gpu_dtype()
    return value.to(dtype=dtype)


def _kernel_dtype(value: Tensor) -> Tensor:
    """CPU kernels are float32; CUDA fallbacks follow the incoming dtype."""

    if value.device.type == "cpu" and value.dtype != torch.float32:
        return value.float()
    return value


# ---------------------------------------------------------------------------
# Retained-activation encodings
# ---------------------------------------------------------------------------


_STORAGE = "bit"


@contextmanager
def exact_evidence(storage: str = "fp32"):
    """Diagnostic: retain activations exactly instead of as bits.

    A diagnostic, not a training mode: it answers "is the packed encoding
    costing my model anything?" by running the same steps with the same seed
    both ways. It gives up the entire memory saving, so nothing should ship
    inside it.

        with qste.exact_evidence():
            ...                      # identical arithmetic, x kept in fp32

    ``storage="int8"`` is the intermediate point, kept because it is what the
    comparison is measured against.
    """

    global _STORAGE
    if storage not in ("fp32", "int8", "bit"):
        raise ValueError("storage must be 'fp32', 'int8', or 'bit'")
    previous, _STORAGE = _STORAGE, storage
    try:
        yield
    finally:
        _STORAGE = previous


def encode_activation(flat: Tensor, storage: str | None = None) -> tuple[Tensor, Tensor | None]:
    """Reduce the activation to whatever backward actually needs.

    Returns ``(payload, aux)``. ``aux`` holds the per-sample constants that
    turn the payload back into an activation: one row for ``int8`` (a scale),
    two for ``bit`` (an offset and a scale).
    """

    storage = storage or _STORAGE

    if storage == "fp32":
        return flat, None
    if storage == "int8":
        scale = flat.abs().amax(dim=1).clamp_min(1e-8).div_(127.0)
        payload = (flat / scale.unsqueeze(1)).round_().clamp_(-127, 127).to(torch.int8)
        return payload, scale.unsqueeze(0)
    packed, offset, scale = kernels.pack_affine_rows(_kernel_dtype(flat))
    return packed, torch.stack((offset, scale))


def decoded_evidence(
    grad: Tensor, payload: Tensor, aux: Tensor | None, storage: str, columns: int
) -> Tensor:
    """``gradᵀ @ x`` from whatever forward retained, as float32."""

    if storage == "fp32":
        return grad.float().t() @ payload.float()
    if storage == "int8":
        return (grad.float() * aux[0].float().unsqueeze(1)).t() @ payload.float()

    # x[n] = offset[n] + scale[n] * sign(x[n] - offset[n]), so the evidence
    # splits into the packed outer product and a rank-one offset correction.
    #
    # The gradient goes over in whatever dtype it arrived in. Forcing float32
    # here cost every mixed-precision host the fastest product in the pass:
    # under autocast the incoming gradient is already half, and upcasting made
    # the one GEMM whose result is stochastically rounded into an INT8
    # coordinate the only one barred from the hardware's reduced-precision
    # path. The kernel decides; see `qste.kernels.device`.
    offset, scale = aux[0].float(), aux[1].float()
    evidence = kernels.evidence_from_packed(grad, payload, columns, row_scale=scale)
    # The offset is constant across a row, so its entire contribution is the
    # column vector grad^T @ offset, broadcast over every column. Exact, and a
    # GEMV rather than a second full-size temporary -- which is also why the
    # offset is cast down to the gradient rather than the gradient up to it.
    correction = torch.mv(grad.t(), offset.to(grad.dtype)).float()
    evidence.add_(correction.unsqueeze(1))
    return evidence


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


class QSTELinearFn(Function):
    @staticmethod
    def forward(ctx, inputs, packed, scale, bias, surface: Surface):
        columns = surface.columns
        output = kernels.packed_linear_affine(
            _kernel_dtype(inputs), packed, _kernel_dtype(scale), bias, columns
        ).to(inputs.dtype)

        flat = inputs.reshape(-1, columns)
        payload, aux = encode_activation(flat)
        storage = _STORAGE
        ctx.save_for_backward(payload, aux, packed, scale)
        ctx.surface = surface
        ctx.storage = storage
        ctx.columns = columns
        ctx.input_shape = inputs.shape
        ctx.input_dtype = inputs.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        payload, aux, packed, scale = ctx.saved_tensors
        surface = ctx.surface
        grad2 = grad_output.reshape(-1, grad_output.shape[-1])

        # The row scale folds into the expanded weight inside the kernel, so
        # this no longer builds a [batch, rows] copy of the gradient.
        grad_inputs = kernels.packed_transpose(
            _kernel_dtype(grad2), packed, ctx.columns, row_scale=scale.detach()
        )
        grad_inputs = grad_inputs.reshape(ctx.input_shape).to(ctx.input_dtype)

        evidence = decoded_evidence(grad2, payload, aux, ctx.storage, ctx.columns)
        # One float matrix is the whole per-surface scratch: read it for the
        # exact scale reduction, then turn it into coordinate evidence in place.
        grad_scale = kernels.packed_row_inner(evidence, packed, ctx.columns).to(scale.dtype)
        evidence.mul_(scale.detach().float().unsqueeze(1))
        surface.consume_evidence(evidence)

        grad_bias = grad2.sum(dim=0).to(ctx.bias_dtype) if ctx.bias_dtype is not None else None
        return grad_inputs, None, grad_scale, grad_bias, None


def qste_linear(
    inputs: Tensor, surface: Surface, bias: Tensor | None = None
) -> Tensor:
    """Functional binary linear. Use this to wire a surface by hand."""

    surface.note_forward()
    return QSTELinearFn.apply(
        _autocast(inputs), surface.packed_sign, surface.scale, bias, surface
    )


# ---------------------------------------------------------------------------
# Dense weight view
# ---------------------------------------------------------------------------


class QSTEWeightFn(Function):
    """A differentiable ``sign * scale`` matrix.

    This exists because real frameworks reach past the module and read
    ``layer.weight`` -- to fuse several projections into one GEMM, to build a
    recurrence matrix, to tie parameters. Those call sites cannot be found and
    rewritten, and if ``.weight`` were a plain detached tensor they would train
    the scale and silently never train the coordinate.

    Routing the weight through autograd instead means ``dL/dW`` arrives here
    and becomes exactly the evidence the module path would have produced. A
    framework that never calls the module still trains correctly.
    """

    @staticmethod
    def forward(ctx, packed, scale, surface: Surface):
        sign = kernels.unpack_rows(packed, surface.columns).to(scale.dtype)
        ctx.save_for_backward(packed, scale)
        ctx.surface = surface
        return sign * scale.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_weight):
        packed, scale = ctx.saved_tensors
        surface = ctx.surface
        grad_weight = grad_weight.float().contiguous()
        # dL/dscale_i = sum_j dL/dW[i,j] * sign[i,j]
        grad_scale = kernels.packed_row_inner(grad_weight, packed, surface.columns)
        # dL/dsign[i,j] = dL/dW[i,j] * scale_i, which is the same coordinate
        # evidence the linear path computes as scale * (grad^T @ x).
        evidence = grad_weight.mul_(scale.detach().float().unsqueeze(1))
        surface.consume_evidence(evidence)
        return None, grad_scale.to(scale.dtype), None


def qste_weight(surface: Surface) -> Tensor:
    """The dense float matrix equivalent to ``surface``, with gradients."""

    surface.note_forward()
    return QSTEWeightFn.apply(surface.packed_sign, surface.scale, surface)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class QSTEEmbeddingFn(Function):
    @staticmethod
    def forward(ctx, ids, packed, scale, surface: Surface):
        output = kernels.packed_embedding(ids, packed, _kernel_dtype(scale), surface.columns)
        ctx.save_for_backward(ids, packed, scale)
        ctx.surface = surface
        return output.to(scale.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        ids, packed, scale = ctx.saved_tensors
        surface = ctx.surface
        columns = surface.columns
        # index_add_ over the vocabulary is the only full matrix here, and it
        # is immediately reused as the evidence tensor.
        evidence = torch.zeros(
            surface.rows, columns, device=grad_output.device, dtype=torch.float32
        )
        evidence.index_add_(0, ids.reshape(-1).long(), grad_output.reshape(-1, columns).float())
        grad_scale = kernels.packed_row_inner(evidence, packed, columns).to(scale.dtype)
        evidence.mul_(scale.detach().float().unsqueeze(1))
        surface.consume_evidence(evidence)
        return None, None, grad_scale, None


def qste_embedding(ids: Tensor, surface: Surface) -> Tensor:
    surface.note_forward()
    return QSTEEmbeddingFn.apply(ids, surface.packed_sign, surface.scale, surface)
