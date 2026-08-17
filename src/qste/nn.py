"""Activations that do not pin a full-precision tensor for backward.

Without this module QSTE's memory saving is real per operation and invisible at
the peak.

In ``x -> Linear -> ReLU -> Linear``, torch's ReLU saves its own output so
backward can zero the gradient where that output was not positive. The saved
output is also the next linear's input, so the full-precision activation stays
alive for the whole backward pass whatever the linear does with it. A QSTE
linear packing its input to one bit per element does not replace that tensor --
it adds a second, smaller one beside it. Measured end to end, a converted stack
used *more* peak memory than the same stack with packing off.

What backward actually needs is much smaller. A saturating activation needs one
bit per element (was the output positive) where torch spends thirty-two. A
smooth activation needs its local derivative, which is bounded and survives
eight bits with room to spare. Dropout needs its mask, one bit. So:

    ReLU, ReLU6, Hardtanh   1 bit per element, exact
    Dropout                 1 bit per element, exact
    GELU, SiLU, Tanh, ...   8 bits per element, derivative quantized per row

With these in the stack nothing holds the activation, and it dies at the end of
the layer that produced it. That is when the per-operation saving reaches the
peak.

Adoption stays one line::

    qste.convert(model)                          # linears and activations
    qste.convert(model, activations=False)       # linears only

and for models that call the functional forms instead of holding modules::

    with qste.packed_activations():
        loss = model(batch)
    loss.backward()

The context packs only activations whose input is the direct output of a
converted QSTE layer. Functional activations elsewhere in a mixed model remain
ordinary PyTorch operations.

Everything here is device-agnostic: it is written against the kernel dispatch
layer, so it runs native on CPU, native on GPU, and on the pure-torch reference
path anywhere else.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from . import kernels

# Bound at import, before anything can patch them. `packed_activations` swaps
# `F.gelu` and `F.silu` for the wrappers below, and those wrappers have to be
# able to call the real thing for the forward value -- otherwise the patch makes
# the activation call itself.
_TORCH_RELU = F.relu
_TORCH_RELU6 = F.relu6
_TORCH_GELU = F.gelu
_TORCH_SILU = F.silu
_TORCH_LEAKY_RELU = F.leaky_relu
_TORCH_HARDTANH = F.hardtanh
_TORCH_HARDSIGMOID = F.hardsigmoid
_TORCH_HARDSWISH = F.hardswish
_TORCH_ELU = F.elu
_TORCH_SELU = F.selu
_TORCH_CELU = F.celu
_TORCH_SOFTPLUS = F.softplus
_TORCH_MISH = F.mish
_TORCH_DROPOUT = F.dropout
_TORCH_TORCH_RELU = torch.relu

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_TANH_COEFFICIENT = 0.044715


def _flatten(value: Tensor) -> tuple[Tensor, tuple[int, ...], int]:
    """Rank-2 view over the last dimension. Packing is per row."""

    if value.ndim == 0:
        # A scalar (e.g. a learned per-tensor gain flowing through a packed
        # activation) has no last dimension. Treat it as a 1x1 row so packing,
        # the row scale, and the backward reshape all still apply.
        return value.reshape(1, 1).contiguous(), value.shape, 1
    if value.ndim == 1:
        return value.reshape(1, -1).contiguous(), value.shape, value.shape[0]
    columns = value.shape[-1]
    return value.reshape(-1, columns).contiguous(), value.shape, columns


def _mark_activation_input(value: Tensor) -> Tensor:
    """Tag a converted layer's ordinary Tensor output without propagating it."""

    value._qste_activation_input = True
    return value


def _take_activation_input(value: Tensor) -> Tensor | None:
    return value if getattr(value, "_qste_activation_input", False) else None


# ---------------------------------------------------------------------------
# Exact: one bit per element
# ---------------------------------------------------------------------------


class _MaskedFn(Function):
    """Backward is ``grad`` where a bit is set. Nothing else is retained."""

    @staticmethod
    def forward(ctx, inputs, lower, upper):
        output = inputs if upper is None else inputs.clamp_max(upper)
        output = output.clamp_min(lower)
        flat, shape, columns = _flatten(output)
        # Strictly greater: an element sitting exactly on the clamp has zero
        # gradient, and `>=` would leak one through at the boundary.
        keep = flat > lower if upper is None else (flat > lower) & (flat < upper)
        ctx.save_for_backward(kernels.pack_bits(keep))
        ctx.shape = shape
        ctx.columns = columns
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (packed,) = ctx.saved_tensors
        flat = grad_output.reshape(-1, ctx.columns).contiguous()
        return kernels.apply_bits(flat, packed, ctx.columns).view(ctx.shape), None, None


class _PiecewiseFn(Function):
    """One bit per element, for any activation that is linear in pieces.

    ``_MaskedFn`` above covers the clamp family, where the output is the input
    and the gradient is one or zero. Plenty of activations are piecewise linear
    without being clamps: leaky ReLU has a non-zero slope below the knee,
    hard sigmoid has a slope of one sixth and an output that is not the input
    at all. The retained information is identical in every case -- which side
    of the knee each element fell on, one bit -- so the only thing that varies
    is the forward and the two slopes, and those go in a table rather than in
    the class.
    """

    @staticmethod
    def forward(ctx, inputs, kind, lower, upper):
        forward, inside_slope, outside_slope = _PIECEWISE[kind]
        output = forward(inputs, lower, upper)
        flat, shape, columns = _flatten(inputs)
        # Strictly inside: an element sitting exactly on a knee takes the outer
        # slope, which is what torch does and what `>=` would get wrong.
        inside = flat > lower
        if upper is not None:
            inside = inside & (flat < upper)
        ctx.save_for_backward(kernels.pack_bits(inside))
        ctx.shape, ctx.columns = shape, columns
        ctx.slopes = (inside_slope, outside_slope)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (packed,) = ctx.saved_tensors
        inside_slope, outside_slope = ctx.slopes
        flat = grad_output.reshape(-1, ctx.columns).contiguous()
        kept = kernels.apply_bits(flat, packed, ctx.columns).view(ctx.shape)
        if outside_slope == 0.0:
            return kept * inside_slope, None, None, None
        # grad*outside everywhere, plus the difference where the bit is set.
        return (
            grad_output * outside_slope + kept * (inside_slope - outside_slope),
            None, None, None,
        )


# kind: (forward, slope inside the interval, slope outside it). Clamps go
# through _MaskedFn instead, which does not need the slopes; this table is for
# the piecewise-linear activations whose output is not simply the clamped input.
_PIECEWISE: dict[str, tuple] = {
    "hardsigmoid": (lambda x, lo, hi: _TORCH_HARDSIGMOID(x), 1.0 / 6.0, 0.0),
}


class _FusableReLU(Tensor):
    """A ReLU output that fuses when squared, and is a plain tensor otherwise.

    ``relu(x).square()`` is the one shape packing cannot help as written:
    square's backward needs ``2*relu(x)``, a magnitude, so the float ReLU
    output stays retained whatever ReLU did. Fused, it costs one unsigned byte
    per element instead of four -- 2.37x beyond what converting the weights
    gives on a wide MLP.

    Fusing requires seeing both calls together, and they are two statements in
    somebody else's model. So ReLU returns a tensor that remembers its input:
    square it and the fused kernel runs from that input, do anything else and
    it behaves as the tensor ReLU would have returned.

    The masked ReLU still runs, so its one-bit tape is briefly allocated and
    dropped when the fused path discards it. One bit against four bytes is not
    worth a lazier design that could get the graph wrong.

    **Known behaviour.** Every operation on this returns a plain tensor, so it
    cannot spread -- but a ReLU output that is the final value of a forward
    reaches the caller as this class. ``isinstance(out, Tensor)`` holds, and
    arithmetic, autograd, printing and device movement are unchanged (the tests
    cover scaling, offsetting, cubing and reducing). ``type(out) is
    torch.Tensor`` does not hold, and pickling a bare ReLU output would need
    this class importable to load again.
    """

    @staticmethod
    def __new__(cls, data: Tensor, pre_activation: Tensor):
        held = data.as_subclass(cls)
        held._pre_activation = pre_activation
        return held

    def __reduce_ex__(self, protocol):
        # Saved as an ordinary tensor. Otherwise a checkpoint containing a bare
        # ReLU output would need this class importable to load again, which is
        # a dependency nobody agreed to and would only show up on the restore.
        with torch._C.DisableTorchFunctionSubclass():
            return self.as_subclass(Tensor).__reduce_ex__(protocol)

    def plain(self) -> Tensor:
        """This, as an ordinary tensor. Nothing else here needs calling."""

        with torch._C.DisableTorchFunctionSubclass():
            return self.as_subclass(Tensor)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        held = args[0] if args else None
        if isinstance(held, cls) and getattr(held, "_pre_activation", None) is not None:
            if func in (Tensor.square, torch.square) and len(args) == 1:
                return _ReluSquareFn.apply(held._pre_activation)
            # ``x ** 2`` is the same expression written the other way.
            if func is Tensor.__pow__ and len(args) == 2 and args[1] == 2:
                return _ReluSquareFn.apply(held._pre_activation)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


def relu(inputs: Tensor) -> Tensor:
    """``F.relu``, retaining one bit per element instead of one float."""

    return _FusableReLU(_MaskedFn.apply(inputs, 0.0, None), inputs)


def relu6(inputs: Tensor) -> Tensor:
    return _MaskedFn.apply(inputs, 0.0, 6.0)


def leaky_relu(inputs: Tensor, negative_slope: float = 0.01,
               inplace: bool = False) -> Tensor:
    if inplace:  # cannot retain what it overwrites; torch keeps its behaviour
        return _TORCH_LEAKY_RELU(inputs, negative_slope, True)
    return _LeakyFn.apply(inputs, negative_slope)


class _LeakyFn(Function):
    """One bit per element. Below the knee the gradient is the slope, not zero."""

    @staticmethod
    def forward(ctx, inputs, negative_slope):
        flat, shape, columns = _flatten(inputs)
        ctx.save_for_backward(kernels.pack_bits(flat > 0))
        ctx.shape, ctx.columns, ctx.slope = shape, columns, negative_slope
        return _TORCH_LEAKY_RELU(inputs, negative_slope, False)

    @staticmethod
    def backward(ctx, grad_output):
        (packed,) = ctx.saved_tensors
        flat = grad_output.reshape(-1, ctx.columns).contiguous()
        kept = kernels.apply_bits(flat, packed, ctx.columns).view(ctx.shape)
        return grad_output * ctx.slope + kept * (1.0 - ctx.slope), None


def hardtanh(inputs: Tensor, min_val: float = -1.0, max_val: float = 1.0,
             inplace: bool = False) -> Tensor:
    if inplace:
        return _TORCH_HARDTANH(inputs, min_val, max_val, True)
    return _MaskedFn.apply(inputs, min_val, max_val)


class _ReluSquareFn(Function):
    """``relu(x) ** 2``, retaining one UNSIGNED byte per element.

    Written as two operations -- ``F.relu(x).square()`` -- this is the most
    expensive activation a model can have, and packing alone cannot help.
    Square's backward needs ``2*relu(x)``, a magnitude and not a sign, so
    autograd keeps the full float ReLU output whatever ReLU retained. The
    packed mask then lands on top as pure addition and the pair costs 3% MORE
    than torch.

    Fused, it becomes one operation retaining one tensor, and that tensor is a
    derivative -- the thing this library already shrinks. ``2*relu(x)`` is
    quantized against a per-row maximum, as the smooth activations quantize
    theirs.

    The byte is unsigned because ``relu(x)`` is never negative. A signed byte
    would spend one of its eight bits on a constant sign and resolve the
    magnitude into 127 levels; unsigned resolves it into 255 for the same
    memory, halving the quantization error because the function's range is
    known.
    """

    @staticmethod
    def forward(ctx, inputs):
        activated = inputs.clamp_min(0)
        flat, shape, columns = _flatten(activated)
        scale = flat.amax(dim=1).clamp_min(1e-8).div_(255.0)
        payload = flat.div(scale.unsqueeze(1)).round_().clamp_(0, 255).to(torch.uint8)
        ctx.save_for_backward(payload, scale)
        ctx.shape, ctx.columns = shape, columns
        return activated * activated

    @staticmethod
    def backward(ctx, grad_output):
        payload, scale = ctx.saved_tensors
        activated = payload.to(grad_output.dtype).mul_(
            scale.to(grad_output.dtype).unsqueeze(1)
        )
        return grad_output * 2.0 * activated.view(ctx.shape)


def relu_square(inputs: Tensor) -> Tensor:
    """``relu(x) ** 2`` as one op, retaining a byte per element instead of four.

    Under :func:`packed_activations` a model writing ``F.relu(x).square()``
    already reaches this through :class:`_FusableReLU`, with no source change.
    Call it directly when the fusion should not depend on that context::

        branch = fc2(qste.nn.relu_square(fc1(norm(hidden))))
    """

    return _ReluSquareFn.apply(inputs)


def hardsigmoid(inputs: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return _TORCH_HARDSIGMOID(inputs, True)
    return _PiecewiseFn.apply(inputs, "hardsigmoid", -3.0, 3.0)


class _DropoutFn(Function):
    @staticmethod
    def forward(ctx, inputs, probability):
        keep = torch.rand_like(inputs, dtype=torch.float32) >= probability
        flat, shape, columns = _flatten(keep)
        ctx.save_for_backward(kernels.pack_bits(flat))
        ctx.shape = shape
        ctx.columns = columns
        ctx.gain = 1.0 / (1.0 - probability)
        return inputs * keep.to(inputs.dtype) * ctx.gain

    @staticmethod
    def backward(ctx, grad_output):
        (packed,) = ctx.saved_tensors
        flat = grad_output.reshape(-1, ctx.columns).contiguous()
        masked = kernels.apply_bits(flat, packed, ctx.columns)
        return masked.mul_(ctx.gain).view(ctx.shape), None


def dropout(inputs: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """``F.dropout``, keeping the mask as bits rather than as bytes."""

    if not training or p == 0.0:
        return inputs
    if not 0.0 <= p < 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    return _DropoutFn.apply(inputs, p)


# ---------------------------------------------------------------------------
# Smooth: eight bits per element, on the derivative itself
# ---------------------------------------------------------------------------
#
# Quantizing the derivative beats quantizing the input at no extra cost. A
# smooth activation's derivative is bounded (GELU's lives in about
# [-0.13, 1.13]) and is what backward multiplies by, so eight bits against a
# per-row peak leaves about 0.2% relative error on the multiplier -- well under
# the noise the gradient already carries.


def _gelu_derivative(inputs: Tensor, approximate: str) -> Tensor:
    if approximate == "tanh":
        inner = _SQRT_2_OVER_PI * (inputs + _TANH_COEFFICIENT * inputs.pow(3))
        tanh = torch.tanh(inner)
        left = 0.5 * (1.0 + tanh)
        right = (
            0.5
            * inputs
            * (1.0 - tanh * tanh)
            * _SQRT_2_OVER_PI
            * (1.0 + 3.0 * _TANH_COEFFICIENT * inputs * inputs)
        )
        return left + right
    cumulative = 0.5 * (1.0 + torch.erf(inputs / _SQRT_2))
    return cumulative + inputs * torch.exp(-0.5 * inputs * inputs) * _INV_SQRT_2PI


def _silu_derivative(inputs: Tensor) -> Tensor:
    sigmoid = torch.sigmoid(inputs)
    return sigmoid * (1.0 + inputs * (1.0 - sigmoid))


def _tanh_derivative(output: Tensor) -> Tensor:
    return 1.0 - output * output


def _sigmoid_derivative(output: Tensor) -> Tensor:
    return output * (1.0 - output)


def _elu_derivative(inputs: Tensor, output: Tensor, alpha: float) -> Tensor:
    # d/dx of alpha*(exp(x)-1) is alpha*exp(x), which is output + alpha.
    return torch.where(inputs > 0, torch.ones_like(inputs), output + alpha)


def _softplus_derivative(inputs: Tensor, beta: float) -> Tensor:
    return torch.sigmoid(beta * inputs)


def _mish_derivative(inputs: Tensor) -> Tensor:
    softplus = F.softplus(inputs)
    tanh = torch.tanh(softplus)
    return tanh + inputs * (1.0 - tanh * tanh) * torch.sigmoid(inputs)


def _hardswish_derivative(inputs: Tensor) -> Tensor:
    # x*hardsigmoid(x): zero below -3, one above 3, (2x+3)/6 between.
    inner = (2.0 * inputs + 3.0) / 6.0
    return torch.where(
        inputs <= -3.0, torch.zeros_like(inputs),
        torch.where(inputs >= 3.0, torch.ones_like(inputs), inner),
    )


# Every smooth activation, as (forward, derivative). The derivative receives
# both the input and the output, since some are cheaper from one and some from
# the other -- tanh' is 1-y^2, gelu' needs x -- and computing it from whichever
# is already present is the difference between retaining one tensor and two.
#
# Adding an activation is one entry. The int8 derivative is a 4x saving on
# whatever a framework calls, independent of which function it is.
_SMOOTH: dict[str, tuple] = {
    "gelu": (lambda x, a: _TORCH_GELU(x, approximate=a),
             lambda x, y, a: _gelu_derivative(x, a)),
    "silu": (lambda x, a: _TORCH_SILU(x),
             lambda x, y, a: _silu_derivative(x)),
    "tanh": (lambda x, a: torch.tanh(x),
             lambda x, y, a: _tanh_derivative(y)),
    "sigmoid": (lambda x, a: torch.sigmoid(x),
                lambda x, y, a: _sigmoid_derivative(y)),
    "elu": (lambda x, a: _TORCH_ELU(x, a),
            lambda x, y, a: _elu_derivative(x, y, a)),
    "celu": (lambda x, a: _TORCH_CELU(x, a),
             lambda x, y, a: torch.where(x > 0, torch.ones_like(x),
                                         (y / a + 1.0))),
    "selu": (lambda x, a: _TORCH_SELU(x),
             lambda x, y, a: torch.where(
                 x > 0, torch.full_like(x, _SELU_SCALE),
                 y + _SELU_SCALE * _SELU_ALPHA)),
    "softplus": (lambda x, a: _TORCH_SOFTPLUS(x, a),
                 lambda x, y, a: _softplus_derivative(x, a)),
    "mish": (lambda x, a: _TORCH_MISH(x),
             lambda x, y, a: _mish_derivative(x)),
    "hardswish": (lambda x, a: _TORCH_HARDSWISH(x),
                  lambda x, y, a: _hardswish_derivative(x)),
}

_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805


class _SmoothFn(Function):
    """Retain the local derivative as INT8 with a learned-free per-row scale."""

    @staticmethod
    def forward(ctx, inputs, kind, approximate):
        entry = _SMOOTH.get(kind)
        if entry is None:  # pragma: no cover - guarded by the public wrappers
            raise ValueError(f"unknown smooth activation {kind!r}")
        forward, derivative_of = entry
        output = forward(inputs, approximate)
        derivative = derivative_of(inputs.float(), output.float(), approximate)

        flat, shape, columns = _flatten(derivative)
        scale = flat.abs().amax(dim=1).clamp_min(1e-8).div_(127.0)
        payload = flat.div_(scale.unsqueeze(1)).round_().clamp_(-127, 127).to(torch.int8)
        ctx.save_for_backward(payload, scale)
        ctx.shape = shape
        ctx.columns = columns
        return output

    @staticmethod
    def backward(ctx, grad_output):
        payload, scale = ctx.saved_tensors
        derivative = payload.to(grad_output.dtype).mul_(scale.to(grad_output.dtype).unsqueeze(1))
        return grad_output * derivative.view(ctx.shape), None, None


class _AnyElementwiseFn(Function):
    """Pack the derivative of an activation nobody wrote down.

    The tables above are an optimization. A closed-form derivative is cheaper
    than asking for one, and a piecewise-linear activation packs to a bit
    instead of a byte -- but neither is required, and a library whose coverage
    is a list only covers the architectures someone already thought of.

    The saving does not need the function's identity. Whatever ``f`` is, its
    backward multiplies by ``f'(x)``, and ``f'(x)`` comes from running ``f``
    once on a detached input and asking autograd. Quantized, the retained tape
    is one byte per element for any ``f``, including one written after this.

    ``f`` must be elementwise, which is checked: ``grad(f(x).sum(), x)`` equals
    ``f'(x)`` only when each output depends on its own input alone. A softmax
    or a norm would silently return the row's sum, so a shape mismatch raises.
    A non-elementwise function that happens to preserve shape is the caller's
    to know about.
    """

    @staticmethod
    def forward(ctx, inputs, function):
        with torch.enable_grad():
            probe = inputs.detach().requires_grad_(True)
            output = function(probe)
            if output.shape != probe.shape:
                raise ValueError(
                    "qste.nn.packed() needs an elementwise function; this one "
                    f"maps {tuple(probe.shape)} to {tuple(output.shape)}"
                )
            (derivative,) = torch.autograd.grad(output.sum(), probe)

        flat, shape, columns = _flatten(derivative.float())
        scale = flat.abs().amax(dim=1).clamp_min(1e-8).div_(127.0)
        payload = flat.div_(scale.unsqueeze(1)).round_().clamp_(-127, 127).to(torch.int8)
        ctx.save_for_backward(payload, scale)
        ctx.shape, ctx.columns = shape, columns
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output):
        payload, scale = ctx.saved_tensors
        derivative = payload.to(grad_output.dtype).mul_(
            scale.to(grad_output.dtype).unsqueeze(1)
        )
        return grad_output * derivative.view(ctx.shape), None


def elementwise(function, inputs: Tensor) -> Tensor:
    """``function(inputs)``, retaining a byte per element instead of a float.

    For any elementwise activation, named or not::

        qste.nn.elementwise(lambda t: t * torch.tanh(F.softplus(t)), x)
        qste.nn.elementwise(lambda t: F.relu(t) ** 3, x)

    The tables handle what they know faster, since a closed form beats asking
    autograd and a mask beats a byte. This covers everything else.
    """

    return _AnyElementwiseFn.apply(inputs, function)


def packed(function):
    """``function``, wrapped so it retains a byte per element. Elementwise only.

        block_activation = qste.nn.packed(lambda t: F.relu(t).square())
        ...
        hidden = fc2(block_activation(fc1(hidden)))
    """

    def wrapper(inputs: Tensor) -> Tensor:
        return elementwise(function, inputs)

    wrapper.__name__ = getattr(function, "__name__", "packed")
    wrapper.__doc__ = f"Packed: {getattr(function, '__doc__', None) or function!r}"
    return wrapper


def gelu(inputs: Tensor, approximate: str = "none") -> Tensor:
    return _SmoothFn.apply(inputs, "gelu", approximate)


def silu(inputs: Tensor) -> Tensor:
    return _SmoothFn.apply(inputs, "silu", "none")


def tanh(inputs: Tensor) -> Tensor:
    return _SmoothFn.apply(inputs, "tanh", "none")


def elu(inputs: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    if inplace:
        return _TORCH_ELU(inputs, alpha, True)
    return _SmoothFn.apply(inputs, "elu", alpha)


def celu(inputs: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    if inplace:
        return _TORCH_CELU(inputs, alpha, True)
    return _SmoothFn.apply(inputs, "celu", alpha)


def selu(inputs: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return _TORCH_SELU(inputs, True)
    return _SmoothFn.apply(inputs, "selu", 1.0)


def softplus(inputs: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    return _SmoothFn.apply(inputs, "softplus", beta)


def mish(inputs: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return _TORCH_MISH(inputs, True)
    return _SmoothFn.apply(inputs, "mish", 1.0)


def hardswish(inputs: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return _TORCH_HARDSWISH(inputs, True)
    return _SmoothFn.apply(inputs, "hardswish", 1.0)


def sigmoid(inputs: Tensor) -> Tensor:
    return _SmoothFn.apply(inputs, "sigmoid", "none")


# ---------------------------------------------------------------------------
# Modules, shaped exactly like the ones they replace
# ---------------------------------------------------------------------------


class ReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        # Accepted and ignored: packing needs the value, and an in-place ReLU
        # saves nothing here because nothing full-precision survives anyway.
        self.inplace = False
        _ = inplace

    def forward(self, inputs: Tensor) -> Tensor:
        return relu(inputs)


class ReLU6(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = False

    def forward(self, inputs: Tensor) -> Tensor:
        return relu6(inputs)


class GELU(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, inputs: Tensor) -> Tensor:
        return gelu(inputs, self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class SiLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = False

    def forward(self, inputs: Tensor) -> Tensor:
        return silu(inputs)


class Tanh(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return tanh(inputs)


class Sigmoid(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return sigmoid(inputs)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, inputs: Tensor) -> Tensor:
        return leaky_relu(inputs, self.negative_slope)


class Hardtanh(nn.Module):
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        super().__init__()
        self.min_val, self.max_val = min_val, max_val

    def forward(self, inputs: Tensor) -> Tensor:
        return hardtanh(inputs, self.min_val, self.max_val)


class Hardsigmoid(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return hardsigmoid(inputs)


class ELU(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: Tensor) -> Tensor:
        return elu(inputs, self.alpha)


class CELU(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs: Tensor) -> Tensor:
        return celu(inputs, self.alpha)


class SELU(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return selu(inputs)


class Softplus(nn.Module):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta, self.threshold = beta, threshold

    def forward(self, inputs: Tensor) -> Tensor:
        return softplus(inputs, self.beta, self.threshold)


class Mish(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return mish(inputs)


class Hardswish(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return hardswish(inputs)


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = False

    def forward(self, inputs: Tensor) -> Tensor:
        return dropout(inputs, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


# Exact replacements first: these change nothing about what the model computes
# or what its gradient is, so converting them is never a decision.
EXACT_REPLACEMENTS: dict[type, type] = {
    nn.ReLU: ReLU,
    nn.ReLU6: ReLU6,
    nn.LeakyReLU: LeakyReLU,
    nn.Hardtanh: Hardtanh,
    nn.Hardsigmoid: Hardsigmoid,
    nn.Dropout: Dropout,
}

# These quantize the retained derivative. The forward is untouched; the
# gradient multiplier carries about 0.2% relative error.
SMOOTH_REPLACEMENTS: dict[type, type] = {
    nn.GELU: GELU,
    nn.SiLU: SiLU,
    nn.Tanh: Tanh,
    nn.Sigmoid: Sigmoid,
    nn.ELU: ELU,
    nn.CELU: CELU,
    nn.SELU: SELU,
    nn.Softplus: Softplus,
    nn.Mish: Mish,
    nn.Hardswish: Hardswish,
}

REPLACEMENTS: dict[type, type] = {**EXACT_REPLACEMENTS, **SMOOTH_REPLACEMENTS}


def replace(module: nn.Module) -> nn.Module | None:
    """A packed equivalent of ``module``, or ``None`` if there is not one."""

    target = REPLACEMENTS.get(type(module))
    if target is None:
        return None
    if target is GELU:
        return GELU(getattr(module, "approximate", "none"))
    if target is Dropout:
        return Dropout(getattr(module, "p", 0.5))
    if target is LeakyReLU:
        return LeakyReLU(getattr(module, "negative_slope", 0.01))
    if target is Hardtanh:
        return Hardtanh(getattr(module, "min_val", -1.0),
                        getattr(module, "max_val", 1.0))
    if target in (ELU, CELU):
        return target(getattr(module, "alpha", 1.0))
    if target is Softplus:
        return Softplus(getattr(module, "beta", 1.0),
                        getattr(module, "threshold", 20.0))
    return target()


# ---------------------------------------------------------------------------
# Functional call sites
# ---------------------------------------------------------------------------


def _select_converted(original, packed):
    """Use ``packed`` only for a value emitted directly by a QSTE module."""

    def selected(inputs: Tensor, *args, **kwargs) -> Tensor:
        marked = _take_activation_input(inputs)
        return original(inputs, *args, **kwargs) if marked is None else packed(
            marked, *args, **kwargs
        )

    return selected


def _packed_relu(inputs: Tensor, inplace: bool = False) -> Tensor:
    return _TORCH_RELU(inputs, True) if inplace else relu(inputs)


def _packed_relu6(inputs: Tensor, inplace: bool = False) -> Tensor:
    return _TORCH_RELU6(inputs, True) if inplace else relu6(inputs)


def _packed_silu(inputs: Tensor, inplace: bool = False) -> Tensor:
    return _TORCH_SILU(inputs, True) if inplace else silu(inputs)


def _packed_dropout(inputs: Tensor, p: float = 0.5, training: bool = True,
                    inplace: bool = False) -> Tensor:
    return _TORCH_DROPOUT(inputs, p, training, True) if inplace else dropout(
        inputs, p, training
    )


_context_relu = _select_converted(_TORCH_RELU, _packed_relu)
_context_relu6 = _select_converted(_TORCH_RELU6, _packed_relu6)
_context_leaky_relu = _select_converted(_TORCH_LEAKY_RELU, leaky_relu)
_context_hardtanh = _select_converted(_TORCH_HARDTANH, hardtanh)
_context_hardsigmoid = _select_converted(_TORCH_HARDSIGMOID, hardsigmoid)
_context_torch_relu = _select_converted(_TORCH_TORCH_RELU, relu)
_context_gelu = _select_converted(_TORCH_GELU, gelu)
_context_silu = _select_converted(_TORCH_SILU, _packed_silu)
_context_elu = _select_converted(_TORCH_ELU, elu)
_context_celu = _select_converted(_TORCH_CELU, celu)
_context_selu = _select_converted(_TORCH_SELU, selu)
_context_softplus = _select_converted(_TORCH_SOFTPLUS, softplus)
_context_mish = _select_converted(_TORCH_MISH, mish)
_context_hardswish = _select_converted(_TORCH_HARDSWISH, hardswish)
_context_dropout = _select_converted(_TORCH_DROPOUT, _packed_dropout)


_PATCHES = (
    # One bit per element.
    (F, "relu", _context_relu),
    (F, "relu6", _context_relu6),
    (F, "leaky_relu", _context_leaky_relu),
    (F, "hardtanh", _context_hardtanh),
    (F, "hardsigmoid", _context_hardsigmoid),
    (torch, "relu", _context_torch_relu),
    # Int8 derivative.
    (F, "gelu", _context_gelu),
    (F, "silu", _context_silu),
    (F, "elu", _context_elu),
    (F, "celu", _context_celu),
    (F, "selu", _context_selu),
    (F, "softplus", _context_softplus),
    (F, "mish", _context_mish),
    (F, "hardswish", _context_hardswish),
    # Its own mask.
    (F, "dropout", _context_dropout),
)
# `torch.sigmoid` and `torch.tanh` stay unpatched. They are general
# mathematical functions, not activation entry points: a gating term, a
# normalization, or torch's own internals may call them, and quantizing a
# derivative there would change results the caller never asked about. Use
# `qste.nn.Sigmoid` or `qste.nn.Tanh` where they are genuinely activations.


@contextmanager
def packed_activations():
    """Pack functional activations directly following converted QSTE layers.

    ``convert`` can only reach activations a model holds as submodules. Plenty
    of code writes ``F.gelu(x)`` inline instead, and those call sites cannot be
    found by walking the module tree. Converted layers tag their output tensors,
    and wrapping the forward lets these call sites consume that tag::

        with qste.packed_activations():
            loss = model(batch).mean()
        loss.backward()

    An activation whose input did not come directly from a converted layer is
    routed to the original PyTorch function unchanged. The patch is scoped and
    restored on the way out, including on an exception. In-place variants
    (``F.relu_``) are left alone, since they cannot retain what they overwrite.
    """

    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in _PATCHES]
    for owner, name, replacement in _PATCHES:
        setattr(owner, name, replacement)
    try:
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)


__all__ = [
    "Dropout",
    "EXACT_REPLACEMENTS",
    "GELU",
    "elementwise",
    "packed",
    "relu_square",
    "REPLACEMENTS",
    "ReLU",
    "ReLU6",
    "SMOOTH_REPLACEMENTS",
    "SiLU",
    "Sigmoid",
    "Tanh",
    "dropout",
    "gelu",
    "packed_activations",
    "relu",
    "relu6",
    "replace",
    "sigmoid",
    "silu",
    "tanh",
]
