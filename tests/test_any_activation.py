"""Activations nobody wrote down.

The tables in ``qste.nn`` are an optimization -- a closed-form derivative is
cheaper than asking for one, and a piecewise-linear activation packs to a bit
rather than a byte. They are not the mechanism, and a library whose coverage is
a list can only be agnostic about the architectures somebody already thought of.

Whatever ``f`` is, its backward multiplies by ``f'(x)``, and ``f'(x)`` comes
from running ``f`` once on a detached input and asking autograd. Quantize that
and the retained tape is one byte per element for any elementwise ``f`` at all,
including one invented after this file was written.

Every expression here is deliberately absent from every table.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import qste
from qste import nn as qnn

UNKNOWN = {
    "relu_cubed": lambda t: F.relu(t) ** 3,
    "mish_by_hand": lambda t: t * torch.tanh(F.softplus(t)),
    "sin_gated": lambda t: torch.sin(t) * torch.sigmoid(2 * t),
    "gelu_with_residual": lambda t: F.gelu(t) + 0.1 * t,
    "squared_relu": lambda t: F.relu(t).square(),
    "clamped_swish": lambda t: (t * torch.sigmoid(t)).clamp(-1.0, 4.0),
}


def _retained(build):
    """Bytes still alive once the forward returns.

    Weak references on purpose. A plain count includes the probe forward's own
    intermediates, which ``autograd.grad`` frees immediately -- counting those
    made a 4x saving read as 0.8x, and sent me looking for a bug in the packer
    that was really a bug in the ruler.
    """

    refs: list[tuple] = []

    def pack(tensor):
        refs.append((weakref.ref(tensor), tensor.numel() * tensor.element_size()))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        output = build()
    gc.collect()
    total = sum(size for ref, size in refs if ref() is not None)
    # `output` has to stay alive across the sum: it owns the graph, the graph
    # owns the saved tensors, and dropping it first makes every weak reference
    # dead and every measurement zero -- for both sides, so the comparison
    # looks like a failure of the packer rather than of the ruler.
    del output
    return total


@pytest.mark.parametrize("name", sorted(UNKNOWN))
def test_forward_is_untouched(name):
    function = UNKNOWN[name]
    x = torch.randn(64, 128, dtype=torch.double)
    assert torch.equal(function(x), qnn.elementwise(function, x))


@pytest.mark.parametrize("name", sorted(UNKNOWN))
def test_gradient_is_within_quantization(name):
    function = UNKNOWN[name]
    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.double) * 2
    upstream = torch.randn_like(x)

    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    function(a).backward(upstream)
    qnn.elementwise(function, b).backward(upstream)

    error = ((a.grad - b.grad).abs().max() / a.grad.abs().max()).item()
    assert error < 0.02, f"{name}: {error:.2e}"
    similarity = F.cosine_similarity(a.grad.flatten(), b.grad.flatten(), dim=0)
    assert similarity.item() > 0.999, name


@pytest.mark.parametrize("name", sorted(UNKNOWN))
def test_it_retains_less_than_torch(name):
    function = UNKNOWN[name]
    x = torch.randn(256, 512, requires_grad=True)
    plain = _retained(lambda: function(x))
    packed = _retained(lambda: qnn.elementwise(function, x))
    assert plain / packed > 3.0, f"{name}: {plain / packed:.1f}x"


def test_a_non_elementwise_function_is_refused():
    """``grad(f(x).sum(), x)`` is ``f'(x)`` only when each output depends on its
    own input alone. A softmax would silently get the row's sum instead, so the
    shape is checked rather than the caller trusted."""

    x = torch.randn(8, 16, requires_grad=True)
    with pytest.raises(ValueError, match="elementwise"):
        qnn.elementwise(lambda t: t.sum(dim=-1), x)


def test_packed_wraps_a_callable_for_reuse():
    activation = qnn.packed(lambda t: F.relu(t) ** 3)
    x = torch.randn(32, 64, dtype=torch.double)
    assert torch.equal(activation(x), F.relu(x) ** 3)


def test_a_model_with_an_unheard_of_activation_learns():
    torch.manual_seed(0)
    exotic = qnn.packed(lambda t: torch.sin(t) * torch.sigmoid(2 * t))

    class Block(nn.Module):
        def __init__(self, width=64, ratio=4):
            super().__init__()
            self.fc1 = nn.Linear(width, width * ratio, bias=False)
            self.fc2 = nn.Linear(width * ratio, width, bias=False)

        def forward(self, x):
            return x + self.fc2(exotic(self.fc1(x)))

    model = nn.Sequential(nn.Linear(32, 64), Block(), Block(), nn.Linear(64, 8))
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model)
    floats = torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=3e-3)

    inputs = torch.randn(256, 32)
    targets = torch.randint(0, 8, (256,))
    losses = []
    for _ in range(150):
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        floats.step()
        coordinates.step()
        floats.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * 0.7, losses[-1]
