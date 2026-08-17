"""Every activation QSTE claims, checked against torch's own gradient.

The framework's claim is that any architecture can adopt it. Five activations
is not any architecture -- SwiGLU is the modern default, mobile stacks are
hard-sigmoid and hard-swish, half of vision is leaky ReLU, and a recurrent core
is as likely to gate with softplus as with anything else. So the two mechanisms
that were already here are table-driven now, and the tables are filled in.

Both mechanisms retain something smaller than the tensor torch would keep:

  one bit per element, for anything piecewise linear -- the retained fact is
  which side of the knee an element fell on, and that is a bit whatever the
  slopes on either side are;

  an int8 derivative with a per-row scale, for anything smooth -- the retained
  fact is a number, and eight bits of it carry the gradient to about a quarter
  of a percent.

This file is the guard on both. A wrong derivative is the worst failure mode
this library has: the forward stays exact, the model still trains, and it
trains somewhere slightly wrong for reasons no loss curve will ever explain.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import qste
from qste import nn as qnn

# name, torch callable, qste callable, kwargs, tolerance on the gradient.
#
# Three tiers, and the middle one is not a fudge. A pure 0/1 mask reproduces
# torch bit for bit because there is no arithmetic to round. A mask carrying a
# slope multiplies by it, which rounds -- and for hard sigmoid torch is the one
# that rounds: its backward hardcodes 1/6 as a float32 constant, so on a double
# input torch says 0.166666671634 and this says 0.166666666667. The knees land
# in identical places, which is the claim that matters. An int8 derivative
# carries 127 levels, so ~0.8% worst case on the multiplier.
EXACT, SLOPE, QUANTIZED = 0.0, 1e-6, 0.02
CASES = [
    ("relu", F.relu, qnn.relu, {}, EXACT),
    ("relu6", F.relu6, qnn.relu6, {}, EXACT),
    ("hardtanh", F.hardtanh, qnn.hardtanh, {}, EXACT),
    ("leaky_relu", F.leaky_relu, qnn.leaky_relu, {"negative_slope": 0.1}, SLOPE),
    ("hardsigmoid", F.hardsigmoid, qnn.hardsigmoid, {}, SLOPE),
    ("gelu", F.gelu, qnn.gelu, {}, QUANTIZED),
    ("silu", F.silu, qnn.silu, {}, QUANTIZED),
    ("elu", F.elu, qnn.elu, {"alpha": 1.3}, QUANTIZED),
    ("celu", F.celu, qnn.celu, {"alpha": 1.3}, QUANTIZED),
    ("selu", F.selu, qnn.selu, {}, QUANTIZED),
    ("softplus", F.softplus, qnn.softplus, {"beta": 1.4}, QUANTIZED),
    ("mish", F.mish, qnn.mish, {}, QUANTIZED),
    ("hardswish", F.hardswish, qnn.hardswish, {}, QUANTIZED),
]


def _inputs(seed=0):
    # Wide enough to straddle every knee: -3, 0, 3 and 6 all fall inside.
    torch.manual_seed(seed)
    return torch.randn(64, 128, dtype=torch.double) * 3


@pytest.mark.parametrize("name,reference,ours,kwargs,tolerance", CASES)
def test_forward_is_not_approximated(name, reference, ours, kwargs, tolerance):
    """The forward is torch's, bit for bit. Only the retained tape changes."""

    x = _inputs()
    assert torch.equal(reference(x, **kwargs), ours(x, **kwargs)), name


@pytest.mark.parametrize("name,reference,ours,kwargs,tolerance", CASES)
def test_gradient_matches_torch(name, reference, ours, kwargs, tolerance):
    x = _inputs()
    upstream = torch.randn_like(x)

    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    reference(a, **kwargs).backward(upstream)
    ours(b, **kwargs).backward(upstream)

    error = ((a.grad - b.grad).abs().max() / upstream.abs().max()).item()
    if tolerance == EXACT:
        assert torch.equal(a.grad, b.grad), f"{name}: {error:.2e}"
    else:
        assert error < tolerance, f"{name}: {error:.2e}"


PIECEWISE = [case for case in CASES if case[4] != QUANTIZED]


@pytest.mark.parametrize("name,reference,ours,kwargs,tolerance", PIECEWISE)
def test_the_knees_are_in_the_same_place(name, reference, ours, kwargs, tolerance):
    """Which elements get no gradient at all, which is what a knee decides.

    Separate from the value check because it is the part that cannot be a
    tolerance. A slope one ulp out is rounding; a knee one element out is a
    different function, and it would hide inside any tolerance loose enough to
    admit torch's float32 1/6.

    Piecewise only. A smooth activation has no knees, and quantizing its
    derivative to int8 rounds the smallest multipliers to exactly zero -- so a
    zero-pattern comparison there tests the rounding, not the function.
    """

    x = _inputs(seed=3)
    upstream = torch.ones_like(x)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    reference(a, **kwargs).backward(upstream)
    ours(b, **kwargs).backward(upstream)
    assert torch.equal(a.grad == 0, b.grad == 0), name


@pytest.mark.parametrize("name,reference,ours,kwargs,tolerance", CASES)
def test_it_actually_retains_less(name, reference, ours, kwargs, tolerance):
    """The whole point, measured rather than asserted.

    Resolved through ``F`` inside the context on purpose: binding the function
    before entering it keeps the unpatched one, which is how a first version of
    this measurement reported 1.0x for everything except ReLU -- and ReLU only
    escaped because ``F.relu`` reaches ``torch.relu``, which is patched
    separately.
    """

    source = nn.Sequential(nn.Linear(512, 512, bias=False))
    qste.convert(source, activations=False)
    x = source(torch.randn(256, 512, requires_grad=True))

    def retained(context):
        seen: dict[int, int] = {}
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (seen.__setitem__(id(t), t.numel() * t.element_size()), t)[1],
            lambda t: t,
        ), context:
            getattr(F, name)(x, **kwargs)
        return sum(seen.values())

    plain = retained(nullcontext())
    packed = retained(qste.packed_activations())
    floor = 3.0 if tolerance == QUANTIZED else 24.0  # 4x and 32x
    assert plain / packed > floor, f"{name}: {plain / packed:.1f}x"


def test_every_patched_function_has_a_module_and_the_reverse():
    """A framework that writes ``nn.GELU()`` and one that writes ``F.gelu`` get
    the same treatment, or the claim of being architecture-agnostic is a claim
    about coding style."""

    patched = {name for owner, name, _ in qnn._PATCHES if owner is F}
    patched -= {"dropout"}  # its module is covered but it is not an activation
    modules = {
        target.__name__.lower() for target in qnn.REPLACEMENTS.values()
    }
    missing = sorted(
        name for name in patched if name.replace("_", "") not in modules
    )
    assert not missing, f"functional-only, no module equivalent: {missing}"


def test_replace_carries_constructor_arguments():
    """A LeakyReLU with a custom slope must not silently become the default."""

    for module, attribute, value in [
        (nn.LeakyReLU(0.2), "negative_slope", 0.2),
        (nn.ELU(alpha=1.7), "alpha", 1.7),
        (nn.CELU(alpha=0.4), "alpha", 0.4),
        (nn.Hardtanh(-2.0, 2.0), "min_val", -2.0),
        (nn.Softplus(beta=2.0), "beta", 2.0),
        (nn.Dropout(0.3), "p", 0.3),
    ]:
        swapped = qnn.replace(module)
        assert swapped is not None, type(module).__name__
        assert getattr(swapped, attribute) == value, type(module).__name__


def test_a_converted_model_with_every_activation_still_learns():
    """End to end, one of each, gradients flowing through all of them."""

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(32, 64), nn.LeakyReLU(0.1),
        nn.Linear(64, 64), nn.ELU(),
        nn.Linear(64, 64), nn.Hardswish(),
        nn.Linear(64, 8),
    )
    qste.convert(model)
    optimizer = qste.QSTEOptimizer(model)
    floats = torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=3e-3)

    inputs = torch.randn(256, 32)
    targets = torch.randint(0, 8, (256,))
    losses = []
    for _ in range(120):
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()
        floats.step()
        optimizer.step()
        floats.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * 0.7, losses[-1]


# ---------------------------------------------------------------------------
# Squared ReLU
# ---------------------------------------------------------------------------


def test_relu_square_forward_is_exact():
    x = _inputs(seed=7)
    assert torch.equal(F.relu(x).square(), qnn.relu_square(x))


def test_relu_square_gradient_is_within_quantization():
    """Against the gradient it approximates, not against the upstream.

    Dividing the error by the upstream gradient rather than by the gradient
    being approximated made this look like a 2.3% method when it is a 0.24%
    one -- the two differ by the factor the derivative itself carries.
    """

    x = _inputs(seed=7)
    upstream = torch.randn_like(x)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    F.relu(a).square().backward(upstream)
    qnn.relu_square(b).backward(upstream)

    error = (a.grad - b.grad).abs()
    assert (error.max() / a.grad.abs().max()).item() < 0.01
    assert (error.sum() / a.grad.abs().sum()).item() < 0.01
    similarity = F.cosine_similarity(a.grad.flatten(), b.grad.flatten(), dim=0)
    assert similarity.item() > 0.9999


def test_relu_square_zeroes_nothing_that_matters():
    """Quantization rounds the smallest multipliers to zero. Bound the damage.

    Elements just above the knee have a derivative near zero, and a per-row
    scale sends them to exactly zero. That is inherent and it is fine -- what
    would not be fine is those elements carrying real gradient, so the test is
    on the mass they represent rather than on their count.
    """

    x = _inputs(seed=11)
    upstream = torch.randn_like(x)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    F.relu(a).square().backward(upstream)
    qnn.relu_square(b).backward(upstream)

    lost = (a.grad != 0) & (b.grad == 0)
    assert (a.grad[lost].abs().sum() / a.grad.abs().sum()).item() < 1e-3
    # And nothing gains a gradient it should not have: below the knee is zero.
    assert torch.equal((a.grad == 0) & (x <= 0), (b.grad == 0) & (x <= 0))


def test_relu_square_uses_an_unsigned_byte():
    """relu(x) is never negative, so a signed byte wastes a bit on a constant.

    255 levels rather than 127, for the same memory. If this ever regresses to
    int8 the error doubles silently, so the retained dtype is asserted.
    """

    x = _inputs(seed=13).float().requires_grad_(True)
    kept = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: (kept.append(t), t)[1], lambda t: t
    ):
        qnn.relu_square(x)
    assert torch.uint8 in {t.dtype for t in kept}, [t.dtype for t in kept]


def test_relu_square_retains_less_than_the_two_call_form():
    """The whole reason it exists: written as two calls, packing cannot help."""

    def retained(body):
        x = torch.randn(256, 512, requires_grad=True)
        seen: dict[int, int] = {}
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (seen.__setitem__(id(t), t.numel() * t.element_size()), t)[1],
            lambda t: t,
        ):
            body(x)
        return sum(seen.values())

    two_calls = retained(lambda x: F.relu(x).square())
    fused = retained(qnn.relu_square)
    assert two_calls / fused > 3.5, f"{two_calls / fused:.2f}x"


def test_a_model_using_relu_square_learns():
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self, width=64, ratio=4):
            super().__init__()
            self.fc1 = nn.Linear(width, width * ratio, bias=False)
            self.fc2 = nn.Linear(width * ratio, width, bias=False)

        def forward(self, x):
            return x + self.fc2(qnn.relu_square(self.fc1(x)))

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


# ---------------------------------------------------------------------------
# Fusing relu(x).square() without the model's help
# ---------------------------------------------------------------------------


def _fusable_cases():
    return [
        ("square()", lambda t: F.relu(t).square(), True),
        ("** 2", lambda t: F.relu(t) ** 2, True),
        ("plain relu", lambda t: F.relu(t), False),
        ("relu * 3", lambda t: F.relu(t) * 3, False),
        ("relu + 1", lambda t: F.relu(t) + 1, False),
        ("relu ** 3", lambda t: F.relu(t) ** 3, False),
        ("relu.sum()", lambda t: F.relu(t).sum(), False),
    ]


@pytest.mark.parametrize("name,body,fused", _fusable_cases())
def test_the_fusable_relu_is_a_tensor_everywhere_else(name, body, fused):
    """Squaring it fuses. Anything else must behave as ReLU's output did.

    A subclass that leaks changes the meaning of code it was supposed to be
    invisible to, so the check is on every other shape a ReLU output turns up
    in -- scaled, offset, cubed, reduced -- not only on the one being fused.
    """

    x = _inputs(seed=5)
    upstream_ref = body(x.clone().requires_grad_(True))
    grad_seed = torch.randn_like(upstream_ref)

    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    body(a).backward(grad_seed)
    with qste.packed_activations():
        out = body(b)
        # Anything DERIVED from a ReLU output must be an ordinary tensor, so
        # the wrapper cannot spread through a model. A bare `F.relu(x)` is the
        # documented exception: it is the wrapper itself, and isinstance holds.
        assert isinstance(out, torch.Tensor), name
        if name != "plain relu":
            assert type(out) is torch.Tensor, f"{name} leaked {type(out).__name__}"
        out.backward(grad_seed)

    assert torch.equal(body(x), out.detach()), name
    error = ((a.grad - b.grad).abs().max() / a.grad.abs().max()).item()
    if fused:
        assert error < 0.01, f"{name}: {error:.2e}"
    else:
        assert torch.equal(a.grad, b.grad), f"{name}: {error:.2e}"


def test_fusing_needs_no_change_to_the_model():
    """The measurement this exists for, on the expression dabsn actually has.

    Not `relu_square(...)` -- the two-call form, written by a model that knows
    nothing about this library, wrapped only in the context manager.
    """

    def retained(context):
        torch.manual_seed(0)
        model = nn.Sequential()
        model.fc1 = nn.Linear(256, 1536, bias=False)
        model.fc2 = nn.Linear(1536, 256, bias=False)
        qste.convert(model)
        x = torch.randn(256, 256)
        seen: dict[int, int] = {}
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (seen.__setitem__(id(t), t.numel() * t.element_size()), t)[1],
            lambda t: t,
        ), context:
            model.fc2(F.relu(model.fc1(x)).square())
        return sum(seen.values())

    assert retained(nullcontext()) / retained(qste.packed_activations()) > 2.0


def test_a_model_whose_source_never_changes_still_learns():
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self, width=64, ratio=4):
            super().__init__()
            self.fc1 = nn.Linear(width, width * ratio, bias=False)
            self.fc2 = nn.Linear(width * ratio, width, bias=False)

        def forward(self, x):
            # Exactly how dabsn writes it. Untouched.
            return x + self.fc2(F.relu(self.fc1(x)).square())

    model = nn.Sequential(nn.Linear(32, 64), Block(), Block(), nn.Linear(64, 8))
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model)
    floats = torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=3e-3)

    inputs = torch.randn(256, 32)
    targets = torch.randint(0, 8, (256,))
    losses = []
    for _ in range(150):
        with qste.packed_activations():
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
        floats.step()
        coordinates.step()
        floats.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * 0.7, losses[-1]


def test_a_bare_relu_output_saves_and_loads_as_an_ordinary_tensor():
    """A checkpoint must not acquire a dependency on this library.

    Pickling a subclass records the class, so a checkpoint holding a bare ReLU
    output would need qste importable to load -- and would fail on the restore,
    which is the worst place to find out. It reduces to a plain tensor.
    """

    import io

    x = torch.randn(8, 16)
    with qste.packed_activations():
        bare = F.relu(x)

    buffer = io.BytesIO()
    torch.save(bare, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert type(loaded) is torch.Tensor
    assert torch.equal(loaded, F.relu(x))


def test_nothing_escapes_a_realistic_block():
    """The case that actually matters: a ReLU consumed by the next layer.

    The wrapper exists for exactly one expression and is consumed by it. A
    model has to end on a bare ReLU to ever hold one, and none of these do.
    """

    torch.manual_seed(0)
    model = nn.Sequential()
    model.f1 = nn.Linear(16, 96, bias=False)
    model.f2 = nn.Linear(96, 16, bias=False)
    qste.convert(model)
    x = torch.randn(8, 16)

    with qste.packed_activations():
        squared = model.f2(F.relu(model.f1(x)).square())
        plain_relu = model.f2(F.relu(model.f1(x)))
        gated = model.f2(F.relu(model.f1(x)) * 2)

    for name, out in [("squared", squared), ("relu", plain_relu), ("gated", gated)]:
        assert type(out) is torch.Tensor, f"{name} returned {type(out).__name__}"
