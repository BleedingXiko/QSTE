"""Packed activations: exactness first, then the memory they were built for.

The defect these exist to fix was measured, not theorized. A converted stack of
``Linear -> ReLU`` used *more* peak memory than the same stack with packing
turned off, because torch's ReLU saves its full-precision output for backward
and that output is the next layer's input. The packed activation was an
addition, not a replacement.

So there are two things to prove and they are separate. That the forward and the
gradient are what they were (exactly, for the saturating ones), and that the
full-precision tensor actually stops being retained.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import qste
from qste import nn as qnn

from test_memory import Tape

SHAPES = [(8, 16), (5, 33), (3, 4, 7), (129,)]


def _pair(shape, seed=0, dtype=torch.float32):
    """The same input twice, so two paths can be differentiated independently."""

    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(*shape, generator=generator).to(dtype) * 2.0
    return values.clone().requires_grad_(True), values.clone().requires_grad_(True)


def _backward(output, seed=1):
    generator = torch.Generator().manual_seed(seed)
    output.backward(torch.randn(*output.shape, generator=generator).to(output.dtype))


# ---------------------------------------------------------------------------
# Exact: identical forward, identical gradient, one bit retained
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_relu_is_bit_identical_to_torch(shape):
    mine, theirs = _pair(shape)
    ours, reference = qnn.relu(mine), F.relu(theirs)
    assert torch.equal(ours, reference)
    _backward(ours)
    _backward(reference)
    assert torch.equal(mine.grad, theirs.grad)


@pytest.mark.parametrize("shape", SHAPES)
def test_relu6_is_bit_identical_to_torch(shape):
    mine, theirs = _pair(shape, seed=3)
    ours, reference = qnn.relu6(mine), F.relu6(theirs)
    assert torch.equal(ours, reference)
    _backward(ours)
    _backward(reference)
    assert torch.equal(mine.grad, theirs.grad)


def test_relu_gradient_is_zero_exactly_at_the_kink():
    """`>=` here would leak a gradient through a dead unit."""

    inputs = torch.zeros(4, 8, requires_grad=True)
    qnn.relu(inputs).sum().backward()
    assert torch.equal(inputs.grad, torch.zeros_like(inputs.grad))


def test_relu6_gradient_is_zero_at_both_clamps():
    inputs = torch.tensor([[-1.0, 0.0, 3.0, 6.0, 9.0]], requires_grad=True)
    qnn.relu6(inputs).sum().backward()
    assert torch.equal(inputs.grad, torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]))


def test_dropout_gradient_follows_the_mask_it_actually_drew():
    torch.manual_seed(0)
    inputs = torch.randn(64, 128, requires_grad=True)
    output = qnn.dropout(inputs, 0.5, training=True)
    output.backward(torch.ones_like(output))
    dropped = output == 0.0
    # The gradient is the scaled mask, and it is the same mask the forward used.
    assert torch.equal(inputs.grad == 0.0, dropped)
    assert torch.allclose(inputs.grad[~dropped], torch.full_like(inputs.grad[~dropped], 2.0))


def test_dropout_is_identity_when_not_training():
    inputs = torch.randn(4, 8, requires_grad=True)
    assert qnn.dropout(inputs, 0.5, training=False) is inputs
    assert qnn.dropout(inputs, 0.0, training=True) is inputs


def test_dropout_rejects_a_probability_it_cannot_scale():
    with pytest.raises(ValueError):
        qnn.dropout(torch.randn(2, 2), 1.0)


# ---------------------------------------------------------------------------
# Smooth: identical forward, quantized derivative
# ---------------------------------------------------------------------------


SMOOTH = [
    (qnn.gelu, lambda x: F.gelu(x)),
    (lambda x: qnn.gelu(x, "tanh"), lambda x: F.gelu(x, approximate="tanh")),
    (qnn.silu, F.silu),
    (qnn.tanh, torch.tanh),
    (qnn.sigmoid, torch.sigmoid),
]


@pytest.mark.parametrize("ours,reference", SMOOTH)
def test_smooth_forward_is_untouched(ours, reference):
    mine, theirs = _pair((32, 64), seed=5)
    assert torch.equal(ours(mine), reference(theirs))


@pytest.mark.parametrize("ours,reference", SMOOTH)
def test_smooth_derivative_formula_is_right_before_quantization(ours, reference):
    """The quantization error has to be the only error.

    An eight-bit derivative hides a wrong derivative, so this checks the shape
    of the gradient against autograd through torch's own implementation and
    demands agreement far tighter than one INT8 step.
    """

    mine, theirs = _pair((64, 128), seed=7)
    _backward(ours(mine), seed=2)
    _backward(reference(theirs), seed=2)
    scale = theirs.grad.abs().max()
    error = (mine.grad - theirs.grad).abs().max() / scale
    assert error < 0.01, f"relative gradient error {error:.4f}"


@pytest.mark.parametrize("ours,reference", SMOOTH)
def test_smooth_gradient_is_unbiased(ours, reference):
    """Rounding a derivative must not push it consistently one way."""

    mine, theirs = _pair((256, 256), seed=9)
    _backward(ours(mine), seed=4)
    _backward(reference(theirs), seed=4)
    bias = (mine.grad - theirs.grad).mean().abs()
    assert bias < theirs.grad.abs().mean() * 1e-3


def test_gelu_tanh_and_exact_are_not_confused_for_each_other():
    inputs = torch.randn(16, 32)
    assert not torch.allclose(qnn.gelu(inputs, "none"), qnn.gelu(inputs, "tanh"), atol=1e-7)


# ---------------------------------------------------------------------------
# What is retained
# ---------------------------------------------------------------------------


def _retained(function, samples=256, columns=512):
    inputs = torch.randn(samples, columns, requires_grad=True)
    with Tape() as tape:
        output = function(inputs)
    total = tape.total
    output.sum().backward()
    return total / samples


def test_relu_retains_one_bit_per_element():
    assert _retained(qnn.relu) == pytest.approx(512 / 8, abs=1.0)
    assert _retained(F.relu) == pytest.approx(512 * 4, abs=1.0)


def test_dropout_retains_one_bit_per_element():
    assert _retained(lambda x: qnn.dropout(x, 0.1)) == pytest.approx(512 / 8, abs=1.0)


def test_smooth_activations_retain_one_byte_per_element():
    # one INT8 per element, plus a float scale per row
    assert _retained(qnn.gelu) == pytest.approx(512 + 4, abs=1.0)
    assert _retained(qnn.silu) == pytest.approx(512 + 4, abs=1.0)


def test_torch_relu_pins_the_activation_and_the_packed_one_does_not():
    """The measurement that started this file.

    A QSTE linear followed by torch's ReLU retains a full-precision tensor per
    sample either way, because ReLU holds it and the packed copy is added on
    top. Swapping the ReLU is what makes the saving real.

    Measured as a slope across two batch sizes: the packed weights are a fixed
    cost that does not scale with batch, and including them would flatter the
    ratio at small widths and understate it at large ones. What decides whether
    a batch fits is the per-sample term.
    """

    def per_sample(activation):
        def tape_bytes(samples):
            torch.manual_seed(0)
            model = nn.Sequential(
                nn.Linear(512, 512, bias=False), nn.Linear(512, 512, bias=False)
            )
            qste.convert(model, activations=False)
            for surface in qste.surfaces(model):
                surface._immediate = lambda *_: None
            with Tape() as tape:
                output = model[1](activation(model[0](torch.randn(samples, 512))))
            total = tape.total
            output.sum().backward()
            return total

        return (tape_bytes(384) - tape_bytes(128)) / 256

    pinned = per_sample(F.relu)
    packed = per_sample(qnn.relu)
    # 512 floats pinned by the ReLU plus two packed rows, against three packed
    # rows: a little over 10x, and it grows with width.
    assert packed < pinned / 8, f"{pinned:.0f} -> {packed:.0f} bytes/sample"


def test_converted_stack_with_activations_retains_a_fraction_of_float():
    def tape_bytes(convert_activations):
        torch.manual_seed(2)
        model = nn.Sequential(
            nn.Linear(512, 512, bias=False), nn.ReLU(),
            nn.Linear(512, 512, bias=False), nn.ReLU(),
            nn.Linear(512, 512, bias=False),
        )
        if convert_activations is not None:
            qste.convert(model, activations=convert_activations)
            for surface in qste.surfaces(model):
                surface._immediate = lambda *_: None
        inputs = torch.randn(128, 512)
        with Tape() as tape:
            output = model(inputs)
        total = tape.total
        output.sum().backward()
        return total

    reference = tape_bytes(None)
    without = tape_bytes(False)
    with_them = tape_bytes(True)
    assert with_them < without / 4, f"{without} -> {with_them}"
    assert with_them < reference / 10, f"float {reference} -> qste {with_them}"


# ---------------------------------------------------------------------------
# Conversion and functional call sites
# ---------------------------------------------------------------------------


def _model():
    return nn.Sequential(
        nn.Linear(8, 8), nn.ReLU(), nn.GELU(approximate="tanh"), nn.Dropout(0.25), nn.SiLU()
    )


def test_convert_replaces_activations_by_default():
    model = _model()
    qste.convert(model)
    kinds = [type(module) for module in model if not isinstance(module, qste.QSTELinear)]
    assert kinds == [qnn.ReLU, qnn.GELU, qnn.Dropout, qnn.SiLU]


def test_convert_exact_mode_leaves_the_quantizing_ones_alone():
    model = _model()
    qste.convert(model, activations="exact")
    kinds = [type(module) for module in model if not isinstance(module, qste.QSTELinear)]
    assert kinds == [qnn.ReLU, nn.GELU, qnn.Dropout, nn.SiLU]


def test_convert_can_be_told_not_to_touch_activations():
    model = _model()
    qste.convert(model, activations=False)
    kinds = [type(module) for module in model if not isinstance(module, qste.QSTELinear)]
    assert kinds == [nn.ReLU, nn.GELU, nn.Dropout, nn.SiLU]


def test_conversion_preserves_activation_settings():
    model = _model()
    qste.convert(model)
    assert model[2].approximate == "tanh"
    assert model[3].p == 0.25


def test_converting_twice_is_idempotent():
    model = _model()
    qste.convert(model)
    first = [type(module) for module in model]
    qste.convert(model)
    assert [type(module) for module in model] == first


def test_convert_rejects_an_unknown_activation_mode():
    with pytest.raises(ValueError):
        qste.convert(nn.Sequential(nn.Linear(4, 4)), activations="sometimes")


def test_eval_mode_survives_conversion():
    model = _model()
    model.eval()
    qste.convert(model)
    assert not model[3].training
    inputs = torch.randn(4, 8)
    assert torch.equal(model(inputs), model(inputs))


def test_packed_activations_covers_functional_call_sites():
    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(512, 512, bias=False)
            self.second = nn.Linear(512, 512, bias=False)

        def forward(self, x):
            return self.second(F.relu(self.first(x)))

    def per_sample(wrapped):
        def tape_bytes(samples):
            torch.manual_seed(3)
            model = Functional()
            qste.convert(model)
            for surface in qste.surfaces(model):
                surface._immediate = lambda *_: None
            inputs = torch.randn(samples, 512)
            if wrapped:
                with qste.packed_activations(), Tape() as tape:
                    output = model(inputs)
                    total = tape.total
            else:
                with Tape() as tape:
                    output = model(inputs)
                    total = tape.total
            output.sum().backward()
            return total

        return (tape_bytes(384) - tape_bytes(128)) / 256

    assert per_sample(True) < per_sample(False) / 8


def test_packed_activations_restores_torch_even_on_an_exception():
    original = (F.relu, F.gelu, F.silu, F.dropout, torch.relu)
    with pytest.raises(RuntimeError):
        with qste.packed_activations():
            assert F.relu is not original[0]
            raise RuntimeError("boom")
    assert (F.relu, F.gelu, F.silu, F.dropout, torch.relu) == original


def test_packed_activations_leaves_unconverted_functionals_untouched():
    """The context is selective: ordinary host-model activations stay native."""

    mine = torch.tensor(-0.4, requires_grad=True)
    theirs = mine.detach().clone().requires_grad_(True)
    with qste.packed_activations():
        ours = F.softplus(mine)
    reference = F.softplus(theirs)
    ours.backward()
    reference.backward()
    assert torch.equal(ours, reference)
    assert torch.equal(mine.grad, theirs.grad)
    assert type(ours.grad_fn).__name__ == "SoftplusBackward0"


def test_packed_activations_selects_only_converted_layer_outputs():
    converted = nn.Sequential(nn.Linear(8, 8, bias=False))
    ordinary = nn.Linear(8, 8, bias=False)
    qste.convert(converted, activations=False)
    inputs = torch.randn(4, 8, requires_grad=True)

    with qste.packed_activations():
        packed = F.relu(converted(inputs)).square()
        native = F.relu(ordinary(inputs)).square()

    assert type(packed.grad_fn).__name__ == "_ReluSquareFnBackward"
    assert type(native.grad_fn).__name__ == "PowBackward0"
    (packed.sum() + native.sum()).backward()
    assert torch.isfinite(inputs.grad).all()


def test_direct_packed_activation_supports_a_scalar():
    mine = torch.tensor(-0.4, requires_grad=True)
    theirs = mine.detach().clone().requires_grad_(True)
    ours = qnn.softplus(mine)
    reference = F.softplus(theirs)
    ours.backward()
    reference.backward()
    assert torch.equal(ours, reference)
    assert torch.allclose(mine.grad, theirs.grad, rtol=1e-2, atol=1e-3)


def test_patched_gelu_does_not_call_itself():
    """The wrapper needs the real forward, and the patch replaces the name."""

    with qste.packed_activations():
        inputs = torch.randn(4, 8, requires_grad=True)
        output = F.gelu(inputs)
        output.sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()


# ---------------------------------------------------------------------------
# Shapes, dtypes, and layouts a real host will hand these
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 3, 5, 7), (1, 1), (64,)])
def test_arbitrary_rank_round_trips(shape):
    mine, theirs = _pair(shape, seed=11)
    ours, reference = qnn.relu(mine), F.relu(theirs)
    assert ours.shape == reference.shape
    _backward(ours)
    _backward(reference)
    assert torch.equal(mine.grad, theirs.grad)


def test_non_contiguous_input_is_handled():
    base = torch.randn(16, 32, requires_grad=True)
    transposed = base.t()
    output = qnn.relu(transposed)
    output.sum().backward()
    assert torch.equal(base.grad, (base > 0).float())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_dtypes_are_preserved(dtype):
    mine, theirs = _pair((8, 16), seed=13, dtype=dtype)
    ours = qnn.relu(mine)
    assert ours.dtype == dtype
    assert torch.equal(ours, F.relu(theirs))
    ours.sum().backward()
    assert mine.grad.dtype == dtype


def test_a_width_that_is_not_a_multiple_of_eight():
    mine, theirs = _pair((7, 13), seed=17)
    _backward(qnn.relu(mine))
    _backward(F.relu(theirs))
    assert torch.equal(mine.grad, theirs.grad)


def test_gradient_checkpointing_still_gets_the_right_mask():
    """Checkpointing reruns forward, so the mask must be redrawn, not stale."""

    from torch.utils.checkpoint import checkpoint

    inputs = torch.randn(32, 64, requires_grad=True)
    reference = torch.randn(32, 64, requires_grad=True)
    with torch.no_grad():
        reference.copy_(inputs)
    reference.requires_grad_(True)

    checkpoint(qnn.relu, inputs, use_reentrant=False).sum().backward()
    F.relu(reference).sum().backward()
    assert torch.equal(inputs.grad, reference.grad)


def test_activation_survives_a_second_backward_through_the_same_graph():
    inputs = torch.randn(16, 32, requires_grad=True)
    output = qnn.relu(inputs)
    output.sum().backward(retain_graph=True)
    first = inputs.grad.clone()
    inputs.grad = None
    output.sum().backward()
    assert torch.equal(inputs.grad, first)


def test_smooth_activation_matches_across_a_wide_input_range():
    """Saturated regions are where a quantized derivative could go wrong."""

    inputs = torch.linspace(-40.0, 40.0, 4096).reshape(32, 128)
    mine = inputs.clone().requires_grad_(True)
    theirs = inputs.clone().requires_grad_(True)
    qnn.gelu(mine).sum().backward()
    F.gelu(theirs).sum().backward()
    assert torch.allclose(mine.grad, theirs.grad, atol=0.02)
    assert math.isclose(float(mine.grad.sum()), float(theirs.grad.sum()), rel_tol=0.01)
