"""Gradients must be the ones a float linear over the same signs would give.

QSTE's approximation lives in exactly one place -- what forward retains for
the evidence outer product -- and nowhere else. The input gradient, the scale
gradient, and the bias gradient are exact no matter what forward retained,
and these tests pin that down rather than trusting it.

``exact_evidence`` is the diagnostic hook that lets a test run the identical
arithmetic with the activation kept in full, so "the packed path agrees with
the exact one" is a measurement here and not an assertion.
"""

import pytest
import torch
import torch.nn as nn

import qste
from qste import kernels
from qste.functional import exact_evidence, qste_linear, qste_weight
from qste.surface import Surface

STORAGE = ["fp32", "int8", "bit"]


def _surface(rows=24, columns=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, columns, generator=generator) / columns**0.5
    return Surface(weight)


def _dense(surface):
    """The float linear this surface is exactly equivalent to."""

    return kernels.unpack_rows(surface.packed_sign.data, surface.columns) * surface.scale.detach().unsqueeze(1)


@pytest.mark.parametrize("storage", STORAGE)
def test_forward_equals_dense_linear(storage):
    """Forward never looks at the retained activation at all."""

    with exact_evidence(storage):
        surface = _surface()
        inputs = torch.randn(7, surface.columns)
        bias = torch.randn(surface.rows)
        expected = inputs @ _dense(surface).t() + bias
        assert torch.allclose(qste_linear(inputs, surface, bias), expected, atol=1e-4)


@pytest.mark.parametrize("storage", STORAGE)
def test_input_gradient_is_exact(storage):
    """grad_input never touches the retained activation either."""

    with exact_evidence(storage):
        surface = _surface(seed=1)
        inputs = torch.randn(9, surface.columns, requires_grad=True)
        reference = inputs.detach().clone().requires_grad_(True)

        qste_linear(inputs, surface).square().sum().backward()
        (reference @ _dense(surface).t()).square().sum().backward()
        assert torch.allclose(inputs.grad, reference.grad, atol=1e-4)


@pytest.mark.parametrize("storage", STORAGE)
def test_bias_gradient_is_exact(storage):
    with exact_evidence(storage):
        surface = _surface(seed=2)
        bias = torch.zeros(surface.rows, requires_grad=True)
        inputs = torch.randn(5, surface.columns)
        grad_output = torch.randn(5, surface.rows)
        (qste_linear(inputs, surface, bias) * grad_output).sum().backward()
        assert torch.allclose(bias.grad, grad_output.sum(0), atol=1e-5)


@exact_evidence()
def test_scale_gradient_is_exact_when_the_activation_is_kept():
    surface = _surface(seed=3)
    inputs = torch.randn(6, surface.columns)
    grad_output = torch.randn(6, surface.rows)

    (qste_linear(inputs, surface) * grad_output).sum().backward()
    signs = kernels.unpack_rows(surface.packed_sign.data, surface.columns)
    projection = inputs @ signs.t()
    expected_scale = (grad_output * projection).sum(0)
    # d scale / d log_scale = scale
    expected = expected_scale * surface.scale.detach()
    assert torch.allclose(surface.log_scale.grad, expected, atol=1e-3)


@exact_evidence()
def test_evidence_is_the_exact_outer_product_when_the_activation_is_kept():
    surface = _surface(seed=4)
    captured = []
    surface._immediate = lambda _, evidence: captured.append(evidence.clone())

    inputs = torch.randn(6, surface.columns)
    grad_output = torch.randn(6, surface.rows)
    (qste_linear(inputs, surface) * grad_output).sum().backward()

    expected = (grad_output.t() @ inputs) * surface.scale.detach().unsqueeze(1)
    assert torch.allclose(captured[0], expected, atol=1e-3)


def test_bit_evidence_keeps_the_pairing_not_just_the_sum():
    """The failure mode this design exists to avoid.

    Reducing either operand before the outer product -- summing grad over the
    batch, say -- loses which input caused which gradient. Bit storage reduces
    the *precision* of one operand and keeps the pairing, so its evidence still
    correlates strongly with the exact outer product. A batch-summed rank-one
    surrogate does not.
    """

    surface = _surface(rows=32, columns=64, seed=5)
    captured = []
    surface._immediate = lambda _, evidence: captured.append(evidence.clone())

    generator = torch.Generator().manual_seed(6)
    inputs = torch.randn(128, surface.columns, generator=generator)
    grad_output = torch.randn(128, surface.rows, generator=generator)
    (qste_linear(inputs, surface) * grad_output).sum().backward()

    exact = (grad_output.t() @ inputs) * surface.scale.detach().unsqueeze(1)
    rank_one = torch.outer(grad_output.sum(0), inputs.sum(0)) * surface.scale.detach().unsqueeze(1)

    def cosine(a, b):
        return float(
            torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
        )

    assert cosine(captured[0], exact) > 0.75
    assert cosine(rank_one, exact) < 0.2


@pytest.mark.parametrize("storage", STORAGE)
def test_evidence_sign_agreement_with_exact(storage):
    """Coordinate updates only need the direction to be right most of the time."""

    with exact_evidence(storage):
        surface = _surface(rows=32, columns=64, seed=7)
        captured = []
        surface._immediate = lambda _, evidence: captured.append(evidence.clone())

        generator = torch.Generator().manual_seed(8)
        inputs = torch.randn(256, surface.columns, generator=generator)
        grad_output = torch.randn(256, surface.rows, generator=generator)
        (qste_linear(inputs, surface) * grad_output).sum().backward()

    exact = (grad_output.t() @ inputs) * surface.scale.detach().unsqueeze(1)
    agreement = (captured[0].sign() == exact.sign()).float().mean()
    floor = {"fp32": 0.999, "int8": 0.95, "bit": 0.70}[storage]
    assert float(agreement) > floor


def test_the_shipped_path_is_the_packed_one():
    """No config can turn the memory saving off; only the diagnostic can."""

    import qste.functional as functional

    assert functional._STORAGE == "bit"
    assert not hasattr(qste.QSTEConfig(), "evidence_storage")
    with exact_evidence():
        assert functional._STORAGE == "fp32"
    assert functional._STORAGE == "bit"


def test_weight_property_trains_the_coordinate():
    """A host framework that reads ``.weight`` gets real coordinate evidence.

    This is the compatibility guarantee that makes the library adoptable by a
    framework that fuses projections by hand instead of calling the modules.
    """

    surface = _surface(rows=16, columns=24, seed=9)
    captured = []
    surface._immediate = lambda _, evidence: captured.append(evidence.clone())

    inputs = torch.randn(5, surface.columns)
    grad_output = torch.randn(5, surface.rows)
    weight = qste_weight(surface)
    (torch.nn.functional.linear(inputs, weight) * grad_output).sum().backward()

    expected = (grad_output.t() @ inputs) * surface.scale.detach().unsqueeze(1)
    assert captured, "reading .weight produced no coordinate evidence"
    assert torch.allclose(captured[0], expected, atol=1e-3)


@exact_evidence()
def test_weight_path_and_module_path_agree():
    surface = _surface(rows=16, columns=24, seed=10)
    inputs = torch.randn(5, surface.columns)
    grad_output = torch.randn(5, surface.rows)

    through_module = []
    surface._immediate = lambda _, evidence: through_module.append(evidence.clone())
    (qste_linear(inputs, surface) * grad_output).sum().backward()
    scale_grad_module = surface.log_scale.grad.clone()
    surface.log_scale.grad = None

    through_weight = []
    surface._immediate = lambda _, evidence: through_weight.append(evidence.clone())
    weight = qste_weight(surface)
    (torch.nn.functional.linear(inputs, weight) * grad_output).sum().backward()

    assert torch.allclose(through_module[0], through_weight[0], atol=1e-3)
    assert torch.allclose(scale_grad_module, surface.log_scale.grad, atol=1e-3)


@exact_evidence()
def test_tied_surface_sums_its_uses():
    """Autograd requires the sum over uses, never one use or their average."""

    surface = _surface(rows=12, columns=16, seed=11)
    captured = []
    surface._immediate = lambda _, evidence: captured.append(evidence.clone())

    inputs_a = torch.randn(4, surface.columns)
    inputs_b = torch.randn(4, surface.columns)
    grad_a = torch.randn(4, surface.rows)
    grad_b = torch.randn(4, surface.rows)
    output = (qste_linear(inputs_a, surface) * grad_a).sum() + (
        qste_linear(inputs_b, surface) * grad_b
    ).sum()
    output.backward()

    expected = ((grad_a.t() @ inputs_a) + (grad_b.t() @ inputs_b)) * surface.scale.detach().unsqueeze(1)
    assert len(captured) == 1, "a tied surface must apply once, after all uses"
    assert torch.allclose(captured[0], expected, atol=1e-2)


@exact_evidence()
def test_embedding_forward_and_evidence():
    surface = _surface(rows=40, columns=16, seed=12)
    captured = []
    surface._immediate = lambda _, evidence: captured.append(evidence.clone())

    ids = torch.randint(0, 40, (3, 5))
    grad_output = torch.randn(3, 5, 16)
    output = qste.qste_embedding(ids, surface)
    assert torch.allclose(output, _dense(surface)[ids], atol=1e-5)

    (output * grad_output).sum().backward()
    expected = torch.zeros(40, 16)
    expected.index_add_(0, ids.reshape(-1), grad_output.reshape(-1, 16))
    expected.mul_(surface.scale.detach().unsqueeze(1))
    assert torch.allclose(captured[0], expected, atol=1e-4)


@pytest.mark.parametrize("storage", STORAGE)
def test_no_nans_on_degenerate_inputs(storage):
    surface = _surface(seed=13)
    captured = []
    surface._immediate = lambda _, evidence: captured.append(evidence.clone())
    inputs = torch.zeros(4, surface.columns, requires_grad=True)
    with exact_evidence(storage):
        qste_linear(inputs, surface).sum().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(captured[0]).all()
    assert torch.isfinite(surface.log_scale.grad).all()
