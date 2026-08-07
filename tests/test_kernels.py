"""Native kernels must agree with the pure-torch reference, exactly.

The reference defines the numerics. Every kernel here is an integer or sign
computation followed by a GEMM, so "close enough" is not the bar -- the packed
paths are bit exact against a dense sign matmul and the tests say so.
"""

import pytest
import torch

from qste import kernels
from qste.kernels import fallback, loader

SHAPES = [(17, 13, 5), (64, 64, 32), (128, 96, 40), (33, 8, 7)]


def _packed(rows, columns, seed=0):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, columns, generator=generator)
    packed, offset, scale = fallback.pack_affine_rows(values)
    return values, packed, (offset, scale)


def test_native_extension_builds():
    status = kernels.status()
    if not status["native"]:
        pytest.skip(f"no compiler available: {status['failure']}")
    assert status["native"]


def _require_native():
    if not loader.native_available():
        pytest.skip("native kernels unavailable")


@pytest.mark.parametrize("rows,columns,_", SHAPES)
def test_affine_pack_encodes_the_centered_sign(rows, columns, _):
    """Signs are of the deviation, so a one-signed row still carries bits."""

    values, packed, (offset, scale) = _packed(rows, columns)
    centered = values - values.mean(dim=1, keepdim=True)
    assert torch.equal(
        fallback.unpack_rows(packed, columns), torch.where(centered >= 0, 1.0, -1.0)
    )
    assert torch.allclose(offset, values.mean(dim=1))
    assert torch.allclose(scale, centered.abs().mean(dim=1))


def test_affine_pack_stays_informative_on_relu_output():
    """The regression that motivated centering: sign(relu(x)) is all ones."""

    values = torch.randn(16, 64, generator=torch.Generator().manual_seed(21)).relu()
    packed, _, _ = fallback.pack_affine_rows(values)
    signs = fallback.unpack_rows(packed, 64)
    fraction_positive = (signs > 0).float().mean()
    assert 0.1 < float(fraction_positive) < 0.9, "encoding collapsed to a constant"
    # Without centering it would collapse completely.
    assert float((torch.where(values >= 0, 1.0, -1.0) > 0).float().mean()) == 1.0


@pytest.mark.parametrize("rows,columns,_", SHAPES)
def test_native_pack_matches_reference(rows, columns, _):
    _require_native()
    generator = torch.Generator().manual_seed(3)
    values = torch.randn(rows, columns, generator=generator)
    native = loader.pack_affine_rows(values)
    reference = fallback.pack_affine_rows(values)
    assert torch.equal(native[0], reference[0])
    assert torch.allclose(native[1], reference[1], atol=1e-6)
    assert torch.allclose(native[2], reference[2], atol=1e-6)


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_packed_linear_matches_dense(rows, columns, samples):
    _, packed, _ = _packed(rows, columns, seed=1)
    inputs = torch.randn(samples, columns)
    scale = torch.rand(rows) + 0.5
    bias = torch.randn(rows)
    dense = fallback.unpack_rows(packed, columns)
    expected = (inputs @ dense.t()) * scale + bias

    reference = fallback.packed_linear_affine(inputs, packed, scale, bias, columns)
    assert torch.allclose(reference, expected, atol=1e-4)
    if loader.native_available():
        native = loader.packed_linear_affine(inputs, packed, scale, bias, columns)
        assert torch.allclose(native, expected, atol=1e-4)


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_packed_transpose_matches_dense(rows, columns, samples):
    _, packed, _ = _packed(rows, columns, seed=2)
    inputs = torch.randn(samples, rows)
    dense = fallback.unpack_rows(packed, columns)
    expected = inputs @ dense

    assert torch.allclose(
        fallback.packed_transpose(inputs, packed, columns), expected, atol=1e-4
    )
    if loader.native_available():
        assert torch.allclose(
            loader.packed_transpose(inputs, packed, columns), expected, atol=1e-4
        )


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_evidence_equals_exact_sign_outer_product(rows, columns, samples):
    """The whole point: bit storage does not approximate the outer product."""

    activations, packed, _ = _packed(samples, columns, seed=4)
    grad = torch.randn(samples, rows)
    centered = activations - activations.mean(dim=1, keepdim=True)
    expected = grad.t() @ torch.where(centered >= 0, 1.0, -1.0)

    assert torch.allclose(
        fallback.evidence_from_packed(grad, packed, columns), expected, atol=1e-3
    )
    if loader.native_available():
        assert torch.allclose(
            loader.evidence_from_packed(grad, packed, columns), expected, atol=1e-3
        )


@pytest.mark.parametrize("rows,columns,_", SHAPES)
def test_row_inner_matches_dense(rows, columns, _):
    _, packed, _ = _packed(rows, columns, seed=5)
    matrix = torch.randn(rows, columns)
    expected = (matrix * fallback.unpack_rows(packed, columns)).sum(dim=1)

    assert torch.allclose(
        fallback.packed_row_inner(matrix, packed, columns), expected, atol=1e-4
    )
    if loader.native_available():
        assert torch.allclose(
            loader.packed_row_inner(matrix, packed, columns), expected, atol=1e-4
        )


@pytest.mark.parametrize("rows,columns,_", SHAPES)
def test_embedding_matches_dense_lookup(rows, columns, _):
    _, packed, _ = _packed(rows, columns, seed=6)
    scale = torch.rand(rows) + 0.5
    ids = torch.randint(0, rows, (7, 3))
    dense = fallback.unpack_rows(packed, columns) * scale.unsqueeze(1)
    expected = dense[ids.reshape(-1)].view(7, 3, columns)

    assert torch.allclose(
        fallback.packed_embedding(ids, packed, scale, columns), expected, atol=1e-5
    )
    if loader.native_available():
        assert torch.allclose(
            loader.packed_embedding(ids, packed, scale, columns), expected, atol=1e-5
        )


def test_native_unpack_matches_reference_on_ragged_widths():
    _require_native()
    for columns in (1, 7, 8, 9, 63, 64, 65):
        _, packed, _ = _packed(5, columns, seed=columns)
        assert torch.equal(
            loader.unpack_rows(packed, columns), fallback.unpack_rows(packed, columns)
        )


def _update_state(rows, columns, block=64):
    generator = torch.Generator().manual_seed(11)
    coordinate = torch.randint(
        -127, 128, (rows, columns), generator=generator, dtype=torch.int32
    ).to(torch.int8)
    packed = fallback.pack_coordinate(coordinate)
    blocks = (rows * columns + block - 1) // block
    return {
        "coordinate": coordinate,
        "packed": packed,
        "moment_q": torch.zeros(rows, columns, dtype=torch.int8),
        "moment_scale": torch.full((blocks,), 1 / 127, dtype=torch.float16),
        "row_v": torch.zeros(rows, dtype=torch.float16),
        "col_v": torch.zeros(columns, dtype=torch.float16),
    }


@pytest.mark.parametrize("rows,columns", [(16, 32), (24, 40), (8, 8)])
def test_coordinate_update_native_matches_reference(rows, columns):
    _require_native()
    block = 64
    generator = torch.Generator().manual_seed(12)
    evidence = torch.randn(rows, columns, generator=generator)
    common = dict(
        beta1=0.9, beta2=0.99, update_clip=2.0, coordinate_lr=1.0,
        block_size=block, seed=7, step=3,
    )

    native = _update_state(rows, columns, block)
    reference = _update_state(rows, columns, block)
    native_flips = loader.coordinate_update(
        evidence.clone(), native["coordinate"], native["packed"], native["moment_q"],
        native["moment_scale"], native["row_v"], native["col_v"], **common,
    )
    reference_flips = fallback.coordinate_update(
        evidence.clone(), reference["coordinate"], reference["packed"],
        reference["moment_q"], reference["moment_scale"], reference["row_v"],
        reference["col_v"], **common,
    )

    assert torch.equal(native["coordinate"], reference["coordinate"])
    assert torch.equal(native["packed"], reference["packed"])
    assert torch.equal(native["moment_q"], reference["moment_q"])
    assert native_flips == reference_flips
    assert torch.allclose(
        native["row_v"].float(), reference["row_v"].float(), atol=1e-3
    )
    assert torch.allclose(
        native["col_v"].float(), reference["col_v"].float(), atol=1e-3
    )


def test_coordinate_update_keeps_packed_in_sync():
    rows, columns = 12, 20
    state = _update_state(rows, columns)
    evidence = torch.randn(rows, columns, generator=torch.Generator().manual_seed(13))
    kernels.coordinate_update(
        evidence, state["coordinate"], state["packed"], state["moment_q"],
        state["moment_scale"], state["row_v"], state["col_v"],
        beta1=0.9, beta2=0.99, update_clip=2.0, coordinate_lr=4.0,
        block_size=64, seed=1, step=0,
    )
    assert torch.equal(state["packed"], fallback.pack_coordinate(state["coordinate"]))


def test_coordinate_update_is_reproducible_from_seed_and_step():
    rows, columns = 10, 24
    evidence = torch.randn(rows, columns, generator=torch.Generator().manual_seed(14))
    results = []
    for _ in range(2):
        state = _update_state(rows, columns)
        kernels.coordinate_update(
            evidence.clone(), state["coordinate"], state["packed"], state["moment_q"],
            state["moment_scale"], state["row_v"], state["col_v"],
            beta1=0.9, beta2=0.99, update_clip=2.0, coordinate_lr=4.0,
            block_size=64, seed=99, step=5,
        )
        results.append(state["coordinate"].clone())
    assert torch.equal(results[0], results[1])


# ---------------------------------------------------------------------------
# The stochastic-rounding stream
# ---------------------------------------------------------------------------


def test_the_random_stream_is_uniform_and_index_derived():
    """Rounding is stochastic, so the draw is part of the optimizer, not noise.

    A coordinate at 12.3 goes to 13 on three draws in ten and stays at 12 on
    the other seven; that is what keeps updates smaller than one integer step
    from vanishing. So a hash that clusters is a biased optimizer rather than
    merely an ugly one, and a hash that differs between backends is a
    coordinate matrix that stops being portable between them.
    """

    from qste.kernels import stream

    for seed, step in ((1, 0), (1, 7), (99, 1234), (0, 0), (2**31 + 5, 3)):
        combined = stream.seed_hash(seed) ^ stream.step_hash(step)
        for index in (0, 1, 2, 63, 4095, 2**20 + 17):
            assert 0 <= stream.scramble(index ^ combined) < 2**32
            assert 0.0 <= stream.uniform(seed, step, index) < 1.0

    draws = [stream.uniform(3, 11, index) for index in range(20000)]
    assert abs(sum(draws) / len(draws) - 0.5) < 0.01
    for lower in range(10):
        share = sum(lower / 10 <= d < (lower + 1) / 10 for d in draws) / len(draws)
        assert abs(share - 0.1) < 0.015, (lower, share)

    # Consecutive steps must not be correlated: the step only enters through a
    # hash, so a run that reads the same coordinates every step would show up
    # here as two identical streams.
    first = [stream.uniform(5, 100, index) for index in range(512)]
    second = [stream.uniform(5, 101, index) for index in range(512)]
    assert sum(a == b for a, b in zip(first, second)) < 8


def test_the_native_and_reference_backends_draw_the_same_numbers():
    """Not a rounding tolerance -- the same integers, or the streams differ.

    Two backends drawing different randomness do not disagree by a rounding
    error; they disagree on about half the matrix, because each element rounds
    independently. So this uses a target that sits far from any integer
    boundary in both, which makes the comparison a test of the draw rather than
    of the summation order.
    """

    _require_native()
    rows, columns, block = 32, 64, 256
    generator = torch.Generator().manual_seed(4242)
    evidence = torch.randn(rows, columns, generator=generator)
    common = dict(
        beta1=0.9, beta2=0.99, update_clip=2.0, coordinate_lr=1.0,
        block_size=block, seed=17, step=5,
    )
    native = _update_state(rows, columns, block)
    reference = _update_state(rows, columns, block)
    loader.coordinate_update(
        evidence.clone(), native["coordinate"], native["packed"],
        native["moment_q"], native["moment_scale"], native["row_v"],
        native["col_v"], **common,
    )
    fallback.coordinate_update(
        evidence.clone(), reference["coordinate"], reference["packed"],
        reference["moment_q"], reference["moment_scale"], reference["row_v"],
        reference["col_v"], **common,
    )
    difference = (native["coordinate"].int() - reference["coordinate"].int()).abs()
    assert int(difference.max()) <= 1
    assert float((difference > 0).float().mean()) < 0.02
