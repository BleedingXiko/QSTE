"""Conformance on whatever GPU is present. Skipped entirely when none is.

The rest of the suite runs on CPU and proves the numerics. This file proves the
*other* implementation computes the same thing, on hardware, across the shapes,
dtypes and batch sizes a real host produces -- and that the memory claim holds
at the peak, which is where it failed once before and where it matters.

Nothing here is written for a particular device. Every reference is the pure
torch path evaluated on the same tensors, so this file is as meaningful on a
laptop GPU as on a datacentre part, and a device nobody has tried yet either
passes it or reports exactly which kernel disagreed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import qste
from qste import kernels
from qste import nn as qnn
from qste.functional import qste_linear
from qste.kernels import device as device_module
from qste.kernels import fallback

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")

DEVICE = "cuda"

# Deliberately awkward: widths that are not multiples of eight, single rows,
# single columns, a batch of one, and a shape wider than it is tall.
SHAPES = [
    (64, 128, 256),
    (17, 13, 5),
    (1, 1, 1),
    (3, 8, 1),
    (128, 7, 33),
    (256, 1024, 64),
    (33, 2048, 129),
]


def _random(*shape, seed=0, dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(DEVICE).to(dtype)


def _packed(rows, columns, seed=0):
    packed, _, _ = fallback.pack_affine_rows(_random(rows, columns, seed=seed))
    return packed


def _relative(got, expected):
    scale = expected.abs().max().clamp_min(1e-12)
    return float((got.float() - expected.float()).abs().max() / scale)


@pytest.fixture(scope="module", autouse=True)
def _profile_once():
    """Build the device profile before any test times anything."""

    kernels.warm(DEVICE)
    yield
    device_module.reset()


@pytest.fixture
def exact_reduction(monkeypatch):
    """Force the evidence product to fp32, for bit-level comparisons."""

    monkeypatch.setenv("QSTE_REDUCTION_DTYPE", "fp32")
    device_module.reset()
    yield
    device_module.reset()


# ---------------------------------------------------------------------------
# The derived profile itself
# ---------------------------------------------------------------------------


def test_the_profile_is_derived_and_sane():
    profile = kernels.profile(DEVICE)
    assert profile.partitions >= 1
    assert profile.scratch_bytes >= 2 << 20
    assert profile.reduction_dtype in (torch.float32, torch.float16, torch.bfloat16)
    # It must have actually asked the hardware, not guessed.
    assert profile.probe not in ("", None)
    assert profile.tile_rows(4096, 2048, torch.float32) >= 1


def test_tiling_never_exceeds_the_scratch_it_derived():
    profile = kernels.profile(DEVICE)
    for columns in (8, 2048, 65536):
        rows = profile.tile_rows(1 << 20, columns, torch.float32)
        assert rows * columns * 4 <= profile.scratch_bytes or rows == 1


def test_a_huge_row_still_produces_a_usable_tile():
    """A single row wider than the whole scratch budget must not tile to zero."""

    profile = kernels.profile(DEVICE)
    assert profile.tile_rows(4, profile.scratch_bytes, torch.float32) == 1


# ---------------------------------------------------------------------------
# Kernel parity against the reference implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_packed_linear_matches_reference(rows, columns, samples):
    packed = _packed(rows, columns, seed=1)
    inputs = _random(samples, columns, seed=2)
    scale = _random(rows, seed=3).abs() + 0.5
    bias = _random(rows, seed=4)
    expected = fallback.packed_linear_affine(inputs, packed, scale, bias, columns)
    got = kernels.packed_linear_affine(inputs, packed, scale, bias, columns)
    assert _relative(got, expected) < 1e-5


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_packed_transpose_matches_reference(rows, columns, samples):
    packed = _packed(rows, columns, seed=5)
    grad = _random(samples, rows, seed=6)
    scale = _random(rows, seed=7).abs() + 0.5
    for row_scale in (None, scale):
        expected = fallback.packed_transpose(grad, packed, columns, row_scale)
        got = kernels.packed_transpose(grad, packed, columns, row_scale)
        assert _relative(got, expected) < 1e-5


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_evidence_matches_reference(exact_reduction, rows, columns, samples):
    packed = _packed(samples, columns, seed=8)
    grad = _random(samples, rows, seed=9)
    scale = _random(samples, seed=10).abs() + 0.5
    for row_scale in (None, scale):
        expected = fallback.evidence_from_packed(grad, packed, columns, row_scale)
        got = kernels.evidence_from_packed(grad, packed, columns, row_scale)
        assert _relative(got, expected) < 1e-5


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_evidence_in_the_dtype_the_device_chose(rows, columns, samples):
    """The default path may run reduced precision. It may not run wrong.

    The evidence is preconditioned and stochastically rounded into an INT8
    coordinate immediately, so a small relative error here is invisible by the
    time it is used -- but a systematic one, or a silent underflow to zero,
    would stop the coordinate moving at all. Both are what this checks.
    """

    packed = _packed(samples, columns, seed=11)
    grad = _random(samples, rows, seed=12)
    expected = fallback.evidence_from_packed(grad, packed, columns)
    got = kernels.evidence_from_packed(grad, packed, columns)
    assert _relative(got, expected) < 0.02
    assert torch.isfinite(got).all()
    if expected.abs().sum() > 0:
        assert got.abs().sum() > 0, "reduced precision underflowed the whole product"


def test_evidence_survives_a_gradient_far_below_fp16_range():
    """A tiny gradient must not be rounded to nothing by a fast dtype.

    This is the failure mode that would make reduced precision unusable and
    invisible: the loss keeps falling because the continuous parameters still
    train, while no coordinate ever moves again.
    """

    samples, rows, columns = 256, 64, 128
    packed = _packed(samples, columns, seed=13)
    grad = _random(samples, rows, seed=14) * 1e-12
    expected = fallback.evidence_from_packed(grad, packed, columns)
    got = kernels.evidence_from_packed(grad, packed, columns)
    assert got.abs().sum() > 0
    assert _relative(got, expected) < 0.02


def test_evidence_survives_a_gradient_far_above_fp16_range():
    samples, rows, columns = 256, 64, 128
    packed = _packed(samples, columns, seed=15)
    grad = _random(samples, rows, seed=16) * 1e12
    got = kernels.evidence_from_packed(grad, packed, columns)
    assert torch.isfinite(got).all()
    assert _relative(got, fallback.evidence_from_packed(grad, packed, columns)) < 0.02


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_packed_row_inner_matches_reference(rows, columns, samples):
    del samples
    packed = _packed(rows, columns, seed=17)
    matrix = _random(rows, columns, seed=18)
    expected = fallback.packed_row_inner(matrix, packed, columns)
    got = kernels.packed_row_inner(matrix, packed, columns)
    assert _relative(got, expected) < 1e-5


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_pack_affine_matches_reference(rows, columns, samples):
    del samples
    values = _random(rows, columns, seed=19)
    packed, offset, scale = kernels.pack_affine_rows(values)
    want_packed, want_offset, want_scale = fallback.pack_affine_rows(values)
    assert torch.equal(packed, want_packed)
    assert _relative(offset, want_offset) < 1e-5
    assert _relative(scale, want_scale) < 1e-5


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_pack_coordinate_matches_reference(rows, columns, samples):
    del samples
    coordinate = (_random(rows, columns, seed=20) * 60).to(torch.int8)
    assert torch.equal(
        kernels.pack_coordinate(coordinate), fallback.pack_coordinate(coordinate)
    )


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_bit_pack_and_mask_match_reference(rows, columns, samples):
    del samples
    mask = _random(rows, columns, seed=21) > 0
    packed = kernels.pack_bits(mask)
    assert torch.equal(packed, fallback.pack_bits(mask))
    values = _random(rows, columns, seed=22)
    assert torch.equal(
        kernels.apply_bits(values, packed, columns), values * mask
    )


def test_packed_embedding_matches_reference():
    vocabulary, width = 4096, 129
    packed = _packed(vocabulary, width, seed=23)
    scale = _random(vocabulary, seed=24).abs() + 0.1
    ids = torch.randint(0, vocabulary, (7, 11), device=DEVICE)
    expected = fallback.packed_embedding(ids, packed, scale, width)
    got = kernels.packed_embedding(ids, packed, scale, width)
    assert _relative(got, expected) < 1e-5


# ---------------------------------------------------------------------------
# Batch sizes and dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("samples", [1, 2, 3, 7, 64, 100, 511, 1024, 4097])
def test_every_batch_size_is_correct(samples):
    """Including the ones nobody rounds to. Batch is a runtime argument."""

    rows, columns = 64, 96
    packed = _packed(rows, columns, seed=25)
    inputs = _random(samples, columns, seed=samples)
    scale = _random(rows, seed=26).abs() + 0.5
    expected = fallback.packed_linear_affine(inputs, packed, scale, None, columns)
    got = kernels.packed_linear_affine(inputs, packed, scale, None, columns)
    assert _relative(got, expected) < 1e-5


@pytest.mark.parametrize("samples", [1, 2, 3, 5, 8, 16, 33, 64])
@pytest.mark.parametrize("rows,columns", [(2048, 2048), (64, 96), (17, 13), (1, 129)])
def test_fused_small_batch_matches_the_expanded_path(samples, rows, columns):
    """The path that consumes packed bits without ever writing them out.

    This is where a binary weight is categorically better rather than merely
    equal, so it is also where a subtle indexing error would be most tempting
    to miss: it only runs at small batch, which no throughput benchmark covers.
    """

    cuda = kernels.cuda_backend()
    if cuda is None:
        pytest.skip("GPU kernels unavailable")
    packed = _packed(rows, columns, seed=43)
    inputs = _random(samples, columns, seed=44)
    scale = _random(rows, seed=45).abs() + 0.5
    # Both blockings, because which one runs is decided by a measurement and a
    # correctness test may not be left to depend on how a device happens to time.
    for bias in (None, _random(rows, seed=46)):
        for split in (False, True):
            fused = cuda._run_small_batch(inputs, packed, scale, bias, columns, split)
            assert fused is not None, (
                f"fused kernel refused (split={split}): {cuda._FUSED_ERROR!r}"
            )
            expected = fallback.packed_linear_affine(
                inputs, packed, scale, bias, columns
            )
            assert _relative(fused, expected) < 1e-4


def test_the_fused_path_compiles_at_a_single_row_and_a_single_column():
    """The shape a decode step actually hands it.

    Triton folds a runtime integer argument whose value happens to be ``1``
    into a compile-time constant, so the one-sample case takes a different path
    through the compiler than every other size -- and it is the case this
    kernel exists for.
    """

    cuda = kernels.cuda_backend()
    if cuda is None:
        pytest.skip("GPU kernels unavailable")
    for rows, columns in ((1, 1), (1, 8), (8, 1), (1, 2048)):
        packed = _packed(rows, columns, seed=53)
        scale = _random(rows, seed=54).abs() + 0.5
        inputs = _random(1, columns, seed=55)
        got = cuda._run_small_batch(inputs, packed, scale, None, columns, False)
        assert got is not None, f"{rows}x{columns}: {cuda._FUSED_ERROR!r}"
        expected = fallback.packed_linear_affine(inputs, packed, scale, None, columns)
        assert _relative(got, expected) < 1e-4


def test_the_fused_choice_is_measured_and_remembered():
    """Which path wins is settled with a stopwatch, not a formula."""

    cuda = kernels.cuda_backend()
    if cuda is None:
        pytest.skip("GPU kernels unavailable")
    cuda.forget()  # the resolved decision is cached too
    packed = _packed(1024, 1024, seed=47)
    scale = _random(1024, seed=48).abs() + 0.5
    inputs = _random(1, 1024, seed=49)
    kernels.packed_linear_affine(inputs, packed, scale, None, 1024)
    assert cuda._FUSED_CHOICE, "no decision was recorded"
    # The fused path offers several blockings and each is timed under its own
    # name -- "fused-n32" rather than "fused" -- so that a report says which
    # one won. The family is what is fixed, not the exact label.
    assert all(
        choice == cuda.EXPANDED
        or choice == cuda.TILED
        or choice.startswith(cuda.FUSED)
        for choice in cuda._FUSED_CHOICE.values()
    ), cuda._FUSED_CHOICE
    # And the recorded decision must be reused, not re-measured.
    before = dict(cuda._FUSED_CHOICE)
    kernels.packed_linear_affine(inputs, packed, scale, None, 1024)
    assert cuda._FUSED_CHOICE == before


def test_large_batch_never_takes_the_fused_path():
    """Above the candidate ceiling the expansion is the only route."""

    cuda = kernels.cuda_backend()
    if cuda is None:
        pytest.skip("GPU kernels unavailable")
    limit = kernels.profile(DEVICE).fused_batch_limit
    cuda.forget()  # the resolved decision is cached too
    packed = _packed(256, 256, seed=50)
    scale = _random(256, seed=51).abs() + 0.5
    inputs = _random(limit + 64, 256, seed=52)
    kernels.packed_linear_affine(inputs, packed, scale, None, 256)
    assert not cuda._FUSED_CHOICE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_and_backward_in_every_float_dtype(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bf16")
    rows, columns, samples = 64, 128, 32
    packed = _packed(rows, columns, seed=27)
    inputs = _random(samples, columns, seed=28, dtype=dtype)
    scale = (_random(rows, seed=29).abs() + 0.5).to(dtype)
    out = kernels.packed_linear_affine(inputs, packed, scale, None, columns)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()

    grad = _random(samples, rows, seed=30, dtype=dtype)
    back = kernels.packed_transpose(grad, packed, columns, scale)
    assert back.dtype == dtype
    assert torch.isfinite(back).all()


def test_higher_rank_inputs_keep_their_shape():
    rows, columns = 32, 64
    packed = _packed(rows, columns, seed=31)
    inputs = _random(2, 3, 5, columns, seed=32)
    scale = _random(rows, seed=33).abs() + 0.5
    out = kernels.packed_linear_affine(inputs, packed, scale, None, columns)
    assert out.shape == (2, 3, 5, rows)


def test_non_contiguous_inputs_are_handled():
    rows, columns, samples = 32, 64, 48
    packed = _packed(rows, columns, seed=34)
    base = _random(columns, samples, seed=35)
    inputs = base.t()
    scale = _random(rows, seed=36).abs() + 0.5
    expected = fallback.packed_linear_affine(inputs.contiguous(), packed, scale, None, columns)
    got = kernels.packed_linear_affine(inputs, packed, scale, None, columns)
    assert _relative(got, expected) < 1e-5


# ---------------------------------------------------------------------------
# Autograd and the optimizer, end to end on device
# ---------------------------------------------------------------------------


def _stack(width=128, depth=3, device=DEVICE, seed=0, activations=True):
    torch.manual_seed(seed)
    model = nn.Sequential(
        *[
            layer
            for _ in range(depth)
            for layer in (nn.Linear(width, width, bias=False), nn.ReLU())
        ]
    ).to(device)
    qste.convert(model, activations=activations)
    return model


def test_a_full_step_runs_and_moves_coordinates():
    model = _stack()
    coordinates = qste.QSTEOptimizer(model)
    before = [s.coordinate.clone() for s in qste.surfaces(model)]
    model(_random(64, 128, seed=37)).square().mean().backward()
    flips = coordinates.step()
    after = [s.coordinate for s in qste.surfaces(model)]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    assert flips > 0


def test_the_same_seed_produces_the_same_coordinates():
    """Stochastic rounding is hashed from the index, so a step is reproducible."""

    def run():
        model = _stack(seed=5)
        coordinates = qste.QSTEOptimizer(model)
        for _ in range(3):
            model(_random(32, 128, seed=38)).square().mean().backward()
            coordinates.step()
        return [s.coordinate.clone() for s in qste.surfaces(model)]

    for first, second in zip(run(), run()):
        assert torch.equal(first, second)


def test_gpu_optimizer_agrees_with_the_cpu_reference():
    """Two independent implementations of the same step, same evidence.

    What has to match exactly is the *random stream*: rounding is stochastic,
    so a coordinate at 12.3 goes to 13 on three draws in ten, and two backends
    drawing different numbers do not disagree by a rounding error -- they
    disagree on half the matrix. That is what a mismatched hash looks like, and
    it is what this test caught: the C++ path mixed the seed in 64 bits while
    the Triton path mixed it in 32, so a coordinate matrix moved between them
    stopped being reproducible.

    What cannot match exactly is the arithmetic that produces the target. The
    two backends sum the column statistics in different orders, so the target
    lands a part in ten million apart, and where that straddles the draw the
    two round to different integers. So: the coordinates agree, or differ by
    one step in the handful of places where a float boundary decides. Demanding
    more than that would be demanding that two different reduction orders
    produce identical floats, which is not a property either backend has.
    """

    rows, columns = 96, 160
    torch.manual_seed(6)
    evidence = torch.randn(rows, columns)
    coordinate = torch.randint(-100, 100, (rows, columns), dtype=torch.int8)

    def run(device):
        grad = evidence.clone().to(device)
        coord = coordinate.clone().to(device)
        packed = kernels.pack_coordinate(coord)
        state = dict(
            moment_q=torch.zeros_like(coord),
            moment_scale=torch.full(
                ((rows * columns + 255) // 256,), 1 / 127,
                dtype=torch.float16, device=device,
            ),
            row_v=torch.zeros(rows, dtype=torch.float16, device=device),
            col_v=torch.zeros(columns, dtype=torch.float16, device=device),
        )
        flips = kernels.coordinate_update(
            grad, coord, packed, state["moment_q"], state["moment_scale"],
            state["row_v"], state["col_v"], beta1=0.9, beta2=0.99,
            update_clip=2.0, coordinate_lr=1.0, block_size=256, seed=1, step=0,
        )
        return coord.cpu(), packed.cpu(), int(flips)

    gpu_coordinate, gpu_packed, gpu_flips = run(DEVICE)
    cpu_coordinate, cpu_packed, cpu_flips = run("cpu")

    difference = (gpu_coordinate.int() - cpu_coordinate.int()).abs()
    assert int(difference.max()) <= 1, "the two backends drew different randomness"
    disagreed = float((difference > 0).float().mean())
    assert disagreed < 0.02, f"{disagreed:.1%} of coordinates differ, not a boundary"
    # The signs are what the forward reads, so they may drift only where a
    # coordinate sat on zero and the boundary sent the two backends apart.
    signs = (gpu_packed != cpu_packed).float().mean()
    assert float(signs) < 0.05
    assert abs(gpu_flips - cpu_flips) <= max(4, int(0.05 * max(cpu_flips, 1)))




def test_autocast_runs_the_whole_layer_in_the_autocast_dtype():
    model = _stack(width=256)
    coordinates = qste.QSTEOptimizer(model)
    inputs = _random(64, 256, seed=39)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(inputs)
    assert output.dtype == torch.float16
    output.float().square().mean().backward()
    assert coordinates.step() > 0


def test_gradient_checkpointing_still_trains_the_coordinate():
    from torch.utils.checkpoint import checkpoint

    model = _stack(width=128, depth=2)
    coordinates = qste.QSTEOptimizer(model)
    inputs = _random(32, 128, seed=40).requires_grad_(True)
    checkpoint(model, inputs, use_reentrant=False).square().mean().backward()
    assert coordinates.step() > 0


def test_graph_capture_of_a_coordinate_step():
    """The optimizer records without a host synchronization inside it."""

    rows, columns = 64, 128
    torch.manual_seed(7)
    grad = torch.randn(rows, columns, device=DEVICE)
    coordinate = torch.randint(-100, 100, (rows, columns), dtype=torch.int8, device=DEVICE)
    packed = kernels.pack_coordinate(coordinate)
    moment_q = torch.zeros_like(coordinate)
    moment_scale = torch.full(
        ((rows * columns + 255) // 256,), 1 / 127, dtype=torch.float16, device=DEVICE
    )
    row_v = torch.zeros(rows, dtype=torch.float16, device=DEVICE)
    col_v = torch.zeros(columns, dtype=torch.float16, device=DEVICE)
    flips = torch.zeros((), dtype=torch.int32, device=DEVICE)
    enabled = torch.ones((), dtype=torch.uint8, device=DEVICE)
    step = torch.zeros((), dtype=torch.int64, device=DEVICE)

    def one_step():
        kernels.coordinate_update(
            grad.clone(), coordinate, packed, moment_q, moment_scale, row_v, col_v,
            beta1=0.9, beta2=0.99, update_clip=2.0, coordinate_lr=1.0,
            block_size=256, seed=1, step=step, flips=flips, update_enabled=enabled,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            one_step()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        one_step()
    graph.replay()
    torch.cuda.synchronize()
    assert int(flips.item()) >= 0


# ---------------------------------------------------------------------------
# Peak memory, which is the claim
# ---------------------------------------------------------------------------


def _peak(build, samples=512, width=1024, depth=4):
    """High-water allocation of one steady-state step.

    The warmup step runs *before* the counter is reset, and that is not
    cosmetic. The first step through a QSTE model pays several one-time costs
    that have nothing to do with steady-state memory: the device profile times
    a matmul, Triton compiles and caches, and the small-batch chooser runs both
    candidate paths to see which is faster. Counting those makes the first
    configuration measured look worse than whichever ran second, which is a
    measurement artefact and not a property of anything.
    """

    model, optimizers = build(width, depth)
    inputs = torch.randn(samples, width, device=DEVICE)

    def step():
        model(inputs).square().mean().backward()
        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            if hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad(set_to_none=True)

    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del model, optimizers, inputs
    torch.cuda.empty_cache()
    return peak


def _bytes_per_sample(build, width=1024, depth=4):
    """The term that decides whether a batch fits, from the slope of two runs.

    An absolute peak at one size mixes a fixed cost -- weights, optimizer
    state, the packed copies -- with the part that grows with the batch, and at
    a small width the fixed part dominates and drowns the thing being measured.
    The slope isolates it.
    """

    small = _peak(build, samples=512, width=width, depth=depth)
    large = _peak(build, samples=2048, width=width, depth=depth)
    return (large - small) / (2048 - 512)


def _float_build(width, depth):
    torch.manual_seed(8)
    model = nn.Sequential(
        *[
            layer
            for _ in range(depth)
            for layer in (nn.Linear(width, width, bias=False), nn.ReLU())
        ]
    ).to(DEVICE)
    return model, [torch.optim.AdamW(model.parameters(), lr=1e-3)]


def _qste_build(activations):
    def build(width, depth):
        model = _stack(width=width, depth=depth, seed=8, activations=activations)
        return model, [
            torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=1e-3),
            qste.QSTEOptimizer(model),
        ]

    return build


def test_peak_memory_is_lower_than_float_end_to_end():
    """Not bytes on the tape -- the number the allocator actually reports."""

    float_peak = _peak(_float_build)
    qste_peak = _peak(_qste_build(True))
    assert qste_peak < float_peak, f"float {float_peak} vs qste {qste_peak}"


def test_packing_the_activations_is_what_makes_the_peak_drop():
    """The regression this whole activation layer exists for.

    With torch's ReLU in the stack the full-precision activation stays resident
    and the packed copy is added on top, so converting the linears alone costs
    *more* than not converting them at all -- the packed input is an addition,
    not a replacement, because the tensor it was packed from is pinned by the
    nonlinearity downstream of it.

    Measured as a slope rather than an absolute, because that is the claim.
    Packing changes what the model retains per sample; it does not change the
    weights or the optimizer state, and at a small width those fixed costs are
    most of the number and would decide the comparison instead.
    """

    without = _bytes_per_sample(_qste_build(False))
    with_them = _bytes_per_sample(_qste_build(True))
    assert with_them < without * 0.9, f"unpacked {without} vs packed {with_them}"


def test_a_larger_batch_fits_where_float_would_not():
    """The saving converts into batch, which is the point of it."""

    small = _peak(_qste_build(True), samples=512)
    large = _peak(_qste_build(True), samples=2048)
    float_small = _peak(_float_build, samples=512)
    # Four times the batch on QSTE against one on float.
    assert large < float_small, f"qste@2048 {large} vs float@512 {float_small}"
    assert small < large


# ---------------------------------------------------------------------------
# Packed activations on device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 128), (7, 33), (3, 5, 129)])
def test_packed_relu_matches_torch_on_device(shape):
    values = _random(*shape, seed=41)
    mine = values.clone().requires_grad_(True)
    theirs = values.clone().requires_grad_(True)
    seed = torch.randn(*shape, device=DEVICE)
    qnn.relu(mine).backward(seed)
    torch.relu(theirs).backward(seed)
    assert torch.equal(mine.grad, theirs.grad)


def test_packed_smooth_activation_matches_torch_on_device():
    values = _random(128, 256, seed=42)
    mine = values.clone().requires_grad_(True)
    theirs = values.clone().requires_grad_(True)
    seed = torch.randn(128, 256, device=DEVICE)
    qnn.gelu(mine).backward(seed)
    torch.nn.functional.gelu(theirs).backward(seed)
    assert _relative(mine.grad, theirs.grad) < 0.01


def test_dropout_mask_is_consistent_between_forward_and_backward():
    torch.manual_seed(9)
    inputs = torch.randn(256, 512, device=DEVICE, requires_grad=True)
    output = qnn.dropout(inputs, 0.3, training=True)
    output.backward(torch.ones_like(output))
    assert torch.equal(inputs.grad == 0.0, output == 0.0)


# ---------------------------------------------------------------------------
# The whole thing learns
# ---------------------------------------------------------------------------


def test_a_converted_model_learns_on_device():
    torch.manual_seed(10)
    classes, width, samples = 8, 128, 512
    inputs = torch.randn(samples, width, device=DEVICE)
    targets = torch.randint(0, classes, (samples,), device=DEVICE)

    model = nn.Sequential(
        nn.Linear(width, width), nn.ReLU(), nn.Linear(width, classes)
    ).to(DEVICE)
    qste.convert(model)
    continuous = torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=3e-3)
    coordinates = qste.QSTEOptimizer(model)

    losses = []
    for _ in range(120):
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        continuous.step()
        coordinates.step()
        continuous.zero_grad(set_to_none=True)
        losses.append(float(loss))

    import math

    chance = math.log(classes)
    assert losses[0] < chance * 1.5
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * 0.7, losses[-1]


# ---------------------------------------------------------------------------
# Being recordable inside somebody else's graph
# ---------------------------------------------------------------------------


def test_an_undecided_shape_inside_a_capture_takes_the_slow_path():
    """The failure qste.warmup() exists to prevent, pinned as a fact.

    QSTE decides which small-batch implementation to use with a stopwatch, and
    a stopwatch means nothing inside a capture. So an undecided shape falls
    back to the expansion -- correct, and about twice as slow -- and bakes that
    into the host's graph permanently and silently.
    """

    cuda = kernels.cuda_backend()
    if cuda is None:
        pytest.skip("GPU kernels unavailable")
    cuda.forget()  # the resolved decision is cached too
    packed = _packed(512, 512, seed=61)
    scale = _random(512, seed=62).abs() + 0.5
    inputs = _random(1, 512, seed=63)

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        pass
    torch.cuda.current_stream().wait_stream(side)
    try:
        with torch.cuda.graph(graph):
            kernels.packed_linear_affine(inputs, packed, scale, None, 512)
    except Exception:
        pytest.skip("this build will not capture the linear")
    assert set(cuda._FUSED_CHOICE.values()) <= {cuda.EXPANDED}


def test_warmup_decides_before_a_capture_can_bake_in_the_fallback():
    """After warmup the decision exists, so the capture records what won."""

    cuda = kernels.cuda_backend()
    if cuda is None:
        pytest.skip("GPU kernels unavailable")
    cuda.forget()  # the resolved decision is cached too
    model = _stack(width=256, depth=2)
    inputs = _random(1, 256, seed=64)

    assert qste.undecided(model, inputs), "nothing had been measured yet"
    decided = qste.warmup(model, inputs)
    assert decided, "warmup measured nothing"
    assert not qste.undecided(model, inputs), "a shape is still undecided"
    assert qste.decisions() == decided


def test_warmup_refuses_to_run_inside_a_capture():
    """It measures; measuring inside a capture is meaningless, so it says so."""

    model = _stack(width=128, depth=1)
    inputs = _random(1, 128, seed=65)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            with pytest.raises(RuntimeError, match="before a capture"):
                qste.warmup(model, inputs)
    except RuntimeError as error:
        if "before a capture" not in str(error):
            pytest.skip(f"this build will not capture: {error}")


# ---------------------------------------------------------------------------
# Retaining expanded weights
# ---------------------------------------------------------------------------


@pytest.fixture
def no_retention():
    """Every test here must leave the budget where it found it."""

    yield
    qste.retain(0)


@pytest.mark.parametrize("samples", [1, 8, 32, 64, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_retaining_a_weight_does_not_change_the_answer(samples, dtype, no_retention):
    """The whole claim is that it is the same number, arrived at cheaper."""

    rows = columns = 256
    torch.manual_seed(samples)
    surface = qste.Surface(torch.randn(rows, columns), qste.QSTEConfig()).cuda()
    packed, scale = surface.packed_sign.data, surface.scale.detach()
    inputs = torch.randn(samples, columns, device="cuda", dtype=dtype)

    qste.retain(0)
    with torch.no_grad():
        cold = kernels.packed_linear_affine(inputs, packed, scale, None, columns)

    qste.retain(64 << 20)
    with torch.no_grad():
        first = kernels.packed_linear_affine(inputs, packed, scale, None, columns)
        second = kernels.packed_linear_affine(inputs, packed, scale, None, columns)

    # Not bit-equality against the cold run. Retention changes what the paths
    # cost relative to each other, so the stopwatch is entitled to pick a
    # different kernel -- and a different kernel sums the same products in a
    # different order. Demanding identical bits here would be demanding that
    # the framework never improve, which is the opposite of the point.
    scale_of = cold.abs().max().clamp_min(1e-6)
    assert (cold - first).abs().max() / scale_of < 1e-3, "retention changed the answer"
    # Between two retained calls there is no such licence: same path, same
    # weight, same order. A difference here is a stale or corrupted entry.
    assert torch.equal(first, second), "a cached weight gave a different answer"


def test_a_zero_budget_holds_nothing(no_retention):
    rows = columns = 256
    surface = qste.Surface(torch.randn(rows, columns), qste.QSTEConfig()).cuda()
    inputs = torch.randn(8, columns, device="cuda")

    qste.retain(0)
    with torch.no_grad():
        for _ in range(3):
            kernels.packed_linear_affine(
                inputs, surface.packed_sign.data, surface.scale.detach(), None, columns
            )
    assert qste.retained_stats()["resident_bytes"] == 0
    assert qste.retained_stats()["entries"] == 0


def test_the_budget_is_actually_a_bound(no_retention):
    """Ten distinct weights, room for about two. It must not grow past it."""

    columns = 512
    budget = 3 * columns * columns * 4 // 2  # room for roughly one and a half
    qste.retain(budget)
    inputs = torch.randn(4, columns, device="cuda")
    for seed in range(10):
        torch.manual_seed(seed)
        surface = qste.Surface(torch.randn(columns, columns), qste.QSTEConfig()).cuda()
        with torch.no_grad():
            kernels.packed_linear_affine(
                inputs, surface.packed_sign.data, surface.scale.detach(), None, columns
            )
    assert qste.retained_stats()["resident_bytes"] <= budget


def test_a_written_weight_is_not_served_from_the_cache(no_retention):
    """A torch-level write. The easy half, and the half that misled me.

    This passed while the cache was badly broken, because ``bitwise_xor_`` is a
    torch op and bumps the version counter the key was relying on. The real
    writer is a Triton kernel and bumps nothing. See the test below.
    """

    columns = 256
    surface = qste.Surface(torch.randn(columns, columns), qste.QSTEConfig()).to(DEVICE)
    packed, scale = surface.packed_sign.data, surface.scale.detach()
    inputs = torch.randn(8, columns, device=DEVICE)

    qste.retain(64 << 20)
    with torch.no_grad():
        before = kernels.packed_linear_affine(inputs, packed, scale, None, columns)
        packed.bitwise_xor_(torch.full_like(packed, 0xFF))
        after = kernels.packed_linear_affine(inputs, packed, scale, None, columns)

    assert not torch.equal(before, after), "served a stale weight after a write"
    # Flipping every bit negates every sign, so the product negates exactly.
    assert torch.allclose(after, -before, atol=1e-4)


def test_the_optimizer_invalidates_what_it_wrote(no_retention):
    """The write that actually happens, by the thing that actually writes.

    The coordinate step is a Triton kernel writing packed bits through a
    pointer. Torch's version counter does not move, so a cache keyed on it
    serves the weight the model started with, forever. Training then runs to
    completion at chance with no error anywhere -- which is what it did.

    Asserting on the forward rather than on cache statistics: the claim is that
    the model multiplies by its current weights, not that some counter moved.
    """

    columns = 128
    torch.manual_seed(3)
    layer = qste.QSTELinear.from_linear(
        nn.Linear(columns, columns, bias=False)
    ).to(DEVICE)
    coordinates = qste.QSTEOptimizer(layer)
    inputs = torch.randn(32, columns, device=DEVICE)

    qste.retain(64 << 20)
    with torch.no_grad():
        before = layer(inputs).clone()

    # A real step, with a real gradient, through the real optimizer.
    layer(inputs).square().sum().backward()
    flipped = coordinates.step()

    with torch.no_grad():
        after = layer(inputs)

    assert flipped, "the step changed no bits, so this proves nothing"
    assert not torch.equal(before, after), (
        f"{flipped} bits changed and the forward did not: served a stale weight"
    )


def test_gradients_are_the_same_with_retention_on(no_retention):
    """The claim is not that the backward avoids the cache. It uses it.

    An earlier version of this asserted the hit count did not move during a
    backward, on the reasoning that retention is an inference trade. That was
    wrong about its own design: autograd runs the backward with grad mode off,
    which is exactly the condition the cache serves under, and it *should*
    serve it -- the packed bits cannot change between a forward and its own
    backward, so re-expanding them is pure waste.

    What matters is not where the weight came from but whether it was the right
    one. So: the same gradient, computed both ways.
    """

    columns = 128
    torch.manual_seed(11)
    layer = qste.QSTELinear.from_linear(
        nn.Linear(columns, columns, bias=False)
    ).to(DEVICE)
    inputs = torch.randn(16, columns, device=DEVICE)

    def gradients():
        held = inputs.clone().requires_grad_(True)
        layer.surface.zero_grad()
        layer(held).square().sum().backward()
        # The coordinate has no ``.grad``: what a binary weight accumulates is
        # evidence, which is the thing the coordinate step consumes. That is
        # the tensor to compare -- ``.grad`` would compare nothing.
        return held.grad.clone(), layer.surface.flush_evidence().clone()

    qste.retain(0)
    cold_input, cold_coordinate = gradients()

    qste.retain(64 << 20)
    held_input, held_coordinate = gradients()

    assert qste.retained_stats()["hits"] + qste.retained_stats()["misses"] > 0, (
        "the backward never touched the cache, so this proves nothing"
    )
    assert torch.equal(cold_input, held_input), "input gradient changed"
    assert torch.equal(cold_coordinate, held_coordinate), "weight gradient changed"


def test_a_retained_model_still_learns(no_retention):
    """End to end with retention on. It is allowed to be pointless during
    training -- every step writes the packed bits and invalidates what it
    cached -- but it is not allowed to be wrong."""

    import math

    torch.manual_seed(10)
    qste.retain(64 << 20)
    classes, width, samples = 8, 128, 512
    inputs = torch.randn(samples, width, device=DEVICE)
    targets = torch.randint(0, classes, (samples,), device=DEVICE)

    model = nn.Sequential(
        nn.Linear(width, width), nn.ReLU(), nn.Linear(width, classes)
    ).to(DEVICE)
    qste.convert(model)
    continuous = torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=3e-3)
    coordinates = qste.QSTEOptimizer(model)

    losses = []
    for _ in range(120):
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        continuous.step()
        coordinates.step()
        continuous.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))

    assert losses[0] < math.log(classes) * 1.5
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 * 0.7, losses[-1]
