"""The CUDA launchers, exercised without a GPU.

Every CUDA entry point is tiling and cuBLAS calls wrapped around the expansion.
Those kernels cannot run here, but everything around them -- the tiling
arithmetic, the slicing of packed rows, where the row scale is folded in, the
accumulation order, which blockings are offered -- is ordinary Python and is
where an off-by-one actually lives.

So: substitute a torch implementation of the expansion and run the real
launchers. If the machine does have a GPU the same tests run against the real
kernel too.
"""

import importlib
import sys
import types

import pytest
import torch

from qste.kernels import fallback

SHAPES = [(17, 13, 5), (64, 96, 128), (128, 64, 40), (33, 8, 7)]


@pytest.fixture(scope="module")
def cuda_module():
    """``qste.kernels.cuda`` with Triton stubbed and the expansion in torch.

    The stub is only enough to import the module; ``expand_dense`` is then
    replaced outright, so the launchers under test are the real ones.
    """

    if "triton" not in sys.modules:
        triton = types.ModuleType("triton")
        language = types.ModuleType("triton.language")

        def jit(fn=None, **_):
            """As permissive as Triton's, and no more.

            A no-op ``jit`` makes a kernel decorated once and a kernel
            decorated twice into the same object, and makes a kernel decorated
            zero times work fine. Both of those are import-time failures on
            real hardware, and both have shipped: one removed a decorator and
            took the split path out of existence, the other doubled one and
            took the entire GPU backend out of existence -- each caught only
            after a T4 run, because on a CPU there was nothing to notice.

            Triton refuses to decorate something that is already a kernel; it
            calls ``inspect`` on it and gets a ``JITFunction``. So does this.
            """

            if fn is None:
                return jit
            if getattr(fn, "_stub_already_jitted", False):
                raise TypeError(
                    "module, class, method, function, traceback, frame, or "
                    "code object was expected, got JITFunction"
                )
            fn._stub_already_jitted = True
            return fn

        triton.jit = jit
        triton.autotune = lambda **_: (lambda f: f)
        triton.Config = lambda *a, **k: None
        triton.cdiv = lambda a, b: -(-a // b)
        triton.next_power_of_2 = lambda n: 1 << max(0, int(n - 1)).bit_length()
        language.constexpr = int
        for name in ("float32", "float16", "int64", "uint8", "int32", "uint32"):
            setattr(language, name, name)
        triton.language = language
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = language

    module = importlib.import_module("qste.kernels.cuda")

    def expand_dense(packed, columns, *, scale=None, dtype=torch.float32):
        signs = fallback.unpack_rows(packed, columns, dtype=dtype)
        if scale is not None:
            signs = signs * scale.to(dtype).unsqueeze(1)
        return signs

    module.expand_dense = expand_dense
    return module


def _packed(rows, columns, seed=0):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, columns, generator=generator)
    packed, _, _ = fallback.pack_affine_rows(values)
    return packed


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_linear_matches_the_reference(cuda_module, rows, columns, samples):
    packed = _packed(rows, columns, seed=1)
    inputs = torch.randn(samples, columns)
    scale = torch.rand(rows) + 0.5
    bias = torch.randn(rows)
    expected = fallback.packed_linear_affine(inputs, packed, scale, bias, columns)
    got = cuda_module.packed_linear_affine(inputs, packed, scale, bias, columns)
    assert torch.allclose(got, expected, atol=1e-4)


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_transpose_matches_the_reference(cuda_module, rows, columns, samples):
    packed = _packed(rows, columns, seed=2)
    inputs = torch.randn(samples, rows)
    expected = fallback.packed_transpose(inputs, packed, columns)
    got = cuda_module.packed_transpose(inputs, packed, columns)
    assert torch.allclose(got, expected, atol=1e-4)


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_transpose_row_scale_folds_in_correctly(cuda_module, rows, columns, samples):
    """grad_input = (grad * scale) @ sign, however the scale gets applied."""

    packed = _packed(rows, columns, seed=3)
    inputs = torch.randn(samples, rows)
    scale = torch.rand(rows) + 0.5
    dense = fallback.unpack_rows(packed, columns)
    expected = (inputs * scale) @ dense

    assert torch.allclose(
        fallback.packed_transpose(inputs, packed, columns, scale), expected, atol=1e-4
    )
    assert torch.allclose(
        cuda_module.packed_transpose(inputs, packed, columns, scale), expected, atol=1e-4
    )


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_evidence_matches_the_reference(cuda_module, rows, columns, samples):
    packed = _packed(samples, columns, seed=4)
    grad = torch.randn(samples, rows)
    scale = torch.rand(samples) + 0.5
    for row_scale in (None, scale):
        expected = fallback.evidence_from_packed(grad, packed, columns, row_scale)
        got = cuda_module.evidence_from_packed(grad, packed, columns, row_scale)
        assert torch.allclose(got, expected, atol=1e-3)


def test_tiling_actually_triggers_and_stays_correct(cuda_module, monkeypatch):
    """Force a scratch budget small enough that every path must tile."""

    from qste.kernels import device as device_module

    tiny = device_module.DeviceProfile(
        kind="cuda", name="tiling probe", partitions=8, scratch_bytes=512,
        reduction_dtype=torch.float32, probe="test",
    )
    monkeypatch.setattr(device_module, "profile", lambda _device=None: tiny)
    monkeypatch.setattr(cuda_module._device, "profile", lambda _device=None: tiny)
    # The launcher remembers what it resolved for a shape, and what it resolved
    # came from the profile that was in place at the time. Substituting a
    # profile without saying so leaves those answers standing.
    cuda_module.forget()
    monkeypatch.setattr(
        cuda_module, "_LAUNCH", {}, raising=False
    )
    rows, columns, samples = 96, 64, 80
    packed = _packed(rows, columns, seed=5)
    scale = torch.rand(rows) + 0.5

    assert cuda_module._tile(rows, columns, torch.float32, "cpu") < rows, "test did not tile"

    inputs = torch.randn(samples, columns)
    assert torch.allclose(
        cuda_module.packed_linear_affine(inputs, packed, scale, None, columns),
        fallback.packed_linear_affine(inputs, packed, scale, None, columns),
        atol=1e-4,
    )
    grads = torch.randn(samples, rows)
    assert torch.allclose(
        cuda_module.packed_transpose(grads, packed, columns, scale),
        fallback.packed_transpose(grads, packed, columns, scale),
        atol=1e-4,
    )
    activations = _packed(samples, columns, seed=6)
    sample_scale = torch.rand(samples) + 0.5
    assert torch.allclose(
        cuda_module.evidence_from_packed(grads, activations, columns, sample_scale),
        fallback.evidence_from_packed(grads, activations, columns, sample_scale),
        atol=1e-3,
    )


def test_every_kernel_still_carries_its_decorator(cuda_module):
    """A kernel that loses ``@triton.jit`` is invisible until it reaches a GPU.

    Under the stub these tests run against, ``triton.jit`` is the identity, so
    an undecorated kernel imports cleanly, passes every test here, and only
    fails when a launcher subscripts it with a grid on real hardware -- which
    is somebody else's machine, at the end of a queue, after a compile.

    Not hypothetical. Editing this file with a line-range script removed the
    decorator from ``_small_batch_epilogue`` while leaving the function itself
    intact. Every CPU test passed. It reached a T4 as fourteen failures and a
    ``TypeError: 'function' object is not subscriptable`` in the middle of a
    benchmark, and the split path had silently stopped existing.

    A kernel is identified by what it does rather than by what it is called:
    anything that reads ``tl.program_id`` or goes through ``tl.load`` or
    ``tl.store`` is device code and needs the decorator.
    """

    import ast
    from pathlib import Path

    tree = ast.parse(Path(cuda_module.__file__).read_text())
    naked = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.dump(node)
        device_code = any(
            f"attr='{name}'" in body for name in ("program_id", "store", "load")
        )
        decorators = [d for d in node.decorator_list if "jit" in ast.dump(d)]
        if device_code and not decorators:
            naked.append(f"{node.name} (line {node.lineno}) has none")
        if len(decorators) > 1:
            naked.append(f"{node.name} (line {node.lineno}) has {len(decorators)}")
    assert not naked, f"wrong number of @triton.jit: {naked}"


def test_a_stray_decorator_between_kernels_is_caught(cuda_module):
    """The failure mode that took out the whole GPU backend, made a test.

    A decorator on its own, with no function under it, applies to whatever
    comes next -- so an edit that leaves ``@triton.jit`` dangling above another
    ``@triton.jit`` decorates a kernel twice. Triton raises at import, the
    loader catches it, and every GPU kernel silently reverts to the torch
    reference: the benchmark still runs, still says LEARNED, and reports a
    train step three times slower with no error anywhere in it.

    Checking the parsed decorator list is not enough on its own, because a
    dangling decorator is a syntax-level thing that ``ast`` folds into the next
    definition. So the source is read for two in a row.
    """

    from pathlib import Path

    lines = Path(cuda_module.__file__).read_text().splitlines()
    doubled = []
    for index in range(1, len(lines)):
        if not lines[index].startswith("@triton.jit"):
            continue
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous >= 0 and lines[previous].startswith("@triton.jit"):
            doubled.append(f"lines {previous + 1} and {index + 1}")
    assert not doubled, f"@triton.jit applied twice at {doubled}"


def test_no_hand_written_gemm_competes_with_cublas(cuda_module):
    """cuBLAS owns every product it is *able* to compute.

    The first CUDA implementation tiled its own GEMMs in Triton and lost to
    cuBLAS by 2.4x to 6x, so the rule since has been to expand the packed
    operand into bounded scratch and hand the product over. That rule holds
    everywhere cuBLAS can take the operand.

    It cannot take a packed one. There is no cuBLAS entry point for a weight
    stored at one bit per element, and materializing a dense copy for it is
    exactly the cost the packed paths exist to remove -- so in those kernels,
    and only there, the multiply is written here. The guard is therefore not
    "no ``tl.dot`` anywhere", which would forbid the one case that is
    justified; it is that the dense paths still route through torch.

    The exemption was withdrawn once, after ``_packed_tiled`` lost at every
    batch and both precisions on a T4. It is back, because "lost on a T4" is
    not "loses". That part has no fp32 matrix instruction and a 1660 Ti has no
    tensor cores at all, while an A100 has TF32 and Blackwell is different
    again -- a kernel that consumes packed bits directly is timed per device
    precisely so that no one part gets to speak for the rest.
    """

    import ast
    from pathlib import Path

    source = Path(cuda_module.__file__).read_text()
    assert "input_precision" not in source

    packed_only = {"_packed_tiled"}
    tree = ast.parse(source)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name in packed_only:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and inner.attr == "dot"
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "tl"
            ):
                offenders.append(node.name)
    assert not offenders, (
        "these compute a product cuBLAS could have done, by hand: "
        f"{sorted(set(offenders))}"
    )

    # And the exception has to still be the exception.
    assert source.count("tl.dot") == 1


def test_no_single_address_atomics_remain(cuda_module):
    """Every reduction writes a private partial.

    The optimizer used to accumulate three separate reductions into one address
    with ``tl.atomic_add``, from tens of thousands of programs. It cost 19 ms of
    a 157 ms step, and it would cost something similar on any parallel machine,
    because contention on a single cache line is not a property of one vendor's
    card. Partials plus a small second pass is the portable answer.
    """

    from pathlib import Path

    code = [
        line
        for line in Path(cuda_module.__file__).read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not [line for line in code if "atomic_add" in line]


def test_no_device_specific_branching(cuda_module):
    """No kernel decides anything from what the hardware is called.

    Scratch size, reduction width, and the evidence dtype are derived per
    device in ``qste.kernels.device``. If a capability check or an
    architecture name shows up in the kernel file, the framework has started
    being fast on one card instead of correct on all of them.
    """

    from pathlib import Path

    source = Path(cuda_module.__file__).read_text().lower()
    for marker in (
        "get_device_capability",
        "is_bf16_supported",
        "get_device_name",
        "sm_75",
        "sm_80",
        "sm_90",
        "device_name ==",
    ):
        assert marker not in source, f"device-specific branch in cuda.py: {marker}"


def test_shape_arguments_are_not_specialized(cuda_module):
    """Batch-varying sizes must be runtime arguments, not constexpr.

    A ``tl.constexpr`` size recompiles the kernel for every distinct value it
    is ever called with. Widths are fine -- a layer has one. Batch, sample
    count and token count are not: specializing on those turns a host with
    variable-length inputs into a compile loop, which is a correctness-adjacent
    performance cliff that only shows up in production.
    """

    import ast
    import inspect

    def constexpr_arguments(kernel):
        # Triton wraps the function in a JITFunction, which inspect cannot read
        # directly; the original is on `.fn`. Parsed rather than string-matched,
        # because "N: tl.constexpr" is a substring of "BN: tl.constexpr".
        function = getattr(kernel, "fn", kernel)
        definition = ast.parse(inspect.getsource(function).strip()).body[0]
        return {
            argument.arg
            for argument in definition.args.args
            if "constexpr" in ast.dump(argument.annotation or ast.Pass())
        }

    for kernel, forbidden in (
        (cuda_module._expand_flat, "Total"),
        (cuda_module._expand_tiled, "Rows"),
        (cuda_module._apply_bit_mask, "Total"),
        (cuda_module._pack_bit_rows, "R"),
        (cuda_module._packed_row_inner, "N"),
        (cuda_module._packed_embedding, "Total"),
        (cuda_module._row_col_squares, "N"),
        (cuda_module._precondition, "Count"),
        (cuda_module._packed_small_batch, "M"),
        (cuda_module._packed_small_batch, "N"),
        # Element counts, which are products of dimensions and so can exceed
        # what an int32 literal holds -- a vocabulary-sized surface is not an
        # exotic case. As constexprs they would not compile at that size.
        (cuda_module._coordinate_and_pack, "Count"),
        (cuda_module._coordinate_and_pack, "PaddedCount"),
        (cuda_module._coordinate_and_pack, "PackedBytes"),
    ):
        assert forbidden not in constexpr_arguments(kernel), (
            f"{getattr(kernel, '__name__', kernel)} specializes on {forbidden}"
        )


def test_no_kernel_calls_a_method_on_a_runtime_scalar(cuda_module):
    """The bug that cost a benchmark run, made unrepeatable.

    Triton folds a runtime integer argument whose *value* happens to be ``1``
    into a compile-time constant. That is invisible until it happens: the
    argument stops being a Triton scalar and becomes a Python ``int``, and a
    Python ``int`` has no ``.to()``. So ``mask = index < R.to(tl.int64) * C``
    compiles for every shape except the single-row one -- which is exactly the
    shape a one-token decode step hands it, and exactly the shape this
    framework claims to be good at.

    The fix in every case is to pass a bound the kernel only ever compares
    against, so the multiplication happens on the host where the type is known.
    """

    import ast
    from pathlib import Path

    # Read from source rather than from the imported objects: under the Triton
    # stub these tests run against, a decorated kernel is not wrapped and there
    # is nothing to introspect, so an object walk would inspect nothing and
    # pass without having looked.
    tree = ast.parse(Path(cuda_module.__file__).read_text())
    offenders = []
    examined = 0
    for definition in ast.walk(tree):
        if not isinstance(definition, ast.FunctionDef):
            continue
        if not any(
            "jit" in ast.dump(decorator) for decorator in definition.decorator_list
        ):
            continue
        examined += 1
        runtime = {
            argument.arg
            for argument in definition.args.args
            if "constexpr" not in ast.dump(argument.annotation or ast.Pass())
        }
        for node in ast.walk(definition):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "to"
                and isinstance(node.value, ast.Name)
                and node.value.id in runtime
            ):
                offenders.append(f"{definition.name}: {node.value.id}.to(...)")
    # A walk that found nothing because it inspected nothing is not a passing
    # test, it is a silent one.
    assert examined >= 10, f"only {examined} kernels were inspected"
    assert not offenders, (
        "a runtime scalar is specialized to a Python int when it equals 1, "
        f"and Python ints have no .to(): {offenders}"
    )


def test_no_kernel_contains_an_integer_literal_that_needs_a_type(cuda_module):
    """The other half of the same lesson, and the more expensive half.

    Triton types a bare integer literal as int32. Anything from 2**31 up does
    not fit, and the kernel does not compile -- so ``value & 0xFFFFFFFF``, the
    mask whose entire job is to make 32-bit arithmetic mean the same thing on
    every backend, is itself the line that only works on some of them. It broke
    the coordinate optimizer outright, and it broke it on hardware rather than
    here, because the stub these tests run against never types anything.

    So: every integer constant inside a kernel stays below 2**31, and the two
    places that genuinely need a full 32-bit value get it as a runtime
    argument, where Triton types it from the value it has. Shift counts and
    lane widths are exempt -- they are small by construction.
    """

    import ast
    from pathlib import Path

    limit = 2 ** 31
    tree = ast.parse(Path(cuda_module.__file__).read_text())
    offenders = []
    examined = 0
    for definition in ast.walk(tree):
        if not isinstance(definition, ast.FunctionDef):
            continue
        if not any(
            "jit" in ast.dump(decorator) for decorator in definition.decorator_list
        ):
            continue
        examined += 1
        for node in ast.walk(definition):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and not isinstance(node.value, bool)
                and abs(node.value) >= limit
            ):
                offenders.append(f"{definition.name}: {node.value:#x}")

    assert examined >= 10, f"only {examined} kernels were inspected"
    assert not offenders, (
        "Triton types a bare integer literal as int32, so these do not "
        f"compile: {offenders}"
    )


@pytest.mark.parametrize("rows,columns,samples", SHAPES)
def test_bit_mask_roundtrip(cuda_module, rows, columns, samples):
    """pack_bits/apply_bits, the pair a packed activation lives on."""

    del samples
    generator = torch.Generator().manual_seed(11)
    values = torch.randn(rows, columns, generator=generator)
    mask = values > 0
    packed = fallback.pack_bits(mask)
    grad = torch.randn(rows, columns, generator=generator)

    expected = grad * mask
    assert torch.equal(fallback.apply_bits(grad, packed, columns), expected)
    assert torch.equal(fallback.unpack_bits(packed, columns), mask.float())


# ---------------------------------------------------------------------------
# The fused small-batch path, without a GPU
# ---------------------------------------------------------------------------
#
# This kernel cannot run under the Triton stub -- it needs real reductions and
# a real static_range. But its blocking is pure arithmetic and that is where
# the bugs are: a split that does not reach the last column, a partial slot two
# programs both claim, a row block that runs past the matrix. So the blocking
# is a separate function, checked here directly and then emulated element for
# element against the reference.

SMALL_BATCH_SHAPES = [
    (1, 2048, 2048), (1, 64, 96), (3, 17, 13), (8, 1, 129), (64, 2048, 2048),
    (5, 33, 7), (2, 128, 8), (7, 129, 255), (1, 1, 1), (16, 300, 1000),
]


def _profile(partitions=640, scratch=16 << 20):
    from qste.kernels import device as device_module

    return device_module.DeviceProfile(
        kind="cuda", name="plan probe", partitions=partitions,
        scratch_bytes=scratch, reduction_dtype=torch.float32, probe="test",
    )


@pytest.mark.parametrize("samples,rows,columns", SMALL_BATCH_SHAPES)
def test_small_batch_blocking_covers_every_column(cuda_module, samples, rows, columns):
    lanes, groups, block_n, block_bytes, row_programs, splits, chunk = (
        cuda_module.small_batch_plan(samples, rows, columns, _profile())
    )
    packed_bytes = (columns + 7) // 8
    assert splits * chunk >= packed_bytes, "the last columns are never visited"
    assert row_programs * block_n >= rows, "the last rows are never visited"
    assert groups * lanes >= samples, "the last samples are never visited"
    assert chunk % block_bytes == 0, "a split does not end on a block boundary"
    assert lanes & (lanes - 1) == 0 and lanes <= 8
    assert block_n & (block_n - 1) == 0
    assert block_bytes & (block_bytes - 1) == 0


def test_small_batch_blocking_fills_a_wide_device(cuda_module):
    """A batch-one product must not be tiled into a handful of programs.

    The first version tiled only over output rows, which gave 32 programs on a
    40-processor device -- a twentieth of the parallelism it needed -- and lost
    to the expansion it was written to beat.
    """

    profile = _profile(partitions=640)
    _, groups, _, _, row_programs, splits, _ = cuda_module.small_batch_plan(
        1, 2048, 2048, profile
    )
    assert row_programs * groups * splits >= profile.partitions


def test_a_wide_enough_grid_does_not_pay_for_a_split(cuda_module):
    """Splitting buys parallelism with a second pass, so it is a last resort.

    At this size a launch costs about what the kernel costs, so a split that
    was not needed to fill the device is a pure loss -- it turns one launch
    into two and adds a partial buffer nothing asked for.
    """

    profile = _profile(partitions=64)
    *_, splits, _ = cuda_module.small_batch_plan(1, 4096, 4096, profile)
    assert splits == 1, "narrowing the tile already filled the device"


def test_small_batch_partials_stay_within_the_scratch_budget(cuda_module):
    """Splitting harder must not turn into allocating more than the budget."""

    profile = _profile(scratch=16 << 20)
    for samples, rows, columns in SMALL_BATCH_SHAPES:
        lanes, groups, _, _, _, splits, _ = cuda_module.small_batch_plan(
            samples, rows, columns, profile
        )
        one_slot = groups * lanes * rows * 4
        assert splits * one_slot <= max(profile.scratch_bytes // 4, one_slot)


def _emulate_small_batch(cuda_module, inputs, packed, scale, bias, columns, profile):
    """Every load, mask and store the kernel performs, in torch.

    Slow and exact, and the only pre-flight this kernel gets: it cannot run
    under the Triton stub, so without this the first time its indexing is
    exercised is on hardware. It catches the class of error a throughput
    benchmark never sees -- a row block past the end of the matrix, a split
    whose chunk overshoots, a lane past the end of the batch contributing to a
    slot that gets stored.
    """

    samples, rows = inputs.shape[0], packed.shape[0]
    packed_bytes = packed.shape[1]
    lanes, groups, block_n, block_bytes, row_programs, splits, chunk = (
        cuda_module.small_batch_plan(samples, rows, columns, profile)
    )
    bit = torch.arange(8)
    padded = groups * lanes
    partial = torch.zeros(splits, padded, rows)

    for row_program in range(row_programs):
        n = row_program * block_n + torch.arange(block_n)
        live_n = n < rows
        for group in range(groups):
            for split in range(splits):
                accumulator = torch.zeros(block_n, lanes)
                for offset in range(0, chunk, block_bytes):
                    byte_index = split * chunk + offset + torch.arange(block_bytes)
                    live_b = byte_index < packed_bytes
                    column = byte_index[:, None] * 8 + bit[None, :]
                    live_c = live_b[:, None] & (column < columns)

                    byte = torch.zeros(block_n, block_bytes, dtype=torch.int64)
                    live_rows = live_n.nonzero().flatten()
                    live_bytes = live_b.nonzero().flatten()
                    if live_rows.numel() and live_bytes.numel():
                        byte[live_rows[:, None], live_bytes[None, :]] = packed[
                            n[live_rows][:, None], byte_index[live_bytes][None, :]
                        ].long()
                    sign = ((byte[:, :, None] >> bit[None, None, :]) & 1).float() * 2 - 1
                    sign = torch.where(live_c[None, :, :], sign, torch.zeros(()))

                    for slot in range(lanes):
                        # Clamped, exactly as the kernel does -- a lane past the
                        # end of the batch reads a real row and the store drops it.
                        sample = min(group * lanes + slot, samples - 1)
                        value = torch.zeros(block_bytes, 8)
                        live_cols = live_c.nonzero()
                        if live_cols.numel():
                            value[live_cols[:, 0], live_cols[:, 1]] = inputs[sample][
                                column[live_cols[:, 0], live_cols[:, 1]]
                            ]
                        accumulator[:, slot] += (sign * value[None, :, :]).sum((1, 2))

                for index in live_rows.tolist():
                    for slot in range(lanes):
                        sample = group * lanes + slot
                        if sample < samples:
                            partial[split, sample, n[index]] = accumulator[index, slot]

    out = partial.sum(0)[:samples] * scale.view(1, rows)
    if bias is not None:
        out = out + bias.view(1, rows)
    return out


@pytest.mark.parametrize("samples,rows,columns", SMALL_BATCH_SHAPES)
def test_small_batch_kernel_emulated_against_the_reference(
    cuda_module, samples, rows, columns
):
    torch.manual_seed(samples + rows + columns)
    inputs = torch.randn(samples, columns)
    packed = _packed(rows, columns, seed=91)
    scale = torch.rand(rows) + 0.5
    bias = torch.randn(rows)

    got = _emulate_small_batch(
        cuda_module, inputs, packed, scale, bias, columns, _profile()
    )
    want = fallback.packed_linear_affine(inputs, packed, scale, bias, columns)
    # Relative: the split changes the summation order, so a 2048-term reduction
    # differs in the last bits and an absolute bound would just be a shape test.
    error = (got - want).abs().max() / want.abs().max().clamp_min(1e-12)
    assert error < 1e-5, error


@pytest.mark.parametrize("partitions", [64, 640, 6400])
def test_the_emulated_kernel_is_right_on_every_width_of_device(
    cuda_module, partitions
):
    """The blocking is derived from the device, so the indexing has to hold for
    every blocking the derivation can produce -- including the split path on a
    narrow device and the single-launch path on a wide one."""

    torch.manual_seed(partitions)
    samples, rows, columns = 5, 300, 1000
    inputs = torch.randn(samples, columns)
    packed = _packed(rows, columns, seed=77)
    scale = torch.rand(rows) + 0.5
    got = _emulate_small_batch(
        cuda_module, inputs, packed, scale, None, columns, _profile(partitions)
    )
    want = fallback.packed_linear_affine(inputs, packed, scale, None, columns)
    error = (got - want).abs().max() / want.abs().max().clamp_min(1e-12)
    assert error < 1e-5, error


def test_the_triton_hash_arithmetic_matches_the_definition():
    """The kernel's spelling of the hash, evaluated here, against the reference.

    The Triton implementation cannot say ``& 0xFFFFFFFF``, so it says
    ``value - ((value >> 32) << 32)`` instead, and it replaces the golden-ratio
    salt with one that fits in an int32. Both are rewrites of arithmetic that
    has to come out bit-identical on three backends, and neither can be checked
    by running the kernel without a GPU. So the rewrite is evaluated here, in
    the same order and with the same 64-bit semantics, and compared against the
    plain-Python definition.

    This does not prove the kernel compiles. It proves that if it compiles it
    computes the right numbers, which is the half that a source-level guard
    cannot reach.
    """

    from qste.kernels import stream

    def low32(value):
        # Exactly what the kernel does, valid only for non-negative values --
        # which is the precondition the kernel's own docstring states.
        return value - ((value >> 32) << 32)

    def kernel_hash(value):
        value = low32((value ^ 61) ^ (value >> 16))
        value = low32(value + (value << 3))
        value = low32(value ^ (value >> 4))
        value = low32(value * stream.MULTIPLIER)
        return low32(value ^ (value >> 15))

    for value in [0, 1, 2, 61, 255, 4095, 2**16, 2**24, 2**31, 2**31 + 7,
                  2**32 - 1, 12345678, 987654321]:
        assert kernel_hash(value) == stream.scramble(value), hex(value)
        # Every intermediate has to stay inside a signed 64-bit register, or
        # the rewrite is only correct on paper.
        assert 0 <= kernel_hash(value) < 2**32
    assert stream.MULTIPLIER < 2**31 and stream.SALT < 2**31

    # And the full draw, the way the kernel assembles it from its three parts.
    for seed, step in ((1, 0), (7, 3), (2**31 + 11, 99)):
        seed_part = stream.seed_hash(seed)
        for index in (0, 1, 4095, 2**20 + 17, 2**33 + 5):
            step_part = kernel_hash(low32(step) ^ stream.SALT)
            counter = low32(index) ^ (seed_part ^ step_part)
            assert kernel_hash(counter) == stream.scramble(
                index ^ stream.seed_hash(seed) ^ stream.step_hash(step)
            )


def test_no_kernel_reads_a_module_level_global(cuda_module):
    """A kernel may not reach outside itself for a constant.

    Triton rejects any global that is not a ``tl.constexpr`` instance, and that
    is not a rule you find out about gradually -- the kernel simply does not
    compile. It is worth stating why the obvious intuition is wrong: ``tl``
    itself is a module global and resolves fine, which makes it look as though
    globals work in general. They do not. Modules and jitted functions are
    resolved specially; a plain Python int is not.

    So a constant a kernel needs is either a literal in the body or an argument
    passed in. Both places are checked elsewhere in this file -- the literal
    against the definition it copies, the argument against the type it has to
    carry -- and this test is the one that says there is no third place.
    """

    import ast
    import builtins
    from pathlib import Path

    tree = ast.parse(Path(cuda_module.__file__).read_text())
    kernels = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any("jit" in ast.dump(x) for x in node.decorator_list)
    }
    allowed = {"tl", "triton"} | kernels | set(dir(builtins))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in kernels:
            continue
        local = {argument.arg for argument in node.args.args}
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, (ast.Store, ast.Del)):
                local.add(inner.id)
        used = {
            inner.id
            for inner in ast.walk(node)
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load)
        }
        for name in sorted(used - local - allowed):
            offenders.append(f"{node.name}: {name}")

    assert len(kernels) >= 10, f"only {len(kernels)} kernels were inspected"
    assert not offenders, (
        "Triton cannot read a module-level Python value from inside a kernel; "
        f"inline it or pass it as an argument: {offenders}"
    )


def test_the_inlined_salt_matches_the_stream_definition(cuda_module):
    """The literal in the kernel, read back out of the source, against the source
    of truth it copies.

    It has to be a literal -- see the test above -- so this is the only way to
    stop the Triton copy drifting from the C++ and Python ones. Three backends
    have to draw the same numbers or a checkpoint stops being portable between
    them, and the constant is the part a refactor would silently change.
    """

    import ast
    from pathlib import Path

    from qste.kernels import stream

    tree = ast.parse(Path(cuda_module.__file__).read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_coordinate_and_pack":
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Constant)
                and isinstance(inner.value, int)
                and inner.value == stream.SALT
            ):
                found.append(inner.value)

    assert found, (
        f"the kernel no longer contains stream.SALT ({stream.SALT:#x}); "
        "if it was renamed or moved, this test has to move with it"
    )
    assert stream.SALT < 2 ** 31 and stream.MULTIPLIER < 2 ** 31


# ---------------------------------------------------------------------------
# The mid-batch tiled path, without a GPU
# ---------------------------------------------------------------------------

TILED_SHAPES = [
    (64, 2048, 2048), (16, 64, 96), (17, 33, 13), (128, 300, 1000),
    (33, 129, 255), (256, 512, 512), (16, 16, 8), (48, 1, 129),
]


@pytest.mark.parametrize("samples,rows,columns", TILED_SHAPES)
def test_tiled_blocking_covers_every_element(cuda_module, samples, rows, columns):
    block_m, block_n, block_bytes, m_programs, n_programs = cuda_module.tiled_plan(
        samples, rows, columns, _profile()
    )
    assert m_programs * block_m >= samples, "the last samples are never visited"
    assert n_programs * block_n >= rows, "the last outputs are never visited"
    # A matrix-multiply instruction will not accept a tile below sixteen on any
    # side, and the contraction is block_bytes * 8 wide.
    assert block_m >= 16 and block_n >= 16 and block_bytes * 8 >= 16
    for value in (block_m, block_n, block_bytes):
        assert value & (value - 1) == 0


def _emulate_tiled(cuda_module, inputs, packed, scale, bias, columns, profile):
    """Every load, reshape and store the tiled kernel performs, in torch.

    The reshape is the part worth emulating rather than trusting: the kernel
    unpacks signs into a ``[BN, BB, 8]`` tile and flattens it to ``[BN, BK]``,
    and that is only the right answer if the flattened order matches the column
    order the sample tile was gathered in. Off by one byte and it computes a
    plausible, wrong product at every shape.
    """

    samples, rows = inputs.shape[0], packed.shape[0]
    packed_bytes = packed.shape[1]
    block_m, block_n, block_bytes, m_programs, n_programs = cuda_module.tiled_plan(
        samples, rows, columns, profile
    )
    bit = torch.arange(8)
    out = torch.zeros(samples, rows)

    for program_m in range(m_programs):
        rows_m = program_m * block_m + torch.arange(block_m)
        live_m = rows_m < samples
        for program_n in range(n_programs):
            outputs = program_n * block_n + torch.arange(block_n)
            live_n = outputs < rows
            accumulator = torch.zeros(block_m, block_n)

            for start in range(0, packed_bytes, block_bytes):
                byte_index = start + torch.arange(block_bytes)
                live_b = byte_index < packed_bytes
                column = byte_index[:, None] * 8 + bit[None, :]
                live_c = live_b[:, None] & (column < columns)
                flat_column = column.reshape(-1)
                flat_live = live_c.reshape(-1)

                sample_tile = torch.zeros(block_m, block_bytes * 8)
                live_rows = live_m.nonzero().flatten()
                live_cols = flat_live.nonzero().flatten()
                if live_rows.numel() and live_cols.numel():
                    sample_tile[live_rows[:, None], live_cols[None, :]] = inputs[
                        rows_m[live_rows][:, None], flat_column[live_cols][None, :]
                    ]

                byte = torch.zeros(block_n, block_bytes, dtype=torch.int64)
                live_outs = live_n.nonzero().flatten()
                live_bytes = live_b.nonzero().flatten()
                if live_outs.numel() and live_bytes.numel():
                    byte[live_outs[:, None], live_bytes[None, :]] = packed[
                        outputs[live_outs][:, None], byte_index[live_bytes][None, :]
                    ].long()
                signs = ((byte[:, :, None] >> bit[None, None, :]) & 1).float() * 2 - 1
                signs = torch.where(live_c[None, :, :], signs, torch.zeros(()))
                accumulator += sample_tile @ signs.reshape(block_n, -1).t()

            value = accumulator * scale[outputs.clamp(max=rows - 1)][None, :]
            if bias is not None:
                value = value + bias[outputs.clamp(max=rows - 1)][None, :]
            for i in live_m.nonzero().flatten().tolist():
                for j in live_n.nonzero().flatten().tolist():
                    out[rows_m[i], outputs[j]] = value[i, j]
    return out


@pytest.mark.parametrize("samples,rows,columns", TILED_SHAPES)
def test_tiled_kernel_emulated_against_the_reference(
    cuda_module, samples, rows, columns
):
    torch.manual_seed(samples * 7 + rows + columns)
    inputs = torch.randn(samples, columns)
    packed = _packed(rows, columns, seed=53)
    scale = torch.rand(rows) + 0.5
    bias = torch.randn(rows)

    got = _emulate_tiled(cuda_module, inputs, packed, scale, bias, columns, _profile())
    want = fallback.packed_linear_affine(inputs, packed, scale, bias, columns)
    error = (got - want).abs().max() / want.abs().max().clamp_min(1e-12)
    assert error < 1e-5, error


@pytest.mark.parametrize("partitions", [40, 640, 6400])
def test_the_tiled_kernel_is_right_on_every_width_of_device(cuda_module, partitions):
    """The blocking is derived from the device, so every blocking it can
    produce has to index correctly -- including the narrow tiles a wide device
    asks for."""

    torch.manual_seed(partitions)
    samples, rows, columns = 48, 300, 1000
    inputs = torch.randn(samples, columns)
    packed = _packed(rows, columns, seed=59)
    scale = torch.rand(rows) + 0.5
    got = _emulate_tiled(
        cuda_module, inputs, packed, scale, None, columns, _profile(partitions)
    )
    want = fallback.packed_linear_affine(inputs, packed, scale, None, columns)
    error = (got - want).abs().max() / want.abs().max().clamp_min(1e-12)
    assert error < 1e-5, error


EXPAND_SHAPES = [
    (2048, 2048), (1, 1), (1, 4096), (4096, 1), (13, 7), (300, 1000),
    (64, 8), (33, 255), (2, 129),
]


@pytest.mark.parametrize("rows,columns", EXPAND_SHAPES)
def test_every_expand_blocking_covers_every_element(cuda_module, rows, columns):
    packed_bytes = (columns + 7) // 8
    plans = cuda_module.expand_plan(rows, columns, _profile())

    assert plans[0] == ("flat", 1024, None), (
        "the first plan is what a captured region and an untimed shape both "
        "get, so it has to be the form with the longest measured record"
    )
    assert len(plans) == len(set(plans)), "a duplicate plan is a wasted timing pass"

    for plan in plans:
        kind, first, second = plan
        grid = cuda_module.expand_grid(plan, rows, columns)
        if kind == "flat":
            assert grid[0] * first >= rows * columns, "the last elements are missed"
            assert first & (first - 1) == 0
            continue
        assert grid[0] * first >= rows, "the last rows are never written"
        assert grid[1] * second >= packed_bytes, "the last columns are missed"
        for value in (first, second):
            assert value >= 1 and value & (value - 1) == 0
        # Held in registers as float32. The first version of this bounded the
        # tile at 2048 elements on a register-count argument and shipped a
        # regression: less work per element bought nothing once occupancy paid
        # for it. The bound is lower now, and more to the point it is no longer
        # load-bearing -- the stopwatch picks, and this only keeps the
        # candidates it picks between sane.
        assert first * second * 8 <= 512


def _emulate_expand(cuda_module, plan, packed, scale, columns):
    """Every load, reshape and store one blocking performs, in torch.

    The reshape is the part worth checking rather than trusting. The tiled
    kernel unpacks a ``[BR, BB, 8]`` tile and flattens it to ``[BR, BB * 8]``,
    and that is the right answer only if the flattened order matches the column
    order the store addresses. Off by one bit and every weight in the model is
    permuted within its byte -- which still trains, to a worse place, and reads
    as a bad hyperparameter rather than a bug.
    """

    rows, packed_bytes = packed.shape
    kind, first, second = plan
    bit = torch.arange(8)
    out = torch.zeros(rows, columns)

    if kind == "flat":
        (programs,) = cuda_module.expand_grid(plan, rows, columns)
        flat = out.reshape(-1)
        for program in range(programs):
            index = program * first + torch.arange(first)
            live = index < rows * columns
            held = index[live]
            row = held // columns
            column = held - row * columns
            byte = packed[row, column >> 3].long()
            sign = ((byte >> (column & 7)) & 1).float() * 2 - 1
            if scale is not None:
                sign = sign * scale[row].float()
            flat[held] = sign
        return out

    row_programs, byte_programs = cuda_module.expand_grid(plan, rows, columns)
    for program_r in range(row_programs):
        row = program_r * first + torch.arange(first)
        live_row = row < rows
        for program_b in range(byte_programs):
            byte_index = program_b * second + torch.arange(second)
            live_byte = byte_index < packed_bytes
            column = byte_index[:, None] * 8 + bit[None, :]
            live_column = live_byte[:, None] & (column < columns)

            byte = torch.zeros(first, second, dtype=torch.int64)
            live_rows = live_row.nonzero().flatten()
            live_bytes = live_byte.nonzero().flatten()
            if live_rows.numel() and live_bytes.numel():
                byte[live_rows[:, None], live_bytes[None, :]] = packed[
                    row[live_rows][:, None], byte_index[live_bytes][None, :]
                ].long()

            sign = ((byte[:, :, None] >> bit[None, None, :]) & 1).float() * 2 - 1
            if scale is not None:
                held = scale[row.clamp(max=rows - 1)].float()
                sign = sign * held[:, None, None]

            flat_column = column.reshape(-1)
            flat_live = live_column.reshape(-1)
            flat_sign = sign.reshape(first, -1)
            live_cols = flat_live.nonzero().flatten()
            if live_rows.numel() and live_cols.numel():
                out[row[live_rows][:, None], flat_column[live_cols][None, :]] = (
                    flat_sign[live_rows[:, None], live_cols[None, :]]
                )
    return out


@pytest.mark.parametrize("rows,columns", EXPAND_SHAPES)
def test_every_expand_blocking_gives_the_same_answer(cuda_module, rows, columns):
    """All of them, not just the one that happens to win here.

    The whole point of offering several is that a different part picks a
    different one, so a blocking that is only correct because it is never
    chosen on this machine is a bug waiting for someone else's hardware.
    """

    packed = _packed(rows, columns, seed=rows * 3 + columns)
    scale = torch.rand(rows) + 0.5
    want = fallback.unpack_rows(packed, columns) * scale.unsqueeze(1)
    for plan in cuda_module.expand_plan(rows, columns, _profile()):
        got = _emulate_expand(cuda_module, plan, packed, scale, columns)
        assert torch.equal(got, want), plan


@pytest.mark.parametrize("partitions", [40, 640, 6400])
def test_expand_is_right_on_every_width_of_device(cuda_module, partitions):
    rows, columns = 300, 1000
    packed = _packed(rows, columns, seed=partitions)
    want = fallback.unpack_rows(packed, columns)
    for plan in cuda_module.expand_plan(rows, columns, _profile(partitions)):
        got = _emulate_expand(cuda_module, plan, packed, None, columns)
        assert torch.equal(got, want), plan
