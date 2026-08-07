"""GPU kernels, in Triton.

Triton compiles one source for every generation the installed toolchain knows
about, including ROCm, without a toolkit matching the host torch. Nothing here
requires a particular instruction -- no tensor-core intrinsic, no bf16, no fp8,
no ``_int_mm`` -- so the memory win lands on any device that runs torch, and
faster hardware only makes it faster.

**Nothing in this file targets a specific device.** No capability check, no
architecture branch, no tuned constant. The three numbers that would otherwise
be tuned -- scratch budget for an expansion, partial accumulators in a
reduction, dtype for the evidence product -- come from
:mod:`qste.kernels.device`, derived from what the hardware reports and from
timing it. An untested device gets asked the same questions as a tested one.

**No GEMM is written here.** Every matrix product goes to the vendor BLAS. The
kernels in that path either expand packed signs into a bounded scratch buffer
first, or consume the packed operand directly where cuBLAS has no entry point.
See the note above ``_expand_flat``: a hand-rolled GEMM for an operand cuBLAS
can already take measured 2.4x to 6x slower, and packing was never the cost.

Triton carries what BLAS has no equivalent for -- expansion, bit packing and
masking, the embedding gather, the row-wise reduction, and the fused coordinate
optimizer.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - depends on the install
    raise ImportError("QSTE CUDA kernels require Triton") from error

from . import device as _device
from . import stream as _stream


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------
#
# The one Triton kernel in the matmul path is not a matmul. Every product goes
# to cuBLAS.
#
# An earlier version hand-rolled three tiled GEMMs that unpacked signs inside
# the K-loop, and measured 2.4x to 6x slower than cuBLAS: the dot ran outside
# the tensor cores and the packed load re-read the same byte for all eight of
# its bits. Packing was never the cost -- 0.138 ms against a 7.6 ms GEMM.
#
# So expansion writes the packed operand into a bounded scratch buffer and
# cuBLAS takes it from there. Expansion is pure bandwidth, a few percent of the
# GEMM it feeds, and the operand still crosses forward-to-backward at one bit
# per element.


@triton.jit
def _expand_flat(Packed, Scale, Out, Total, C: tl.constexpr,
                 PB: tl.constexpr, BLOCK: tl.constexpr, SCALED: tl.constexpr):
    """packed [R, PB] -> dense +-1 [R, C], one program per output element.

    Folding the row scale in here is free: the store happens either way, and
    it saves the caller a full-size multiply on the other operand.

    This form re-reads a packed byte once for each of its eight bits, which
    looks like eight times the loads. The eight lanes wanting a byte are
    adjacent, so the coalescer merges them into one transaction and the
    redundancy costs instruction issue, not bandwidth. The kernel is bound by
    the *store* -- a dense matrix going out at four bytes an element -- so
    spending issue to keep the store simple wins on the hardware measured so
    far. :func:`_expand_tiled` makes the opposite trade and
    :func:`expand_dense` times both.

    ``Total`` is the element count, passed at runtime. Specializing on it would
    recompile the kernel for every distinct row count a model is fed, turning a
    variable-length host into a compile loop. ``C`` and ``PB`` stay constexpr,
    since a layer has one width and the compiler can strength-reduce the
    division by it.

    The value passed is a *bound*, multiplied out on the host, which avoids a
    Triton hazard: a runtime integer argument whose value is ``1`` becomes a
    compile-time Python ``int``, which has no ``.to()``. A kernel doing
    arithmetic on its own shape argument compiles for every shape except the
    single-row one and dies there -- the shape a one-token decode step hands
    it. A bound that is only ever compared against never triggers this.
    """

    # int64 throughout: a vocabulary-sized surface has more elements than an
    # int32 index can address, and the overflow would be silent.
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = index < Total
    row = index // C
    column = index - row * C
    byte = tl.load(Packed + row * PB + (column >> 3), mask=mask, other=0)
    sign = ((byte >> (column & 7)) & 1).to(tl.float32) * 2.0 - 1.0
    if SCALED:
        sign = sign * tl.load(Scale + row, mask=mask, other=1.0).to(tl.float32)
    tl.store(Out + index, sign, mask=mask)


@triton.jit
def _expand_tiled(Packed, Scale, Out, Rows, C: tl.constexpr, PB: tl.constexpr,
                  BR: tl.constexpr, BB: tl.constexpr, SCALED: tl.constexpr):
    """The same expansion, tiled over ``[rows, bytes]``.

    Each program loads a byte once into a ``[BR, BB, 8]`` tile, the eight signs
    fall out of a shift, the row index comes from the grid so there is no
    division, and the scale is one broadcast load per row instead of one per
    element. That is strictly less work per output element than
    :func:`_expand_flat` does.

    It still lost on a T4, at both precisions, and by enough that it dragged the
    whole inference table down when it was the only implementation. Less work
    per element is not the same as less time: a tile wide enough to amortize the
    byte load holds thousands of floats in registers, and what that buys in
    issue it gives back in occupancy. Which way that trade lands is a property
    of the part, not of the arithmetic, so both live here and the stopwatch
    picks. ``Rows`` is a runtime bound for the reason given in
    :func:`_expand_flat`.
    """

    row = tl.program_id(0) * BR + tl.arange(0, BR)
    byte_index = tl.program_id(1) * BB + tl.arange(0, BB)
    bit = tl.arange(0, 8)

    live_row = row < Rows
    live_byte = byte_index < PB
    column = byte_index[:, None] * 8 + bit[None, :]
    live_column = live_byte[:, None] & (column < C)

    # [BR, BB, 1] -- one load per byte of the weight. int64 on the row index
    # because a vocabulary-sized surface has more elements than an int32 can
    # address and the overflow would be silent.
    byte = tl.load(
        Packed + row[:, None, None].to(tl.int64) * PB + byte_index[None, :, None],
        mask=live_row[:, None, None] & live_byte[None, :, None],
        other=0,
    )
    sign = ((byte >> bit[None, None, :]) & 1).to(tl.float32) * 2.0 - 1.0
    if SCALED:
        scale = tl.load(Scale + row, mask=live_row, other=1.0).to(tl.float32)
        sign = sign * scale[:, None, None]

    # The eight bits of a byte are eight adjacent columns, so a row of this
    # tile is BB*8 contiguous elements of the output and the store coalesces.
    flat_column = tl.reshape(column, (BB * 8,))
    live_flat = tl.reshape(live_column, (BB * 8,))
    tl.store(
        Out + row[:, None].to(tl.int64) * C + flat_column[None, :],
        tl.reshape(sign, (BR, BB * 8)),
        mask=live_row[:, None] & live_flat[None, :],
    )


@triton.jit
def _apply_bit_mask(Values, Packed, Out, Total, C: tl.constexpr,
                    PB: tl.constexpr, BLOCK: tl.constexpr):
    """``values`` where the bit is set, zero where it is not.

    The entire backward of a saturating activation. ReLU's gradient is the
    input gradient exactly where the output was positive, and "was it positive"
    is one bit -- so a QSTE stack keeps 1/32nd of what torch keeps, and the
    full-precision activation actually dies instead of being pinned by the
    nonlinearity while the packed copy sits next to it.

    ``Total`` rather than a row count, for the reason on ``_expand_flat``.
    """

    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = index < Total
    row = index // C
    column = index - row * C
    byte = tl.load(Packed + row * PB + (column >> 3), mask=mask, other=0)
    keep = ((byte >> (column & 7)) & 1) != 0
    value = tl.load(Values + index, mask=mask, other=0.0)
    tl.store(Out + index, tl.where(keep, value, tl.zeros_like(value)), mask=mask)


@triton.jit
def _pack_bit_rows(Mask, Packed, R, K: tl.constexpr, PB: tl.constexpr,
                   BB: tl.constexpr):
    """One bit per element of a boolean mask, least significant bit first."""

    row = tl.program_id(0).to(tl.int64)
    byte_index = tl.program_id(1) * BB + tl.arange(0, BB)
    lane = tl.arange(0, 8)
    column = byte_index[:, None] * 8 + lane[None, :]
    live = (byte_index[:, None] < PB) & (column < K) & (row < R)
    value = tl.load(Mask + row * K + column, mask=live, other=0)
    byte = tl.sum(tl.where(live & (value != 0), 1 << lane[None, :], 0), axis=1)
    tl.store(Packed + row * PB + byte_index, byte.to(tl.uint8),
             mask=(byte_index < PB) & (row < R))


@triton.jit
def _packed_small_batch(X, Packed, Scale, Bias, Out, Partial, M, N,
                        K: tl.constexpr, PB: tl.constexpr,
                        BN: tl.constexpr, BB: tl.constexpr, LANES: tl.constexpr,
                        MPAD: tl.constexpr, CHUNK: tl.constexpr,
                        SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr):
    """``x @ signT * scale + bias`` without ever expanding the weight.

    This is the one place where a binary weight is categorically better rather
    than merely equal, and expanding would throw it away.

    At large batch a product is compute bound: the weight is read once and
    reused across every sample, so expanding it costs a rounding error and BLAS
    schedules the arithmetic better than anything written by hand. At small
    batch there is no arithmetic to speak of -- the whole cost is dragging the
    weight out of memory -- and a weight that is one bit per element is a
    thirty-second of the traffic. Expanding it first means writing the dense
    matrix out and reading it back, which is *more* traffic than the float path
    ever needed, which is why the expansion route loses here by a factor of two
    against cuBLAS on a weight it has to materialize.

    Three things decide whether that theoretical advantage survives contact,
    and the first two versions of this kernel each lost to one of them:

    *The byte is loaded once, not once per bit.* Indexing the packed operand by
    column re-reads the same byte for all eight of its bits, so a tile that
    needs 512 bytes issues 4096 loads. The tile is built from a ``[BN, BB, 8]``
    view instead: one load per byte, and the eight signs fall out of a shift.

    *The epilogue is in here.* At this size the kernel is tens of microseconds
    and a launch is roughly ten, so scaling, bias and the cast are three more
    launches against work that should be one. When the grid is wide enough not
    to need a split, this writes the finished result and nothing follows it.

    *The grid has to be wide.* Tiling only over output rows leaves a few dozen
    programs, and a few dozen programs cannot saturate a device that wants
    thousands. Rows per program shrink until the grid reaches the width the
    device reports, and only if that is still not enough does the contraction
    split -- because a split costs a second pass, and a second pass at this
    size costs as much as the kernel.
    """

    row_program = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)

    n = row_program * BN + tl.arange(0, BN)
    live_n = n < N
    bit = tl.arange(0, 8)
    accumulator = tl.zeros((BN, LANES), tl.float32)

    start = split * CHUNK
    for offset in range(0, CHUNK, BB):
        byte_index = start + offset + tl.arange(0, BB)
        live_b = byte_index < PB
        column = byte_index[:, None] * 8 + bit[None, :]
        live_c = live_b[:, None] & (column < K)

        # [BN, BB, 1] -- one load per byte of the weight, which is the whole
        # bandwidth argument for a packed operand actually being taken.
        byte = tl.load(
            Packed + n[:, None, None].to(tl.int64) * PB + byte_index[None, :, None],
            mask=live_n[:, None, None] & live_b[None, :, None],
            other=0,
        )
        sign = ((byte >> bit[None, None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        sign = tl.where(live_c[None, :, :], sign, 0.0)

        # One pass over the weight tile serves every sample this program owns,
        # which is why the batch is grouped rather than put on the grid whole.
        for slot in tl.static_range(LANES):
            sample = group * LANES + slot
            # Clamped rather than masked: a lane past the end of the batch
            # reads a real row and computes a real number, and the store
            # discards it. Masking on a rank-zero condition instead would put
            # a scalar and a tile in the same boolean, which is the sort of
            # thing that compiles on one Triton and not the next.
            value = tl.load(
                X + tl.minimum(sample, M - 1).to(tl.int64) * K + column,
                mask=live_c,
                other=0.0,
            ).to(tl.float32)
            contribution = tl.sum(tl.sum(sign * value[None, :, :], axis=2), axis=1)
            accumulator += tl.where(
                tl.arange(0, LANES)[None, :] == slot, contribution[:, None], 0.0
            )

    samples = group * LANES + tl.arange(0, LANES)
    live = live_n[:, None] & (samples[None, :] < M)
    if SPLIT == 1:
        # Nothing follows this launch: the row scale, the bias and the cast to
        # the caller's dtype all happen while the accumulator is still live.
        result = accumulator * tl.load(Scale + n, mask=live_n, other=0.0)[:, None]
        if HAS_BIAS:
            result += tl.load(Bias + n, mask=live_n, other=0.0).to(tl.float32)[:, None]
        tl.store(
            Out + samples[None, :].to(tl.int64) * N + n[:, None],
            result.to(Out.dtype.element_ty),
            mask=live,
        )
    else:
        # Private per split; a second pass combines them. No atomics, for the
        # same reason the optimizer has none.
        tl.store(
            Partial + split.to(tl.int64) * MPAD * N
            + samples[None, :].to(tl.int64) * N + n[:, None],
            accumulator,
            mask=live,
        )




@triton.jit
def _packed_tiled(X, Packed, Scale, Bias, Out, M, N,
                  K: tl.constexpr, PB: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BB: tl.constexpr,
                  HAS_BIAS: tl.constexpr, DOT: tl.constexpr):
    """The middle. Enough samples to be arithmetic, too few to hide an expansion.

    This is the batch range where QSTE was slower than float and there was no
    good reason for it, so it is worth being exact about what goes wrong there.

    The expansion route writes the whole dense weight out before cuBLAS reads
    it. That costs about the same every time -- roughly seventeen megabytes of
    traffic at this width -- whether the layer is doing one row of work or ten
    thousand. At ten thousand rows the product itself takes nine milliseconds
    and the expansion is a rounding error. At sixty-four rows the product takes
    a tenth of that and the expansion is most of the cost, so a format storing
    a thirty-second of the bytes ends up losing to one storing all of them.

    The lane kernel above avoids the expansion but cannot cover this range: it
    keeps its accumulator per sample, so the samples it can hold at once are
    bounded by what one program can carry, and past about eight it either
    spills or re-reads the weight once per group.

    What is different here is that the product is genuinely arithmetic, so it
    is worth structuring it as one -- unpack a tile of signs and hand it to a
    real matrix multiply, rather than reducing sample by sample. The weight
    still crosses memory packed, once, at one bit per element; only the tile
    that is live in a program is ever dense.

    That does mean a matrix multiply written here rather than handed to cuBLAS,
    which is the thing this file otherwise refuses to do -- and the refusal
    still stands everywhere it can. cuBLAS lost to nothing here because cuBLAS
    cannot be used at all: it has no way to read an operand that is one bit per
    element, and materializing one for it is precisely the cost being removed.
    """

    program_m = tl.program_id(0)
    program_n = tl.program_id(1)
    rows = program_m * BM + tl.arange(0, BM)
    outputs = program_n * BN + tl.arange(0, BN)
    live_rows = rows < M
    live_outputs = outputs < N
    bit = tl.arange(0, 8)

    accumulator = tl.zeros((BM, BN), tl.float32)
    for start in range(0, PB, BB):
        byte_index = start + tl.arange(0, BB)
        live_byte = byte_index < PB
        column = byte_index[:, None] * 8 + bit[None, :]
        live_column = live_byte[:, None] & (column < K)
        # Flattened row-major, so entry i of the flat tile is column
        # ``start * 8 + i`` -- the same order the sign tile unpacks into.
        flat_column = tl.reshape(column, (BB * 8,))
        flat_live = tl.reshape(live_column, (BB * 8,))

        samples = tl.load(
            X + rows[:, None].to(tl.int64) * K + flat_column[None, :],
            mask=live_rows[:, None] & flat_live[None, :],
            other=0.0,
        )
        # One load per byte of weight, eight signs out of each.
        byte = tl.load(
            Packed + outputs[:, None, None].to(tl.int64) * PB
            + byte_index[None, :, None],
            mask=live_outputs[:, None, None] & live_byte[None, :, None],
            other=0,
        )
        signs = ((byte >> bit[None, None, :]) & 1).to(tl.float32) * 2.0 - 1.0
        signs = tl.where(live_column[None, :, :], signs, 0.0)
        accumulator += tl.dot(
            samples.to(DOT),
            tl.trans(tl.reshape(signs, (BN, BB * 8))).to(DOT),
            out_dtype=tl.float32,
        )

    result = accumulator * tl.load(Scale + outputs, mask=live_outputs, other=0.0)[None, :]
    if HAS_BIAS:
        result += tl.load(
            Bias + outputs, mask=live_outputs, other=0.0
        ).to(tl.float32)[None, :]
    tl.store(
        Out + rows[:, None].to(tl.int64) * N + outputs[None, :],
        result.to(Out.dtype.element_ty),
        mask=live_rows[:, None] & live_outputs[None, :],
    )


@triton.jit
def _small_batch_epilogue(Partial, Scale, Bias, Out, N,
                          MPAD: tl.constexpr, SPLIT: tl.constexpr,
                          BLOCK: tl.constexpr, HAS_BIAS: tl.constexpr):
    """Sum the split partials and finish the row. One launch, not three."""

    # One program per sample, so the sample index is in range by construction
    # and only the row block needs a bound.
    sample = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = n < N
    total = tl.zeros((BLOCK,), tl.float32)
    for split in range(0, SPLIT):
        total += tl.load(
            Partial + split.to(tl.int64) * MPAD * N + sample.to(tl.int64) * N + n,
            mask=live, other=0.0,
        )
    total *= tl.load(Scale + n, mask=live, other=0.0)
    if HAS_BIAS:
        total += tl.load(Bias + n, mask=live, other=0.0).to(tl.float32)
    tl.store(Out + sample.to(tl.int64) * N + n, total.to(Out.dtype.element_ty),
             mask=live)


def small_batch_plan(samples: int, rows: int, columns: int, profile,
                     allow_split: bool = True):
    """Blocking for the fused path. Pure arithmetic, so it is testable anywhere.

    Returns ``(lanes, groups, block_n, block_bytes, row_programs, splits,
    chunk)``. The kernel's correctness lives here: an index that fails to cover
    ``columns``, or a split writing another split's slot, fails silently and
    only at shapes nobody benchmarks.

    ``allow_split`` is not a knob the caller is expected to answer. Splitting
    the contraction trades a second pass for a wider grid, and the better side
    depends on launch overhead against memory bandwidth on the actual part, so
    both plans are built, timed once, and the winner remembered. See
    ``_choose_path``.
    """

    packed_bytes = (columns + 7) // 8

    # Samples per program. The weight tile is re-read once per group, so a
    # group of one turns the batch into extra weight traffic and a group of the
    # whole batch turns the accumulator into a register problem; the cap is
    # what a program can hold, not anything about the device.
    lanes = min(8, max(1, triton.next_power_of_2(samples)))
    groups = triton.cdiv(samples, lanes)

    # Rows per program shrink until the grid is as wide as the device says it
    # wants -- that is the cheap way to fill the machine, because it costs
    # nothing but a smaller tile.
    wanted = max(1, triton.cdiv(profile.partitions, groups))
    block_n = max(8, triton.next_power_of_2(max(1, triton.cdiv(rows, wanted))))
    block_n = min(block_n, max(8, triton.next_power_of_2(rows)))
    block_bytes = min(16, max(1, triton.next_power_of_2(packed_bytes)))

    # ... and then shrink further if the tile would not fit in registers. The
    # sign tile is block_n x block_bytes x 8 elements and every lane multiplies
    # through all of it, so what a program holds live grows with the product of
    # the two -- and a tile that spills is slower than no tile at all. The
    # ceiling is the largest configuration measured to beat the expansion
    # (0.7x at four times this, 1.4x at exactly it), which is the honest thing
    # to call it: a register bound found by measurement, not a machine constant.
    while block_n > 8 and block_n * block_bytes * 8 * lanes > 8192:
        block_n //= 2
    return _small_batch_fields(samples, rows, columns, profile, lanes,
                               block_n, block_bytes, allow_split)


def _small_batch_fields(samples, rows, columns, profile, lanes, block_n,
                        block_bytes, allow_split=True):
    """Everything a plan implies once its three free choices are made.

    Split out so that a blocking can be *proposed* rather than only derived.
    The fields below are not independent -- the number of programs decides
    whether a split is worth taking, the split decides the chunk -- so a
    proposal that set them by hand would be a plan the kernel cannot run.
    """

    packed_bytes = (columns + 7) // 8
    lanes = max(1, min(8, triton.next_power_of_2(max(1, lanes))))
    block_n = max(8, triton.next_power_of_2(max(1, block_n)))
    block_bytes = max(1, min(16, triton.next_power_of_2(max(1, block_bytes))))
    groups = triton.cdiv(samples, lanes)
    row_programs = triton.cdiv(rows, block_n)

    # Splitting the contraction is the expensive way to fill it: it buys
    # parallelism with a second pass over the partials, and at this size a pass
    # costs about what the kernel costs. So it is a last resort, taken only
    # when narrowing the tile has already bottomed out and the grid is still
    # short of what the device reports.
    programs = row_programs * groups
    splits = max(1, min(triton.cdiv(packed_bytes, block_bytes),
                        triton.cdiv(profile.partitions, programs)))
    # ... and the per-split partials are real memory, so they answer to the same
    # scratch budget every other buffer here does.
    padded = groups * lanes
    affordable = max(1, (profile.scratch_bytes // 4) // max(1, padded * rows))
    splits = 1 << max(0, (min(splits, affordable) - 1).bit_length())
    if not allow_split:
        splits = 1
    chunk = triton.cdiv(triton.cdiv(packed_bytes, splits), block_bytes) * block_bytes
    return lanes, groups, block_n, block_bytes, row_programs, splits, chunk


def small_batch_variants(samples: int, rows: int, columns: int, profile,
                         allow_split: bool = True):
    """Blockings worth timing for the fused path, the derived one first.

    The formula above fills the grid and then shrinks the tile until it fits a
    register budget that was itself found by measurement. It is a good guess.
    It is still a guess, and the two most recent guesses of this kind were both
    wrong: the expansion's blocking, hand-derived twice, was beaten by a
    proposal the stopwatch found -- and the winner is a different one in fp32
    than in fp16 on the same card, which no formula taking neither dtype nor
    part as input can express.

    So the neighbours get offered too. Deliberately only two of them: every
    distinct blocking is a separate Triton compile, and a search that explores
    thirty of them per shape would spend more time compiling than the kernel
    will ever save. One step narrower and one step wider is enough to tell
    whether the derived point is on a slope or at the bottom.
    """

    base = small_batch_plan(samples, rows, columns, profile, allow_split)
    lanes, _, block_n, block_bytes, *_ = base
    plans = [base]
    for candidate_n in (max(8, block_n // 2), block_n * 2):
        if candidate_n * block_bytes * 8 * lanes > 8192:
            continue
        plan = _small_batch_fields(samples, rows, columns, profile, lanes,
                                   candidate_n, block_bytes, allow_split)
        if plan not in plans:
            plans.append(plan)
    return plans




def tiled_plan(samples: int, rows: int, columns: int, profile):
    """Blocking for the mid-batch path. Pure arithmetic, testable anywhere.

    Returns ``(block_m, block_n, block_bytes, m_programs, n_programs)``. The
    sample block is capped by what a program can hold; the output block then
    shrinks until the grid is as wide as the device says it wants, which is the
    same rule the lane kernel follows and for the same reason -- a tile that is
    generous on a part with few processors starves one with many.
    """

    packed_bytes = (columns + 7) // 8
    # Sixteen is the floor a matrix-multiply instruction accepts. Below it
    # there is nothing to tile and the lane kernel handles the shape.
    block_m = min(64, max(16, triton.next_power_of_2(samples)))
    m_programs = triton.cdiv(samples, block_m)

    wanted = max(1, triton.cdiv(profile.partitions, m_programs))
    block_n = max(16, triton.next_power_of_2(max(1, triton.cdiv(rows, wanted))))
    block_n = min(block_n, 64, max(16, triton.next_power_of_2(rows)))
    block_bytes = min(8, max(2, triton.next_power_of_2(packed_bytes)))
    return block_m, block_n, block_bytes, m_programs, triton.cdiv(rows, block_n)


def _run_tiled(flat, packed, scale, bias, columns):
    """The mid-batch path, or ``None`` when this build will not run it.

    Declining is allowed, as with every candidate. A Triton that will not
    compile a matrix multiply at these shapes, or a device without the
    instruction, falls back to the expansion route, which is always available
    and always correct.
    """

    global _FUSED_ERROR
    samples, rows = flat.shape[0], packed.shape[0]
    plan = tiled_plan(samples, rows, columns, _device.profile(flat.device))
    block_m, block_n, block_bytes, m_programs, n_programs = plan

    flat = flat.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    out = torch.empty(samples, rows, device=flat.device, dtype=flat.dtype)
    # The multiply runs in the caller's precision. An fp16 host gets tensor
    # cores, an fp32 host keeps fp32. Narrowing a caller's forward to buy
    # throughput is their decision to make, not this library's.
    dot_dtype = tl.float16 if flat.dtype == torch.float16 else tl.float32
    try:
        _packed_tiled[(m_programs, n_programs)](
            flat, packed, scale, scale if bias is None else bias, out,
            samples, rows, K=columns, PB=packed.shape[1],
            BM=block_m, BN=block_n, BB=block_bytes,
            HAS_BIAS=bias is not None, DOT=dot_dtype, num_warps=4,
        )
    except Exception as error:  # pragma: no cover - depends on the Triton build
        _FUSED_ERROR = error
        return None
    return out



def _run_small_batch(flat, packed, scale, bias, columns, allow_split=True,
                     plan=None):
    """The fused path, or ``None`` when this build cannot run it.

    A Triton version that rejects the kernel is a reason to take the general
    path, not a reason to fail: the expansion route is always available and
    always correct.
    """

    global _FUSED_ERROR
    samples, rows = flat.shape[0], packed.shape[0]
    if plan is None:
        plan = small_batch_plan(samples, rows, columns,
                                _device.profile(flat.device),
                                allow_split=allow_split)
    lanes, groups, block_n, block_bytes, row_programs, splits, chunk = plan

    if not flat.is_contiguous():
        flat = flat.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()
    out = torch.empty(samples, rows, device=flat.device, dtype=flat.dtype)
    if splits == 1:
        partial = out  # unused by the kernel; the branch is compiled out
    else:
        partial = torch.empty(splits, groups * lanes, rows,
                              device=flat.device, dtype=torch.float32)
    try:
        _packed_small_batch[(row_programs, groups, splits)](
            flat, packed, scale, scale if bias is None else bias, out, partial,
            samples, rows, K=columns, PB=packed.shape[1],
            BN=block_n, BB=block_bytes, LANES=lanes, MPAD=groups * lanes,
            CHUNK=chunk, SPLIT=splits, HAS_BIAS=bias is not None, num_warps=4,
        )
        if splits > 1:
            block = min(1024, triton.next_power_of_2(rows))
            _small_batch_epilogue[(samples, triton.cdiv(rows, block))](
                partial, scale, scale if bias is None else bias, out,
                rows, MPAD=groups * lanes, SPLIT=splits, BLOCK=block,
                HAS_BIAS=bias is not None, num_warps=4,
            )
    except Exception as error:  # pragma: no cover - depends on the Triton build
        # Kept, not swallowed. A discarded compile error is indistinguishable
        # from "the fused path measured slower", and reading one as the other
        # cost two benchmark runs.
        _FUSED_ERROR = error
        return None
    return out


# The last reason the fused path declined, for tests and for anyone wondering
# why a device fell back. ``None`` means it has never refused.
_FUSED_ERROR: Exception | None = None


# Small-batch products have several possible implementations, and which one
# wins is settled with a stopwatch, once per shape per device, then remembered.
#
# A formula here -- arithmetic intensity against machine balance -- would encode
# an assumption about the hardware. Timing the shape in front of it does not,
# which is how the CPU backend ended up with no fused path (measured at 0.12x)
# while this one degrades gracefully on untested devices.
#
# The split candidate is a real trade with no device-independent answer: it
# widens the grid and pays for that with a second pass costing a launch. Which
# side wins depends on how a part's launch overhead compares to its memory
# bandwidth.
#
# Every implementation is a *candidate*, never a replacement. One that fails to
# compile, or compiles and loses, is timed once and never chosen again on that
# device -- so a new kernel aimed at a slow batch range cannot regress the
# ranges that were already fast.
#
# Losing kernels are kept. A kernel that loses on one part says nothing about a
# part with different arithmetic: some cards have no wide-float matrix
# instruction and run fp32 on plain multiply-adds, others in the same
# generation ship with their matrix units fused off, and newer parts move the
# register file, scheduler and instruction set together. Keeping a kernel that
# loses somewhere costs one timing pass. Deleting one that would have won on an
# untested part costs that part's entire gain, invisibly.
EXPANDED, FUSED, FUSED_SPLIT, TILED = "expanded", "fused", "fused-split", "tiled"
_FUSED_CHOICE: dict[tuple, str] = {}

# What every candidate measured, kept alongside what won. Without it, a path
# missing from a report is indistinguishable from a path that never ran, and
# those call for opposite responses: one means the kernel is slower than the
# alternative, the other means it failed to compile on this build and said
# nothing.
_FUSED_TIMINGS: dict[tuple, dict[str, float]] = {}

# The blocking behind each named fused candidate, so the winner can be launched
# again without re-deriving which proposal it was.
_FUSED_PLANS: dict[tuple, dict[str, tuple]] = {}


def timings(device=None) -> dict[tuple, dict[str, float]]:
    """Every candidate's measured milliseconds, keyed the same way as the choice.

    ``inf`` means the candidate declined -- it did not compile, or the device
    lacks an instruction it needs. :data:`_FUSED_ERROR` holds the last reason.
    """

    if device is None:
        return dict(_FUSED_TIMINGS)
    # ``torch.device("cuda")`` has no index, but the keys were built from a
    # tensor, which always does. Resolving it here rather than at the call site
    # is the difference between an empty report and a real one.
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    return {key: value for key, value in _FUSED_TIMINGS.items() if key[0] == index}


def _time_path(function, iterations=5) -> float:
    for _ in range(2):
        if function() is None:
            return float("inf")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iterations


def _choose_path(flat, packed, scale, bias, columns) -> str:
    samples, rows = flat.shape[0], packed.shape[0]
    # Retention belongs in the key, since it changes what the paths cost
    # relative to each other. At batch one in fp16 the lane kernel beat an
    # expansion recomputed on every call, then kept winning after that
    # expansion became a dictionary lookup at half the cost, because the
    # ranking had been measured under the old economics. Clearing the table on
    # every budget change fixed that and broke benchmarks that toggle the
    # budget to compare both regimes, which then printed an empty table. Keyed,
    # both regimes are measured once and both stay visible.
    key = (flat.device.index, rows, columns, samples, bias is not None,
           flat.dtype, _RETAIN_BYTES > 0)
    decided = _FUSED_CHOICE.get(key)
    if decided is not None:
        return decided
    capturing = getattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    if capturing():
        return EXPANDED  # never time or allocate inside a captured region
    profile = _device.profile(flat.device)
    candidates = {
        TILED: lambda: _run_tiled(flat, packed, scale, bias, columns),
        EXPANDED: lambda: _expanded_linear(flat, packed, scale, bias, columns),
    }
    # Each blocking is timed under its own name, so a report says which one
    # won and not merely that "fused" did. Neighbouring plans exist to show
    # whether the derived point is the bottom, which one shared label hides.
    for split in (False, True):
        for index, plan in enumerate(small_batch_variants(
            samples, rows, columns, profile, allow_split=split
        )):
            name = f"{FUSED_SPLIT if split else FUSED}-n{plan[2]}"
            if name in candidates:
                continue
            candidates[name] = (
                lambda plan=plan, split=split: _run_small_batch(
                    flat, packed, scale, bias, columns, split, plan
                )
            )
    _FUSED_PLANS[key] = {
        f"{FUSED_SPLIT if split else FUSED}-n{plan[2]}": (plan, split)
        for split in (False, True)
        for plan in small_batch_variants(samples, rows, columns, profile,
                                         allow_split=split)
    }
    try:
        measured = {name: _time_path(run) for name, run in candidates.items()}
    except Exception:  # pragma: no cover - a build that cannot time is a build
        _FUSED_CHOICE[key] = EXPANDED  # that takes the path which always works
        return EXPANDED
    _FUSED_TIMINGS[key] = measured
    _FUSED_CHOICE[key] = min(measured, key=measured.get)
    return _FUSED_CHOICE[key]


# How much of an operand may be expanded at once, how many independent
# accumulators a reduction splits into, and where the small-batch crossover
# falls all come from :mod:`qste.kernels.device` -- derived from what the device
# reports and from timing it, never from which device it is.


def _tile(rows: int, columns: int, dtype: torch.dtype, device=None) -> int:
    return _device.profile(device or torch.device("cuda")).tile_rows(rows, columns, dtype)


def _partitions(device, work: int) -> int:
    """Independent accumulator slots for a reduction over ``work`` programs."""

    return max(1, min(int(work), _device.profile(device).partitions))


def expand_plan(rows: int, columns: int, profile=None):
    """Blockings the expansion is willing to try, cheapest guess first.

    Returns a list of ``(kind, first, second)`` -- ``("flat", block, None)`` for
    the element-per-program form, ``("tiled", block_rows, block_bytes)`` for the
    byte-once one. Pure arithmetic, testable anywhere.

    A list, because the winner is a property of the part and not of the
    arithmetic. Fewer instructions per element does not mean fewer
    microseconds: a tile wide enough to amortize the byte load holds thousands
    of floats in registers, and can give back in occupancy several times what
    it saves in issue.

    Flat entries come first, so the default -- what a captured region gets, and
    what runs before anything is timed -- has the longest measured record.
    """

    packed_bytes = (columns + 7) // 8
    plans = [("flat", 1024, None), ("flat", 512, None), ("flat", 2048, None)]

    widest = max(1, min(16, triton.next_power_of_2(packed_bytes)))
    block_bytes = widest
    while block_bytes >= 1:
        block_rows = max(1, min(16, triton.next_power_of_2(max(1, rows))))
        while block_rows > 1 and block_rows * block_bytes * 8 > 512:
            block_rows //= 2
        if profile is not None:
            # A bias-sized or head-sized matrix can otherwise land on a couple
            # of programs and leave the device idle for the whole launch.
            while (
                block_rows > 1
                and triton.cdiv(rows, block_rows)
                * triton.cdiv(packed_bytes, block_bytes)
                < profile.partitions
            ):
                block_rows //= 2
        candidate = ("tiled", block_rows, block_bytes)
        if candidate not in plans:
            plans.append(candidate)
        if block_bytes == 1:
            break
        block_bytes //= 2
    return plans


def expand_grid(plan, rows: int, columns: int):
    """The launch grid for one plan. Separated so it can be checked on a CPU."""

    kind, first, second = plan
    if kind == "flat":
        return (triton.cdiv(rows * columns, first),)
    packed_bytes = (columns + 7) // 8
    return triton.cdiv(rows, first), triton.cdiv(packed_bytes, second)


_EXPAND_CHOICE: dict[tuple, tuple] = {}
_EXPAND_TIMINGS: dict[tuple, dict[str, float]] = {}


def _run_expand(plan, packed, scale, out, columns):
    rows, packed_bytes = packed.shape
    kind, first, second = plan
    grid = expand_grid(plan, rows, columns)
    if kind == "flat":
        _expand_flat[grid](
            packed, packed if scale is None else scale, out,
            rows * columns, C=columns, PB=packed_bytes, BLOCK=first,
            SCALED=scale is not None, num_warps=4,
        )
    else:
        _expand_tiled[grid](
            packed, packed if scale is None else scale, out,
            rows, C=columns, PB=packed_bytes, BR=first, BB=second,
            SCALED=scale is not None, num_warps=4,
        )
    return out


def _choose_expand(packed, scale, out, columns) -> tuple:
    """Time every blocking once for this shape and remember the winner.

    The expansion sits under the forward at any batch BLAS should own, under
    the input gradient, and under the evidence product, so a blocking that is
    wrong here is wrong three times per step. It was previously a formula, and
    the formula was hand-derived twice and wrong twice -- the second time by
    enough to cost a benchmark run. Everything else in this file that could not
    be predicted is measured instead, and this is no different.
    """

    rows = packed.shape[0]
    key = (packed.device.index, rows, columns, out.dtype, scale is not None)
    decided = _EXPAND_CHOICE.get(key)
    if decided is not None:
        return decided

    default = ("flat", 1024, None)
    capturing = getattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    if capturing():
        return default  # never time or allocate inside a captured region

    plans = expand_plan(rows, columns, _device.profile(packed.device))
    measured = {}
    for plan in plans:
        try:
            measured[plan] = _time_path(
                lambda plan=plan: _run_expand(plan, packed, scale, out, columns)
            )
        except Exception:  # a blocking this build will not compile is not a bug
            measured[plan] = float("inf")
    live = {plan: cost for plan, cost in measured.items() if cost != float("inf")}
    _EXPAND_TIMINGS[key] = {f"{k}-{a}-{b}": v for (k, a, b), v in measured.items()}
    _EXPAND_CHOICE[key] = min(live, key=live.get) if live else default
    return _EXPAND_CHOICE[key]


def expand_timings(device=None) -> dict[tuple, dict[str, float]]:
    """What each blocking measured, keyed by shape. See :func:`timings`."""

    if device is None:
        return dict(_EXPAND_TIMINGS)
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    return {key: value for key, value in _EXPAND_TIMINGS.items() if key[0] == index}


def expand_dense(packed: Tensor, columns: int, *, scale: Tensor | None = None,
                 dtype: torch.dtype = torch.float32, out: Tensor | None = None
                 ) -> Tensor:
    rows = packed.shape[0]
    if out is None:
        out = torch.empty(rows, columns, device=packed.device, dtype=dtype)
    return _run_expand(
        _choose_expand(packed, scale, out, columns), packed, scale, out, columns
    )


# ---------------------------------------------------------------------------
# Not re-expanding a weight that has not changed
# ---------------------------------------------------------------------------
#
# Expansion is a fixed cost that depends only on the weight's size, so it does
# not shrink with the batch. On a T4 at width 2048 it runs 0.088 ms in fp32 and
# 0.059 ms in fp16, while the entire fp16 product at batch 32 takes 0.050 ms.
# In that range the expansion costs more than the multiply it feeds, and no
# kernel work changes that -- expanding to feed a small product loses by
# construction.
#
# The fix is expanding once instead of every call. At inference the packed
# weight is frozen: a generation loop calls the same layer with the same bits
# thousands of times, recomputing a value that was already correct.
#
# ``retain`` keeps expanded weights in a buffer with a byte budget. A call
# whose weight is already resident goes straight to cuBLAS and costs what the
# float path costs, at any batch and any precision. It is off by default,
# because what it spends is the memory this library exists to save, so the
# caller sets the budget. Storage stays one bit per element either way: the
# cache is scratch and the packed weight remains the source of truth.
_RETAINED: dict[tuple, Tensor] = {}
_RETAIN_BYTES = 0
_RETAINED_HITS = 0
_RETAINED_MISSES = 0


def retain(budget_bytes: int) -> None:
    """Keep expanded weights, up to ``budget_bytes``. ``0`` disables and frees.

    Sized in bytes rather than layers because layers are not the same size and
    the caller is trading against a real memory limit, not a count.
    """

    global _RETAIN_BYTES
    _RETAIN_BYTES = max(0, int(budget_bytes))

    if _RETAIN_BYTES == 0:
        _RETAINED.clear()
    else:
        _evict()


def retained_stats() -> dict:
    """Hits, misses, resident bytes and the budget. For deciding if it paid."""

    resident = sum(d.numel() * d.element_size() for _, d in _RETAINED.values())
    return {
        "hits": _RETAINED_HITS, "misses": _RETAINED_MISSES,
        "resident_bytes": resident, "budget_bytes": _RETAIN_BYTES,
        "entries": len(_RETAINED),
    }


def _evict() -> None:
    # Insertion-ordered, so the oldest goes first. A weight that is called every
    # step and one that is called once are not distinguished, which is the right
    # simplification for a forward pass that visits every layer in order.
    resident = sum(d.numel() * d.element_size() for _, d in _RETAINED.values())
    for key in list(_RETAINED):
        if resident <= _RETAIN_BYTES:
            return
        _, held = _RETAINED.pop(key)
        resident -= held.numel() * held.element_size()


def invalidate(packed: Tensor) -> None:
    """Drop anything expanded from these bits. Call after writing them.

    Required, not defensive. The natural way to detect a changed weight is
    torch's version counter, and **a Triton kernel writing through a pointer
    does not bump it.** The coordinate optimizer is such a kernel: every step
    wrote new bits, the counter stayed put, the cache kept serving the weight
    the model started with, and training ran to completion at chance -- no
    error, no warning, plausible loss numbers, nothing learned.

    So the writer announces its writes. The counter stays in the key since it
    costs nothing and does catch torch-level writes, but nothing depends on it.
    """

    if not _RETAINED:
        return
    address = packed.data_ptr()
    for key in [key for key in _RETAINED if key[0] == address]:
        del _RETAINED[key]


def _expanded_weight(packed, scale, columns, dtype):
    """The dense weight, from the cache when it is there and allowed."""

    global _RETAINED_HITS, _RETAINED_MISSES
    if _RETAIN_BYTES == 0 or torch.is_grad_enabled():
        return expand_dense(packed, columns, scale=scale, dtype=dtype)

    key = (
        packed.data_ptr(), packed._version, packed.shape, columns, dtype,
        None if scale is None else (scale.data_ptr(), scale._version),
    )
    held = _RETAINED.get(key)
    if held is not None:
        _RETAINED_HITS += 1
        return held[1]

    _RETAINED_MISSES += 1
    dense = expand_dense(packed, columns, scale=scale, dtype=dtype)
    cost = dense.numel() * dense.element_size()
    if cost <= _RETAIN_BYTES:
        # The packed tensor is kept alive alongside its expansion, and not for
        # sentiment: the key starts with an address, and a freed tensor's
        # address is handed straight back out by the caching allocator. Holding
        # a reference makes that address unreusable for as long as anything is
        # keyed on it, which is the difference between a stale hit being
        # unlikely and being impossible. It costs a thirty-second of what the
        # expansion beside it already costs.
        _RETAINED[key] = (packed, dense)
        _evict()
    return dense


def pack_bits(mask: Tensor) -> Tensor:
    # Viewed, not converted: torch stores bool as one byte, so this is the same
    # storage read as uint8 rather than an 8x temporary of the thing being
    # packed to save memory.
    source = mask.contiguous()
    source = source.view(torch.uint8) if source.dtype == torch.bool else source
    rows, columns = source.shape
    packed_bytes = (columns + 7) // 8
    packed = torch.empty(rows, packed_bytes, device=source.device, dtype=torch.uint8)
    block = min(128, triton.next_power_of_2(packed_bytes))
    _pack_bit_rows[(rows, triton.cdiv(packed_bytes, block))](
        source, packed, rows, K=columns, PB=packed_bytes, BB=block, num_warps=4
    )
    return packed


def apply_bits(values: Tensor, packed: Tensor, columns: int) -> Tensor:
    values = values.contiguous()
    packed = packed.contiguous()
    rows = values.shape[0]
    out = torch.empty_like(values)
    block = 1024
    _apply_bit_mask[(triton.cdiv(rows * columns, block),)](
        values, packed, out, rows * columns, C=columns, PB=packed.shape[1],
        BLOCK=block,
        num_warps=4,
    )
    return out



@triton.jit
def _packed_row_inner(Matrix, Packed, Out, N, K: tl.constexpr,
                      PB: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    acc = tl.zeros((BK,), tl.float32)
    for k0 in range(0, K, BK):
        columns = k0 + tl.arange(0, BK)
        value = tl.load(Matrix + row * K + columns, mask=columns < K, other=0.0)
        byte = tl.load(Packed + row * PB + (columns >> 3), mask=columns < K, other=0)
        acc += value.to(tl.float32) * (
            (((byte >> (columns & 7)) & 1).to(tl.float32)) * 2.0 - 1.0
        )
    tl.store(Out + row, tl.sum(acc, axis=0).to(Out.dtype.element_ty), mask=row < N)


@triton.jit
def _packed_embedding(Ids, Packed, Scale, Out, Total,
                      K: tl.constexpr, PB: tl.constexpr, BLOCK: tl.constexpr):
    # ``Total``, not a token count, for the reason on ``_expand_flat``:
    # a one-token step would otherwise be the shape that fails to compile.
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = index < Total
    position = index // K
    column = index - position * K
    token = tl.load(Ids + position, mask=mask, other=0).to(tl.int64)
    byte = tl.load(Packed + token * PB + (column >> 3), mask=mask, other=0)
    sign = (((byte >> (column & 7)) & 1).to(tl.float32) * 2.0 - 1.0)
    tl.store(Out + index, sign * tl.load(Scale + token, mask=mask, other=0.0).to(tl.float32),
             mask=mask)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


@triton.jit
def _pack_affine_rows(Source, Packed, Offset, Scale, K: tl.constexpr,
                      PB: tl.constexpr, BB: tl.constexpr):
    """One-bit affine encoding of one row. See ``cpu.cpp`` for why it centers.

    One program owns a whole row, so both reductions are register-local and
    no atomics or second launch are needed.
    """

    row = tl.program_id(0)
    lane = tl.arange(0, 8)
    total = tl.zeros((), tl.float32)
    for b0 in range(0, PB, BB):
        byte_index = b0 + tl.arange(0, BB)
        column = byte_index[:, None] * 8 + lane[None, :]
        live = (byte_index[:, None] < PB) & (column < K)
        value = tl.load(Source + row * K + column, mask=live, other=0.0).to(tl.float32)
        total += tl.sum(tl.sum(tl.where(live, value, 0.0), axis=1), axis=0)
    mean = total / K

    deviation = tl.zeros((), tl.float32)
    for b0 in range(0, PB, BB):
        byte_index = b0 + tl.arange(0, BB)
        column = byte_index[:, None] * 8 + lane[None, :]
        live = (byte_index[:, None] < PB) & (column < K)
        centered = tl.load(Source + row * K + column, mask=live, other=0.0).to(tl.float32) - mean
        deviation += tl.sum(tl.sum(tl.where(live, tl.abs(centered), 0.0), axis=1), axis=0)
        byte = tl.sum(tl.where(live & (centered >= 0.0), 1 << lane[None, :], 0), axis=1)
        tl.store(Packed + row * PB + byte_index, byte.to(tl.uint8), mask=byte_index < PB)
    tl.store(Offset + row, mean)
    tl.store(Scale + row, deviation / K)


@triton.jit
def _pack_coordinate_rows(Source, Packed, K: tl.constexpr, PB: tl.constexpr,
                          BB: tl.constexpr):
    """Sign bits of an INT8 coordinate matrix. No centering: it is signed."""

    row = tl.program_id(0)
    byte_index = tl.program_id(1) * BB + tl.arange(0, BB)
    lane = tl.arange(0, 8)
    column = byte_index[:, None] * 8 + lane[None, :]
    live = (byte_index[:, None] < PB) & (column < K)
    value = tl.load(Source + row * K + column, mask=live, other=0)
    byte = tl.sum(tl.where(live & (value >= 0), 1 << lane[None, :], 0), axis=1)
    tl.store(Packed + row * PB + byte_index, byte.to(tl.uint8), mask=byte_index < PB)


# ---------------------------------------------------------------------------
# Coordinate optimizer
# ---------------------------------------------------------------------------


# No atomics in this section.
#
# An earlier version accumulated every reduction into a single address with
# tl.atomic_add, from as many programs as the matrix had blocks -- tens of
# thousands, all serialized on one cache line. It cost 19 ms per step on a
# six-layer model whose entire forward and backward took 137 ms, and would cost
# about that on any parallel machine, since contention on one address is not a
# property of a particular card.
#
# Every reduction below writes a private partial, and a small second pass
# combines them: a few hundred kilobytes of scratch and one extra launch,
# against a hard serialization everywhere.


@triton.jit
def _row_col_squares(Grad, RowSum, ColPartial, N, K: tl.constexpr,
                     BM: tl.constexpr, BK: tl.constexpr):
    """Row and column sums of squares in one pass, no atomics.

    Each program owns a strip of rows: its row sums are register-local, and
    its column contributions go to a private slice that the caller reduces.
    """

    pid = tl.program_id(0)
    rows = (pid * BM + tl.arange(0, BM)).to(tl.int64)
    live_row = rows < N
    acc = tl.zeros((BM,), tl.float32)
    for k0 in range(0, K, BK):
        columns = k0 + tl.arange(0, BK)
        live = live_row[:, None] & (columns[None, :] < K)
        value = tl.load(Grad + rows[:, None] * K + columns[None, :], mask=live, other=0.0)
        square = value * value
        acc += tl.sum(square, axis=1)
        tl.store(ColPartial + pid.to(tl.int64) * K + columns,
                 tl.sum(square, axis=0), mask=columns < K)
    tl.store(RowSum + rows, acc, mask=live_row)


@triton.jit
def _finish_factors(Sum, State, UpdateEnabled, Divisor: tl.constexpr,
                    Count: tl.constexpr, Beta2: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < Count
    mean = tl.load(Sum + index, mask=mask, other=0.0) / Divisor + 1e-12
    old = tl.load(State + index, mask=mask, other=0.0).to(tl.float32)
    updated = old * Beta2 + mean * (1.0 - Beta2)
    tl.store(Sum + index, updated, mask=mask)
    tl.store(State + index, updated, mask=mask & (tl.load(UpdateEnabled) != 0))


@triton.jit
def _precondition(Grad, RowV, ColV, RowMean, SquarePartial, Count,
                  K: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0)
    index = program.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = index < Count
    row = index // K
    column = index - row * K
    grad = tl.load(Grad + index, mask=mask, other=0.0)
    row_v = tl.maximum(tl.load(RowV + row, mask=mask, other=1e-10).to(tl.float32), 1e-10)
    col_v = tl.maximum(tl.load(ColV + column, mask=mask, other=1e-10).to(tl.float32), 1e-10)
    update = grad * tl.sqrt(tl.maximum(tl.load(RowMean), 1e-10) / (row_v * col_v))
    tl.store(Grad + index, update, mask=mask)
    tl.store(SquarePartial + program, tl.sum(tl.where(mask, update * update, 0.0), axis=0))


@triton.jit
def _moment_and_scale(Grad, MomentQ, OldMomentScale, MomentScale,
                      UpdateSquareSum, MomentPartial, UpdateEnabled, Count,
                      BlockSize: tl.constexpr,
                      Beta1: tl.constexpr, UpdateClip: tl.constexpr):
    block = tl.program_id(0)
    index = block.to(tl.int64) * BlockSize + tl.arange(0, BlockSize).to(tl.int64)
    mask = index < Count
    update_rms = tl.sqrt(tl.maximum(tl.load(UpdateSquareSum) / Count, 1e-16))
    divisor = tl.maximum(1.0, update_rms / UpdateClip)
    update = tl.load(Grad + index, mask=mask, other=0.0) / divisor
    previous = tl.load(MomentQ + index, mask=mask, other=0).to(tl.float32)
    old_scale = tl.load(OldMomentScale + block).to(tl.float32)
    moment = previous * old_scale * Beta1 + update * (1.0 - Beta1)
    tl.store(Grad + index, moment, mask=mask)
    tl.store(MomentPartial + block, tl.sum(tl.where(mask, moment * moment, 0.0), axis=0))
    maximum = tl.maximum(tl.max(tl.where(mask, tl.abs(moment), 0.0), axis=0), 1e-6)
    tl.store(MomentScale + block, maximum / 127.0, mask=tl.load(UpdateEnabled) != 0)


@triton.jit
def _low32(value):
    """The low 32 bits of a non-negative 64-bit value.

    ``value & 0xFFFFFFFF`` is the obvious way to write this and it does not
    compile. Triton types a bare integer literal as int32, and 0xFFFFFFFF does
    not fit in one -- so the mask whose whole purpose is to make this
    arithmetic portable is itself the part that is not. Two shifts say the same
    thing with literals no compiler has to make a choice about.

    Two preconditions, both enforced by the caller.

    The value must be **int64**. Shifting an int32 right by 32 is undefined and
    Triton warns instead of erroring, so an int32 argument does not fail -- it
    silently returns something other than the low 32 bits. That shipped once:
    the caller built its index from ``tl.arange``, which is int32, giving a
    rounding stream of garbage that ran, trained, and did not learn.

    The value must be **non-negative**, which is what makes the arithmetic
    shift a logical one. ``_hash32`` keeps its running value below 2**32 and
    its widest intermediate is 2**32 * 0x27D4EB2D, comfortably inside a signed
    64-bit register.
    """

    return value - ((value >> 32) << 32)


@triton.jit
def _hash32(value):
    """A 32-bit avalanche, evaluated in 64-bit registers and masked each step.

    Masking explicitly, instead of relying on a 32-bit type's wraparound, is
    what keeps a checkpoint portable. The same hash written against ``uint32``
    in C++ and Triton's ``int32`` disagrees twice over: shifts are logical on
    one and arithmetic on the other once the high bit is set, and signed
    overflow wraps on a GPU while being undefined to a C++ optimizer.

    Every constant here stays below 2**31, so nothing depends on how a backend
    types an integer literal. An earlier version used 0xFFFFFFFF and 0x9E3779B9
    directly and failed to compile.

    This is what makes a coordinate matrix trained on a GPU continue on a CPU
    and land on the same integers. It is also why the update is
    rank-identical under DDP without communicating anything: the randomness is
    a function of (seed, step, flat index), not of an RNG that each rank
    advances at its own rate.
    """

    value = _low32((value ^ 61) ^ (value >> 16))
    value = _low32(value + (value << 3))
    value = _low32(value ^ (value >> 4))
    value = _low32(value * 0x27D4EB2D)
    return _low32(value ^ (value >> 15))


@triton.jit
def _round_nearest_even(value):
    lower = tl.floor(value)
    fraction = value - lower
    increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    return lower + increment.to(tl.float32)


@triton.jit
def _coordinate_and_pack(
    Moment, Coordinate, Packed, MomentQ, MomentScale, MomentSquareSum, FlipPartial,
    StepCounter, UpdateEnabled,
    # Counts, not dimensions, and therefore runtime. A dimension is bounded by
    # what a layer is wide; an element count is their product, and a surface
    # with more than 2**31 elements is an ordinary vocabulary embedding at
    # scale. As a constexpr those become bare literals, which Triton types as
    # int32 -- the same failure that took the optimizer out entirely, one line
    # away from where it happened. Runtime arguments are typed from the value.
    Count, PaddedCount, PackedBytes,
    # Runtime for the same reason, and here the distinction is load-bearing. A
    # hashed seed is a full 32-bit value, so roughly half of all seeds produce
    # one that does not fit in an int32 -- and a constexpr integer is a bare
    # literal whose type Triton picks. Passed as an argument it is typed from
    # the value it actually has, which is the one thing guaranteed to be right.
    SeedHash,
    N: tl.constexpr, K: tl.constexpr, PB: tl.constexpr,
    BlockSize: tl.constexpr, ByteBlock: tl.constexpr,
    CoordinateLR: tl.constexpr,
):
    program = tl.program_id(0)
    byte_index = program * ByteBlock + tl.arange(0, ByteBlock)
    row = byte_index // PB
    byte_column = byte_index - row * PB
    lane = tl.arange(0, 8)
    column = byte_column[:, None] * 8 + lane[None, :]
    row = row[:, None]
    # int64, and not only because the hash below shifts it by 32 -- an int32
    # shifted by its own width is undefined, and Triton says so in a warning
    # nobody reads, which is how a silently garbage random stream shipped. It
    # also has to be int64 to address a surface at all: this is a flat element
    # index, so a vocabulary-sized embedding passes 2**31 and an int32 wraps
    # into another row's memory without a word of complaint.
    index = row.to(tl.int64) * K + column
    byte_mask = byte_index < PackedBytes
    mask = byte_mask[:, None] & (row < N) & (column < K) & (index < Count)

    raw_moment = tl.load(Moment + index, mask=mask, other=0.0)
    moment_rms = tl.sqrt(tl.maximum(tl.load(MomentSquareSum) / PaddedCount, 1e-16))
    coordinate = tl.load(Coordinate + index, mask=mask, other=1).to(tl.float32)
    target = coordinate - CoordinateLR * raw_moment / moment_rms
    # The step is read from device memory rather than baked in, so a captured
    # graph draws fresh randomness on every replay instead of repeating the one
    # step that happened to be current when it was recorded.
    step_value = _low32(tl.load(StepCounter).to(tl.int64))
    # 0x2545F491 is qste.kernels.stream.SALT, written out rather than
    # referenced. A kernel cannot read a module-level Python constant at all --
    # Triton rejects any global that is not a tl.constexpr instance -- so the
    # one place this is allowed to live is inline. The test suite reads this
    # literal back out of the source and compares it to the definition, which
    # is how it is kept from drifting without depending on a mechanism that
    # does not exist.
    hash_seed = SeedHash ^ _hash32(step_value ^ 0x2545F491)
    counter = _low32(index) ^ hash_seed
    lower = tl.floor(target)
    random = _hash32(counter).to(tl.float32) * (1.0 / 4294967296.0)
    next_coordinate = tl.maximum(
        tl.minimum(lower + (random < target - lower).to(tl.float32), 127.0), -127.0
    )
    enabled = tl.load(UpdateEnabled) != 0
    commit = mask & enabled
    changed = commit & ((coordinate >= 0.0) != (next_coordinate >= 0.0))
    tl.store(FlipPartial + program, tl.sum(tl.sum(changed.to(tl.int32), axis=1), axis=0))
    tl.store(Coordinate + index, next_coordinate.to(tl.int8), mask=commit)

    scale = tl.maximum(
        tl.load(MomentScale + index // BlockSize, mask=mask, other=1e-12).to(tl.float32),
        1e-12,
    )
    quantized = tl.maximum(tl.minimum(_round_nearest_even(raw_moment / scale), 127.0), -127.0)
    tl.store(MomentQ + index, quantized.to(tl.int8), mask=commit)

    bits = tl.where(mask & (next_coordinate >= 0.0), 1 << lane[None, :], 0)
    next_byte = tl.sum(bits, axis=1).to(tl.uint8)
    tl.store(Packed + byte_index, next_byte, mask=byte_mask & enabled)


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def _flat(tensor: Tensor, width: int) -> Tensor:
    return tensor.reshape(-1, width).contiguous()


_LAUNCH: dict[tuple, tuple] = {}


def forget() -> None:
    """Drop every remembered decision, so the next call measures again.

    Needed by anything that moves the ground the measurements stood on -- a
    test substituting a device profile, or a caller re-timing after the
    machine's state changed. Retention does not need it, since the budget is
    part of the key and both regimes coexist.
    """

    _LAUNCH.clear()
    _FUSED_CHOICE.clear()
    _FUSED_PLANS.clear()
    _FUSED_TIMINGS.clear()
    _EXPAND_CHOICE.clear()
    _EXPAND_TIMINGS.clear()


def _resolve(key, flat, packed, scale, bias, columns):
    """Settle everything about this shape once: path, blocking, tiling.

    Kept separate from the hot path below because it is the expensive half and
    it runs once per shape. Everything it returns is a plain value the launcher
    can act on without asking anything else.
    """

    rows, samples = packed.shape[0], flat.shape[0]
    profile = _device.profile(flat.device)
    tile = profile.tile_rows(rows, columns, flat.dtype)

    if samples > profile.fused_batch_limit:
        _LAUNCH[key] = resolved = (EXPANDED, None, False, tile)
        return resolved

    chosen = _choose_path(flat, packed, scale, bias, columns)
    plans = _FUSED_PLANS.get(key, {})

    def dress(name):
        held, split = plans.get(name, (None, name.startswith(FUSED_SPLIT)))
        return (name, held, split, tile)

    # The candidates above were timed as kernels, but what ships is a kernel
    # plus its launcher, and at small batch the launcher is not a rounding
    # error. At one row in fp16 the best kernel measured 0.036 ms against a
    # float path at 0.040 -- a win -- while the call shipped 0.060. Ranking on
    # kernel time answers which kernel is fastest, not which call is.
    #
    # So the finalists are re-timed through the real entry point, with the
    # decision installed so the timed call takes it. Only the top few, since
    # this is the expensive half of a once-per-shape path and a candidate that
    # lost by a factor on the kernel will not win it back on dispatch.
    measured = _FUSED_TIMINGS.get(key, {})
    finalists = [
        name for _, name in sorted(
            (cost, name) for name, cost in measured.items() if cost != float("inf")
        )[:3]
    ] or [chosen]

    best, winner = float("inf"), dress(chosen)
    try:
        for name in finalists:
            candidate = dress(name)
            _LAUNCH[key] = candidate  # so the timed call dispatches, not re-resolves
            cost = _time_path(
                lambda: packed_linear_affine(flat, packed, scale, bias, columns)
            )
            if cost < best:
                best, winner = cost, candidate
    except Exception:  # a build that cannot time keeps the kernel-time winner
        winner = dress(chosen)

    _FUSED_CHOICE[key] = winner[0]
    _LAUNCH[key] = winner
    return winner


def packed_linear_affine(
    inputs: Tensor, packed: Tensor, scale: Tensor, bias: Tensor | None, columns: int
) -> Tensor:
    # Every line here runs on every forward of every converted layer. With the
    # weight already expanded, the best measured kernel for a single row was
    # 0.040 ms and the call shipped 0.070 -- thirty microseconds of Python
    # around a forty microsecond kernel, none of it arithmetic.
    #
    # So the shape resolves once and every later call is one dictionary lookup.
    # Previously each call fetched a device profile twice (building a
    # torch.device and formatting a string key each time), recomputed a tile,
    # and built and hashed two separate seven-field tuples to find the path and
    # then its blocking. The conversions stay, since they return the same
    # tensor when there is nothing to do, but nothing is derived twice.
    if not packed.is_contiguous():
        packed = packed.contiguous()
    flat = _flat(inputs, columns)
    rows = packed.shape[0]
    if scale.dtype is not torch.float32 or scale.device != flat.device:
        scale = scale.to(device=flat.device, dtype=torch.float32)
    if not scale.is_contiguous():
        scale = scale.contiguous()

    key = (flat.device.index, rows, columns, flat.shape[0], bias is not None,
           flat.dtype, _RETAIN_BYTES > 0)
    resolved = _LAUNCH.get(key)
    if resolved is None:
        resolved = _resolve(key, flat, packed, scale, bias, columns)
    chosen, plan, split, tile = resolved

    if chosen != EXPANDED:
        if chosen == TILED:
            tiled = _run_tiled(flat, packed, scale, bias, columns)
            if tiled is not None:
                return tiled.view(*inputs.shape[:-1], rows)
        else:
            fused = _run_small_batch(
                flat, packed, scale, bias, columns, split, plan
            )
            if fused is not None:
                return fused.view(*inputs.shape[:-1], rows)
    return _expanded_linear(flat, packed, scale, bias, columns, tile).view(
        *inputs.shape[:-1], rows
    )


def _expanded_linear(flat, packed, scale, bias, columns, tile=None):
    rows = packed.shape[0]
    if tile is None:
        tile = _tile(rows, columns, flat.dtype, flat.device)

    # The common case: the whole weight fits the scratch budget, so there is
    # one tile and nothing to accumulate. It gets its own path because the loop
    # below is written for the accumulating case, and that structure is
    # expensive here in a way that never shows up as a slow kernel.
    #
    # The loop allocates the output, lets cuBLAS allocate a second full-size
    # result, copies one into the other, then makes a third pass for the bias:
    # three trips over a [batch, rows] tensor where one would do. At batch 512
    # and width 2048 that is about twelve megabytes of traffic for nothing, on
    # the forward of every layer wide enough to take this route.
    if tile >= rows:
        weight = _expanded_weight(packed, scale, columns, flat.dtype)
        if bias is None:
            return flat @ weight.t()
        # cuBLAS applies the bias inside the epilogue, so this is the same
        # single pass rather than a product followed by an addition.
        return torch.addmm(bias.to(flat.dtype), flat, weight.t())

    out = torch.empty(flat.shape[0], rows, device=flat.device, dtype=flat.dtype)
    for start in range(0, rows, tile):
        stop = min(start + tile, rows)
        weight = _expanded_weight(
            packed[start:stop], scale[start:stop], columns, flat.dtype
        )
        # Written straight into the output slice: ``out[:, a:b] = x @ w.t()``
        # would make cuBLAS allocate its own result and then copy it here.
        torch.mm(flat, weight.t(), out=out[:, start:stop])
    if bias is not None:
        out.add_(bias.to(flat.dtype).view(1, rows))
    return out


def packed_transpose(
    inputs: Tensor, packed: Tensor, columns: int, row_scale: Tensor | None = None
) -> Tensor:
    packed = packed.contiguous()
    rows = packed.shape[0]
    flat = _flat(inputs, rows)
    # The same scaled matrix the forward built. Folding the scale in here
    # rather than pre-multiplying the gradient saves a [batch, rows] temporary.
    if row_scale is not None:
        row_scale = row_scale.to(device=flat.device, dtype=torch.float32).contiguous()
    tile = _tile(rows, columns, flat.dtype, flat.device)

    # One tile means nothing accumulates, so the zero-fill and the accumulating
    # multiply are both pure overhead -- a full-size write of zeros followed by
    # a read-modify-write of the same tensor, to add a product to nothing. This
    # is the input gradient, so it runs on every backward of every layer.
    if tile >= rows:
        weight = _expanded_weight(packed, row_scale, columns, flat.dtype)
        return (flat @ weight).view(*inputs.shape[:-1], columns)

    out = torch.zeros(flat.shape[0], columns, device=flat.device, dtype=flat.dtype)
    for start in range(0, rows, tile):
        stop = min(start + tile, rows)
        weight = _expanded_weight(
            packed[start:stop],
            None if row_scale is None else row_scale[start:stop],
            columns, flat.dtype,
        )
        out.addmm_(flat[:, start:stop], weight)
    return out.view(*inputs.shape[:-1], columns)


def evidence_from_packed(
    grad: Tensor, packed: Tensor, columns: int, row_scale: Tensor | None = None
) -> Tensor:
    """``gradᵀ @ sign(x)``, expanding the packed activation a tile at a time.

    Tiling is over samples, so the scratch is bounded no matter how long the
    sequence is, while the tensor that survived from forward is still one bit
    per element.
    """

    grad = grad.contiguous()
    packed = packed.contiguous()
    samples, rows = grad.shape
    profile = _device.profile(grad.device)

    # The one product in QSTE whose result does not stay a float: it is
    # preconditioned and stochastically rounded into an INT8 coordinate within
    # microseconds. Its mantissa carries nothing, so it can run in whatever
    # dtype the machine multiplies fastest -- measured once, per machine.
    #
    # Its *exponent range* does matter, since a gradient that underflows to
    # zero is a coordinate that never moves. bf16 has fp32's range and needs
    # nothing. fp16 does not, so the gradient is first scaled onto a power of
    # two, which is exact in binary floating point and undone exactly at the
    # end.
    compute = grad.dtype if grad.dtype in (torch.float16, torch.bfloat16) else profile.reduction_dtype
    factor = None
    if compute == torch.float16 and grad.dtype != torch.float16:
        factor = _device.rescale_factor(grad)

    accumulator = torch.zeros(rows, columns, device=grad.device, dtype=torch.float32)
    tile = _tile(samples, columns, compute, grad.device)
    if row_scale is not None:
        row_scale = row_scale.to(device=grad.device, dtype=torch.float32).contiguous()
    for start in range(0, samples, tile):
        stop = min(start + tile, samples)
        signs = expand_dense(
            packed[start:stop], columns,
            scale=None if row_scale is None else row_scale[start:stop],
            dtype=compute,
        )
        block = grad[start:stop]
        block = block if factor is None else block * factor
        # The running total stays fp32 whatever the operands are. Inside one
        # tile cuBLAS already accumulates in fp32; this keeps that true across
        # tiles as well, so a long sequence does not lose the tail of its sum.
        accumulator.add_(torch.mm(block.to(compute).t(), signs))
    return accumulator if factor is None else accumulator.div_(factor)


def pack_affine_rows(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:

    values = values.contiguous()
    rows, columns = values.shape
    packed_bytes = (columns + 7) // 8
    packed = torch.empty(rows, packed_bytes, device=values.device, dtype=torch.uint8)
    offset = torch.empty(rows, device=values.device, dtype=torch.float32)
    scale = torch.empty(rows, device=values.device, dtype=torch.float32)
    block = min(128, triton.next_power_of_2(packed_bytes))
    _pack_affine_rows[(rows,)](
        values, packed, offset, scale, K=columns, PB=packed_bytes, BB=block,
        num_warps=4,
    )
    return packed, offset, scale


def pack_coordinate(coordinate: Tensor) -> Tensor:
    coordinate = coordinate.contiguous()
    rows, columns = coordinate.shape
    packed_bytes = (columns + 7) // 8
    packed = torch.empty(rows, packed_bytes, device=coordinate.device, dtype=torch.uint8)
    block = min(128, triton.next_power_of_2(packed_bytes))
    _pack_coordinate_rows[(rows, triton.cdiv(packed_bytes, block))](
        coordinate, packed, K=columns, PB=packed_bytes, BB=block, num_warps=4
    )
    return packed


def packed_row_inner(matrix: Tensor, packed: Tensor, columns: int) -> Tensor:
    matrix = matrix.contiguous()
    packed = packed.contiguous()
    rows = matrix.shape[0]
    out = torch.empty(rows, device=matrix.device, dtype=matrix.dtype)
    block = min(1024, triton.next_power_of_2(columns))
    _packed_row_inner[(rows,)](
        matrix, packed, out, rows, K=columns, PB=packed.shape[1], BK=block,
        num_warps=8 if block >= 512 else 4,
    )
    return out


def packed_embedding(ids: Tensor, packed: Tensor, scale: Tensor, columns: int) -> Tensor:
    flat_ids = ids.long().reshape(-1).contiguous()
    packed = packed.contiguous()
    scale = scale.contiguous()
    count = flat_ids.numel()
    out = torch.empty(count, columns, device=ids.device, dtype=scale.dtype)
    block = 256
    _packed_embedding[(triton.cdiv(count * columns, block),)](
        flat_ids, packed, scale, out, count * columns, K=columns,
        PB=packed.shape[1], BLOCK=block, num_warps=4,
    )
    return out.view(*ids.shape, columns)


def coordinate_update(
    gradient: Tensor,
    coordinate: Tensor,
    packed: Tensor,
    moment_q: Tensor,
    moment_scale: Tensor,
    row_v: Tensor,
    col_v: Tensor,
    *,
    beta1: float,
    beta2: float,
    update_clip: float,
    coordinate_lr: float,
    block_size: int,
    seed: int,
    step,
    flips: Tensor | None = None,
    update_enabled: Tensor | None = None,
) -> int | Tensor:
    """One fused coordinate step. Returns flips on device when preallocated.

    ``flips`` and ``update_enabled`` exist for CUDA-graph capture: pass device
    tensors and the whole step records without a host synchronization, and
    warmup replays can be gated off by zeroing ``update_enabled``.
    """

    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("momentum block size must be a positive power of two")
    gradient = gradient.contiguous()
    rows, columns = gradient.shape
    count = gradient.numel()
    blocks = triton.cdiv(count, block_size)
    device = gradient.device
    if update_enabled is None:
        update_enabled = torch.ones((), device=device, dtype=torch.uint8)

    # One pass over the evidence produces both factored statistics. The column
    # partials are private per program, so nothing contends; how many programs
    # there are follows the device's own reported width.
    # Enough strips to fill the device, but never so many that the partial
    # buffer becomes a memory problem in its own right: a surface whose columns
    # are vocabulary-sized would otherwise allocate hundreds of megabytes of
    # partials to avoid an atomic. Both bounds are the device's own numbers.
    profile = _device.profile(device)
    affordable = max(1, profile.scratch_bytes // max(1, columns * 4))
    groups = max(1, min(rows, profile.partitions, affordable))
    strip = max(1, triton.next_power_of_2(triton.cdiv(rows, groups)))
    grid = triton.cdiv(rows, strip)
    row_sum = torch.empty(rows, device=device, dtype=torch.float32)
    # Every slot is written by exactly one program before it is read.
    col_partial = torch.empty(grid, columns, device=device, dtype=torch.float32)
    _row_col_squares[(grid,)](
        gradient, row_sum, col_partial, rows, K=columns,
        BM=strip, BK=min(256, triton.next_power_of_2(columns)), num_warps=4,
    )
    col_sum = col_partial.sum(0)

    vector_block = 256
    _finish_factors[(triton.cdiv(rows, vector_block),)](
        row_sum, row_v, update_enabled, Divisor=float(columns), Count=rows,
        Beta2=beta2, BLOCK=vector_block,
    )
    _finish_factors[(triton.cdiv(columns, vector_block),)](
        col_sum, col_v, update_enabled, Divisor=float(rows), Count=columns,
        Beta2=beta2, BLOCK=vector_block,
    )

    element_block = 1024
    element_programs = triton.cdiv(count, element_block)
    square_partial = torch.empty(element_programs, device=device, dtype=torch.float32)
    _precondition[(element_programs,)](
        gradient, row_sum, col_sum, row_sum.mean(), square_partial, count,
        K=columns, BLOCK=element_block,
    )
    update_square_sum = square_partial.sum()

    moment_partial = torch.empty(blocks, device=device, dtype=torch.float32)
    _moment_and_scale[(blocks,)](
        gradient, moment_q, moment_scale, moment_scale,
        update_square_sum, moment_partial, update_enabled, count,
        BlockSize=block_size, Beta1=beta1, UpdateClip=update_clip,
        num_warps=4,
    )
    moment_square_sum = moment_partial.sum()

    counter = step if torch.is_tensor(step) else torch.tensor(step, device=device, dtype=torch.int64)
    packed_bytes = rows * packed.shape[1]
    byte_block = 32
    byte_programs = triton.cdiv(packed_bytes, byte_block)
    flip_partial = torch.empty(byte_programs, device=device, dtype=torch.int32)
    _coordinate_and_pack[(byte_programs,)](
        gradient, coordinate, packed, moment_q, moment_scale, moment_square_sum,
        flip_partial, counter, update_enabled,
        count, blocks * block_size, packed_bytes, _stream.seed_hash(seed),
        N=rows, K=columns, PB=packed.shape[1],
        BlockSize=block_size, ByteBlock=byte_block,
        CoordinateLR=coordinate_lr, num_warps=4,
    )
    # This kernel just wrote new bits through a pointer, which torch's version
    # counter does not see. Anything expanded from the old bits is now wrong.
    invalidate(packed)
    total = flip_partial.sum()
    if flips is None:
        return int(total.item())
    # A graphed caller reads its own preallocated counter later, without a
    # synchronization here, so the write has to land in that exact tensor.
    flips.copy_(total)
    return flips


__all__ = [
    "apply_bits",
    "coordinate_update",
    "evidence_from_packed",
    "pack_bits",
    "pack_coordinate",
    "pack_affine_rows",
    "packed_embedding",
    "packed_linear_affine",
    "packed_row_inner",
    "packed_transpose",
]
