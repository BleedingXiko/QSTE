"""Pure-torch implementations of every QSTE kernel.

These run anywhere torch runs -- no compiler, no CUDA, no BLAS assumptions --
and they define the numerics the native kernels are tested against. They are
slower, but never wrong, so ``import qste`` succeeds on any machine and a
missing toolchain degrades throughput instead of breaking training.

Every routine here holds bounded scratch. The packed operand is expanded one
tile at a time, exactly as the C++ path does, so choosing the fallback never
reintroduces a full floating-point copy of a binary matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import stream

_ROWS_PER_TILE = 64


def _shifts(device: torch.device) -> Tensor:
    return torch.arange(8, device=device, dtype=torch.uint8)


def unpack_rows(packed: Tensor, columns: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    bits = ((packed.unsqueeze(-1) >> _shifts(packed.device)) & 1).reshape(packed.shape[0], -1)
    return bits[:, :columns].to(dtype).mul_(2).sub_(1)


def pack_bits(mask: Tensor) -> Tensor:
    """One bit per element of a boolean matrix, least significant bit first."""

    if mask.ndim != 2:
        raise ValueError("packing expects a rank-2 tensor")
    rows, columns = mask.shape
    packed = torch.zeros(
        rows, (columns + 7) // 8, device=mask.device, dtype=torch.uint8
    )
    # One lane per bit position keeps the temporary at an eighth of the input
    # instead of materializing an int64 reduction over the whole matrix.
    for shift in range(8):
        lane = mask[:, shift::8]
        if lane.shape[1]:
            packed[:, : lane.shape[1]].bitwise_or_(lane.to(torch.uint8) << shift)
    return packed


def unpack_bits(packed: Tensor, columns: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """The 0/1 companion of :func:`unpack_rows`, which gives +-1."""

    bits = ((packed.unsqueeze(-1) >> _shifts(packed.device)) & 1).reshape(packed.shape[0], -1)
    return bits[:, :columns].to(dtype)


def apply_bits(values: Tensor, packed: Tensor, columns: int) -> Tensor:
    """``values`` where the bit is set, zero where it is not.

    This is the whole backward of a saturating activation: ReLU keeps a
    gradient exactly where its output was positive, and that fact is one bit.
    """

    rows = values.shape[0]
    out = torch.empty_like(values)
    for start in range(0, rows, _ROWS_PER_TILE):
        stop = min(start + _ROWS_PER_TILE, rows)
        bits = unpack_bits(packed[start:stop], columns, dtype=values.dtype)
        out[start:stop] = values[start:stop] * bits
    return out


def _pack_bits(values: Tensor) -> Tensor:
    return pack_bits(values >= 0)


def pack_affine_rows(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """One-bit affine encoding: ``x[n] ~= offset[n] + scale[n] * sign(x[n] - offset[n])``.

    Centering first is what makes this work on one-signed activations. See the
    note in ``cpu.cpp``.
    """

    values = values.float()
    offset = values.mean(dim=1)
    centered = values - offset.unsqueeze(1)
    scale = centered.abs().mean(dim=1)
    return _pack_bits(centered), offset, scale


def pack_coordinate(coordinate: Tensor) -> Tensor:
    return _pack_bits(coordinate)


def packed_linear_affine(
    inputs: Tensor, packed: Tensor, scale: Tensor, bias: Tensor | None, columns: int
) -> Tensor:
    flat = inputs.reshape(-1, columns)
    rows = packed.shape[0]
    out = torch.empty(flat.shape[0], rows, device=inputs.device, dtype=inputs.dtype)
    for start in range(0, rows, _ROWS_PER_TILE):
        stop = min(start + _ROWS_PER_TILE, rows)
        sign = unpack_rows(packed[start:stop], columns, dtype=flat.dtype)
        out[:, start:stop] = flat @ sign.t()
    out.mul_(scale.to(out.dtype).unsqueeze(0))
    if bias is not None:
        out.add_(bias.to(out.dtype).unsqueeze(0))
    return out.view(*inputs.shape[:-1], rows)


def packed_transpose(
    inputs: Tensor, packed: Tensor, columns: int, row_scale: Tensor | None = None
) -> Tensor:
    rows = packed.shape[0]
    flat = inputs.reshape(-1, rows)
    out = torch.zeros(flat.shape[0], columns, device=inputs.device, dtype=inputs.dtype)
    for start in range(0, rows, _ROWS_PER_TILE):
        stop = min(start + _ROWS_PER_TILE, rows)
        sign = unpack_rows(packed[start:stop], columns, dtype=flat.dtype)
        if row_scale is not None:
            sign.mul_(row_scale[start:stop].to(sign.dtype).unsqueeze(1))
        out.addmm_(flat[:, start:stop], sign)
    return out.view(*inputs.shape[:-1], columns)


def evidence_from_packed(
    grad: Tensor, packed: Tensor, columns: int, row_scale: Tensor | None = None
) -> Tensor:
    samples, rows = grad.shape
    evidence = torch.zeros(rows, columns, device=grad.device, dtype=torch.float32)
    for start in range(0, samples, _ROWS_PER_TILE):
        stop = min(start + _ROWS_PER_TILE, samples)
        sign = unpack_rows(packed[start:stop], columns, dtype=torch.float32)
        chunk = grad[start:stop].float()
        if row_scale is not None:
            chunk = chunk * row_scale[start:stop].float().unsqueeze(1)
        evidence.addmm_(chunk.t(), sign)
    return evidence


def packed_row_inner(matrix: Tensor, packed: Tensor, columns: int) -> Tensor:
    rows = matrix.shape[0]
    out = torch.empty(rows, device=matrix.device, dtype=matrix.dtype)
    for start in range(0, rows, _ROWS_PER_TILE):
        stop = min(start + _ROWS_PER_TILE, rows)
        sign = unpack_rows(packed[start:stop], columns, dtype=matrix.dtype)
        out[start:stop] = (matrix[start:stop] * sign).sum(dim=1)
    return out


def packed_embedding(ids: Tensor, packed: Tensor, scale: Tensor, columns: int) -> Tensor:
    flat = ids.reshape(-1)
    rows = unpack_rows(packed[flat], columns, dtype=scale.dtype)
    rows.mul_(scale[flat].unsqueeze(-1))
    return rows.view(*ids.shape, columns)


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
    step: int,
) -> int:
    """Reference coordinate step. Same algorithm as ``cpu.cpp``, same order."""

    rows, columns = gradient.shape
    count = rows * columns
    blocks = (count + block_size - 1) // block_size

    square = gradient.square().add_(1e-12)
    new_row = row_v.float().mul_(beta2).add_(square.mean(1), alpha=1 - beta2)
    new_col = col_v.float().mul_(beta2).add_(square.mean(0), alpha=1 - beta2)
    del square
    row_v.copy_(new_row)
    col_v.copy_(new_col)

    row_factor = new_row.clamp_min(1e-10).mean().sqrt() / new_row.clamp_min(1e-10).sqrt()
    col_factor = new_col.clamp_min(1e-10).rsqrt()
    update = gradient.mul_(row_factor.unsqueeze(1)).mul_(col_factor.unsqueeze(0))
    rms = update.square().mean().sqrt().clamp_min(1e-8)
    update.div_(max(1.0, float(rms / update_clip)))

    padded = torch.zeros(blocks * block_size, device=gradient.device, dtype=torch.float32)
    padded[:count] = update.reshape(-1)
    blocked = padded.view(blocks, block_size)
    previous = moment_q.float().reshape(-1)
    previous_padded = torch.zeros_like(padded)
    previous_padded[:count] = previous
    moment = previous_padded.view(blocks, block_size) * moment_scale.float().unsqueeze(1)
    moment.mul_(beta1).add_(blocked, alpha=1 - beta1)
    moment_rms = moment.square().mean().sqrt().clamp_min(1e-8)

    target = coordinate.float() - (
        moment.reshape(-1)[:count].view_as(coordinate) * (coordinate_lr / moment_rms)
    )
    lower = target.floor()
    # Index-derived hashing, so a step is reproducible from (seed, step) alone
    # and does not depend on how many surfaces ran before this one.
    index = torch.arange(count, device=gradient.device, dtype=torch.int64).view_as(coordinate)
    hash_seed = stream.seed_hash(seed) ^ stream.step_hash(step)
    value = (index ^ hash_seed) & stream.MASK
    value = ((value ^ 61) ^ (value >> 16)) & stream.MASK
    value = (value + (value << 3)) & stream.MASK
    value = (value ^ (value >> 4)) & stream.MASK
    value = (value * stream.MULTIPLIER) & stream.MASK
    value = (value ^ (value >> 15)) & stream.MASK
    # Rounded to fp32 and then scaled by an exact power of two, matching what
    # the other two backends do. Dividing in double would differ in the last
    # bit, and the comparison below is exactly where that bit decides.
    uniform = value.to(torch.float32).mul_(1.0 / 4294967296.0)
    rounded = (lower + (uniform < (target - lower)).to(torch.float32)).clamp_(-127, 127)

    previous_bits = packed.clone()
    coordinate.copy_(rounded.to(torch.int8))
    # The stored FP16 scale is what decodes the moment next step, so quantize
    # against the rounded value, not the FP32 one it came from.
    block_scale = (moment.abs().amax(1).clamp_min(1e-6) / 127.0).to(torch.float16)
    moment_scale.copy_(block_scale.to(moment_scale.dtype))
    decode = block_scale.float().clamp_min(1e-12)
    quantized = (moment / decode.unsqueeze(1)).round().clamp_(-127, 127)
    moment_q.copy_(quantized.reshape(-1)[:count].view_as(moment_q).to(torch.int8))

    packed.copy_(pack_coordinate(coordinate))
    difference = previous_bits.bitwise_xor(packed)
    flips = int(
        ((difference.unsqueeze(-1) >> _shifts(packed.device)) & 1)
        .reshape(rows, -1)[:, :columns]
        .sum()
        .item()
    )
    return flips
