// QSTE CPU kernels.
//
// Design note, because it is the difference between this file and 2000 lines
// of intrinsics: a packed binary matrix is not faster than BLAS when you
// multiply it by hand. It is faster when you *expand* it -- one memcpy per
// byte out of a 256-entry LUT -- and hand the expanded tile to the same GEMM
// the float path would have used. Measured at 1.02x fp32 sgemm at n=2048 and
// 0.97x at n=128, bit exact, with the operand held at one bit per element.
//
// So the only hot loop written out longhand here is the LUT expansion. Every
// product goes through at::mm / at::addmm_, which means whatever BLAS the host
// torch was built against, on whatever ISA, with its own threading.
//
// Scratch is bounded per call by kScratchBytes, so expanding a tile never
// costs the memory that keeping a float matrix would have.

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// How much of a weight may be expanded at once. This is the single number that
// decides whether the CPU path is faster or slower than the float one it
// replaces, and it is not a constant: it is set from Python, by measurement.
//
// The reason it matters so much is that the expansion is only free if nobody
// pays to move it. Expand a tile larger than the last-level cache and the
// sequence becomes write-to-memory then read-back-from-memory, which is twice
// the traffic the float path needed to read its weight once -- so a format
// that stores a thirty-second of the bytes ends up slower than one that stores
// all of them. Expand a tile that fits in cache and the packed bits are the
// only thing that reaches memory at all.
//
// Too small is a different loss: BLAS stops seeing a real GEMM and the call
// overhead dominates. The size that balances those is a property of the
// machine, so qste.kernels.device finds it by timing this function and passes
// it in. Zero means "no measurement available", and falls back to a tile that
// is small enough to be safe on any cache rather than large enough to be fast
// on a particular one.
constexpr int64_t kDefaultScratchBytes = 1 << 20;
constexpr int64_t kMinTileRows = 8;
int64_t g_scratch_bytes = kDefaultScratchBytes;

// There is deliberately no small-batch path here. Consuming packed bits in
// place instead of expanding them wins when the product is bandwidth bound on
// the weight, and the bits are a thirty-second of it -- which is true on a GPU
// and was measured false on a CPU. A scalar sign-extend-and-accumulate loop
// runs at 0.12x to 0.35x of the host BLAS sgemv it would replace, because that
// sgemv is already near the bandwidth bound *and* vectorized, and beating it
// needs the thousands of lines of ISA-specific intrinsics this file exists to
// avoid. Expansion plus BLAS is the right answer here at every batch size.

// byte -> eight +-1 floats, so expansion is a 32-byte memcpy per input byte.
const float* sign_lut() {
  static const std::vector<float> table = [] {
    std::vector<float> values(256 * 8);
    for (int byte = 0; byte < 256; ++byte) {
      for (int bit = 0; bit < 8; ++bit) {
        values[byte * 8 + bit] = ((byte >> bit) & 1) ? 1.0f : -1.0f;
      }
    }
    return values;
  }();
  return table.data();
}

// Expand `rows` packed rows into `out`, which must hold rows * stride * 8
// floats. Trailing pad columns decode to -1 and are dropped by the caller's
// narrow, never by a branch in the inner loop.
void expand(const uint8_t* packed, int64_t rows, int64_t stride, float* out) {
  const float* lut = sign_lut();
  at::parallel_for(0, rows, 16, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const uint8_t* source = packed + row * stride;
      float* destination = out + row * stride * 8;
      for (int64_t byte = 0; byte < stride; ++byte) {
        std::memcpy(destination + byte * 8, lut + source[byte] * 8,
                    8 * sizeof(float));
      }
    }
  });
}

// byte -> eight 0/1 floats, the masking companion of sign_lut().
const float* mask_lut() {
  static const std::vector<float> table = [] {
    std::vector<float> values(256 * 8);
    for (int byte = 0; byte < 256; ++byte) {
      for (int bit = 0; bit < 8; ++bit) {
        values[byte * 8 + bit] = ((byte >> bit) & 1) ? 1.0f : 0.0f;
      }
    }
    return values;
  }();
  return table.data();
}

int64_t tile_rows(int64_t stride, int64_t limit) {
  const int64_t bytes_per_row = stride * 8 * static_cast<int64_t>(sizeof(float));
  const int64_t budget = g_scratch_bytes / std::max<int64_t>(1, bytes_per_row);
  return std::max<int64_t>(1, std::min(limit, std::max(budget, kMinTileRows)));
}

void check_packed(const torch::Tensor& packed, int64_t columns) {
  TORCH_CHECK(packed.device().is_cpu(), "qste CPU kernels require CPU tensors");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8 && packed.dim() == 2,
              "packed signs must be a rank-2 uint8 tensor");
  TORCH_CHECK(packed.size(1) == (columns + 7) / 8,
              "packed sign width does not match the column count");
}

// Expanded view of packed[row0 : row0 + rows] as a [rows, columns] float
// matrix living in `scratch`. The view aliases scratch; it is consumed before
// the next expansion overwrites it.
torch::Tensor expanded_tile(const uint8_t* packed, int64_t row0, int64_t rows,
                            int64_t stride, int64_t columns,
                            torch::Tensor& scratch) {
  expand(packed + row0 * stride, rows, stride, scratch.data_ptr<float>());
  return scratch.narrow(0, 0, rows).narrow(1, 0, columns);
}

}  // namespace

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

// One-bit affine encoding of each row:  x[n, :] ~= offset[n] + scale[n] * s
// where s is the sign of the row's deviation from its own mean.
//
// The offset is not optional. A post-activation tensor is frequently
// one-signed -- everything out of a ReLU is -- and sign(x) of a non-negative
// row is all ones, which turns the evidence outer product into a rank-one
// matrix and destroys exactly the per-example pairing the whole design exists
// to preserve. Signing the deviation instead keeps the encoding informative
// for any activation, and the offset it leaves behind is recovered exactly in
// backward as a rank-one correction, so nothing is approximated away.
std::vector<torch::Tensor> pack_affine_rows(torch::Tensor values) {
  TORCH_CHECK(values.device().is_cpu() && values.dim() == 2,
              "pack_affine_rows expects a rank-2 CPU tensor");
  values = values.to(torch::kFloat32).contiguous();
  const int64_t rows = values.size(0), columns = values.size(1);
  const int64_t stride = (columns + 7) / 8;
  const float inverse = 1.0f / static_cast<float>(std::max<int64_t>(1, columns));
  auto packed = torch::zeros({rows, stride}, values.options().dtype(torch::kUInt8));
  auto offset = torch::empty({rows}, values.options());
  auto scale = torch::empty({rows}, values.options());
  const float* source = values.data_ptr<float>();
  uint8_t* bits = packed.data_ptr<uint8_t>();
  float* offsets = offset.data_ptr<float>();
  float* scales = scale.data_ptr<float>();

  const int64_t full = columns / 8;
  at::parallel_for(0, rows, 8, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const float* input = source + row * columns;
      uint8_t* output = bits + row * stride;
      // Four independent accumulators so the sum vectorizes instead of
      // serializing on one dependency chain.
      float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
      int64_t column = 0;
      for (; column + 4 <= columns; column += 4) {
        s0 += input[column]; s1 += input[column + 1];
        s2 += input[column + 2]; s3 += input[column + 3];
      }
      for (; column < columns; ++column) s0 += input[column];
      const float mean = (s0 + s1 + s2 + s3) * inverse;

      // Build each output byte in a register. The obvious version -- setting
      // one bit at a time with |= into memory -- is a read-modify-write per
      // element and was the single most expensive thing in the forward pass.
      float d0 = 0.0f, d1 = 0.0f;
      for (int64_t byte = 0; byte < full; ++byte) {
        const float* block = input + byte * 8;
        const float c0 = block[0] - mean, c1 = block[1] - mean;
        const float c2 = block[2] - mean, c3 = block[3] - mean;
        const float c4 = block[4] - mean, c5 = block[5] - mean;
        const float c6 = block[6] - mean, c7 = block[7] - mean;
        d0 += std::fabs(c0) + std::fabs(c1) + std::fabs(c2) + std::fabs(c3);
        d1 += std::fabs(c4) + std::fabs(c5) + std::fabs(c6) + std::fabs(c7);
        output[byte] = static_cast<uint8_t>(
            (c0 >= 0.0f) | ((c1 >= 0.0f) << 1) | ((c2 >= 0.0f) << 2) |
            ((c3 >= 0.0f) << 3) | ((c4 >= 0.0f) << 4) | ((c5 >= 0.0f) << 5) |
            ((c6 >= 0.0f) << 6) | ((c7 >= 0.0f) << 7));
      }
      uint8_t tail = 0;
      for (int64_t index = full * 8; index < columns; ++index) {
        const float centered = input[index] - mean;
        d0 += std::fabs(centered);
        if (centered >= 0.0f) tail |= static_cast<uint8_t>(1u << (index - full * 8));
      }
      if (full < stride) output[full] = tail;
      offsets[row] = mean;
      scales[row] = (d0 + d1) * inverse;
    }
  });
  return {packed, offset, scale};
}

// Repack the sign of an INT8 coordinate matrix. Used at surface construction
// and by any caller that mutates coordinates outside the fused optimizer.
torch::Tensor pack_coordinate(torch::Tensor coordinate) {
  TORCH_CHECK(coordinate.device().is_cpu() && coordinate.dim() == 2 &&
                  coordinate.scalar_type() == torch::kChar,
              "pack_coordinate expects a rank-2 CPU int8 tensor");
  coordinate = coordinate.contiguous();
  const int64_t rows = coordinate.size(0), columns = coordinate.size(1);
  const int64_t stride = (columns + 7) / 8;
  auto packed = torch::zeros({rows, stride}, coordinate.options().dtype(torch::kUInt8));
  const int8_t* source = coordinate.data_ptr<int8_t>();
  uint8_t* bits = packed.data_ptr<uint8_t>();
  const int64_t full = columns / 8;
  at::parallel_for(0, rows, 8, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int8_t* input = source + row * columns;
      uint8_t* output = bits + row * stride;
      for (int64_t byte = 0; byte < full; ++byte) {
        const int8_t* block = input + byte * 8;
        output[byte] = static_cast<uint8_t>(
            (block[0] >= 0) | ((block[1] >= 0) << 1) | ((block[2] >= 0) << 2) |
            ((block[3] >= 0) << 3) | ((block[4] >= 0) << 4) |
            ((block[5] >= 0) << 5) | ((block[6] >= 0) << 6) |
            ((block[7] >= 0) << 7));
      }
      uint8_t tail = 0;
      for (int64_t index = full * 8; index < columns; ++index) {
        if (input[index] >= 0) tail |= static_cast<uint8_t>(1u << (index - full * 8));
      }
      if (full < stride) output[full] = tail;
    }
  });
  return packed;
}

// One bit per element of a boolean mask. This is what a saturating activation
// has to remember: ReLU's backward is "keep the gradient exactly where the
// output was positive", which is one bit, not one float. Torch saves the whole
// output tensor instead, and in a Linear->ReLU stack that tensor is also the
// next layer's input -- so without this the packed activation QSTE keeps is
// pure addition and the memory saving never reaches the peak.
torch::Tensor pack_bits(torch::Tensor mask) {
  TORCH_CHECK(mask.device().is_cpu() && mask.dim() == 2 &&
                  mask.scalar_type() == torch::kBool,
              "pack_bits expects a rank-2 CPU bool tensor");
  mask = mask.contiguous();
  const int64_t rows = mask.size(0), columns = mask.size(1);
  const int64_t stride = (columns + 7) / 8;
  auto packed = torch::zeros({rows, stride}, mask.options().dtype(torch::kUInt8));
  const uint8_t* source = reinterpret_cast<const uint8_t*>(mask.data_ptr<bool>());
  uint8_t* bits = packed.data_ptr<uint8_t>();
  const int64_t full = columns / 8;
  at::parallel_for(0, rows, 8, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const uint8_t* input = source + row * columns;
      uint8_t* output = bits + row * stride;
      for (int64_t byte = 0; byte < full; ++byte) {
        const uint8_t* block = input + byte * 8;
        output[byte] = static_cast<uint8_t>(
            (block[0] != 0) | ((block[1] != 0) << 1) | ((block[2] != 0) << 2) |
            ((block[3] != 0) << 3) | ((block[4] != 0) << 4) |
            ((block[5] != 0) << 5) | ((block[6] != 0) << 6) |
            ((block[7] != 0) << 7));
      }
      uint8_t tail = 0;
      for (int64_t index = full * 8; index < columns; ++index) {
        if (input[index] != 0) tail |= static_cast<uint8_t>(1u << (index - full * 8));
      }
      if (full < stride) output[full] = tail;
    }
  });
  return packed;
}

// values, masked by the packed bits. A 256-entry LUT of eight 0/1 floats turns
// the mask expansion into the same 32-byte memcpy the sign path uses.
torch::Tensor apply_bits(torch::Tensor values, torch::Tensor packed,
                         int64_t columns) {
  check_packed(packed, columns);
  TORCH_CHECK(values.scalar_type() == torch::kFloat32 && values.dim() == 2,
              "apply_bits requires a rank-2 float32 matrix");
  TORCH_CHECK(values.size(0) == packed.size(0) && values.size(1) == columns,
              "apply_bits shape mismatch");
  values = values.contiguous();
  packed = packed.contiguous();
  const int64_t rows = values.size(0), stride = packed.size(1);
  const int64_t full = columns / 8;
  auto out = torch::empty_like(values);
  const float* source = values.data_ptr<float>();
  const uint8_t* bits = packed.data_ptr<uint8_t>();
  float* result = out.data_ptr<float>();
  const float* lut = mask_lut();

  at::parallel_for(0, rows, 8, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const float* input = source + row * columns;
      const uint8_t* row_bits = bits + row * stride;
      float* destination = result + row * columns;
      for (int64_t byte = 0; byte < full; ++byte) {
        const float* keep = lut + row_bits[byte] * 8;
        const float* block = input + byte * 8;
        float* target = destination + byte * 8;
        for (int bit = 0; bit < 8; ++bit) target[bit] = block[bit] * keep[bit];
      }
      for (int64_t column = full * 8; column < columns; ++column) {
        destination[column] =
            ((row_bits[full] >> (column - full * 8)) & 1) ? input[column] : 0.0f;
      }
    }
  });
  return out;
}

torch::Tensor unpack_rows(torch::Tensor packed, int64_t columns) {
  check_packed(packed, columns);
  packed = packed.contiguous();
  const int64_t rows = packed.size(0), stride = packed.size(1);
  auto out = torch::empty({rows, stride * 8}, packed.options().dtype(torch::kFloat32));
  expand(packed.data_ptr<uint8_t>(), rows, stride, out.data_ptr<float>());
  return out.narrow(1, 0, columns).contiguous();
}

// ---------------------------------------------------------------------------
// Forward and backward projections
// ---------------------------------------------------------------------------

// The expansion budget, set once from Python after it has been timed on this
// host. Clamped rather than trusted: a caller that asks for a gigabyte of
// scratch has misunderstood what this buffer is for.
void set_scratch_bytes(int64_t bytes) {
  g_scratch_bytes = std::min<int64_t>(std::max<int64_t>(bytes, 4 << 10), 64 << 20);
}

int64_t scratch_bytes() { return g_scratch_bytes; }

// out = (inputs @ sign^T) * scale + bias, with sign expanded one tile at a
// time. Fused: the scale and bias never touch a separate full-size temporary.
torch::Tensor packed_linear_affine(torch::Tensor inputs, torch::Tensor packed,
                                   torch::Tensor scale,
                                   c10::optional<torch::Tensor> bias,
                                   int64_t columns) {
  check_packed(packed, columns);
  TORCH_CHECK(inputs.scalar_type() == torch::kFloat32,
              "qste CPU projection requires float32 activations");
  TORCH_CHECK(inputs.size(-1) == columns, "activation width mismatch");
  packed = packed.contiguous();
  auto flat = inputs.reshape({-1, columns}).contiguous();
  const int64_t rows = packed.size(0), stride = packed.size(1);
  TORCH_CHECK(scale.numel() == rows, "scale must have one entry per output row");

  auto out = torch::empty({flat.size(0), rows}, flat.options());
  const uint8_t* bits = packed.data_ptr<uint8_t>();

  const int64_t tile = tile_rows(stride, rows);
  auto scratch = torch::empty({tile, stride * 8}, flat.options());
  for (int64_t row0 = 0; row0 < rows; row0 += tile) {
    const int64_t count = std::min(tile, rows - row0);
    auto weight = expanded_tile(bits, row0, count, stride, columns, scratch);
    out.narrow(1, row0, count).copy_(at::mm(flat, weight.t()));
  }
  out.mul_(scale.to(torch::kFloat32).view({1, rows}));
  if (bias.has_value()) out.add_(bias.value().to(torch::kFloat32).view({1, rows}));

  std::vector<int64_t> shape(inputs.sizes().begin(), inputs.sizes().end());
  shape.back() = rows;
  return out.view(shape);
}

// out = inputs @ sign, the transpose projection backward needs for grad_input.
// Tiles the contraction dimension and accumulates in place.
torch::Tensor packed_transpose(torch::Tensor inputs, torch::Tensor packed,
                               int64_t columns,
                               c10::optional<torch::Tensor> row_scale) {
  check_packed(packed, columns);
  TORCH_CHECK(inputs.scalar_type() == torch::kFloat32,
              "qste CPU transpose requires float32 inputs");
  packed = packed.contiguous();
  const int64_t rows = packed.size(0), stride = packed.size(1);
  TORCH_CHECK(inputs.size(-1) == rows, "transpose width mismatch");
  auto flat = inputs.reshape({-1, rows}).contiguous();
  const bool scaled = row_scale.has_value();
  if (scaled) {
    TORCH_CHECK(row_scale.value().numel() == rows,
                "transpose row scale must have one entry per packed row");
    row_scale = row_scale.value().to(torch::kFloat32).contiguous();
  }

  auto out = torch::zeros({flat.size(0), columns}, flat.options());
  const int64_t tile = tile_rows(stride, rows);
  auto scratch = torch::empty({tile, stride * 8}, flat.options());
  const uint8_t* bits = packed.data_ptr<uint8_t>();
  for (int64_t row0 = 0; row0 < rows; row0 += tile) {
    const int64_t count = std::min(tile, rows - row0);
    auto weight = expanded_tile(bits, row0, count, stride, columns, scratch);
    // Scaling the expanded weight rather than the incoming gradient keeps the
    // work at rows x columns instead of batch x rows, and needs no temporary:
    // the scratch is already being written.
    if (scaled) {
      weight = weight.mul_(row_scale.value().narrow(0, row0, count).unsqueeze(1));
    }
    out.addmm_(flat.narrow(1, row0, count), weight);
  }
  std::vector<int64_t> shape(inputs.sizes().begin(), inputs.sizes().end());
  shape.back() = columns;
  return out.view(shape);
}

// evidence = grad^T @ sign(x), where sign(x) arrives packed one bit per
// element. This is the call that lets forward drop the activation: the outer
// product is unchanged and exact against the signs, and the operand that had
// to survive from forward to backward is 32x smaller.
torch::Tensor evidence_from_packed(torch::Tensor grad, torch::Tensor packed,
                                   int64_t columns,
                                   c10::optional<torch::Tensor> row_scale) {
  check_packed(packed, columns);
  TORCH_CHECK(grad.scalar_type() == torch::kFloat32 && grad.dim() == 2,
              "evidence grad must be a rank-2 float32 tensor");
  TORCH_CHECK(grad.size(0) == packed.size(0),
              "evidence grad and packed activations disagree on sample count");
  grad = grad.contiguous();
  packed = packed.contiguous();
  const int64_t samples = grad.size(0), rows = grad.size(1);
  const int64_t stride = packed.size(1);
  const bool scaled = row_scale.has_value();
  if (scaled) {
    TORCH_CHECK(row_scale.value().numel() == samples,
                "evidence row scale must have one entry per sample");
    row_scale = row_scale.value().to(torch::kFloat32).contiguous();
  }

  auto evidence = torch::zeros({rows, columns}, grad.options());
  const int64_t tile = tile_rows(stride, samples);
  auto scratch = torch::empty({tile, stride * 8}, grad.options());
  // Per-sample scaling folds into the small operand, one tile at a time, so
  // it never allocates a second copy of the full gradient.
  auto grad_tile = scaled ? torch::empty({tile, rows}, grad.options()) : torch::Tensor();
  const uint8_t* bits = packed.data_ptr<uint8_t>();
  for (int64_t sample0 = 0; sample0 < samples; sample0 += tile) {
    const int64_t count = std::min(tile, samples - sample0);
    auto signs = expanded_tile(bits, sample0, count, stride, columns, scratch);
    auto slice = grad.narrow(0, sample0, count);
    if (scaled) {
      auto scaled_slice = grad_tile.narrow(0, 0, count);
      at::mul_out(scaled_slice, slice,
                  row_scale.value().narrow(0, sample0, count).unsqueeze(1));
      evidence.addmm_(scaled_slice.t(), signs);
    } else {
      evidence.addmm_(slice.t(), signs);
    }
  }
  return evidence;
}

// out[i] = sum_j matrix[i, j] * sign[i, j]. Row-local and memory bound, so it
// stays a scalar loop rather than paying for an expansion.
torch::Tensor packed_row_inner(torch::Tensor matrix, torch::Tensor packed,
                               int64_t columns) {
  check_packed(packed, columns);
  TORCH_CHECK(matrix.scalar_type() == torch::kFloat32 && matrix.dim() == 2,
              "row inner requires a rank-2 float32 matrix");
  TORCH_CHECK(matrix.size(0) == packed.size(0) && matrix.size(1) == columns,
              "row inner shape mismatch");
  matrix = matrix.contiguous();
  packed = packed.contiguous();
  const int64_t rows = matrix.size(0), stride = packed.size(1);
  const int64_t full = columns / 8;
  auto out = torch::empty({rows}, matrix.options());
  const float* values = matrix.data_ptr<float>();
  const uint8_t* bits = packed.data_ptr<uint8_t>();
  float* result = out.data_ptr<float>();
  const float* lut = sign_lut();

  at::parallel_for(0, rows, 4, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const float* value = values + row * columns;
      const uint8_t* row_bits = bits + row * stride;
      float total = 0.0f;
      for (int64_t byte = 0; byte < full; ++byte) {
        const float* sign = lut + row_bits[byte] * 8;
        const float* block = value + byte * 8;
        total += block[0] * sign[0] + block[1] * sign[1] + block[2] * sign[2] +
                 block[3] * sign[3] + block[4] * sign[4] + block[5] * sign[5] +
                 block[6] * sign[6] + block[7] * sign[7];
      }
      for (int64_t column = full * 8; column < columns; ++column) {
        total += value[column] * (((row_bits[full] >> (column - full * 8)) & 1) ? 1.0f : -1.0f);
      }
      result[row] = total;
    }
  });
  return out;
}

// Gather rows of the binary matrix for an embedding lookup.
torch::Tensor packed_embedding(torch::Tensor ids, torch::Tensor packed,
                               torch::Tensor scale, int64_t columns) {
  check_packed(packed, columns);
  TORCH_CHECK(ids.scalar_type() == torch::kLong, "embedding ids must be int64");
  ids = ids.contiguous();
  packed = packed.contiguous();
  scale = scale.to(torch::kFloat32).contiguous();
  const int64_t rows = packed.size(0), stride = packed.size(1);
  TORCH_CHECK(scale.numel() == rows, "embedding scale shape mismatch");
  const int64_t count = ids.numel();
  auto out = torch::empty({count, columns}, scale.options());
  const int64_t* token = ids.data_ptr<int64_t>();
  const uint8_t* bits = packed.data_ptr<uint8_t>();
  const float* scales = scale.data_ptr<float>();
  float* result = out.data_ptr<float>();
  const float* lut = sign_lut();
  const int64_t full = columns / 8;

  at::parallel_for(0, count, 8, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t row = token[index];
      TORCH_CHECK(row >= 0 && row < rows, "embedding id outside the vocabulary");
      const uint8_t* row_bits = bits + row * stride;
      float* destination = result + index * columns;
      const float row_scale = scales[row];
      for (int64_t byte = 0; byte < full; ++byte) {
        const float* sign = lut + row_bits[byte] * 8;
        float* block = destination + byte * 8;
        for (int bit = 0; bit < 8; ++bit) block[bit] = sign[bit] * row_scale;
      }
      for (int64_t column = full * 8; column < columns; ++column) {
        destination[column] =
            (((row_bits[full] >> (column - full * 8)) & 1) ? row_scale : -row_scale);
      }
    }
  });
  std::vector<int64_t> shape(ids.sizes().begin(), ids.sizes().end());
  shape.push_back(columns);
  return out.view(shape);
}

// ---------------------------------------------------------------------------
// Coordinate optimizer
// ---------------------------------------------------------------------------

namespace {

// The stochastic-rounding stream. See qste/kernels/stream.py -- that module is
// the definition and this has to match it bit for bit, because a coordinate
// matrix trained on a GPU has to continue on a CPU and land on the same
// integers. Unsigned throughout, so every shift is logical and every overflow
// wraps; the Triton implementation gets the same arithmetic by evaluating in a
// wider register and masking back to 32 bits.
inline uint32_t scramble(uint32_t value) {
  value = (value ^ 61u) ^ (value >> 16);
  value += value << 3;
  value ^= value >> 4;
  value *= 0x27d4eb2du;
  value ^= value >> 15;
  return value;
}

// The salt is below 2^31 because the Triton implementation cannot express a
// larger literal -- see stream.py. Nothing about the mixing depends on which
// odd constant it is, only on all three backends using the same one.
inline uint32_t stream_seed(int64_t seed, int64_t step) {
  return scramble(static_cast<uint32_t>(seed)) ^
         scramble(static_cast<uint32_t>(step) ^ 0x2545f491u);
}

}  // namespace

// One fused optimizer step for a single surface.
//
// Factored second moment (one row vector, one column vector -- never a full
// matrix), block-quantized INT8 first moment, RMS-clipped update, stochastic
// rounding into the INT8 coordinate, and the packed sign refresh, all in four
// passes over the evidence. `gradient` is scratch: it is consumed in place so
// the caller can release it immediately and no second float matrix exists.
int64_t coordinate_update(torch::Tensor gradient, torch::Tensor coordinate,
                          torch::Tensor packed, torch::Tensor moment_q,
                          torch::Tensor moment_scale, torch::Tensor row_v,
                          torch::Tensor col_v, double beta1_in, double beta2_in,
                          double update_clip_in, double coordinate_lr_in,
                          int64_t block_size, int64_t seed, int64_t step) {
  TORCH_CHECK(gradient.device().is_cpu(), "coordinate_update requires CPU tensors");
  TORCH_CHECK(gradient.scalar_type() == torch::kFloat32 && gradient.dim() == 2,
              "evidence must be a rank-2 float32 tensor");
  TORCH_CHECK(coordinate.scalar_type() == torch::kChar &&
                  moment_q.scalar_type() == torch::kChar &&
                  packed.scalar_type() == torch::kByte,
              "coordinate/momentum/packed dtypes are wrong");
  TORCH_CHECK(moment_scale.scalar_type() == torch::kHalf &&
                  row_v.scalar_type() == torch::kHalf &&
                  col_v.scalar_type() == torch::kHalf,
              "compact optimizer statistics must be float16");
  gradient = gradient.contiguous();
  const int64_t rows = gradient.size(0), columns = gradient.size(1);
  TORCH_CHECK(coordinate.sizes() == gradient.sizes() &&
                  moment_q.sizes() == gradient.sizes(),
              "coordinate state shape mismatch");
  TORCH_CHECK(packed.size(0) == rows && packed.size(1) == (columns + 7) / 8,
              "packed sign shape mismatch");
  const int64_t count = rows * columns;
  const int64_t blocks = (count + block_size - 1) / block_size;
  TORCH_CHECK(moment_scale.numel() == blocks, "momentum scale shape mismatch");
  TORCH_CHECK(row_v.numel() == rows && col_v.numel() == columns,
              "factored statistics shape mismatch");

  float* grad = gradient.data_ptr<float>();
  int8_t* coord = coordinate.data_ptr<int8_t>();
  uint8_t* bits = packed.data_ptr<uint8_t>();
  int8_t* moment_int = moment_q.data_ptr<int8_t>();
  c10::Half* moment_block = moment_scale.data_ptr<c10::Half>();
  c10::Half* row_stat = row_v.data_ptr<c10::Half>();
  c10::Half* col_stat = col_v.data_ptr<c10::Half>();
  const float beta1 = static_cast<float>(beta1_in);
  const float beta2 = static_cast<float>(beta2_in);
  const float update_clip = static_cast<float>(update_clip_in);
  const float coordinate_lr = static_cast<float>(coordinate_lr_in);

  // Pass 1: row and column second moments from one row-major read. Column
  // sums go into per-thread strips so there is no atomic and no column-major
  // traversal of the evidence.
  const int threads = std::max(1, at::get_num_threads());
  std::vector<float> column_partial(static_cast<size_t>(threads) * columns, 0.0f);
  std::vector<float> row_stats(rows), column_stats(columns);
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    float* local = column_partial.data() +
                   static_cast<size_t>(at::get_thread_num()) * columns;
    for (int64_t row = begin; row < end; ++row) {
      const float* line = grad + row * columns;
      float total = 0.0f;
      for (int64_t column = 0; column < columns; ++column) {
        const float square = line[column] * line[column];
        total += square;
        local[column] += square;
      }
      const float updated = static_cast<float>(row_stat[row]) * beta2 +
                            (total / static_cast<float>(columns) + 1e-12f) * (1.0f - beta2);
      row_stats[row] = updated;
      row_stat[row] = updated;
    }
  });
  at::parallel_for(0, columns, 64, [&](int64_t begin, int64_t end) {
    for (int64_t column = begin; column < end; ++column) {
      float total = 0.0f;
      for (int thread = 0; thread < threads; ++thread) {
        total += column_partial[static_cast<size_t>(thread) * columns + column];
      }
      const float updated = static_cast<float>(col_stat[column]) * beta2 +
                            (total / static_cast<float>(rows) + 1e-12f) * (1.0f - beta2);
      column_stats[column] = updated;
      col_stat[column] = updated;
    }
  });

  double row_mean_accumulator = 0.0;
  for (int64_t row = 0; row < rows; ++row) row_mean_accumulator += row_stats[row];
  const float row_mean =
      std::max(static_cast<float>(row_mean_accumulator / static_cast<double>(rows)), 1e-10f);
  const float sqrt_row_mean = std::sqrt(row_mean);
  std::vector<float> row_factor(rows), column_factor(columns);
  for (int64_t row = 0; row < rows; ++row) {
    row_factor[row] = sqrt_row_mean / std::sqrt(std::max(row_stats[row], 1e-10f));
  }
  for (int64_t column = 0; column < columns; ++column) {
    column_factor[column] = 1.0f / std::sqrt(std::max(column_stats[column], 1e-10f));
  }

  // Pass 2: normalize in place and measure the global update RMS.
  double square_sum = 0.0;
  {
    std::vector<double> partial(threads, 0.0);
    at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
      double local = 0.0;
      for (int64_t row = begin; row < end; ++row) {
        float* line = grad + row * columns;
        const float multiplier = row_factor[row];
        // Four float accumulators rather than one double. Accumulating into a
        // double inside the loop makes every iteration depend on the last and
        // stops the whole thing vectorizing; the row's own sum is far too
        // small to need the extra exponent.
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int64_t column = 0;
        for (; column + 4 <= columns; column += 4) {
          const float u0 = line[column] * multiplier * column_factor[column];
          const float u1 = line[column + 1] * multiplier * column_factor[column + 1];
          const float u2 = line[column + 2] * multiplier * column_factor[column + 2];
          const float u3 = line[column + 3] * multiplier * column_factor[column + 3];
          line[column] = u0; line[column + 1] = u1;
          line[column + 2] = u2; line[column + 3] = u3;
          a0 += u0 * u0; a1 += u1 * u1; a2 += u2 * u2; a3 += u3 * u3;
        }
        for (; column < columns; ++column) {
          const float update = line[column] * multiplier * column_factor[column];
          line[column] = update;
          a0 += update * update;
        }
        local += static_cast<double>(a0 + a1) + static_cast<double>(a2 + a3);
      }
      partial[at::get_thread_num()] += local;
    });
    for (double value : partial) square_sum += value;
  }
  const float update_rms = std::sqrt(
      std::max(static_cast<float>(square_sum / static_cast<double>(count)), 1e-16f));
  const float update_divisor = std::max(1.0f, update_rms / update_clip);
  const float inverse_divisor = 1.0f / update_divisor;

  // Pass 3: first moment per block, decoded and re-encoded through its own
  // FP16 scale. The moment is left in `grad` for the final pass.
  double moment_square_sum = 0.0;
  {
    std::vector<double> partial(threads, 0.0);
    at::parallel_for(0, blocks, 1, [&](int64_t begin, int64_t end) {
      double local = 0.0;
      for (int64_t block = begin; block < end; ++block) {
        const int64_t start = block * block_size;
        const int64_t stop = std::min(start + block_size, count);
        const float old_scale = static_cast<float>(moment_block[block]);
        const float decay = old_scale * beta1;
        const float gain = (1.0f - beta1) * inverse_divisor;
        float maximum = 1e-6f;
        float block_square = 0.0f;
        for (int64_t index = start; index < stop; ++index) {
          const float moment =
              static_cast<float>(moment_int[index]) * decay + grad[index] * gain;
          grad[index] = moment;
          block_square += moment * moment;
          maximum = std::max(maximum, std::fabs(moment));
        }
        local += static_cast<double>(block_square);
        moment_block[block] = maximum / 127.0f;
      }
      partial[at::get_thread_num()] += local;
    });
    for (double value : partial) moment_square_sum += value;
  }
  const float moment_rms = std::sqrt(std::max(
      static_cast<float>(moment_square_sum / static_cast<double>(blocks * block_size)),
      1e-16f));
  const float coordinate_step = coordinate_lr / moment_rms;

  // Pass 4: stochastically round the coordinate target, re-quantize the
  // moment, and rewrite the packed sign byte -- one traversal, byte aligned so
  // each output byte is owned by exactly one worker.
  const int64_t stride = packed.size(1);
  const uint32_t hash_seed = stream_seed(seed, step);
  std::vector<int64_t> flip_partial(threads, 0);
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    int64_t local_flips = 0;
    for (int64_t row = begin; row < end; ++row) {
      const int64_t row_base = row * columns;
      // (a) Stochastic rounding and momentum requantization, walked in
      // segments that share one momentum block so the scale reciprocal is
      // loop invariant and the body is straight-line arithmetic.
      int64_t column = 0;
      while (column < columns) {
        const int64_t index = row_base + column;
        const int64_t block = index / block_size;
        const int64_t span = (block + 1) * block_size - index;
        const int64_t segment = std::min(columns, column + span);
        const float inverse_scale =
            1.0f / std::max(static_cast<float>(moment_block[block]), 1e-12f);
        for (int64_t c = column; c < segment; ++c) {
          const int64_t i = row_base + c;
          const float moment = grad[i];
          const float target =
              static_cast<float>(coord[i]) - moment * coordinate_step;
          const float lower = std::floor(target);
          const float uniform =
              static_cast<float>(scramble(static_cast<uint32_t>(i) ^ hash_seed)) *
              (1.0f / 4294967296.0f);
          const float rounded = lower + (uniform < (target - lower) ? 1.0f : 0.0f);
          coord[i] = static_cast<int8_t>(
              std::min(127.0f, std::max(-127.0f, rounded)));
          // rint, not nearbyint: both round half to even under the default
          // mode, and nearbyint has to inspect the rounding mode every call.
          moment_int[i] = static_cast<int8_t>(std::min(
              127.0f, std::max(-127.0f, std::rint(moment * inverse_scale))));
        }
        column = segment;
      }

      // (b) Repack the row's sign bits and count what changed. Byte aligned,
      // so every output byte is owned by exactly one worker.
      const int8_t* line = coord + row_base;
      uint8_t* slot = bits + row * stride;
      for (int64_t byte = 0; byte < stride; ++byte) {
        const int64_t column0 = byte * 8;
        const int64_t limit = std::min<int64_t>(8, columns - column0);
        uint8_t next = 0;
        for (int64_t bit = 0; bit < limit; ++bit) {
          if (line[column0 + bit] >= 0) next |= static_cast<uint8_t>(1u << bit);
        }
        const uint8_t mask =
            static_cast<uint8_t>(limit >= 8 ? 0xFF : ((1u << limit) - 1u));
        const uint8_t previous = slot[byte];
        local_flips +=
            __builtin_popcount(static_cast<unsigned>((previous ^ next) & mask));
        slot[byte] = static_cast<uint8_t>((previous & static_cast<uint8_t>(~mask)) | next);
      }
    }
    flip_partial[at::get_thread_num()] += local_flips;
  });
  int64_t flips = 0;
  for (int64_t value : flip_partial) flips += value;
  return flips;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_affine_rows", &pack_affine_rows, "sign bits of the centered row, plus offset and scale");
  m.def("pack_coordinate", &pack_coordinate, "sign bits of an int8 coordinate matrix");
  m.def("unpack_rows", &unpack_rows, "expand packed signs to +-1 floats");
  m.def("pack_bits", &pack_bits, "one bit per element of a boolean mask");
  m.def("apply_bits", &apply_bits, "values where the bit is set, zero elsewhere");
  m.def("set_scratch_bytes", &set_scratch_bytes,
        "how much of a weight may be expanded at once; measured, not guessed");
  m.def("scratch_bytes", &scratch_bytes, "the expansion budget in force");
  m.def("packed_linear_affine", &packed_linear_affine, "(x @ sign^T) * scale + bias");
  m.def("packed_transpose", &packed_transpose, "x @ sign",
        py::arg("inputs"), py::arg("packed"), py::arg("columns"),
        py::arg("row_scale") = c10::optional<torch::Tensor>());
  m.def("evidence_from_packed", &evidence_from_packed, "grad^T @ sign(x)",
        py::arg("grad"), py::arg("packed"), py::arg("columns"),
        py::arg("row_scale") = c10::optional<torch::Tensor>());
  m.def("packed_row_inner", &packed_row_inner, "row-wise dot against packed signs");
  m.def("packed_embedding", &packed_embedding, "gather scaled binary rows");
  m.def("coordinate_update", &coordinate_update, "fused QSTE coordinate step");
}
