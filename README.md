# QSTE

[![PyPI](https://img.shields.io/pypi/v/qste.svg)](https://pypi.org/project/qste/)
[![Python](https://img.shields.io/pypi/pyversions/qste.svg)](https://pypi.org/project/qste/)
[![CI](https://github.com/BleedingXiko/QSTE/actions/workflows/ci.yml/badge.svg)](https://github.com/BleedingXiko/QSTE/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

True-binary training for PyTorch models. Training precision is inference
precision: there is no float shadow weight and no post-training quantization
step, so the model you deploy is bit-for-bit the model that trained.

Every forward pass, from the first step to deployment, multiplies by a matrix
of `±1` and a learned per-row scale.

> **Status: beta.** The API is stable, the test suite is thorough, and it
> trains. Speed has been measured on CPU and one GPU — see [Where this has
> actually run](#where-this-has-actually-run) for what that covers.

## Adopting it

Two lines and a list of layer names. No base class, no config system, no
training-loop rewrite.

```python
import qste

qste.convert(model, include=["blocks.*.mlp_fc1", "blocks.*.mlp_fc2"])

optimizer   = torch.optim.AdamW(model.parameters())   # yours, unchanged
coordinates = qste.QSTEOptimizer(model)               # owns the binary state

loss.backward()
optimizer.step();       coordinates.step()
optimizer.zero_grad();  coordinates.zero_grad()
```

`convert` mutates the model in place and returns the same object. Layers you
did not name keep training in float. The coordinate optimizer owns the `int8`
coordinates and nothing else — row scales and biases are ordinary parameters
your existing optimizer picks up from `model.parameters()`.

Look before you leap:

```python
print(qste.format_plan(qste.plan(model, include=["blocks.*"])))
```

## What to convert

**This is the decision that determines whether QSTE helps or hurts, and it is
the one part the library cannot make for you.** `convert(model)` with no
`include` converts every `Linear` and `Embedding` it finds, which is the right
default for a quick memory measurement and the wrong default for a model you
intend to train.

A binary matrix has far less capacity than the float matrix it replaced. Where
that is affordable depends on what the matrix does with its output.

### The rule of thumb

| | convert? | why |
|---|---|---|
| MLP / feed-forward matrices | **yes, start here** | most of the parameters, most tolerant of the loss |
| Attention value & output projections | usually fine | their error is averaged before it reaches anything sharp |
| Attention query & key projections | measure first | they feed a softmax, which amplifies perturbation |
| Output head / classifier | usually not | feeds a softmax over the vocabulary |
| Embeddings | usually not | each row sees little gradient; tied heads inherit the head's problem |
| Routers, gates, small projections | no | discrete decisions, and no memory to save anyway |

**In a transformer block, convert the MLP and leave attention alone.** At the
usual ratio of 4 the MLP is `8d²` parameters against attention's `4d²`, so the
MLP alone is **two thirds of the block's weight** — you capture most of the
saving without touching the part where binarization is riskiest.

```python
qste.convert(model, include=["*.mlp.*", "*.feed_forward.*"])
```

### Why attention is different

Attention scores are a softmax over inner products. Softmax is exponentially
sensitive to the *scale* of its logits, so perturbing `Q` and `K` does not
add a little noise to the attention weights — it can move the distribution to
attending somewhere else. `V` and `O` are gentler, because their error is
averaged across the sequence before it reaches anything that sharpens it.

The same argument applies to the output head, which is a softmax over the
vocabulary, and to any router in a mixture-of-experts, where a perturbed score
sends the token to a different expert entirely.

You *can* convert all of it. The library will do it and the model will train.
The point is that "convert everything" trades a lot of quality for the last
third of the memory, and the trade is usually not worth making blind.

### Recurrent models

A recurrent cell written out of `nn.Linear` layers converts like anything else.
Two things to know:

**`torch.nn.GRU` and `torch.nn.LSTM` are not converted.** They hold flat
`weight_ih_l0` / `weight_hh_l0` parameters on a fused module rather than
`Linear` submodules, so `convert()` walks past them. In a model that is only an
LSTM you get an explicit "selected no layers" error; in a mixed model it is
silent, so check `format_plan` rather than assuming.

**Error compounds along the sequence.** Gate matrices run at every timestep, so
their perturbation accumulates instead of being paid once. Check the loss over
a long sequence, not a short one.

### The procedure

Widen the selection rather than narrowing it. Start with the MLP, confirm the
loss curve is where you expect, then add attention projections and confirm
again. `qste.memory_report(model)` tells you what each step bought, so you can
stop when the next step stops being worth it.

## What it costs and what it buys

Per matrix of `R × C` weights:

| | float | QSTE training | QSTE deployed |
|---|---|---|---|
| weights | `4·R·C` | `R·C/8` packed signs | `R·C/8` + `4·R` scales |
| training coordinate | — | `R·C` int8 | dropped |
| optimizer state | `8·R·C` (AdamW) | `~1·R·C` | — |
| **total** | `12·R·C` | `~2.14·R·C` | `~R·C/8` |

At `R = C = 2048`, measured: 50.3 MB float against 9.0 MB training
(**5.6×**), and 0.53 MB deployed (**31.5×**). Optimizer state alone is 4.2 MB
against AdamW's 33.6 MB (**8×**).

Activation memory matters more, because it is the term that scales with batch
and sequence length. Measured at width 1024, bytes retained per sample:

| what is kept | float | QSTE | |
|---|---|---|---|
| linear input | 4096 | 136 (bits + 2 floats) | **30.1×** |
| ReLU output | 4096 | 128 (mask bits) | **32.0×** |
| GELU input | 4096 | 1028 (int8 derivative) | **4.0×** |

There is no setting for this: it is how the library works.

### The activation has to actually die

Those are per-operation numbers and they do not reach the peak on their own.
Torch's `ReLU` saves its full-precision *output* for backward, and that output
is the next layer's input — so the tensor stays resident whatever the linear
does with it, and a packed copy is an *addition*, not a replacement. Measured
end to end, converting the linears alone once used **more** peak memory than
not converting them.

So `qste.nn` replaces the activations too, and `convert` does it by default:

| | torch retains | QSTE retains | |
|---|---|---|---|
| `ReLU`, `ReLU6`, `Hardtanh` | the output, fp32 | 1 bit — was it positive | exact |
| `Dropout` | the mask, 1 byte | 1 bit | exact |
| `GELU`, `SiLU`, `Tanh`, `Sigmoid`, … | the input, fp32 | the local derivative, int8 | forward exact |
| `relu(x).square()` | the output, fp32 | the derivative, 1 unsigned byte | forward exact |

`ReLU` and `Dropout` are bit-identical to torch in both forward and gradient —
the suite asserts `torch.equal`, not `allclose`. The smooth ones have an exact
forward and about 0.2% relative error on the gradient multiplier, which is what
eight bits buys on a bounded smooth function. `activations="exact"` restricts
conversion to the lossless group; `activations=False` leaves them all alone.

Activations called functionally rather than held as modules cannot be found by
walking the module tree, so wrap the forward instead:

```python
with qste.packed_activations():
    loss = model(batch)
loss.backward()
```

This covers **any elementwise activation**, including ones QSTE has never heard
of. For a function it has no closed-form derivative for, it runs the function
once on a detached probe, takes the derivative from autograd, and quantizes
that — one byte per element for anything shaped like `f(x)`:

```python
qste.nn.elementwise(lambda t: t * torch.tanh(F.softplus(t)), x)
block_act = qste.nn.packed(lambda t: F.relu(t) ** 3)
```

The named tables are an optimization over this, not a limit on it.

## Why the activation can be thrown away

The coordinate update needs `gradᵀ @ x` — which input caused which gradient.
`grad` exists only in backward and `x` only in forward, so something has to
survive between them. Reducing either side before they meet destroys the
pairing: sum `grad` over the batch first and you get a rank-one matrix and a
model that sits at chance.

So the pairing is kept exactly and the *operand* is shrunk. Forward retains one
bit per element plus two floats per sample, and backward expands it a tile at a
time inside the same GEMM.

The bit is the sign of each row's deviation from its own mean, not the sign of
the value. Everything out of a ReLU is non-negative, so plain `sign(x)` would
be all ones and the outer product would collapse to rank one — the failure this
design exists to avoid. The offset left behind is recovered exactly in backward
as a rank-one correction.

Nothing else in backward needs `x`: the input gradient is
`(grad * scale) @ sign`, the scale gradient reduces the evidence against the
packed signs, and the bias gradient is a sum.

## Nothing is tuned for one device

There is no compute-capability check, no architecture branch, and no tuned
constant anywhere in the kernels. The numbers that would otherwise be tuned are
derived per device in `kernels/device.py`, from what the hardware reports and
from timing it once:

| | how it is decided |
|---|---|
| GPU scratch ceiling for an expansion | reported cache size and free memory |
| partial accumulators per reduction | reported multiprocessor count |
| dtype of the evidence product | **timed** — fp32 vs fp16 vs bf16 |
| small-batch path | **timed** — fused, fused-with-split, tiled, or expanded |
| CPU expansion budget | **timed** — the tiling that is fastest here |

fp16 exists on a GTX 1080 and is no faster there; it exists on a V100 and is
much faster; bf16 exists on newer parts and on ROCm, where capability tuples
say nothing. A stopwatch answers all of those and a table of device names
answers none. Where nothing wins, fp32 stands.

Losing kernels are kept rather than deleted. A kernel that loses on one card
says nothing about a card with different arithmetic, and the cost of keeping it
is one timing pass on the devices where it loses.

`qste.device_profile()` prints what came out, and the bench leads with it.

## Speed

`python bench/bench.py` prints these for your machine. Everything below is one
run of it, so the numbers agree with each other.

CPU (Apple silicon), width 1024, batch 2048:

| stage | float | QSTE | |
|---|---|---|---|
| forward | 2.442 ms | 2.952 ms | 0.83× |
| grad_input | 2.270 ms | 2.915 ms | 0.78× |
| evidence | 2.737 ms | 2.833 ms | 0.97× |
| relu backward | 0.299 ms | 0.496 ms | 0.60× |
| **full train step** | 35.8 ms | 54.4 ms | **0.66×** |

Inference on the same machine, by batch: 0.20× at batch 1, then 0.65×–0.85×
from batch 8 to 1024.

**On CPU, QSTE trades speed for memory.** The expansion writes the dense weight
out and reads it back — twice the traffic of reading a float weight once. What
that buys is 31.5× less weight memory and 8× less optimizer state, which is the
reason to run a binary model on a CPU at all.

More intrinsics would not change this. A hand-written packed multiply
benchmarked 100× worse than BLAS, and a hand-written packed `sgemv` for the
small-batch case measured 0.12×–0.35×: the host `sgemv` is already vectorized
and near the bandwidth bound, and beating it needs the ISA-specific intrinsics
this project avoids.

On GPU, **no GEMM is written here** — every product goes to the vendor BLAS,
and the kernel in that path expands the packed operand into a bounded scratch
buffer first. An earlier version tiled its own GEMMs in Triton and unpacked
bits inside the K-loop, measuring 2.4×–6× slower than cuBLAS: the dot ran
outside the tensor cores and the packed load re-read the same byte for all
eight of its bits. Packing was never the cost — 0.138 ms against a 7.6 ms GEMM.

The exception is small batch, where there is no arithmetic to amortize and the
whole cost is dragging the weight out of memory. Expanding is worse there,
since it writes the float matrix out and reads it back, while consuming the
bits where they lie moves a thirty-second of the bytes. Which side of the
crossover a shape falls on is settled by timing both, once, per shape.

At inference the packed weight is frozen, so `qste.retain(bytes)` keeps
expanded weights in a bounded cache and a generation loop pays the expansion
once instead of per call. It is off by default, because what it spends is the
memory the library exists to save.

## Where this has actually run

| | status |
|---|---|
| Correctness | 516 tests, nothing mocked. 365 pass on a stock CPU install; 6 more (371 total) need the optional `dabsn` package, and the rest need a GPU. |
| Convergence | verified — models train to convergence, and the bench's own learning check reports `LEARNED` against a stated chance baseline |
| Architectures | transformers, recurrent models and MLP stacks convert and train |
| CPU | measured — the numbers above |
| NVIDIA T4 | measured — full suite green, forward 1.27×, evidence 3.51×, peak memory 2.03× |
| Other GPUs | not yet measured. Speed is chosen at runtime, so an untested card gets a different set of winners, not a different answer. |

Nothing here needs a device to be known in advance. Every kernel is a candidate
that may decline, the expansion route is always available and always correct,
and three numerics implementations (C++, Triton, pure torch) are tested against
each other bit-for-bit. New hardware changes which candidate wins, not whether
the result is right.

`python tools/bundle_cell.py > qste_cell.py` produces one file you can paste
into a hosted notebook. It builds the kernels for that machine, runs the suite
against them, and prints the measurements. It will not print a speed table
without running the tests first.

## Kernels

Three implementations, dispatched by device, numerically identical and tested
against each other bit-for-bit:

- **`kernels/cpu.cpp`** — C++. The only longhand loop is a 256-entry LUT
  expansion, one 32-byte `memcpy` per input byte; every product goes through
  `at::mm`/`at::addmm_`, so it uses whatever BLAS your torch was built with, on
  whatever ISA, with its own threading. Plus a fused coordinate-update kernel.
- **`kernels/cuda.py`** — Triton: expansion, bit packing and masking, the
  embedding gather, the row reduction, the small-batch product, and the fused
  coordinate optimizer. One source compiles for every generation the toolchain
  knows, including ROCm; nothing requires a tensor-core intrinsic, bf16, fp8, or
  `_int_mm`, so the memory win lands on every device.
- **`kernels/device.py`** — the per-device derivation above. The only file that
  asks the hardware anything.
- **`kernels/fallback.py`** — pure torch. Defines the numerics the other two are
  tested against, and means `import qste` never fails.

The C++ builds on first use and caches to `~/.cache/qste`, keyed by source hash.
**Ninja is not required** — if `cpp_extension.load` cannot find it, a plain
setuptools `build_ext` runs instead. `QSTE_KERNELS=torch` skips native entirely;
`QSTE_KERNELS=native` makes a build failure raise instead of degrade.

```python
qste.kernel_status()   # what will run, and why not if it won't
```

## Hosts that read `.weight`

Real model code reaches past the module: fusing three projections into one GEMM,
building a recurrence matrix, tying parameters.

`QSTELinear.weight` is therefore a *differentiable* `sign × scale` matrix routed
through autograd, so `dL/dW` arrives and becomes exactly the evidence the module
path would have produced. A framework that never calls the module still trains.
If `.weight` were a plain detached tensor, such a host would train the scale,
never move a coordinate, and never raise.

The suite converts a real DABSN model — including the `core.W/A/Wg/Ug` matrices
its fused recurrence consumes as raw weights — and asserts every surface moves.

## Distributed

**Read `qste/distributed.py` before running on more than one GPU.** There is one
way to get this wrong and it fails silently.

Coordinate evidence is not an autograd gradient — it never lands in
`parameter.grad`, because the coordinate is `requires_grad=False`. DDP's
gradient all-reduce does not see it, each rank would step its coordinates from
its own local batch, and the ranks would diverge while the loss still went down.

Because that failure is invisible, the safe behaviour is the **default**: when a
process group is initialized, `QSTEOptimizer` all-reduces evidence itself.

```python
coordinates = qste.QSTEOptimizer(model)                           # reduces
coordinates = qste.QSTEOptimizer(model, gradient_reducer=False)   # opt out
```

Stochastic rounding is seeded by `(seed, step, index)` instead of RNG state, so
identical evidence gives byte-identical coordinates on every rank. The suite
spawns two real gloo ranks and checks both directions: bit-identical with a
reducer, demonstrably drifted without one.

For FSDP, keep surfaces out of the flattening and sync them the same way:

```python
model = FullyShardedDataParallel(
    model, ignored_states=qste.distributed.fsdp_ignored_states(model)
)
```

Surfaces are then replicated instead of sharded — one byte per weight per rank,
against the four a float model would have sharded — and everything else shards
normally. `qste.distributed.surfaces_agree(model)` is a cheap debug assert.

## Configuration

`QSTEConfig` holds the training recipe — `coordinate_lr`, `beta1`, `beta2`,
`update_rms_clip`, `momentum_block_size`, `seed`. How the activation is retained
is not a setting.

To find out whether the packed encoding is costing your model anything, measure
it:

```python
with qste.exact_evidence():
    ...   # identical arithmetic and seed, activation kept in fp32
```

A diagnostic, not a mode: it gives up the entire memory saving. The packed
evidence agrees with the exact outer product on ~75% of signs and tracks it to
within noise on every learning test in the suite.

## Gotchas that are handled

- **Multi-rank drift** — the reducer is on by default; see above.
- **Gradient checkpointing** runs forward more times than backward. The
  pending-call count is an optimization only; `step()` flushes unconditionally,
  and checkpointed and non-checkpointed runs produce identical coordinates.
- **Tied weights** share one `Surface`, and their evidence sums across uses as
  autograd requires.
- **Gradient accumulation** — `coordinates.prepare_accumulation()`, asserted to
  match one large batch exactly.
- **AMP** — `coordinates.unscale_(scaler.get_scale())` alongside the parameter
  unscale.
- **Autocast** — forward casts the way a native GEMM would.
- **CUDA graphs** — capture is supported for training. Nothing times, allocates,
  or branches on host state inside a captured region.

## Deployment

```python
packed = qste.export_packed(model)   # drops the int8 coordinate
```

Nothing is re-quantized. The exported module runs the same signs the last
training step wrote.

## Layout

```
src/qste/
  config.py        QSTEConfig
  surface.py       the binary matrix: coordinate, packed signs, row scale
  functional.py    autograd; the retained-activation encodings
  modules.py       QSTELinear, QSTEEmbedding, packed inference variants
  nn.py            activations that retain bits instead of floats
  convert.py       plan / convert / export_packed / memory_report
  optim.py         QSTEOptimizer
  distributed.py   DDP and FSDP
  kernels/         cpu.cpp, cuda.py (Triton), device.py, fallback.py, loader.py
tests/             516 tests (365 host, 6 host+dabsn, 145 GPU)
bench/bench.py     speed and memory on this machine
tools/bundle_cell.py   one paste-ready cell for a hosted notebook
```

## Testing

```bash
pytest
```

Nothing is mocked. The kernel tests compare native against the reference
bit-for-bit, the memory tests hook the autograd tape and count bytes, the
distributed tests spawn real processes, and the learning tests train against a
stated chance baseline.

`tests/test_gpu.py` skips without a GPU and otherwise runs the whole thing on
hardware: every kernel against the reference across awkward shapes, every batch
size from 1 upward, every float dtype, the GPU optimizer against the CPU one bit
for bit, graph capture, checkpointing, and peak memory from the allocator.
