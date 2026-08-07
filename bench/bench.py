"""Measure what QSTE costs and what it buys. No arguments needed.

    python bench/bench.py                   # this device, derived defaults
    python bench/bench.py --width 4096 --batch 8192

Every number is wall clock or the allocator's own high-water mark on real
tensors, and anything worse than the float baseline is printed as worse.

The first thing it prints is the execution profile QSTE *derived* for whatever
device it is running on -- scratch size, reduction width, reduction dtype. None
of those are constants in the source; they come from what the hardware reports
and from timing it once. That line is what makes a result from unfamiliar
hardware interpretable instead of mysterious.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn

import qste
from qste import kernels
from qste import nn as qnn
from qste.config import QSTEConfig
from qste.functional import encode_activation
from qste.kernels import device as device_module
from qste.surface import Surface


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(function, repeats=20, warmup=5, device="cpu"):
    for _ in range(warmup):
        function()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    synchronize(device)
    return (time.perf_counter() - start) / repeats * 1e3


def rule(title):
    print(f"\n{title}\n" + "-" * len(title))


def verdict(ratio):
    return "faster" if ratio > 1.02 else ("parity" if ratio > 0.95 else "slower")


def peak_bytes(build, device, samples, width, depth, steps=2, autocast=False):
    """The allocator's high-water mark for a *steady-state* training step.

    The first step is run before the counter is reset, and that matters. A
    first step through a QSTE model pays one-time costs that are not part of
    steady-state memory at all: the device profile times a matmul, Triton
    compiles, and the small-batch chooser runs every candidate path to see
    which is fastest. Counting those makes whichever configuration is measured
    first look worse, which is an artefact of the harness rather than a
    property of the thing being measured.
    """

    model, optimizers = build(width, depth, device)
    inputs = torch.randn(samples, width, device=device)

    def step():
        if autocast:
            with torch.autocast(device, dtype=torch.float16):
                output = model(inputs)
            output.float().square().mean().backward()
        else:
            model(inputs).square().mean().backward()
        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

    step()
    synchronize(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        step()
    synchronize(device)
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    del model, optimizers, inputs
    if device == "cuda":
        torch.cuda.empty_cache()
    return peak


def stack(width, depth, device, converted, activations=True):
    torch.manual_seed(0)
    model = nn.Sequential(
        *[
            layer
            for _ in range(depth)
            for layer in (nn.Linear(width, width, bias=False), nn.ReLU())
        ]
    ).to(device)
    if converted:
        qste.convert(model, activations=activations)
    return model


def float_build(width, depth, device):
    model = stack(width, depth, device, converted=False)
    return model, [torch.optim.AdamW(model.parameters(), lr=1e-3)]


def qste_build(activations):
    def build(width, depth, device):
        model = stack(width, depth, device, converted=True, activations=activations)
        return model, [
            torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=1e-3),
            qste.QSTEOptimizer(model),
        ]

    return build


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def correctness_table(device, width, batch):
    """Every packed kernel against the dense reference it replaces."""

    rule("correctness: packed kernels vs dense references")
    from qste.kernels import fallback

    generator = torch.Generator(device="cpu").manual_seed(0)
    small = min(width, 512)
    samples = min(batch, 512)
    weight = torch.randn(small, small, generator=generator).to(device)
    inputs = torch.randn(samples, small, generator=generator).to(device)
    grad = torch.randn(samples, small, generator=generator).to(device)
    surface = Surface(weight, QSTEConfig()).to(device)
    packed = surface.packed_sign.data
    scale = surface.scale.detach()
    packed_x, _, _ = kernels.pack_affine_rows(inputs)
    mask = inputs > 0
    bits = kernels.pack_bits(mask)
    # row_inner reduces a matrix against the *weight* bits, so it is shaped
    # like the weight, not like a batch.
    weight_shaped = torch.randn(small, small, generator=generator).to(device)

    checks = [
        ("packed_linear_affine",
         lambda: kernels.packed_linear_affine(inputs, packed, scale, None, small),
         lambda: fallback.packed_linear_affine(inputs, packed, scale, None, small)),
        ("packed_transpose",
         lambda: kernels.packed_transpose(grad, packed, small, scale),
         lambda: fallback.packed_transpose(grad, packed, small, scale)),
        ("packed_row_inner",
         lambda: kernels.packed_row_inner(weight_shaped, packed, small),
         lambda: fallback.packed_row_inner(weight_shaped, packed, small)),
        ("evidence_from_packed",
         lambda: kernels.evidence_from_packed(grad, packed_x, small),
         lambda: fallback.evidence_from_packed(grad, packed_x, small)),
        ("pack_bits / apply_bits",
         lambda: kernels.apply_bits(grad, bits, small),
         lambda: grad * mask),
    ]
    worst = 0.0
    for name, candidate, reference in checks:
        got, want = candidate().float(), reference().float()
        error = float((got - want).abs().max() / want.abs().max().clamp_min(1e-12))
        worst = max(worst, error)
        state = "OK" if error < 2e-2 else "MISMATCH"
        print(f"  {name:<24} max relative error {error:.2e}   {state}")
    return worst


def kernel_table(device, width, batch):
    rule(f"speed: width={width} batch={batch} on {device}")
    generator = torch.Generator(device="cpu").manual_seed(0)
    weight = torch.randn(width, width, generator=generator).to(device)
    inputs = torch.randn(batch, width, generator=generator).to(device)
    grad = torch.randn(batch, width, generator=generator).to(device)
    surface = Surface(weight, QSTEConfig()).to(device)
    packed = surface.packed_sign.data
    scale = surface.scale.detach()
    packed_x, _, _ = kernels.pack_affine_rows(inputs)
    mask = inputs > 0
    bits = kernels.pack_bits(mask)

    rows = [
        ("forward",
         lambda: torch.nn.functional.linear(inputs, weight),
         lambda: kernels.packed_linear_affine(inputs, packed, scale, None, width)),
        ("grad_input",
         lambda: grad @ weight,
         lambda: kernels.packed_transpose(grad, packed, width)),
        ("evidence",
         lambda: grad.t() @ inputs,
         lambda: kernels.evidence_from_packed(grad, packed_x, width)),
        ("relu backward",
         lambda: grad * mask,
         lambda: kernels.apply_bits(grad, bits, width)),
    ]
    print(f"  {'stage':<16}{'float':>10}{'qste':>10}{'ratio':>9}")
    for stage, base, candidate in rows:
        base_ms = timed(base, device=device)
        qste_ms = timed(candidate, device=device)
        ratio = base_ms / qste_ms
        print(f"  {stage:<16}{base_ms:10.3f}{qste_ms:10.3f}{ratio:8.2f}x  {verdict(ratio)}")

    overhead = [
        ("pack_affine_rows", lambda: kernels.pack_affine_rows(inputs)),
        ("pack_bits", lambda: kernels.pack_bits(mask)),
    ]
    for name, function in overhead:
        print(f"  {name:<16}{'':>10}{timed(function, device=device):10.3f}   (overhead)")


def retention_table(width, batch):
    rule(f"retained per sample   width={width}")
    print(f"{'what is kept':<34}{'bytes/sample':>14}{'vs float':>10}")
    inputs = torch.randn(batch, width)

    def measure(payload, aux=None):
        total = payload.numel() * payload.element_size()
        if aux is not None:
            total += aux.numel() * aux.element_size()
        return total / batch

    reference = measure(inputs)
    rows = [("linear input, float (torch)", reference)]
    for storage, label in (("int8", "linear input, int8"), ("bit", "linear input, bits (qste)")):
        payload, aux = encode_activation(inputs, storage)
        rows.append((label, measure(payload, aux)))
    rows.append(("relu output, float (torch)", reference))
    rows.append(("relu mask, bits (qste)", width / 8))
    rows.append(("gelu input, float (torch)", reference))
    rows.append(("gelu derivative, int8 (qste)", width + 4))
    for label, value in rows:
        print(f"{label:<34}{value:>14.1f}{reference / value:>9.1f}x")


def memory_table(device, width, batch, depth):
    rule(f"peak memory   {depth} x [{width}, {width}]   batch={batch}")
    if device != "cuda":
        print("  (allocator high-water mark is only reported on CUDA)")
        return
    configurations = [
        ("float + AdamW", float_build, False),
        ("qste, torch activations", qste_build(False), False),
        ("qste, packed activations", qste_build(True), False),
        ("float + AdamW, autocast", float_build, True),
        ("qste, autocast", qste_build(True), True),
    ]
    baseline = {}
    print(f"  {'configuration':<28}{'peak MB':>10}{'vs float':>10}")
    for name, build, amp in configurations:
        peak = peak_bytes(build, device, batch, width, depth, autocast=amp)
        reference = baseline.setdefault(amp, peak)
        print(f"  {name:<28}{peak / 1e6:>10.1f}{reference / peak:>9.2f}x")


def scaling_table(device, width, depth):
    """Peak memory per sample, from the slope of two real training steps.

    Doubling the batch until the allocator refuses answers the wrong question.
    At an extreme batch the peak is dominated by the *transient* layer output,
    which QSTE does not shrink and never claimed to -- both configurations die
    at the same place and the comparison says nothing. What decides whether a
    real batch fits is the term that scales with it, and that is the slope.

    This is the least flattering table here and it should stay that way. The
    absolute peak improves by about half because QSTE takes out the fixed
    costs -- the float weights, the two AdamW moments, and the retained
    activations, none of which grow with the batch. The slope is a different
    quantity: with almost nothing retained, what is left at the margin is the
    working set of whichever layer is executing, and packed weights do not make
    a layer's own input and output smaller. So the honest reading is that QSTE
    buys most of a device's memory back at any realistic batch and roughly
    nothing at the asymptote, and both of those are worth knowing before
    someone plans a run around the first number.
    """

    rule(f"peak memory per sample   {depth} x [{width}, {width}]")
    if device != "cuda":
        print("  (allocator high-water mark is only reported on CUDA)")
        return
    print(f"  {'configuration':<26}{'bytes/sample':>13}{'fixed MB':>11}"
          f"{'vs float':>10}{'batch in 8 GiB':>16}")
    baseline = None
    for name, build in (
        ("float + AdamW", float_build),
        ("qste, torch activations", qste_build(False)),
        ("qste, packed activations", qste_build(True)),
    ):
        small = peak_bytes(build, device, 1024, width, depth)
        large = peak_bytes(build, device, 4096, width, depth)
        slope = max(1.0, (large - small) / 3072)
        fixed = small - slope * 1024
        baseline = baseline or slope
        room = int(max(0, ((8 << 30) - fixed) / slope))
        print(f"  {name:<26}{slope:>13.0f}{fixed / (1 << 20):>11.1f}"
              f"{baseline / slope:>9.1f}x{room:>16,}")
    print("  (the fixed column is where the saving is; the slope is the working")
    print("   set of the executing layer, which a packed weight does not shrink)")


def step_table(device, width, batch, depth):
    rule(f"train step   {depth} x [{width}, {width}]   batch={batch}")
    torch.manual_seed(0)
    inputs = torch.randn(batch, width, device=device)

    def make(build):
        model, optimizers = build(width, depth, device)

        def one_step():
            model(inputs).square().mean().backward()
            for optimizer in optimizers:
                optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

        return one_step

    def amp(build):
        model, optimizers = build(width, depth, device)

        def one_step():
            with torch.autocast(device, dtype=torch.float16):
                output = model(inputs)
            output.float().square().mean().backward()
            for optimizer in optimizers:
                optimizer.step()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

        return one_step

    rows = [
        ("float + AdamW", make(float_build), None),
        ("qste, torch activations", make(qste_build(False)), None),
        ("qste, packed activations", make(qste_build(True)), None),
    ]
    if device == "cuda":
        rows.append(("float + AdamW, autocast", amp(float_build), None))
        rows.append(("qste, autocast", amp(qste_build(True)), None))

    print(f"  {'configuration':<28}{'ms/step':>10}{'vs float':>10}")
    baseline = None
    for name, function, _ in rows:
        ms = timed(function, repeats=8, warmup=3, device=device)
        if name == "float + AdamW":
            baseline = ms
        marker = ""
        if name.endswith("autocast") and "float" in name:
            marker = "  (autocast baseline)"
        print(f"  {name:<28}{ms:>10.2f}{baseline / ms:>9.2f}x{marker}")


def optimizer_table(device, width, depth):
    """The coordinate step on its own, against the AdamW it replaces."""

    rule(f"optimizer step alone   {depth} x [{width}, {width}]")
    model = stack(width, depth, device, converted=True)
    coordinates = qste.QSTEOptimizer(model)
    surfaces = qste.surfaces(model)
    evidence = [torch.randn(s.rows, s.columns, device=device) for s in surfaces]

    def coordinate_step():
        for surface, value in zip(surfaces, evidence):
            coordinates._apply(surface, value.clone())
        coordinates.step_number += 1

    reference = stack(width, depth, device, converted=False)
    adamw = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    for parameter in reference.parameters():
        parameter.grad = torch.randn_like(parameter)

    adam_ms = timed(adamw.step, repeats=20, warmup=5, device=device)
    qste_ms = timed(coordinate_step, repeats=20, warmup=5, device=device)
    print(f"  {'AdamW':<26}{adam_ms:>10.3f} ms")
    print(f"  {'QSTE coordinate step':<26}{qste_ms:>10.3f} ms{adam_ms / qste_ms:>9.2f}x")
    print(f"  {'optimizer state':<26}{coordinates.state_bytes() / 1e6:>10.1f} MB"
          f"   vs {sum(p.numel() * 8 for p in reference.parameters()) / 1e6:.1f} MB for AdamW")


def _chosen_path(device, width, batch, dtype, retained=None):
    """Which implementation the stopwatch picked, so the ratio can be read.

    ``retained`` selects which regime's decision to report, since the two are
    measured and remembered separately -- holding the expansion changes what
    the paths cost relative to each other, and often changes the winner.
    """

    cuda = kernels.cuda_backend() if device == "cuda" else None
    if cuda is None:
        return "expand + BLAS"
    for key, choice in cuda._FUSED_CHOICE.items():
        if key[1] != width or key[2] != width or key[3] != batch or key[5] != dtype:
            continue
        if retained is not None and key[6] is not retained:
            continue
        if choice == cuda.EXPANDED and cuda._FUSED_ERROR is not None:
            # A fused path that never compiled and a fused path that lost a
            # race read identically in a ratio column. Two benchmark runs
            # were spent reading one as the other.
            return f"expanded ({type(cuda._FUSED_ERROR).__name__})"
        return choice
    return "expanded (over the ceiling)"


def _both_paths(device, width, batch, dtype):
    """What each regime chose. They frequently differ, and that is the point."""

    streamed = _chosen_path(device, width, batch, dtype, retained=False)
    held = _chosen_path(device, width, batch, dtype, retained=True)
    return streamed if streamed == held else f"{streamed} -> {held}"


def decode_table(device, width):
    """Inference, swept across batch and across precision.

    Both axes matter and the second one was missing, which made the first one
    unreadable.

    A packed weight buys memory traffic: a thirty-second of the bytes cross the
    bus. That is decisive when the product is bandwidth bound -- one row, a few
    rows -- and worth nothing when it is compute bound, because the arithmetic
    is the same either way and somebody else's GEMM is tuned for it. So the
    interesting question is where the crossover falls, and the answer depends
    entirely on which arithmetic the device has.

    In fp32 on a part with no fp32 matrix instruction, the packed kernels have
    to do the multiply with plain fused multiply-adds while the vendor GEMM
    does the same work in a routine tuned for years. Saving bandwidth cannot
    pay for that in the middle of the range, and the stopwatch correctly says
    so and expands instead. In fp16 the matrix instruction exists, and both the
    packed kernel and the expansion get much cheaper -- so both sides are
    measured, rather than quoting a packed kernel against an fp32 baseline it
    was never going to beat.

    On a CPU the packed paths are switched off entirely. The host sgemv is
    vectorized and near the bandwidth bound, a portable scalar replacement
    measured at 0.12x to 0.35x of it, and beating it needs the ISA-specific
    intrinsics this project exists to avoid. There the format buys memory, not
    speed, and that is worth measuring rather than hiding.
    """

    generator = torch.Generator(device="cpu").manual_seed(0)
    reference = torch.randn(width, width, generator=generator)
    surface = Surface(reference.clone(), QSTEConfig()).to(device)
    packed = surface.packed_sign.data
    scale = surface.scale.detach()

    dtypes = [torch.float32]
    if device == "cuda":
        dtypes.append(torch.float16)

    for dtype in dtypes:
        name = str(dtype).replace("torch.", "")
        rule(f"inference   [{width}, {width}]   {name}   on {device}")
        weight = reference.to(device).to(dtype)
        # The expansion is quoted on its own line because it is a *floor*, not
        # an overhead: it does not shrink with the batch, so wherever the
        # expanded path wins, the qste column can never go below it. Reading the
        # ratio without it invites tuning the product when the cost is the
        # materialization sitting in front of it.
        backend = kernels.cuda_backend() if device == "cuda" else None
        expand = timed(
            lambda: backend.expand_dense(packed, width, scale=scale, dtype=dtype),
            repeats=50, device=device,
        ) if backend is not None else float("nan")
        print(f"  {'batch':<8}{'float ms':>11}{'qste ms':>11}{'ratio':>9}"
              f"{'retained':>11}{'ratio':>9}   path taken")
        # Swept across the whole range rather than only the small end, because
        # the interesting failure was never at batch one -- it was in the
        # middle, where the product is too big for the lane kernel and too
        # small to bury the cost of expanding a weight.
        #
        # The retained column is the same call with the expansion held across
        # calls instead of redone. That is the trade this library normally
        # refuses -- it spends memory -- so it is quoted beside the streaming
        # number rather than instead of it, and the reader picks.
        batches = (1, 8, 32, 64, 128, 256, 512, 1024)
        samples = {
            batch: torch.randn(batch, width, generator=generator).to(device).to(dtype)
            for batch in batches
        }

        def sweep():
            """One full pass over the batch axis, so the two configurations are
            not interleaved. Turning retention on and off between batches
            churns the allocator by a weight's worth each time, and that lands
            in the timing of whichever configuration was measured second."""

            return {
                batch: timed(
                    lambda batch=batch: kernels.packed_linear_affine(
                        samples[batch], packed, scale, None, width),
                    repeats=50, device=device,
                )
                for batch in batches
            }

        base = {
            batch: timed(
                lambda batch=batch: torch.nn.functional.linear(samples[batch], weight),
                repeats=50, device=device,
            )
            for batch in batches
        }
        with torch.no_grad():
            qste.retain(0)
            streamed = sweep()
            retained = {}
            if backend is not None:
                qste.retain(256 << 20)
                retained = sweep()
                qste.retain(0)

        for batch in batches:
            held = retained.get(batch)
            kept = (f"{held:>11.4f}{base[batch] / held:>8.2f}x"
                    if held is not None else " " * 20)
            print(f"  {batch:<8}{base[batch]:>11.4f}{streamed[batch]:>11.4f}"
                  f"{base[batch] / streamed[batch]:>8.2f}x{kept}"
                  f"   {_both_paths(device, width, batch, dtype)}")
        if backend is not None:
            print(f"  {'expand':<8}{'':>11}{expand:>11.4f}"
                  f"{'':>9}   (the floor under every streamed row)")
            # What the call costs that is not the kernel. The candidate timings
            # launch a kernel directly; the shipped number goes through the
            # launcher. The difference is host work, and at batch one it has
            # been comparable to the kernel itself -- which is invisible unless
            # it is subtracted and printed.
            best = _best_candidate(device, width, 1, dtype, retained=True)
            if best is not None and retained.get(1) is not None:
                print(f"  {'host':<8}{'':>11}{retained[1] - best:>11.4f}"
                      f"{'':>9}   (per call, at batch 1: shipped minus best kernel)")
            _candidate_table(device, width, dtype)


def _best_candidate(device, width, batch, dtype, retained):
    cuda = kernels.cuda_backend()
    if cuda is None:
        return None
    for key, measured in cuda.timings(device).items():
        if (key[1] == width and key[2] == width and key[3] == batch
                and key[5] == dtype and key[6] is retained):
            live = [v for v in measured.values() if v != float("inf")]
            return min(live) if live else None
    return None


def _candidate_table(device, width, dtype):
    """What every candidate measured, not only the one that won.

    A path that never appears in the column above is either slower than the
    alternative or was never able to run at all, and those call for opposite
    responses -- write a different kernel, or find out why this build refuses to
    compile the one that exists. Two runs were spent reading one as the other,
    so the losing times are printed rather than thrown away.
    """

    cuda = kernels.cuda_backend()
    if cuda is None:
        return
    rows = [
        (key[3], key[6], measured)
        for key, measured in cuda.timings(device).items()
        if key[1] == width and key[2] == width and key[5] == dtype
    ]
    if not rows:
        return
    # Whatever names actually got timed, rather than a fixed list: the fused
    # path offers several blockings and each is timed under its own name, so a
    # hard-coded header would hide the one that won.
    names = sorted({name for _, _, row in rows for name in row})
    print()
    print(f"  {'batch':<8}{'held':<6}" + "".join(f"{name:>13}" for name in names))
    for batch, held, measured in sorted(rows):
        cells = ""
        for name in names:
            value = measured.get(name, float("inf"))
            cells += f"{'declined' if value == float('inf') else f'{value:.4f}':>13}"
        print(f"  {batch:<8}{'yes' if held else 'no':<6}{cells}")
    if cuda._FUSED_ERROR is not None:
        print(f"  a declining path last said: {type(cuda._FUSED_ERROR).__name__}: "
              f"{cuda._FUSED_ERROR}")

    # And the blockings underneath, which is where the expanded column comes
    # from. Two hand-derived formulas for this were wrong, the second by enough
    # to cost a run, so the choice is timed and the timings are shown.
    blockings = [
        measured for key, measured in cuda.expand_timings(device).items()
        if key[1] == width and key[2] == width and key[3] == dtype
    ]
    if blockings:
        every = sorted({name for row in blockings for name in row})
        print()
        print(f"  {'expand':<12}" + "".join(f"{name:>16}" for name in every))
        for row in blockings:
            cells = "".join(
                f"{'declined' if row.get(n, float('inf')) == float('inf') else f'{row[n]:.4f}':>16}"
                for n in every
            )
            print(f"  {'':<12}{cells}")


def reduction_ab(device, width, batch):
    """Does the dtype the device picked for the evidence change the answer?"""

    rule("A/B: evidence in the dtype the device chose, vs fp32")
    if device != "cuda":
        print("  (reduced-precision reduction is only selected on GPU)")
        return
    profile = kernels.profile(device)
    if profile.reduction_dtype == torch.float32:
        print("  device chose fp32; nothing to compare")
        return

    import os

    generator = torch.Generator(device="cpu").manual_seed(0)
    inputs = torch.randn(batch, width, generator=generator).to(device)
    grad = torch.randn(batch, width, generator=generator).to(device)
    packed_x, _, _ = kernels.pack_affine_rows(inputs)

    fast = kernels.evidence_from_packed(grad, packed_x, width)
    fast_ms = timed(lambda: kernels.evidence_from_packed(grad, packed_x, width), device=device)

    os.environ["QSTE_REDUCTION_DTYPE"] = "fp32"
    device_module.reset()
    exact = kernels.evidence_from_packed(grad, packed_x, width)
    exact_ms = timed(lambda: kernels.evidence_from_packed(grad, packed_x, width), device=device)
    del os.environ["QSTE_REDUCTION_DTYPE"]
    device_module.reset()

    error = float((fast - exact).abs().max() / exact.abs().max())
    cosine = float(
        torch.nn.functional.cosine_similarity(fast.flatten(), exact.flatten(), dim=0)
    )
    label = str(profile.reduction_dtype).replace("torch.", "")
    print(f"  fp32          {exact_ms:8.3f} ms")
    print(f"  {label:<13} {fast_ms:8.3f} ms   {exact_ms / fast_ms:.2f}x")
    print(f"  max relative difference {error:.2e}   cosine similarity {cosine:.8f}")
    print("  (this product is stochastically rounded into an INT8 coordinate"
          " immediately after)")


def learning_ab(device):
    """Two runs, same seed, one forced to fp32 evidence. Do they learn the same?"""

    rule("A/B: does the reduction dtype change what the model learns")
    import os

    def run():
        torch.manual_seed(0)
        classes, width, samples = 8, 256, 1024
        inputs = torch.randn(samples, width, device=device)
        targets = torch.randint(0, classes, (samples,), device=device)
        model = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, classes)
        ).to(device)
        qste.convert(model)
        continuous = torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=3e-3)
        coordinates = qste.QSTEOptimizer(model)
        losses, flips = [], 0
        for _ in range(150):
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            continuous.step()
            flips += coordinates.step()
            continuous.zero_grad(set_to_none=True)
            losses.append(loss.detach().item())
        return losses, flips

    default_losses, default_flips = run()
    os.environ["QSTE_REDUCTION_DTYPE"] = "fp32"
    device_module.reset()
    exact_losses, exact_flips = run()
    del os.environ["QSTE_REDUCTION_DTYPE"]
    device_module.reset()

    chance = math.log(8)
    print(f"  {'run':<16}{'first 10':>10}{'last 10':>10}{'flips':>12}")
    for label, losses, flips in (
        ("device default", default_losses, default_flips),
        ("forced fp32", exact_losses, exact_flips),
    ):
        print(f"  {label:<16}{sum(losses[:10]) / 10:>10.4f}"
              f"{sum(losses[-10:]) / 10:>10.4f}{flips:>12,}")
    print(f"  chance is {chance:.4f}")
    gap = abs(sum(default_losses[-10:]) - sum(exact_losses[-10:])) / 10
    print(f"  final loss difference {gap:.5f}")


def learning_check(device):
    rule("sanity: does it actually learn")
    torch.manual_seed(0)
    classes, width, samples = 8, 256, 1024
    inputs = torch.randn(samples, width, device=device)
    targets = torch.randint(0, classes, (samples,), device=device)
    model = nn.Sequential(
        nn.Linear(width, width), qnn.ReLU(), nn.Linear(width, width), qnn.GELU(),
        nn.Linear(width, classes),
    ).to(device)
    qste.convert(model)
    continuous = torch.optim.AdamW(list(qste.continuous_parameters(model)), lr=3e-3)
    coordinates = qste.QSTEOptimizer(model)
    losses, flips = [], 0
    for _ in range(200):
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        continuous.step()
        flips += coordinates.step()
        continuous.zero_grad(set_to_none=True)
        losses.append(loss.detach().item())
    chance = math.log(classes)
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    print(f"  chance      {chance:.4f}")
    print(f"  first 10    {first:.4f}")
    print(f"  last 10     {last:.4f}")
    print(f"  sign flips  {flips:,}")
    print(f"  verdict     {'LEARNED' if last < first * 0.7 else 'DID NOT LEARN'}")
    return last < first * 0.7


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--skip-batch-search", action="store_true")
    arguments = parser.parse_args()
    device = arguments.device

    status = kernels.status()
    print(f"torch {torch.__version__}   device {device}")
    print(f"kernels  native_cpu={status['native']}  gpu={status['cuda']}")
    if not status["native"]:
        print(f"  (native CPU build unavailable: {status['failure']})")
    if device == "cuda" and not status["cuda"]:
        print(f"  (GPU kernels unavailable: {status['cuda_failure']})")
    print(f"derived  {qste.device_profile(device)}")

    # Each table is isolated. One that raises used to take the whole benchmark
    # with it, so a single broken kernel cost every number after the point it
    # was first touched -- including, on the run that motivated this, the one
    # table the run existed to read. A failure is now reported where it happens
    # and the rest still prints.
    tables = [
        ("correctness", lambda: correctness_table(device, arguments.width, arguments.batch)),
        ("kernel speed", lambda: kernel_table(device, arguments.width, arguments.batch)),
        ("retention", lambda: retention_table(arguments.width, arguments.batch)),
        ("peak memory", lambda: memory_table(
            device, arguments.width, arguments.batch, arguments.depth)),
        ("memory per sample", lambda: None if arguments.skip_batch_search
            else scaling_table(device, arguments.width, arguments.depth)),
        ("train step", lambda: step_table(
            device, arguments.width, arguments.batch, arguments.depth)),
        ("optimizer", lambda: optimizer_table(device, arguments.width, arguments.depth)),
        ("small-batch inference", lambda: decode_table(device, arguments.width)),
        ("reduction A/B", lambda: reduction_ab(device, arguments.width, arguments.batch)),
        ("learning A/B", lambda: learning_ab(device)),
        ("learning check", lambda: learning_check(device)),
    ]
    failures = []
    for name, table in tables:
        try:
            table()
        except Exception as error:  # noqa: BLE001 - a benchmark reports, it does not raise
            failures.append(name)
            last = str(error).strip().splitlines()
            print(f"\n  [{name} could not run: {type(error).__name__}: "
                  f"{(last[-1] if last else '')[:200]}]")
    if failures:
        print(f"\n{len(failures)} of {len(tables)} tables failed: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
