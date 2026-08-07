"""What actually survives from forward to backward.

The claim QSTE is being built for is a memory claim, so it gets measured
rather than argued. These tests hook the autograd tape and count unique bytes
by storage pointer -- the same thing that decides whether a batch fits.

The number to watch is the *retained activation*: what the QSTE layer forces to
stay alive because backward will need it. Weights and optimizer state are
counted separately, because they do not scale with batch and sequence length
and the activation does.
"""

import pytest
import torch
import torch.nn as nn

import qste
from qste.functional import exact_evidence, qste_linear
from qste.surface import Surface


class Tape:
    """Unique bytes held by the autograd graph, keyed by storage."""

    def __enter__(self):
        self.seen: dict[int, int] = {}

        def pack(tensor):
            self.seen.setdefault(
                tensor.untyped_storage().data_ptr(),
                tensor.untyped_storage().nbytes(),
            )
            return tensor

        self._hook = torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t)
        self._hook.__enter__()
        return self

    def __exit__(self, *args):
        self._hook.__exit__(*args)

    @property
    def total(self) -> int:
        return sum(self.seen.values())


def _tape_bytes(storage, samples, columns=512, rows=512):
    torch.manual_seed(0)
    surface = Surface(torch.randn(rows, columns) / columns**0.5)
    surface._immediate = lambda *_: None
    inputs = torch.randn(samples, columns)
    with exact_evidence(storage), Tape() as tape:
        output = qste_linear(inputs, surface)
    total = tape.total
    output.sum().backward()
    return total


def _per_sample(storage, columns=512, rows=512):
    """Retained bytes per sample, from the slope.

    Differencing two batch sizes removes every fixed cost -- packed weights,
    the row scale, the autograd nodes behind it -- and leaves exactly the term
    that decides how large a batch fits.
    """

    small = _tape_bytes(storage, 128, columns, rows)
    large = _tape_bytes(storage, 384, columns, rows)
    return (large - small) / 256


def test_bit_storage_retains_far_less_per_sample():
    ratio = _per_sample("fp32") / _per_sample("bit")
    # One bit per element is 32x on its own; the per-sample offset and scale
    # cost a little back, and proportionally less the wider the layer.
    assert ratio > 28.0, f"expected ~28x at width 512, measured {ratio:.1f}x"


def test_the_ratio_approaches_thirty_two_as_width_grows():
    narrow = _per_sample("fp32", columns=256, rows=256) / _per_sample(
        "bit", columns=256, rows=256
    )
    wide = _per_sample("fp32", columns=4096, rows=256) / _per_sample(
        "bit", columns=4096, rows=256
    )
    assert wide > narrow
    assert 31.0 < wide <= 32.0, f"measured {wide:.2f}x at width 4096"


def test_int8_storage_retains_about_a_quarter_of_fp32():
    ratio = _per_sample("fp32") / _per_sample("int8")
    assert 3.4 < ratio < 4.1, f"expected ~4x, measured {ratio:.2f}x"


def test_retained_bytes_match_the_arithmetic():
    columns = 512
    assert _per_sample("fp32", columns) == pytest.approx(columns * 4)
    # one bit per element, plus a float offset and a float scale per sample
    assert _per_sample("bit", columns) == pytest.approx(columns / 8 + 8)
    # one int8 per element, plus one float scale per sample
    assert _per_sample("int8", columns) == pytest.approx(columns + 4)


def _stack_tape(storage, depth=4, dim=256, batch=64):
    torch.manual_seed(1)
    layers = nn.Sequential(*[nn.Linear(dim, dim, bias=False) for _ in range(depth)])
    qste.convert(layers)
    for surface in qste.surfaces(layers):
        surface._immediate = lambda *_: None
    inputs = torch.randn(batch, dim)
    with exact_evidence(storage), Tape() as tape:
        output = layers(inputs)
    total = tape.total
    output.sum().backward()
    return total


def test_whole_stack_tape_shrinks():
    """End to end, not just one layer in isolation."""

    fp32 = _stack_tape("fp32")
    bit = _stack_tape("bit")
    assert bit < fp32 / 4, f"stack tape {fp32} -> {bit}"


def test_optimizer_state_is_one_byte_per_weight():
    model = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 128))
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model)
    weights = sum(s.rows * s.columns for s in qste.surfaces(model))
    # int8 moment (1 byte/weight) plus fp16 block scales and row/col vectors
    assert coordinates.state_bytes() < weights * 1.1
    # AdamW over the same weights would hold two fp32 moments
    assert coordinates.state_bytes() < weights * 8 / 6


def test_evidence_buffers_are_not_allocated_in_the_common_case():
    """A once-used surface applies in backward and never holds a buffer."""

    model = nn.Sequential(nn.Linear(64, 64))
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model)
    model(torch.randn(8, 64)).sum().backward()
    for surface in qste.surfaces(model):
        assert surface._evidence is None
        assert surface._call_evidence is None
    coordinates.step()


def test_deferred_mode_allocates_exactly_one_buffer_per_surface():
    model = nn.Sequential(nn.Linear(64, 64))
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model)
    coordinates.prepare_accumulation()
    for _ in range(3):
        model(torch.randn(8, 64)).sum().backward()
    for surface in qste.surfaces(model):
        assert surface._evidence is not None
        assert surface._evidence.shape == (surface.rows, surface.columns)
    coordinates.step()


def test_export_drops_the_coordinate():
    model = nn.Sequential(nn.Linear(512, 512, bias=False))
    qste.convert(model)
    report = qste.memory_report(model)
    assert report["inference_ratio"] > 30.0
    exported = qste.export_packed(model)
    resident = sum(
        b.numel() * b.element_size() for b in exported.buffers()
    )
    assert resident < 512 * 512 * 4 / 25
