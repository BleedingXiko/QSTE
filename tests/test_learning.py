"""It has to actually train, on real objectives, in every storage mode.

Small models and few steps, so the suite stays fast, but real losses against a
stated chance baseline -- not "the loss decreased", which a broken optimizer
can also manage.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import qste
from qste.functional import exact_evidence

STORAGE = ["fp32", "int8", "bit"]


def _train(model, batches, steps, storage="bit", coordinate_lr=6.0, lr=3e-3):
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": coordinate_lr})
    continuous = torch.optim.AdamW(qste.continuous_parameters(model), lr=lr)
    losses = []
    with exact_evidence(storage):
        for step in range(steps):
            inputs, targets = batches(step)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            continuous.zero_grad(set_to_none=True)
            coordinates.zero_grad()
            loss.backward()
            continuous.step()
            coordinates.step()
            losses.append(loss.item())
    return losses


def _final(losses, window=15):
    return sum(losses[-window:]) / window


# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("storage", STORAGE)
def test_classification(storage):
    """Linear-readout classification: a floor every working method clears."""

    torch.manual_seed(0)
    classes, dim = 8, 64
    centroids = torch.randn(classes, dim) * 2.0
    generator = torch.Generator().manual_seed(1)

    def batches(_):
        labels = torch.randint(0, classes, (128,), generator=generator)
        return centroids[labels] + torch.randn(128, dim, generator=generator) * 0.4, labels

    model = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, classes))
    losses = _train(model, batches, 150, storage)
    chance = math.log(classes)
    assert _final(losses) < chance * 0.5, f"{storage}: {_final(losses):.3f} vs {chance:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("storage", STORAGE)
def test_deterministic_token_map(storage):
    """Position-wise token transform. Learnable, and provably not by chance."""

    torch.manual_seed(2)
    vocab, dim = 32, 96
    generator = torch.Generator().manual_seed(3)

    def batches(_):
        ids = torch.randint(0, vocab, (16, 24), generator=generator)
        return ids, (ids * 7 + 3) % vocab

    model = nn.Sequential(
        nn.Embedding(vocab, dim), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, vocab)
    )
    losses = _train(model, batches, 200, storage, coordinate_lr=8.0)
    chance = math.log(vocab)
    assert _final(losses) < chance - 0.6, f"{storage}: {_final(losses):.3f} vs {chance:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("storage", STORAGE)
def test_next_token_on_a_markov_source(storage):
    """Sequence modeling with a real entropy floor to beat."""

    torch.manual_seed(4)
    vocab, dim = 24, 96
    transition = torch.softmax(torch.randn(vocab, vocab) * 2.5, dim=-1)
    entropy = float(-(transition * transition.clamp_min(1e-9).log()).sum(-1).mean())
    generator = torch.Generator().manual_seed(5)

    def batches(_):
        current = torch.randint(0, vocab, (32, 1), generator=generator)
        columns = [current]
        for _ in range(15):
            current = torch.multinomial(
                transition[current.squeeze(-1)], 1, generator=generator
            )
            columns.append(current)
        sequence = torch.cat(columns, dim=1)
        return sequence[:, :-1], sequence[:, 1:]

    model = nn.Sequential(nn.Embedding(vocab, dim), nn.Linear(dim, vocab))
    losses = _train(model, batches, 250, storage, coordinate_lr=8.0)
    chance = math.log(vocab)
    final = _final(losses, 25)
    assert final < chance - 0.3, f"{storage}: {final:.3f} (chance {chance:.3f}, floor {entropy:.3f})"


@pytest.mark.slow
def test_bit_storage_tracks_fp32_storage():
    """The saving must not cost accuracy, which is the whole bargain.

    ``exact_evidence`` is what makes this measurable: identical seeds and
    identical arithmetic, differing only in what forward kept.
    """

    def run(storage):
        torch.manual_seed(6)
        classes, dim = 8, 64
        centroids = torch.randn(classes, dim) * 2.0
        generator = torch.Generator().manual_seed(7)

        def batches(_):
            labels = torch.randint(0, classes, (128,), generator=generator)
            return centroids[labels] + torch.randn(128, dim, generator=generator) * 0.4, labels

        model = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, classes))
        return _final(_train(model, batches, 150, storage))

    exact = run("fp32")
    packed = run("bit")
    assert packed < exact + 0.15, f"bit {packed:.3f} vs fp32 {exact:.3f}"


@pytest.mark.slow
def test_partial_conversion_trains_alongside_float_layers():
    """A model that is half binary must train as a whole."""

    torch.manual_seed(8)
    classes, dim = 6, 48
    centroids = torch.randn(classes, dim) * 2.0
    generator = torch.Generator().manual_seed(9)

    model = nn.Sequential(
        nn.Linear(dim, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, classes)
    )
    qste.convert(model, include=["2"])  # only the middle layer is binary
    assert isinstance(model[0], nn.Linear) and isinstance(model[2], qste.QSTELinear)

    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 6.0})
    continuous = torch.optim.AdamW(qste.continuous_parameters(model), lr=3e-3)
    losses = []
    for _ in range(150):
        labels = torch.randint(0, classes, (128,), generator=generator)
        inputs = centroids[labels] + torch.randn(128, dim, generator=generator) * 0.4
        loss = F.cross_entropy(model(inputs), labels)
        continuous.zero_grad(set_to_none=True)
        loss.backward()
        continuous.step()
        coordinates.step()
        losses.append(loss.item())
    assert _final(losses) < math.log(classes) * 0.5


@pytest.mark.slow
def test_gradient_accumulation_matches_one_large_batch():
    """Deferred evidence must sum to what the single batch would have given.

    The claim is about the evidence, and that is what is asserted on: four
    microbatches accumulate to the same matrix as one pass, to float summation
    order. It is deliberately *not* asserted that the resulting coordinates
    come out bit-identical. Rounding here is stochastic, so an element whose
    target lands within a summation error of its draw rounds up in one run and
    down in the other -- and demanding otherwise would be demanding that two
    different orders of adding the same floats produce the same float, which no
    backend promises. The bit-exact property that does hold is one pass against
    itself, and it is checked separately below.
    """

    def run(microbatches):
        torch.manual_seed(10)
        model = nn.Sequential(nn.Linear(32, 32, bias=False))
        qste.convert(model)
        coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 4.0})
        coordinates.prepare_accumulation()
        generator = torch.Generator().manual_seed(11)
        data = torch.randn(64, 32, generator=generator)
        with exact_evidence():
            for chunk in data.chunk(microbatches):
                (model(chunk).square().sum() / len(data)).backward()
        surface = qste.surfaces(model)[0]
        evidence = surface._evidence.clone()
        coordinates.step()
        return evidence, surface.coordinate.clone()

    one_evidence, one_coordinate = run(1)
    many_evidence, many_coordinate = run(4)

    scale = one_evidence.pow(2).mean().sqrt().clamp_min(1e-12)
    assert float((one_evidence - many_evidence).pow(2).mean().sqrt() / scale) < 1e-4
    assert float((one_evidence - many_evidence).abs().max() / scale) < 1e-2

    difference = (one_coordinate.int() - many_coordinate.int()).abs()
    assert int(difference.max()) <= 1
    assert float((difference > 0).float().mean()) < 0.01


@pytest.mark.slow
def test_one_pass_is_reproducible_to_the_bit():
    """The part that must be exact, kept apart from the part that cannot be.

    Same seed, same data, same order: identical integers. This is the property
    DDP relies on -- every rank draws from ``(seed, step, index)`` and never
    communicates about it -- and it is what would break if the rounding stream
    ever picked up state from a generator instead.
    """

    def run():
        torch.manual_seed(10)
        model = nn.Sequential(nn.Linear(32, 32, bias=False))
        qste.convert(model)
        coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 4.0})
        generator = torch.Generator().manual_seed(11)
        data = torch.randn(64, 32, generator=generator)
        with exact_evidence():
            model(data).square().sum().backward()
        coordinates.step()
        return qste.surfaces(model)[0].coordinate.clone()

    assert torch.equal(run(), run())


@pytest.mark.slow
def test_flips_report_real_movement():
    """Flip count is the honest progress metric for a binary model."""

    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(64, 64))
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 12.0})
    target = torch.randn(16, 64)
    flips = []
    for _ in range(20):
        F.mse_loss(model(torch.randn(16, 64)), target).backward()
        flips.append(coordinates.step())
        model.zero_grad(set_to_none=True)
    assert sum(flips) > 0, "no sign ever changed; the coordinate is not moving"
    assert all(f >= 0 for f in flips)


@pytest.mark.slow
def test_training_survives_a_checkpoint_roundtrip():
    torch.manual_seed(13)
    generator = torch.Generator().manual_seed(14)

    def build():
        torch.manual_seed(13)
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
        qste.convert(model)
        return model, qste.QSTEOptimizer(model, config={"coordinate_lr": 6.0})

    model, coordinates = build()
    data = torch.randn(32, 32, generator=generator)
    labels = torch.randint(0, 4, (32,), generator=generator)
    for _ in range(5):
        F.cross_entropy(model(data), labels).backward()
        coordinates.step()
        model.zero_grad(set_to_none=True)

    saved_model = {k: v.clone() for k, v in model.state_dict().items()}
    saved_optimizer = coordinates.state_dict()

    restored, restored_coordinates = build()
    restored.load_state_dict(saved_model)
    restored_coordinates.load_state_dict(saved_optimizer)

    for target in (model, restored):
        F.cross_entropy(target(data), labels).backward()
    coordinates.step()
    restored_coordinates.step()

    for a, b in zip(qste.surfaces(model), qste.surfaces(restored)):
        assert torch.equal(a.coordinate, b.coordinate)
        assert torch.equal(a.packed_sign, b.packed_sign)
