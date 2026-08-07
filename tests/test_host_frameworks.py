"""Adoption tests: real host frameworks, and the patterns that break naive swaps.

A library that only works on ``nn.Sequential`` is not framework agnostic. The
things below are what actual model code does to its own layers -- read weights
directly, fuse projections, tie parameters, checkpoint, autocast -- and each
one is a way a drop-in replacement can be silently wrong rather than loudly
broken.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import qste

try:
    import dabsn as _dabsn
except Exception:  # pragma: no cover - depends on the environment
    _dabsn = None


def _moved(surfaces, before):
    return sum(int(not torch.equal(a, s.packed_sign)) for a, s in zip(before, surfaces))


# ---------------------------------------------------------------------------
# Hosts that read .weight instead of calling the module
# ---------------------------------------------------------------------------


class FusedHost(nn.Module):
    """Reads ``.weight`` off its children and fuses them into one GEMM.

    DABSN's recurrence does exactly this. A ``QSTELinear`` whose ``weight``
    were a detached tensor would train its scale here and never move a single
    coordinate, and nothing would raise.
    """

    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        fused = torch.cat([self.q.weight, self.k.weight, self.v.weight], dim=0)
        return F.linear(x, fused)


def test_weight_reading_host_moves_every_coordinate():
    torch.manual_seed(0)
    model = FusedHost(32)
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 8.0})
    surfaces = qste.surfaces(model)
    before = [s.packed_sign.clone() for s in surfaces]

    model(torch.randn(8, 32)).square().mean().backward()
    coordinates.step()
    assert _moved(surfaces, before) == len(surfaces) == 3


def test_weight_reading_host_actually_learns():
    torch.manual_seed(1)
    model = FusedHost(24)
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 6.0})
    continuous = torch.optim.AdamW(qste.continuous_parameters(model), lr=3e-3)
    generator = torch.Generator().manual_seed(2)
    inputs = torch.randn(64, 24, generator=generator)
    target = torch.randn(64, 72, generator=generator)

    losses = []
    for _ in range(60):
        loss = F.mse_loss(model(inputs), target)
        continuous.zero_grad(set_to_none=True)
        loss.backward()
        continuous.step()
        coordinates.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.9, f"{losses[0]:.4f} -> {losses[-1]:.4f}"


def test_mixed_host_calls_some_modules_and_reads_other_weights():
    class Mixed(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.called = nn.Linear(dim, dim)
            self.read = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.called(x) + F.linear(x, self.read.weight.t())

    torch.manual_seed(3)
    model = Mixed(16)
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 8.0})
    surfaces = qste.surfaces(model)
    before = [s.packed_sign.clone() for s in surfaces]
    model(torch.randn(8, 16)).square().mean().backward()
    coordinates.step()
    assert _moved(surfaces, before) == 2


# ---------------------------------------------------------------------------
# Other host behaviours
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reentrant", [False, True])
def test_gradient_checkpointing_gives_the_same_coordinates(reentrant):
    """Recomputation runs forward twice. The result must not depend on that.

    A surface that trusted its forward count would sit on evidence it never
    released here and silently never train, with the loss still going down
    because the float parameters kept moving.
    """

    def run(checkpointed):
        torch.manual_seed(4)
        block = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
        qste.convert(block)
        coordinates = qste.QSTEOptimizer(block, config={"coordinate_lr": 8.0})
        # Reentrant checkpointing requires an input that requires grad.
        inputs = torch.randn(
            8, 32, generator=torch.Generator().manual_seed(5)
        ).requires_grad_(True)
        if checkpointed:
            output = torch.utils.checkpoint.checkpoint(
                block, inputs, use_reentrant=reentrant
            )
        else:
            output = block(inputs)
        output.square().mean().backward()
        flips = coordinates.step()
        return [s.packed_sign.clone() for s in qste.surfaces(block)], flips

    plain, plain_flips = run(False)
    checkpointed, checkpointed_flips = run(True)
    assert plain_flips > 0, "the baseline did not move, so the test proves nothing"
    assert checkpointed_flips == plain_flips
    for a, b in zip(plain, checkpointed):
        assert torch.equal(a, b)


def test_more_forwards_than_backwards_still_applies_evidence():
    """Explicitly: a host may call forward and throw the result away."""

    torch.manual_seed(6)
    model = nn.Sequential(nn.Linear(16, 16))
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 8.0})
    surface = qste.surfaces(model)[0]

    model(torch.randn(4, 16))  # discarded, never backwarded
    model(torch.randn(4, 16)).sum().backward()
    before = surface.packed_sign.clone()
    assert coordinates.step() > 0
    assert not torch.equal(before, surface.packed_sign)
    assert surface._pending_calls == 0


def test_autocast_forward_keeps_training_stable():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8))
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        loss = model(torch.randn(8, 32)).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in qste.continuous_parameters(model):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    coordinates.step()


def test_module_moves_across_devices_and_dtypes():
    model = nn.Sequential(nn.Linear(16, 16))
    qste.convert(model)
    model.to(torch.float32)  # a no-op that must not corrupt int8 state
    surface = qste.surfaces(model)[0]
    assert surface.coordinate.dtype == torch.int8
    assert surface.packed_sign.dtype == torch.uint8
    assert surface.log_scale.dtype == torch.float32
    model(torch.randn(4, 16)).sum().backward()


def test_named_parameters_and_numel_stay_sane_for_host_reporting():
    """Hosts count parameters; the count should describe the binary model."""

    model = nn.Sequential(nn.Linear(64, 64, bias=False))
    qste.convert(model)
    names = dict(model.named_parameters())
    assert "0.surface.coordinate" in names
    assert "0.surface.log_scale" in names
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 64, "only the row scale should be a trainable float"


def test_hooks_registered_by_the_host_still_fire():
    model = nn.Sequential(nn.Linear(16, 16))
    qste.convert(model)
    seen = []
    model[0].register_forward_hook(lambda m, i, o: seen.append(o.shape))
    model(torch.randn(4, 16))
    assert seen == [torch.Size([4, 16])]


# ---------------------------------------------------------------------------
# DABSN, the real thing
# ---------------------------------------------------------------------------

requires_dabsn = pytest.mark.skipif(_dabsn is None, reason="dabsn is not installed")


def _dabsn_model(depth=2, hidden=32, state=16):
    torch.manual_seed(0)
    layers = [_dabsn.DABSNLayerSpec(hidden_dim=hidden, state_dim=state) for _ in range(depth)]
    return _dabsn.DABSNModel(
        input_dim=16, out_dim=8, layers=layers, residual=True, mlp_ratio=2.0
    )


@requires_dabsn
@pytest.mark.parametrize(
    "include",
    [
        pytest.param(["*.mlp_fc1", "*.mlp_fc2"], id="mlp"),
        pytest.param(
            ["*.mlp_fc1", "*.mlp_fc2", "*.short_read.out", "*.state_to_hidden"],
            id="mlp+read",
        ),
        pytest.param(None, id="everything"),
    ],
)
def test_dabsn_converts_and_every_surface_moves(include):
    """Including ``core.W/A/Wg/Ug``, which DABSN consumes as raw weights."""

    model = _dabsn_model()
    qste.convert(model, include=include)
    surfaces = qste.surfaces(model)
    assert surfaces
    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 8.0})
    before = [s.packed_sign.clone() for s in surfaces]

    loss = model(torch.randn(2, 12, 16)).square().mean()
    loss.backward()
    coordinates.step()

    assert torch.isfinite(loss)
    assert _moved(surfaces, before) == len(surfaces), "a surface never moved"
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@requires_dabsn
def test_dabsn_fully_binary_trains():
    model = _dabsn_model()
    qste.convert(model)
    coordinates = qste.QSTEOptimizer(model, config={"coordinate_lr": 6.0})
    continuous = torch.optim.AdamW(qste.continuous_parameters(model), lr=3e-3)
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(4, 12, 16, generator=generator)
    target = torch.randn(4, 8, generator=generator)

    losses = []
    for _ in range(40):
        loss = F.mse_loss(model(inputs), target)
        continuous.zero_grad(set_to_none=True)
        loss.backward()
        continuous.step()
        coordinates.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.85, f"{losses[0]:.4f} -> {losses[-1]:.4f}"


@requires_dabsn
def test_dabsn_state_dict_roundtrips():
    model = _dabsn_model()
    qste.convert(model)
    inputs = torch.randn(2, 12, 16)
    with torch.no_grad():
        expected = model(inputs)

    restored = _dabsn_model()
    qste.convert(restored)
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.allclose(restored(inputs), expected, atol=1e-5)


@requires_dabsn
def test_dabsn_plan_lists_the_recurrence_matrices():
    candidates = {c.name for c in qste.plan(_dabsn_model())}
    assert any(name.endswith("core.W") for name in candidates)
    assert any(name.endswith("core.Wg") for name in candidates)
    assert any(name.endswith("mlp_fc1") for name in candidates)
