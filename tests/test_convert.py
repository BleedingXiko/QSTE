"""Conversion must be surgical: only what you named, nothing else touched.

These tests stand in for the adoption promise. A host framework hands over a
model it built its own way; it gets that same object back, with the layers it
chose swapped and every other layer, buffer, and attribute exactly as it was.
"""

import copy

import pytest
import torch
import torch.nn as nn

import qste


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.norm(x + self.fc2(torch.relu(self.fc1(x))) * torch.sigmoid(self.gate(x)))


class Tiny(nn.Module):
    def __init__(self, vocab=48, dim=16, depth=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab)
        self.register_buffer("marker", torch.arange(4))

    def forward(self, ids):
        h = self.embed(ids)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def test_plan_reports_without_mutating():
    model = Tiny()
    before = copy.deepcopy(model.state_dict())
    candidates = qste.plan(model, include=["blocks.*.fc1"])
    assert {c.name for c in candidates if c.selected} == {"blocks.0.fc1", "blocks.1.fc1"}
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert "selected total" in qste.format_plan(candidates)


def test_convert_replaces_only_selected_layers():
    model = Tiny()
    qste.convert(model, include=["blocks.*.fc1", "blocks.*.fc2"])

    assert isinstance(model.blocks[0].fc1, qste.QSTELinear)
    assert isinstance(model.blocks[1].fc2, qste.QSTELinear)
    assert isinstance(model.blocks[0].gate, nn.Linear)
    assert isinstance(model.head, nn.Linear)
    assert isinstance(model.embed, nn.Embedding)
    assert isinstance(model.blocks[0].norm, nn.LayerNorm)
    assert torch.equal(model.marker, torch.arange(4))
    assert model._qste_converted == (
        "blocks.0.fc1", "blocks.0.fc2", "blocks.1.fc1", "blocks.1.fc2",
    )


def test_convert_returns_the_same_object():
    model = Tiny()
    assert qste.convert(model, include=["head"]) is model


def test_exclude_wins_over_include():
    model = Tiny()
    qste.convert(model, include=["blocks.*"], exclude=["*.gate"])
    assert isinstance(model.blocks[0].fc1, qste.QSTELinear)
    assert isinstance(model.blocks[0].gate, nn.Linear)


def test_convert_everything_by_default():
    model = Tiny()
    qste.convert(model)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert not linears
    assert isinstance(model.embed, qste.QSTEEmbedding)


def test_unknown_pattern_raises_rather_than_silently_doing_nothing():
    with pytest.raises(ValueError, match="selected no layers"):
        qste.convert(Tiny(), include=["does.not.exist"])


def test_tied_weights_share_one_surface():
    model = Tiny(vocab=48, dim=16)
    model.head = nn.Linear(16, 48, bias=False)
    model.head.weight = model.embed.weight
    qste.convert(model, include=["embed", "head"])
    assert model.embed.surface is model.head.surface
    assert len(qste.surfaces(model)) == 1


def test_converted_model_runs_forward_and_backward():
    model = Tiny()
    qste.convert(model, include=["blocks.*.fc1", "blocks.*.fc2", "embed"])
    ids = torch.randint(0, 48, (3, 6))
    loss = model(ids).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_coordinates_are_excluded_from_the_hosts_optimizer():
    model = Tiny()
    qste.convert(model, include=["blocks.*.fc1"])
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    for surface in qste.surfaces(model):
        assert id(surface.coordinate) not in trainable
        assert id(surface.packed_sign) not in trainable
        assert id(surface.log_scale) in trainable


def test_state_dict_roundtrips_through_the_hosts_own_saving():
    model = Tiny()
    qste.convert(model, include=["blocks.*.fc1", "embed"])
    ids = torch.randint(0, 48, (2, 5))
    expected = model(ids)

    restored = Tiny()
    qste.convert(restored, include=["blocks.*.fc1", "embed"])
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored(ids), expected, atol=1e-5)


def test_memory_report_counts_the_real_win():
    model = Tiny(vocab=512, dim=128)
    qste.convert(model)
    report = qste.memory_report(model)
    assert report["training_ratio"] > 3.0
    assert report["inference_ratio"] > 25.0
    assert report["inference_mb"] < report["training_mb"] < report["float_mb"]


def test_export_packed_drops_training_state_and_keeps_outputs():
    model = Tiny()
    qste.convert(model, include=["blocks.*.fc1", "blocks.*.fc2", "head"])
    model.eval()
    ids = torch.randint(0, 48, (2, 5))
    with torch.no_grad():
        expected = model(ids)

    exported = qste.export_packed(model)
    with torch.no_grad():
        assert torch.allclose(exported(ids), expected, atol=1e-4)
    assert not any(isinstance(m, qste.Surface) for m in exported.modules())
    assert isinstance(exported.blocks[0].fc1, qste.PackedLinear)


def test_repeated_conversion_is_idempotent():
    model = Tiny()
    qste.convert(model, include=["head"])
    first = model.head
    qste.convert(model, include=["head"])
    assert model.head is first


def test_convert_accepts_a_plain_config_mapping():
    model = Tiny()
    qste.convert(model, include=["head"], config={"coordinate_lr": 4.0})
    assert model.head.surface.cfg.coordinate_lr == 4.0


def test_config_exposes_no_storage_option():
    """What forward retains is the mechanism, not a setting to choose."""

    assert "evidence_storage" not in qste.QSTEConfig().metadata()
    with pytest.raises(TypeError):
        qste.QSTEConfig(evidence_storage="fp32")
