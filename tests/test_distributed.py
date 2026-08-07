"""Multi-rank safety, run for real on gloo with two CPU processes.

The failure this guards against is silent: without an evidence reducer, ranks
step their coordinates from their own local batches and diverge while the loss
keeps falling. So the tests below actually spawn ranks and compare bits, rather
than asserting that a helper returns the right type.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import qste
from qste import distributed as qdist


def _model(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(32, 48), nn.ReLU(), nn.Linear(48, 16))
    qste.convert(model, include=["0", "2"])
    return model


def _run(rank, world, queue, reduce, steps=4):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        model = _model(seed=0)  # identical init on every rank
        # `None` is the default and must resolve to an all-reduce; `False` is
        # the explicit opt-out. Nothing else is passed, on purpose.
        coordinates = qste.QSTEOptimizer(
            model,
            config={"coordinate_lr": 8.0},
            gradient_reducer=None if reduce else False,
        )
        generator = torch.Generator().manual_seed(100 + rank)  # different data
        for _ in range(steps):
            inputs = torch.randn(8, 32, generator=generator)
            model(inputs).square().mean().backward()
            coordinates.step()
            model.zero_grad(set_to_none=True)
        signature = [s.packed_sign.data.clone() for s in qste.surfaces(model)]
        queue.put((rank, [t.numpy().tobytes() for t in signature], qdist.surfaces_agree(model)))
    finally:
        dist.destroy_process_group()


def _spawn(reduce, world=2):
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_run, args=(rank, world, queue, reduce))
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=180) for _ in range(world)]
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0, f"rank exited {process.exitcode}"
    return dict((rank, payload) for rank, *payload in results)


@pytest.mark.slow
def test_default_optimizer_keeps_ranks_bit_identical():
    """No reducer passed. Correct multi-rank behaviour must be the default."""

    results = _spawn(reduce=True)
    assert results[0][0] == results[1][0], "ranks hold different signs"
    assert all(agree for _, agree in results.values())


@pytest.mark.slow
def test_opting_out_of_reduction_really_does_drift():
    """The hazard is real, and this is what the default is protecting against."""

    results = _spawn(reduce=False)
    assert results[0][0] != results[1][0]
    assert not any(agree for _, agree in results.values())


def test_default_reducer_is_absent_on_a_single_process():
    """No process group, no all-reduce, no cost."""

    model = _model()
    assert qste.QSTEOptimizer(model).gradient_reducer is None


def test_explicit_reducer_is_used_verbatim():
    marker = []

    def reducer(evidence):
        marker.append(evidence.shape)
        return evidence

    model = _model()
    coordinates = qste.QSTEOptimizer(model, gradient_reducer=reducer)
    assert coordinates.gradient_reducer is reducer
    model(torch.randn(4, 32)).square().mean().backward()
    coordinates.step()
    assert len(marker) == len(qste.surfaces(model))


def test_reducers_are_identity_when_not_distributed():
    evidence = torch.randn(4, 6)
    assert torch.equal(qdist.mean_evidence()(evidence.clone()), evidence)
    assert torch.equal(qdist.sum_evidence()(evidence.clone()), evidence)
    assert qdist.surfaces_agree(_model()) is True
    assert qdist.broadcast_surfaces(_model()) == 0


def test_fsdp_ignored_states_covers_every_surface_tensor():
    model = _model()
    ignored = qdist.fsdp_ignored_states(model)
    ignored_ids = {id(p) for p in ignored}
    for surface in qste.surfaces(model):
        assert id(surface.coordinate) in ignored_ids
        assert id(surface.packed_sign) in ignored_ids
        assert id(surface.log_scale) in ignored_ids
    # Nothing outside the surfaces is hidden from FSDP.
    surface_ids = {
        id(t)
        for s in qste.surfaces(model)
        for t in (s.coordinate, s.packed_sign, s.log_scale)
    }
    assert ignored_ids == surface_ids
    assert qdist.fsdp_ignored_modules(model) == qste.surfaces(model)


def test_replicated_bytes_is_one_byte_per_weight_plus_signs():
    model = _model()
    weights = sum(s.rows * s.columns for s in qste.surfaces(model))
    signs = sum(s.rows * ((s.columns + 7) // 8) for s in qste.surfaces(model))
    scales = sum(s.rows * 4 for s in qste.surfaces(model))
    assert qdist.replicated_bytes(model) == weights + signs + scales


def test_ddp_wrapping_leaves_coordinates_out_of_the_gradient_path():
    """DDP only reduces ``requires_grad`` params; coordinates must not be there."""

    model = _model()
    reduced = [name for name, p in model.named_parameters() if p.requires_grad]
    assert not any(name.endswith("coordinate") for name in reduced)
    assert not any(name.endswith("packed_sign") for name in reduced)
    assert any(name.endswith("log_scale") for name in reduced)
