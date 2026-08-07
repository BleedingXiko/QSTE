"""The GPU tests, checked for calls that cannot work, without a GPU.

Six consecutive hardware runs failed on tests that had never executed. Not
kernel bugs -- ``qste.nn.QSTELinear`` when the class is on the package,
``QSTEOptimizer(lr=...)`` when it takes a model and reads its config,
``convert()`` on a bare ``nn.Linear`` that has no children to select,
``coordinate.grad`` on a parameter that never receives one. Every single one
was decidable here, instantly, and instead each cost a queue slot, three
minutes of somebody else's GPU, and a report that said FAIL.

The GPU tests are skipped on a machine without a GPU, so nothing about them is
exercised at all -- not even whether the names they use exist. This reads them
as source and checks what can be checked: that every attribute reached for on
``qste`` is really there, and that every call into the package matches the
signature it is calling.

It cannot prove a GPU test passes. It can prove one is not dead on arrival,
which is the failure that has actually been happening.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import qste

GPU_TESTS = Path(__file__).with_name("test_gpu.py")


def _tree():
    return ast.parse(GPU_TESTS.read_text())


def _dotted(node):
    """``qste.QSTELinear.from_linear`` -> ['qste', 'QSTELinear', 'from_linear']."""

    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def test_every_name_reached_for_on_qste_exists():
    missing = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Attribute):
            continue
        parts = _dotted(node)
        if not parts or parts[0] != "qste":
            continue
        held = qste
        for step, name in enumerate(parts[1:], start=1):
            if not hasattr(held, name):
                missing.append(".".join(parts[: step + 1]))
                break
            held = getattr(held, name)
    assert not missing, f"these do not exist: {sorted(set(missing))}"


def test_every_call_into_qste_matches_its_signature():
    """Keyword arguments and arity, against the real thing.

    ``QSTEOptimizer(model.parameters(), lr=0.05)`` is torch's shape, not this
    package's, and it took a hardware run to find that out.
    """

    wrong = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Call):
            continue
        parts = _dotted(node.func)
        if not parts or parts[0] != "qste" or len(parts) < 2:
            continue
        held = qste
        for name in parts[1:]:
            held = getattr(held, name, None)
            if held is None:
                break
        if held is None or not callable(held):
            continue
        try:
            signature = inspect.signature(held)
        except (TypeError, ValueError):  # a builtin or a C extension
            continue
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            continue
        if any(keyword.arg is None for keyword in node.keywords):
            continue
        try:
            signature.bind_partial(
                *[None] * len(node.args),
                **{keyword.arg: None for keyword in node.keywords},
            )
        except TypeError as error:
            wrong.append(f"{'.'.join(parts)}(...): {error}")
    assert not wrong, "calls that cannot bind:\n  " + "\n  ".join(sorted(set(wrong)))


@pytest.mark.parametrize(
    "expression",
    [
        # Every construction the GPU tests build a model with. These are the
        # exact lines that failed on hardware, run where they are free.
        "qste.QSTELinear.from_linear(nn.Linear(8, 8, bias=False))",
        "qste.QSTEOptimizer(qste.QSTELinear.from_linear(nn.Linear(8, 8)))",
        "qste.Surface(torch.randn(8, 8), qste.QSTEConfig())",
        "qste.convert(nn.Sequential(nn.Linear(8, 8)))",
    ],
)
def test_the_constructions_the_gpu_tests_use_actually_construct(expression):
    import torch
    import torch.nn as nn

    assert eval(expression, {"qste": qste, "torch": torch, "nn": nn}) is not None


def test_a_surface_accumulates_evidence_rather_than_a_gradient():
    """``coordinate.grad`` is ``None`` and always will be.

    A binary weight has no float to differentiate; what the backward leaves
    behind is evidence, and that is what the coordinate step consumes. A test
    that reaches for ``.grad`` here is comparing ``None`` to ``None`` and
    passing, or crashing on ``.clone()`` -- which is what happened.
    """

    import torch
    import torch.nn as nn

    layer = qste.QSTELinear.from_linear(nn.Linear(16, 16, bias=False))
    layer(torch.randn(4, 16)).square().sum().backward()

    assert layer.surface.coordinate.grad is None
    evidence = layer.surface.flush_evidence()
    assert evidence is not None and evidence.shape == (16, 16)
    assert evidence.abs().sum() > 0


# Two traps that bind cleanly and fail at run time, so the checks above cannot
# see them. Both cost a hardware run. They are named explicitly rather than
# generalised, because a narrow check that fires is worth more than a broad one
# that does not.


def _unwrap(call):
    """``nn.Linear(...).to(DEVICE)`` -> the ``nn.Linear(...)`` inside it.

    Written out because the obvious version does not work: ``_dotted`` walks
    attribute chains down to a Name, and the receiver here is a Call, so it
    returns None and every check downstream quietly does nothing. That is
    exactly how this check passed on the bug it was written for.
    """

    methods = {"to", "cuda", "cpu", "float", "half", "double", "eval", "train"}
    while (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in methods
        and isinstance(call.func.value, ast.Call)
    ):
        call = call.func.value
    return call


def _assignments(function):
    """``name -> the call it was assigned from``, within one function body."""

    bound = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = node.value
    return bound


def test_convert_is_never_handed_a_module_with_nothing_to_convert():
    """``convert()`` walks children. A bare ``nn.Linear`` has none.

    It selects nothing and raises "convert() selected no layers". Twice now a
    test has built a single layer and handed it straight over, when what it
    wanted was ``QSTELinear.from_linear`` or a ``Sequential`` around it.
    """

    leaves = {"Linear", "Embedding", "Conv1d", "Conv2d", "LayerNorm"}
    offenders = []
    for function in ast.walk(_tree()):
        if not isinstance(function, ast.FunctionDef):
            continue
        bound = _assignments(function)
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            parts = _dotted(node.func)
            if parts != ["qste", "convert"] or not node.args:
                continue
            argument = node.args[0]
            source = bound.get(argument.id) if isinstance(argument, ast.Name) else argument
            if not isinstance(source, ast.Call):
                continue
            built = _dotted(_unwrap(source).func)
            if built and built[-1] in leaves:
                offenders.append(f"{function.name}: convert({'.'.join(built)}(...))")
    assert not offenders, (
        "convert() walks children and these have none, so it selects nothing "
        f"and raises: {sorted(set(offenders))}"
    )


def test_nothing_reads_a_gradient_that_is_always_none():
    """``surface.coordinate.grad`` and ``surface.packed_sign.grad`` are None.

    Neither is differentiated -- a binary weight has no float to differentiate
    -- and what the backward leaves is evidence. Reading ``.grad`` here either
    compares None to None and passes for the wrong reason, or crashes on
    ``.clone()``. Both have happened in the same test.
    """

    offenders = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Attribute) or node.attr != "grad":
            continue
        inner = node.value
        if isinstance(inner, ast.Attribute) and inner.attr in {
            "coordinate", "packed_sign"
        }:
            offenders.append(f"line {node.lineno}: .{inner.attr}.grad")
    assert not offenders, (
        "these are always None; use surface.flush_evidence() instead: "
        f"{sorted(set(offenders))}"
    )
