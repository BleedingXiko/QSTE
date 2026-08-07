"""Model surgery: swap chosen ``nn.Linear``/``nn.Embedding`` for QSTE ones.

The whole adoption story. No base class to inherit, no config file, no training
loop to rewrite: hand :func:`convert` a model and the names of the layers you
want binary, and get the same model object back with those layers replaced.

    qste.plan(model)                      # what would convert, and what it costs
    qste.convert(model, include=[...])    # do it

Selection is by qualified module name with shell-style globs
(``"blocks.*.mlp_fc1"``). Anything unselected keeps training in float.

Which layers to name is a modelling decision, and the README's "What to
convert" section is the guidance: MLP matrices first, attention projections
only after measuring, output heads and embeddings usually not. Converting
everything works and costs more quality than it saves memory.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch.nn as nn

from .config import QSTEConfig, coerce_config
from .modules import PackedEmbedding, PackedLinear, QSTEEmbedding, QSTELinear
from .surface import Surface

DEFAULT_TYPES = (nn.Linear, nn.Embedding)


@dataclass(frozen=True)
class Candidate:
    """One convertible layer and what converting it would buy."""

    name: str
    kind: str
    rows: int
    columns: int
    selected: bool
    tied_to: str | None = None

    @property
    def parameters(self) -> int:
        return self.rows * self.columns

    @property
    def float_bytes(self) -> int:
        return self.parameters * 4

    @property
    def qste_bytes(self) -> int:
        """Packed signs plus the INT8 coordinate plus the row scale."""

        return self.rows * ((self.columns + 7) // 8) + self.parameters + self.rows * 4

    @property
    def packed_bytes(self) -> int:
        """What survives export, once the coordinate is dropped."""

        return self.rows * ((self.columns + 7) // 8) + self.rows * 4


def _matches(name: str, patterns: Sequence[str] | None) -> bool:
    if patterns is None:
        return True
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


_CONVERTED = (QSTELinear, QSTEEmbedding)


def _walk(model: nn.Module, types: tuple[type, ...]) -> Iterator[tuple[str, nn.Module]]:
    """Convertible layers, plus ones already converted, by qualified name."""

    for name, module in model.named_modules():
        if name and isinstance(module, types + _CONVERTED):
            yield name, module


def _shape(module: nn.Module) -> tuple[int, int] | None:
    """Rank-2 shape without reading ``.weight`` on a converted module.

    ``QSTELinear.weight`` is a property that builds a differentiable matrix and
    marks a pending backward. Planning must never trigger that.
    """

    if isinstance(module, _CONVERTED):
        surface = module.surface
        return surface.rows, surface.columns
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    return int(weight.shape[0]), int(weight.shape[1])


def plan(
    model: nn.Module,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    *,
    types: tuple[type, ...] = DEFAULT_TYPES,
) -> list[Candidate]:
    """Report what :func:`convert` would do, without doing it."""

    seen: dict[int, str] = {}
    out: list[Candidate] = []
    for name, module in _walk(model, types):
        shape = _shape(module)
        if shape is None:
            continue
        key = id(module.surface) if isinstance(module, _CONVERTED) else id(module.weight)
        tied = seen.get(key)
        if tied is None:
            seen[key] = name
        out.append(
            Candidate(
                name=name,
                kind=type(module).__name__,
                rows=shape[0],
                columns=shape[1],
                selected=_matches(name, include)
                and not (exclude and _matches(name, exclude)),
                tied_to=tied,
            )
        )
    return out


def format_plan(candidates: Sequence[Candidate]) -> str:
    """A table a human can read before committing to a conversion."""

    if not candidates:
        return "no convertible layers found"
    width = max(len(c.name) for c in candidates)
    lines = [f"{'layer':<{width}}  {'shape':>17}  {'float':>10}  {'qste':>10}  sel"]
    float_total = qste_total = 0
    for c in candidates:
        marker = "yes" if c.selected else " - "
        if c.tied_to is not None:
            marker += f" (tied to {c.tied_to})"
        lines.append(
            f"{c.name:<{width}}  {c.rows:>8} x {c.columns:<6}  "
            f"{c.float_bytes / 1e6:>9.1f}M  {c.qste_bytes / 1e6:>9.1f}M  {marker}"
        )
        if c.selected and c.tied_to is None:
            float_total += c.float_bytes
            qste_total += c.qste_bytes
    ratio = float_total / qste_total if qste_total else 0.0
    lines.append(
        f"{'selected total':<{width}}  {'':>17}  {float_total / 1e6:>9.1f}M  "
        f"{qste_total / 1e6:>9.1f}M  {ratio:.2f}x"
    )
    return "\n".join(lines)


def convert(
    model: nn.Module,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    *,
    config: QSTEConfig | None = None,
    types: tuple[type, ...] = DEFAULT_TYPES,
    activations: bool | str = True,
    verbose: bool = False,
) -> nn.Module:
    """Replace the selected layers in ``model`` with QSTE equivalents.

    Args:
        model: Any ``nn.Module``. Mutated in place and also returned.
        include: Glob patterns over qualified module names. ``None`` selects
            every convertible layer.
        exclude: Glob patterns to skip, applied after ``include``.
        config: Numerics for the new surfaces. One config is shared by all of
            them, and is recorded on the model for the optimizer to find.
        types: Module classes eligible for conversion.
        activations: Also replace activation modules with the packed
            equivalents in :mod:`qste.nn`. On by default, because torch's ReLU
            saves its full-precision output for backward and that output is the
            next layer's input -- so it stays resident whatever the linear
            does, and the packed copy is pure saving.

            ``True`` replaces the exact ones (ReLU, ReLU6, Dropout -- identical
            forward, identical gradient, one bit per element) and the smooth
            ones (GELU, SiLU, Tanh, Sigmoid -- identical forward, retained
            derivative quantized to eight bits). ``"exact"`` replaces only the
            first group, which is numerically identical to torch. ``False``
            leaves every activation alone.
        verbose: Print the conversion table.

    Returns:
        The same model object.

    Tied weights become one shared :class:`~qste.surface.Surface`, so a tied
    embedding and readout stay tied and their gradients sum as autograd
    requires.

    Walking the module tree cannot reach activations a model calls functionally
    (``F.gelu(x)`` instead of a held ``nn.GELU``). Wrap the forward pass in
    :func:`qste.packed_activations` for those.
    """

    cfg = coerce_config(config)
    targets = [
        name
        for name, module in _walk(model, types)
        if _shape(module) is not None
        and _matches(name, include)
        and not (exclude and _matches(name, exclude))
    ]
    if not targets:
        raise ValueError(
            "convert() selected no layers; check `include` against "
            f"{[name for name, _ in _walk(model, types)][:8]}"
        )

    lookup = dict(model.named_modules())
    shared: dict[int, Surface] = {}
    converted: list[str] = list(getattr(model, "_qste_converted", ()))
    for name in targets:
        module = lookup[name]
        if isinstance(module, _CONVERTED):
            # Already binary; leave it and its surface alone.
            continue
        surface = shared.get(id(module.weight))
        if isinstance(module, nn.Embedding):
            replacement = QSTEEmbedding.from_embedding(module, cfg, surface)
        else:
            replacement = QSTELinear.from_linear(module, cfg, surface)
        shared[id(module.weight)] = replacement.surface
        _set_submodule(model, name, replacement)
        converted.append(name)

    if activations:
        converted.extend(_convert_activations(model, activations))

    model._qste_config = cfg
    model._qste_converted = tuple(converted)
    if verbose:
        print(format_plan(plan(model, include, exclude, types=types)))
    return model


def _convert_activations(model: nn.Module, mode: bool | str) -> list[str]:
    """Swap activation modules for the ones that retain bits, not floats.

    Returns the names replaced. Idempotent: the packed classes are not in the
    replacement table, so converting twice does nothing the second time.
    """

    from . import nn as packed

    if mode is True:
        table = packed.REPLACEMENTS
    elif mode == "exact":
        table = packed.EXACT_REPLACEMENTS
    elif mode == "all":
        table = packed.REPLACEMENTS
    else:
        raise ValueError("activations must be True, False, 'exact', or 'all'")

    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not name or type(module) not in table:
            continue
        substitute = packed.replace(module)
        if substitute is None:
            continue
        substitute.training = module.training
        _set_submodule(model, name, substitute)
        replaced.append(name)
    return replaced


def _set_submodule(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent = model
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def surfaces(model: nn.Module) -> list[Surface]:
    """Every distinct surface in the model, tied weights counted once."""

    seen: dict[int, Surface] = {}
    for module in model.modules():
        if isinstance(module, Surface):
            seen.setdefault(id(module), module)
    return list(seen.values())


def continuous_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Parameters the host's own optimizer should own.

    Coordinates are excluded because they carry ``requires_grad=False`` and
    are stepped by :class:`~qste.optim.QSTEOptimizer`. Row scales are included
    because they are ordinary float parameters and should be trained by
    whatever the host already uses.
    """

    for parameter in model.parameters():
        if parameter.requires_grad:
            yield parameter


def memory_report(model: nn.Module) -> dict[str, float]:
    """Matrix bytes before and after conversion, in megabytes."""

    total_float = total_qste = total_packed = 0
    for surface in surfaces(model):
        stats = surface.memory()
        total_float += stats["float_equivalent"]
        total_qste += stats["packed_sign"] + stats["coordinate"] + stats["log_scale"]
        total_packed += stats["packed_sign"] + stats["log_scale"]
    return {
        "float_mb": total_float / 1e6,
        "training_mb": total_qste / 1e6,
        "inference_mb": total_packed / 1e6,
        "training_ratio": total_float / total_qste if total_qste else 0.0,
        "inference_ratio": total_float / total_packed if total_packed else 0.0,
    }


def export_packed(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    """Drop the training coordinate, keeping only bits, scales, and biases.

    Training precision is inference precision here: the exported module runs
    the same signs the last training step wrote. Nothing is re-quantized, so
    there is no accuracy step at deployment.
    """

    import copy

    target = model if inplace else copy.deepcopy(model)
    for name, module in list(target.named_modules()):
        if isinstance(module, QSTELinear):
            surface = module.surface
            replacement = PackedLinear(
                surface.packed_sign.data.clone(),
                surface.scale.detach().clone(),
                surface.columns,
                None if module.bias is None else module.bias.detach().clone(),
            )
        elif isinstance(module, QSTEEmbedding):
            surface = module.surface
            replacement = PackedEmbedding(
                surface.packed_sign.data.clone(),
                surface.scale.detach().clone(),
                surface.columns,
            )
        else:
            continue
        _set_submodule(target, name, replacement)
    return target
