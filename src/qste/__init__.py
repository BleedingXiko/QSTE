"""QSTE -- true-binary training for any PyTorch model.

Training precision is inference precision. There is no float shadow weight and
no post-training quantization step, so the model you deploy is bit-for-bit the
model that trained.

Adoption is two lines and a subset of layer names::

    import qste

    qste.convert(model, include=["blocks.*.mlp_fc1", "blocks.*.mlp_fc2"])
    optimizer = torch.optim.AdamW(model.parameters())   # unchanged
    coordinates = qste.QSTEOptimizer(model)

    loss.backward()
    optimizer.step();  coordinates.step()
    optimizer.zero_grad();  coordinates.zero_grad()

Everything not named in ``include`` keeps training in float. Nothing else in
the host framework changes.

Choosing that subset is the decision that matters -- see the README's "What to
convert". Wrap the forward in :func:`packed_activations` too, so the retained
activations shrink along with the weights.

Beta. Speed is measured on CPU and one GPU; the README states the scope.
"""

from .config import QSTEConfig, coerce_config
from .convert import (
    Candidate,
    continuous_parameters,
    convert,
    export_packed,
    format_plan,
    memory_report,
    plan,
    surfaces,
)
from . import nn
from .functional import exact_evidence, qste_embedding, qste_linear, qste_weight
from .nn import packed_activations
from .modules import (
    PackedEmbedding,
    PackedLinear,
    QSTEEmbedding,
    QSTELinear,
)
from .capture import (
    decisions,
    invalidate,
    retain,
    retained_stats,
    undecided,
    warmup,
)
from .optim import QSTEOptimizer
from .surface import Surface

__version__ = "0.1.0"

__all__ = [
    "Candidate",
    "PackedEmbedding",
    "PackedLinear",
    "QSTEConfig",
    "QSTEEmbedding",
    "QSTELinear",
    "QSTEOptimizer",
    "Surface",
    "coerce_config",
    "device_profile",
    "continuous_parameters",
    "convert",
    "decisions",
    "exact_evidence",
    "nn",
    "packed_activations",
    "export_packed",
    "format_plan",
    "memory_report",
    "plan",
    "qste_embedding",
    "qste_linear",
    "qste_weight",
    "invalidate",
    "retain",
    "retained_stats",
    "surfaces",
    "undecided",
    "warmup",
    "__version__",
]


def kernel_status() -> dict:
    """Whether the native kernels built, and why not if they did not."""

    from . import kernels

    return kernels.status()


def device_profile(device=None) -> str:
    """The execution parameters QSTE derived for a device, as one line.

    Nothing in QSTE is tuned for a particular card. Scratch size, reduction
    width, and the dtype the evidence product runs in are all derived from what
    the device reports and from timing it once. This prints what that came out
    to, so a number measured on unfamiliar hardware is interpretable.
    """

    import torch

    from . import kernels

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return kernels.profile(device).describe()
