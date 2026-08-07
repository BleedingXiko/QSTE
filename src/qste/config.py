"""Configuration for QSTE surfaces and the coordinate optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QSTEConfig:
    """The training recipe, in one immutable object.

    These are the knobs that change how a binary model learns. How the
    activation is retained for backward is not among them -- that is QSTE's
    mechanism, not a setting, and there is one of it.

    Attributes:
        coordinate_clip: Latent range the INT8 coordinate spans at init.
        coordinate_lr: Step size in coordinate units per optimizer step.
        beta1: First-moment decay for the coordinate optimizer.
        beta2: Second-moment decay for the factored row/column statistics.
        update_rms_clip: Global RMS ceiling applied to a normalized update.
        momentum_block_size: Elements sharing one FP16 momentum scale.
        seed: Base seed for the stochastic rounding of coordinate targets.
        scale_min: Lower clamp on the learned per-row scale.
        scale_max: Upper clamp on the learned per-row scale.
        evidence_accum_dtype: Dtype of the deferred evidence buffer. Only
            allocated when a surface is used more than once per backward or
            when updates are deferred for gradient accumulation.
    """

    coordinate_clip: float = 1.5
    coordinate_lr: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.99
    update_rms_clip: float = 2.0
    momentum_block_size: int = 256
    seed: int = 1
    scale_min: float = 1e-6
    scale_max: float = 4.0
    evidence_accum_dtype: str = "fp16"

    def __post_init__(self) -> None:
        if self.coordinate_clip <= 0 or self.coordinate_lr <= 0:
            raise ValueError("coordinate_clip and coordinate_lr must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if self.update_rms_clip <= 0 or self.momentum_block_size <= 0:
            raise ValueError("update_rms_clip and momentum_block_size must be positive")
        if not 0 < self.scale_min < self.scale_max:
            raise ValueError("scale bounds must satisfy 0 < scale_min < scale_max")
        if self.evidence_accum_dtype not in ("fp16", "fp32"):
            raise ValueError("evidence_accum_dtype must be 'fp16' or 'fp32'")

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def coerce_config(cfg) -> QSTEConfig:
    """Accept ``None``, a mapping, or a config and return a config."""

    if cfg is None:
        return QSTEConfig()
    if isinstance(cfg, QSTEConfig):
        return cfg
    return QSTEConfig(**dict(cfg))
