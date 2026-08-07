"""The random stream the coordinate optimizer rounds against.

A coordinate is an integer and the update that moves it is a fraction of one.
Rounding to nearest would drop every update below half a step, so QSTE rounds
*stochastically*: a coordinate at 12.3 goes to 13 three times in ten and stays
at 12 the rest, preserving the expected value without ever storing anything
between the integers. A biased stream is therefore a biased optimizer.

The draw is a pure function of ``(seed, step, flat index)`` and touches no RNG
state, which buys two properties a generator cannot offer:

* **Every DDP rank draws the same numbers** with no communication and no
  seeding ritual, since no rank advances a shared cursor. A generator would
  need lockstep stepping across ranks whose surfaces are sharded differently,
  and would diverge the first time one took a different number of draws.

* **A captured CUDA graph still draws fresh numbers**, because the step is read
  from device memory during replay instead of baked in at capture.

Three backends implement this -- Python here, Triton for CUDA, C++ for the
native CPU path -- and they must agree bit for bit or checkpoints stop being
portable between them. The obvious 32-bit spelling does not agree: right shift
is arithmetic on a signed type and logical on an unsigned one, and overflow
wraps on a GPU while being undefined to a C++ optimizer. Every implementation
therefore mixes in a wider register and masks back to 32 bits after each step.
"""

from __future__ import annotations

MASK = 0xFFFFFFFF

# Both constants stay below 2**31 because Triton types a bare integer literal
# as int32, and a kernel containing 0x9E3779B9 (the golden-ratio constant this
# originally used) fails to compile. `scramble` does the avalanching, so mixing
# quality is independent of which odd constant is chosen -- all three backends
# choosing the same one is the only requirement.
SALT = 0x2545F491
MULTIPLIER = 0x27D4EB2D


def scramble(value: int) -> int:
    """The 32-bit avalanche, in plain Python. The reference for the other two."""

    value &= MASK
    value = ((value ^ 61) ^ (value >> 16)) & MASK
    value = (value + (value << 3)) & MASK
    value = (value ^ (value >> 4)) & MASK
    value = (value * MULTIPLIER) & MASK
    return (value ^ (value >> 15)) & MASK


def seed_hash(seed: int) -> int:
    """The part of the stream that depends only on the run, not the step.

    Split out because the CUDA kernel takes it as a compile-time constant: the
    seed cannot change during a run, and folding it in on the host saves a hash
    per element on the device.
    """

    return scramble(int(seed) & MASK)


def step_hash(step: int) -> int:
    """The part that changes every step. Read from device memory under capture."""

    return scramble((int(step) & MASK) ^ SALT)


def uniform(seed: int, step: int, index: int) -> float:
    """The draw for one element, as the backends compute it.

    Used only by the tests, and defined here so the three backends are checked
    against one spelling of the reference.
    """

    import struct

    raw = scramble(index ^ seed_hash(seed) ^ step_hash(step))
    # float(raw) rounds to fp32 first and is then scaled by an exact power of
    # two, which is what every backend does; doing the division in double would
    # round differently in the last bit.
    single = struct.unpack("f", struct.pack("f", float(raw)))[0]
    return single * (1.0 / 4294967296.0)
