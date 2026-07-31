"""How much of a memory survives after a given age.

One curve, used by both the forgetting scorer and the retrieval reranker. They
previously disagreed: the scorer used an exponential half-life, the reranker a
hyperbolic ``1 / (1 + age * 0.1)``, so the same memory aged at two different rates
depending on which subsystem was asking.

Both also ignored ``MemoryRecord.decay_rate``, which the write path populates per
memory precisely to separate the stable from the ephemeral (0.01 stable, 0.1 medium,
0.5 ephemeral). Reading it is the point of writing it.
"""

import math

# Applied when a record carries no rate of its own; matches the column default.
DEFAULT_DECAY_RATE = 0.01

# Ceiling on the per-record rate. The write-path prompt asks for 0.01-0.5, but the
# value arrives from an LLM and nothing downstream validates it — an unclamped 50.0
# would zero out a memory's retention within hours.
MAX_DECAY_RATE = 1.0


def retention(age_days: float, decay_rate: float | None = None) -> float:
    """Fraction of a memory's strength remaining after ``age_days``.

    Exponential decay, the standard Ebbinghaus form: 1.0 at age zero, asymptotically
    approaching zero. A rate of 0.01 retains ~70% after a year; 0.5 retains ~1% after
    a week.
    """
    if age_days <= 0:
        return 1.0
    rate = DEFAULT_DECAY_RATE if decay_rate is None else decay_rate
    rate = max(0.0, min(MAX_DECAY_RATE, rate))
    return math.exp(-rate * age_days)


def frequency_score(access_count: int, log_base: float = 10.0) -> float:
    """Retrieval-practice term: how much use has strengthened this memory.

    Log-compressed and capped at 1.0, so a runaway access count cannot dominate a
    ranking — the testing effect is real but it is not unbounded.
    """
    if access_count <= 0:
        return 0.0
    return min(math.log(1 + access_count, log_base), 1.0)
