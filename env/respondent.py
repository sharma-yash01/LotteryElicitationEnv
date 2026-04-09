"""Prospect-theory respondent used by LotteryElicitationEnv."""

from __future__ import annotations

import math

import numpy as np

from .models import Lottery


def prospect_theory_value(x: float, gamma: float, lambda_: float) -> float:
    """Compute value function v(x) under two-parameter prospect theory."""
    if x >= 0.0:
        return x**gamma
    return -lambda_ * ((-x) ** gamma)


def expected_utility(lottery: Lottery, gamma: float, lambda_: float) -> float:
    """Compute expected utility using prospect-theory value over outcomes."""
    return sum(
        outcome.probability * prospect_theory_value(outcome.value, gamma, lambda_)
        for outcome in lottery.outcomes
    )


def respondent_choice(
    lottery_a: Lottery,
    lottery_b: Lottery,
    gamma: float,
    lambda_: float,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> str:
    """Return A/B choice from utility difference with optional logistic noise."""
    eu_a = expected_utility(lottery_a, gamma=gamma, lambda_=lambda_)
    eu_b = expected_utility(lottery_b, gamma=gamma, lambda_=lambda_)
    diff = eu_a - eu_b

    # Deterministic argmax with documented tie-break: choose A on exact tie.
    if noise_std <= 0.0:
        return "A" if diff >= 0.0 else "B"

    # Fechner/random utility interpretation: logistic probability with scale set
    # by noise_std (smaller noise -> sharper deterministic behavior).
    scale = max(noise_std, 1e-8)
    p_choose_a = 1.0 / (1.0 + math.exp(-diff / scale))

    local_rng = rng if rng is not None else np.random.default_rng()
    return "A" if local_rng.random() < p_choose_a else "B"
