"""Random lottery baseline for LotteryElicitationEnv."""

from __future__ import annotations

import numpy as np

try:
    from env.models import (
        Lottery,
        LotteryElicitationAction,
        LotteryElicitationObservation,
        LotteryOutcome,
    )
except ImportError:
    from ..env.models import (
        Lottery,
        LotteryElicitationAction,
        LotteryElicitationObservation,
        LotteryOutcome,
    )


class RandomLotteryBaseline:
    """Randomly proposes lottery pairs; final estimate uses prior midpoint."""

    def __init__(
        self,
        min_outcome_value: float = -50.0,
        max_outcome_value: float = 100.0,
    ):
        self.min_outcome_value = min_outcome_value
        self.max_outcome_value = max_outcome_value

    def _sample_lottery(self, rng: np.random.Generator) -> Lottery:
        p = float(rng.uniform(0.1, 0.9))
        v1 = float(rng.uniform(self.min_outcome_value, self.max_outcome_value))
        v2 = float(rng.uniform(self.min_outcome_value, self.max_outcome_value))
        return Lottery(
            outcomes=[
                LotteryOutcome(value=v1, probability=p),
                LotteryOutcome(value=v2, probability=1.0 - p),
            ]
        )

    def select_action(
        self,
        obs: LotteryElicitationObservation,
        rng: np.random.Generator,
    ) -> LotteryElicitationAction:
        lottery_a = self._sample_lottery(rng)
        lottery_b = self._sample_lottery(rng)

        theta_estimate = None
        if obs.steps_remaining == 1:
            theta_estimate = {
                "gamma": 0.5 * (obs.gamma_range[0] + obs.gamma_range[1]),
                "lambda": 0.5 * (obs.lambda_range[0] + obs.lambda_range[1]),
            }

        return LotteryElicitationAction(
            lottery_a=lottery_a,
            lottery_b=lottery_b,
            theta_estimate=theta_estimate,
        )
