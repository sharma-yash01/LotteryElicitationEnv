"""Fixed Holt-Laury baseline with grid-search fitting."""

from __future__ import annotations

import numpy as np

try:
    from env.holt_laury import HOLT_LAURY_PAIRS
    from env.models import Lottery, LotteryElicitationAction, LotteryElicitationObservation
    from env.respondent import respondent_choice
except ImportError:
    from ..env.holt_laury import HOLT_LAURY_PAIRS
    from ..env.models import Lottery, LotteryElicitationAction, LotteryElicitationObservation
    from ..env.respondent import respondent_choice


def _lottery_from_dict(payload: dict) -> Lottery:
    outcomes = payload.get("outcomes", [])
    return Lottery(outcomes=outcomes)


class HoltLauryFixedBaseline:
    """Presents fixed H-L pairs, then estimates parameters by grid search."""

    def __init__(
        self,
        gamma_range: tuple[float, float] = (0.2, 1.5),
        lambda_range: tuple[float, float] = (1.0, 4.5),
        grid_step: float = 0.01,
    ):
        self.gamma_range = gamma_range
        self.lambda_range = lambda_range
        self.grid_step = grid_step

    def select_action(
        self,
        obs: LotteryElicitationObservation,
        step_idx: int | None = None,
    ) -> LotteryElicitationAction:
        idx = obs.step_idx if step_idx is None else step_idx
        pair_idx = min(idx, len(HOLT_LAURY_PAIRS) - 1)
        lottery_a, lottery_b = HOLT_LAURY_PAIRS[pair_idx]

        theta_estimate = None
        if obs.steps_remaining == 1:
            theta_estimate = self._fit_from_choices(obs.history)

        return LotteryElicitationAction(
            lottery_a=lottery_a,
            lottery_b=lottery_b,
            theta_estimate=theta_estimate,
        )

    def _fit_from_choices(self, history: list[dict]) -> dict[str, float]:
        if not history:
            return {
                "gamma": 0.5 * (self.gamma_range[0] + self.gamma_range[1]),
                "lambda": 0.5 * (self.lambda_range[0] + self.lambda_range[1]),
            }

        gamma_grid = np.arange(self.gamma_range[0], self.gamma_range[1] + 1e-9, self.grid_step)
        lambda_grid = np.arange(
            self.lambda_range[0], self.lambda_range[1] + 1e-9, self.grid_step
        )

        gamma_mid = 0.5 * (self.gamma_range[0] + self.gamma_range[1])
        lambda_mid = 0.5 * (self.lambda_range[0] + self.lambda_range[1])

        best_score = -1
        best_dist = float("inf")
        best_theta = {"gamma": gamma_mid, "lambda": lambda_mid}

        for gamma in gamma_grid:
            for lambda_ in lambda_grid:
                score = 0
                for row in history:
                    lottery_a = _lottery_from_dict(row["lottery_a"])
                    lottery_b = _lottery_from_dict(row["lottery_b"])
                    predicted = respondent_choice(
                        lottery_a=lottery_a,
                        lottery_b=lottery_b,
                        gamma=float(gamma),
                        lambda_=float(lambda_),
                        noise_std=0.0,
                    )
                    score += int(predicted == row["choice"])

                dist = (float(gamma) - gamma_mid) ** 2 + (float(lambda_) - lambda_mid) ** 2
                if score > best_score or (score == best_score and dist < best_dist):
                    best_score = score
                    best_dist = dist
                    best_theta = {"gamma": float(gamma), "lambda": float(lambda_)}

        return best_theta
