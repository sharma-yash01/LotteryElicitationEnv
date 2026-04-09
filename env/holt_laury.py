"""Holt-Laury battery definitions and helpers."""

from __future__ import annotations

from .models import Lottery, LotteryOutcome
from .respondent import respondent_choice


def _hl_pair(p_high: float) -> tuple[Lottery, Lottery]:
    """Create one standard Holt-Laury pair for a given high-outcome probability."""
    p_low = 1.0 - p_high
    safe = Lottery(
        outcomes=[
            LotteryOutcome(value=2.00, probability=p_high),
            LotteryOutcome(value=1.60, probability=p_low),
        ]
    )
    risky = Lottery(
        outcomes=[
            LotteryOutcome(value=3.85, probability=p_high),
            LotteryOutcome(value=0.10, probability=p_low),
        ]
    )
    return safe, risky


HOLT_LAURY_PAIRS: list[tuple[Lottery, Lottery]] = [
    _hl_pair(0.1),
    _hl_pair(0.2),
    _hl_pair(0.3),
    _hl_pair(0.4),
    _hl_pair(0.5),
    _hl_pair(0.6),
    _hl_pair(0.7),
    _hl_pair(0.8),
    _hl_pair(0.9),
    _hl_pair(1.0),
]


def predict_holt_laury_choices(gamma: float, lambda_: float) -> list[str]:
    """Predict choices over all 10 Holt-Laury lottery pairs."""
    return [
        respondent_choice(
            lottery_a=lottery_a,
            lottery_b=lottery_b,
            gamma=gamma,
            lambda_=lambda_,
            noise_std=0.0,
        )
        for lottery_a, lottery_b in HOLT_LAURY_PAIRS
    ]


def holt_laury_accuracy(predicted_choices: list[str], true_choices: list[str]) -> float:
    """Compute fraction of matching Holt-Laury choices."""
    if len(predicted_choices) != len(true_choices):
        raise ValueError("Predicted and true choice vectors must have equal length.")
    if not predicted_choices:
        return 0.0
    correct = sum(p == t for p, t in zip(predicted_choices, true_choices))
    return correct / len(predicted_choices)
