"""Unit tests for prospect-theory respondent arithmetic."""

from __future__ import annotations

from env.holt_laury import predict_holt_laury_choices
from env.models import Lottery, LotteryOutcome
from env.respondent import respondent_choice


def test_risk_neutral_prefers_higher_expected_value():
    lottery_a = Lottery(
        outcomes=[
            LotteryOutcome(value=10.0, probability=0.5),
            LotteryOutcome(value=0.0, probability=0.5),
        ]
    )  # EV = 5
    lottery_b = Lottery(
        outcomes=[
            LotteryOutcome(value=7.0, probability=0.5),
            LotteryOutcome(value=1.0, probability=0.5),
        ]
    )  # EV = 4
    assert respondent_choice(lottery_a, lottery_b, gamma=1.0, lambda_=1.0) == "A"


def test_very_risk_averse_prefers_safer_lottery():
    safe = Lottery(
        outcomes=[
            LotteryOutcome(value=5.0, probability=1.0),
            LotteryOutcome(value=5.0, probability=0.0),
        ]
    )
    risky = Lottery(
        outcomes=[
            LotteryOutcome(value=20.0, probability=0.25),
            LotteryOutcome(value=0.0, probability=0.75),
        ]
    )
    assert respondent_choice(safe, risky, gamma=0.3, lambda_=1.0) == "A"


def test_loss_averse_avoids_negative_outcomes():
    no_loss = Lottery(
        outcomes=[
            LotteryOutcome(value=4.0, probability=1.0),
            LotteryOutcome(value=4.0, probability=0.0),
        ]
    )
    has_loss = Lottery(
        outcomes=[
            LotteryOutcome(value=9.0, probability=0.6),
            LotteryOutcome(value=-6.0, probability=0.4),
        ]
    )
    assert respondent_choice(no_loss, has_loss, gamma=0.9, lambda_=3.0) == "A"


def test_holt_laury_predictions_have_consistent_structure():
    choices = predict_holt_laury_choices(gamma=1.0, lambda_=1.0)
    assert len(choices) == 10
    assert all(choice in {"A", "B"} for choice in choices)
    # For risk-neutral utility under standard H-L setup:
    assert choices[0] == "A"
    assert choices[-1] == "B"
