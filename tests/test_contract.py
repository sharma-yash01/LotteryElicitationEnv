"""Contract tests for LotteryElicitationEnvironment."""

from __future__ import annotations

import pytest

from env.config import EnvConfig
from env.lottery_env import LotteryElicitationEnvironment
from env.models import Lottery, LotteryElicitationAction, LotteryOutcome


def _simple_action(theta_estimate: dict[str, float] | None = None) -> LotteryElicitationAction:
    lottery_a = Lottery(
        outcomes=[
            LotteryOutcome(value=10.0, probability=0.5),
            LotteryOutcome(value=0.0, probability=0.5),
        ]
    )
    lottery_b = Lottery(
        outcomes=[
            LotteryOutcome(value=8.0, probability=0.5),
            LotteryOutcome(value=1.0, probability=0.5),
        ]
    )
    return LotteryElicitationAction(
        lottery_a=lottery_a,
        lottery_b=lottery_b,
        theta_estimate=theta_estimate,
    )


def test_reset_returns_valid_observation():
    env = LotteryElicitationEnvironment(config=EnvConfig(max_steps=3, seed=7))
    obs = env.reset()
    assert obs.step_idx == 0
    assert obs.steps_remaining == 3
    assert obs.max_steps == 3
    assert obs.history == []
    assert obs.done is False


def test_step_accepts_valid_action():
    env = LotteryElicitationEnvironment(config=EnvConfig(max_steps=2, seed=1))
    env.reset()
    obs = env.step(_simple_action())
    assert obs.last_choice in {"A", "B"}
    assert obs.step_idx == 1
    assert obs.done is False


def test_step_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        Lottery(
            outcomes=[
                LotteryOutcome(value=1.0, probability=0.7),
                LotteryOutcome(value=0.0, probability=0.7),
            ]
        )


def test_episode_terminates_after_k_steps():
    env = LotteryElicitationEnvironment(config=EnvConfig(max_steps=2, seed=12))
    env.reset()

    first = env.step(_simple_action())
    second = env.step(_simple_action(theta_estimate={"gamma": 1.0, "lambda": 2.25}))

    assert first.done is False
    assert second.done is True
    assert second.step_idx == 2


def test_final_step_requires_theta_estimate():
    env = LotteryElicitationEnvironment(config=EnvConfig(max_steps=1, seed=21))
    env.reset()
    final_obs = env.step(_simple_action(theta_estimate=None))
    assert final_obs.done is True
    assert final_obs.reward == -2.0


def test_deterministic_with_seed():
    cfg = EnvConfig(max_steps=3, seed=77, noise_std=0.0)
    env_1 = LotteryElicitationEnvironment(config=cfg)
    env_2 = LotteryElicitationEnvironment(config=cfg)

    env_1.reset(seed=77)
    env_2.reset(seed=77)

    choices_1 = []
    choices_2 = []
    for step_idx in range(3):
        is_last = step_idx == 2
        action = _simple_action(
            theta_estimate={"gamma": 0.8, "lambda": 2.0} if is_last else None
        )
        choices_1.append(env_1.step(action).last_choice)
        choices_2.append(env_2.step(action).last_choice)

    assert choices_1 == choices_2


def test_state_contains_ground_truth():
    env = LotteryElicitationEnvironment(config=EnvConfig(seed=5))
    env.reset()
    state = env.state

    assert state.true_gamma is not None
    assert state.true_lambda is not None
    assert 0.2 <= state.true_gamma <= 1.5
    assert 1.0 <= state.true_lambda <= 4.5
