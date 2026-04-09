"""Integration-style tests for LotteryElicitationEnvironment behavior."""

from __future__ import annotations

import pytest

from env.config import EnvConfig
from env.lottery_env import LotteryElicitationEnvironment
from env.models import Lottery, LotteryElicitationAction, LotteryOutcome


def _action(final: bool = False, terminate_early: bool = False) -> LotteryElicitationAction:
    return LotteryElicitationAction(
        lottery_a=Lottery(
            outcomes=[
                LotteryOutcome(value=10.0, probability=0.5),
                LotteryOutcome(value=0.0, probability=0.5),
            ]
        ),
        lottery_b=Lottery(
            outcomes=[
                LotteryOutcome(value=7.0, probability=0.5),
                LotteryOutcome(value=2.0, probability=0.5),
            ]
        ),
        theta_estimate={"gamma": 0.9, "lambda": 2.2} if final else None,
        terminate_early=terminate_early,
    )


def test_episode_runs_and_terminates():
    env = LotteryElicitationEnvironment(config=EnvConfig(max_steps=3, seed=42))
    env.reset()

    obs1 = env.step(_action(final=False))
    obs2 = env.step(_action(final=False))
    obs3 = env.step(_action(final=True))

    assert obs1.done is False
    assert obs2.done is False
    assert obs3.done is True
    assert obs3.step_idx == 3


def test_history_accumulates_each_step():
    env = LotteryElicitationEnvironment(config=EnvConfig(max_steps=2, seed=3))
    obs = env.reset()
    assert len(obs.history) == 0

    obs = env.step(_action(final=False))
    assert len(obs.history) == 1

    obs = env.step(_action(final=True))
    assert len(obs.history) == 2


def test_reward_only_computed_at_terminal_step():
    env = LotteryElicitationEnvironment(config=EnvConfig(max_steps=2, seed=17))
    env.reset()
    first = env.step(_action(final=False))
    second = env.step(_action(final=True))

    assert first.reward == pytest.approx(0.0)
    assert second.reward is not None
    assert second.done is True


def test_invalid_lottery_values_return_penalty():
    env = LotteryElicitationEnvironment(
        config=EnvConfig(max_steps=2, min_outcome_value=-5.0, max_outcome_value=5.0, seed=99)
    )
    env.reset()

    invalid_action = LotteryElicitationAction(
        lottery_a=Lottery(
            outcomes=[
                LotteryOutcome(value=100.0, probability=0.5),
                LotteryOutcome(value=0.0, probability=0.5),
            ]
        ),
        lottery_b=Lottery(
            outcomes=[
                LotteryOutcome(value=1.0, probability=0.5),
                LotteryOutcome(value=0.0, probability=0.5),
            ]
        ),
    )

    obs = env.step(invalid_action)
    assert obs.reward == -1.0
    assert obs.done is False


def test_early_termination_ends_episode_with_efficiency_bonus():
    env = LotteryElicitationEnvironment(config=EnvConfig(max_steps=4, seed=123))
    env.reset()

    obs = env.step(_action(final=True, terminate_early=True))

    assert obs.done is True
    assert obs.step_idx == 1
    assert obs.metadata["reward_breakdown"]["efficiency_bonus"] == pytest.approx(0.75)


def test_reset_curriculum_stage_kwarg_overrides_theta_sampling():
    """Per-reset curriculum_stage=1 fixes lambda at 2.25 even if config.stage is 2."""
    cfg = EnvConfig(max_steps=2, seed=0, curriculum_stage=2)
    env = LotteryElicitationEnvironment(config=cfg)
    env.reset(seed=0, curriculum_stage=1)
    assert env.state.true_lambda == pytest.approx(2.25)

    env2 = LotteryElicitationEnvironment(config=cfg)
    env2.reset(seed=0, curriculum_stage=2)
    assert env2.state.true_lambda != pytest.approx(2.25)
