"""Tests for reward decomposition logic."""

from __future__ import annotations

import pytest

from env.config import EnvConfig
from env.reward import compute_episode_reward


def test_perfect_estimate_has_high_reward():
    config = EnvConfig()
    reward, breakdown = compute_episode_reward(
        theta_estimate={"gamma": 0.8, "lambda": 2.2},
        true_gamma=0.8,
        true_lambda=2.2,
        history=[],
        config=config,
        steps_taken=config.max_steps,
    )
    assert breakdown["gamma_mse"] == pytest.approx(0.0)
    assert breakdown["lambda_mse"] == pytest.approx(0.0)
    assert breakdown["holt_laury_accuracy"] == pytest.approx(1.0)
    assert reward >= 0.5


def test_worse_estimate_gets_lower_reward_than_perfect():
    config = EnvConfig()
    perfect_reward, _ = compute_episode_reward(
        theta_estimate={"gamma": 0.9, "lambda": 2.4},
        true_gamma=0.9,
        true_lambda=2.4,
        history=[],
        config=config,
        steps_taken=config.max_steps,
    )
    imperfect_reward, _ = compute_episode_reward(
        theta_estimate={"gamma": 0.2, "lambda": 4.5},
        true_gamma=0.9,
        true_lambda=2.4,
        history=[],
        config=config,
        steps_taken=config.max_steps,
    )
    assert imperfect_reward < perfect_reward


def test_reward_breakdown_contains_all_components():
    config = EnvConfig()
    _, breakdown = compute_episode_reward(
        theta_estimate={"gamma": 1.0, "lambda": 2.0},
        true_gamma=0.8,
        true_lambda=2.2,
        history=[],
        config=config,
        steps_taken=config.max_steps,
    )
    assert "mse_component" in breakdown
    assert "holt_laury_accuracy" in breakdown
    assert "efficiency_bonus" in breakdown
    assert "total_reward" in breakdown


def test_early_termination_gets_efficiency_bonus():
    config = EnvConfig(max_steps=10)
    reward_early, breakdown_early = compute_episode_reward(
        theta_estimate={"gamma": 0.8, "lambda": 2.2},
        true_gamma=0.8,
        true_lambda=2.2,
        history=[],
        config=config,
        steps_taken=5,
    )
    reward_full, breakdown_full = compute_episode_reward(
        theta_estimate={"gamma": 0.8, "lambda": 2.2},
        true_gamma=0.8,
        true_lambda=2.2,
        history=[],
        config=config,
        steps_taken=10,
    )
    assert breakdown_early["efficiency_bonus"] == pytest.approx(0.5)
    assert breakdown_full["efficiency_bonus"] == pytest.approx(0.0)
    assert reward_early > reward_full
