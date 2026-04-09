"""Baseline evaluation harness for LotteryElicitationEnv."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

try:
    from baselines.holt_laury_fixed import HoltLauryFixedBaseline
    from baselines.random_lottery import RandomLotteryBaseline
    from env.config import EnvConfig
    from env.lottery_env import LotteryElicitationEnvironment
except ImportError:
    from ..baselines.holt_laury_fixed import HoltLauryFixedBaseline
    from ..baselines.random_lottery import RandomLotteryBaseline
    from ..env.config import EnvConfig
    from ..env.lottery_env import LotteryElicitationEnvironment


def _summary(values: list[float | None]) -> dict[str, float]:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}


def _baseline_action(baseline: Any, obs: Any, rng: np.random.Generator):
    if isinstance(baseline, RandomLotteryBaseline):
        return baseline.select_action(obs, rng)
    return baseline.select_action(obs)


def evaluate_baseline(
    baseline: Any,
    env: LotteryElicitationEnvironment,
    n_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run episodes and return aggregate baseline metrics."""
    rng = np.random.default_rng(seed)
    gamma_mse_values: list[float | None] = []
    lambda_mse_values: list[float | None] = []
    hl_acc_values: list[float | None] = []
    episode_rewards: list[float] = []

    for _ in range(n_episodes):
        episode_seed = int(rng.integers(0, 2**32 - 1))
        obs = env.reset(seed=episode_seed)

        while not obs.done:
            action = _baseline_action(baseline, obs, rng)
            obs = env.step(action)

        state = env.state
        gamma_mse_values.append(state.gamma_mse)
        lambda_mse_values.append(state.lambda_mse)
        hl_acc_values.append(state.holt_laury_prediction_accuracy)
        episode_rewards.append(float(state.total_reward))

    return {
        "n_episodes": n_episodes,
        "gamma_mse": _summary(gamma_mse_values),
        "lambda_mse": _summary(lambda_mse_values),
        "holt_laury_accuracy": _summary(hl_acc_values),
        "episode_rewards": {
            "mean": float(np.mean(episode_rewards)),
            "std": float(np.std(episode_rewards)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["random", "holt-laury"], default="random")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()

    config = EnvConfig(max_steps=args.max_steps, seed=args.seed)
    env = LotteryElicitationEnvironment(config=config)

    if args.baseline == "random":
        baseline = RandomLotteryBaseline(
            min_outcome_value=config.min_outcome_value,
            max_outcome_value=config.max_outcome_value,
        )
    else:
        baseline = HoltLauryFixedBaseline(
            gamma_range=config.gamma_range,
            lambda_range=config.lambda_range,
        )

    results = evaluate_baseline(
        baseline=baseline,
        env=env,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
