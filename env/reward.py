"""Reward decomposition for LotteryElicitationEnv."""

from __future__ import annotations

from .config import EnvConfig
from .holt_laury import holt_laury_accuracy, predict_holt_laury_choices


def _safe_norm_sq(low: float, high: float) -> float:
    width = high - low
    return width * width if width > 0.0 else 1.0


def compute_episode_reward(
    theta_estimate: dict[str, float],
    true_gamma: float,
    true_lambda: float,
    history: list[dict],
    config: EnvConfig,
    steps_taken: int,
) -> tuple[float, dict[str, float]]:
    """Compute final episode reward and detailed component breakdown."""
    del history  # Reserved for future information-gain terms.

    est_gamma = float(theta_estimate["gamma"])
    est_lambda = float(theta_estimate["lambda"])

    gamma_mse = (est_gamma - true_gamma) ** 2
    lambda_mse = (est_lambda - true_lambda) ** 2

    gamma_norm = _safe_norm_sq(*config.gamma_range)
    lambda_norm = _safe_norm_sq(*config.lambda_range)

    mse_component = -((gamma_mse / gamma_norm) + (lambda_mse / lambda_norm))

    predicted_choices = predict_holt_laury_choices(est_gamma, est_lambda)
    true_choices = predict_holt_laury_choices(true_gamma, true_lambda)
    hl_accuracy = holt_laury_accuracy(predicted_choices, true_choices)

    steps_saved = max(0, config.max_steps - steps_taken)
    efficiency_bonus = steps_saved / max(1, config.max_steps)

    total_reward = (
        config.mse_weight * mse_component
        + config.holt_laury_weight * hl_accuracy
        + config.efficiency_weight * efficiency_bonus
    )

    breakdown = {
        "gamma_mse": gamma_mse,
        "lambda_mse": lambda_mse,
        "mse_component": mse_component,
        "holt_laury_accuracy": hl_accuracy,
        "efficiency_bonus": efficiency_bonus,
        "total_reward": total_reward,
    }
    return total_reward, breakdown
