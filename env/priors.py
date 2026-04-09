"""Prior distributions for sampling latent preference parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PriorConfig:
    """Configuration for gamma/lambda sampling."""

    gamma_low: float = 0.2
    gamma_high: float = 1.5
    lambda_low: float = 1.0
    lambda_high: float = 4.5
    distribution: str = "uniform"  # uniform | truncated_normal | beta


def _sample_range(
    rng: np.random.Generator,
    low: float,
    high: float,
    distribution: str,
) -> float:
    if distribution == "uniform":
        return float(rng.uniform(low, high))

    if distribution == "truncated_normal":
        mean = 0.5 * (low + high)
        std = (high - low) / 6.0
        return float(np.clip(rng.normal(mean, std), low, high))

    if distribution == "beta":
        x = float(rng.beta(2.0, 2.0))
        return low + x * (high - low)

    raise ValueError(f"Unsupported prior distribution: {distribution}")


def sample_theta(
    config: PriorConfig,
    rng: np.random.Generator,
    curriculum_stage: int = 2,
) -> dict[str, float]:
    """Sample latent theta = {gamma, lambda} from configured prior."""
    gamma = _sample_range(
        rng=rng,
        low=config.gamma_low,
        high=config.gamma_high,
        distribution=config.distribution,
    )

    if curriculum_stage <= 1:
        lambda_fixed = 2.25
        lambda_val = float(np.clip(lambda_fixed, config.lambda_low, config.lambda_high))
    else:
        lambda_val = _sample_range(
            rng=rng,
            low=config.lambda_low,
            high=config.lambda_high,
            distribution=config.distribution,
        )

    return {"gamma": gamma, "lambda": lambda_val}
