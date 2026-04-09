"""Environment configuration for LotteryElicitationEnv."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    """Configurable parameters for lottery elicitation episodes."""

    max_steps: int = 10
    gamma_range: tuple[float, float] = (0.2, 1.5)
    lambda_range: tuple[float, float] = (1.0, 4.5)
    noise_std: float = 0.0
    seed: Optional[int] = None

    # Reward weights.
    mse_weight: float = 1.0
    holt_laury_weight: float = 0.5
    efficiency_weight: float = 0.1

    # Action bounds.
    max_outcome_value: float = 100.0
    min_outcome_value: float = -50.0

    # 1 = gamma only (lambda fixed), 2 = gamma + lambda.
    curriculum_stage: int = 1
