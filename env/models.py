"""OpenEnv models for LotteryElicitation environment."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from openenv.core.env_server.types import Action as _ActionBase
    from openenv.core.env_server.types import Observation as _ObservationBase
    from openenv.core.env_server.types import State as _StateBase
except ImportError:
    _ActionBase = BaseModel
    _ObservationBase = BaseModel
    _StateBase = BaseModel


class LotteryOutcome(BaseModel):
    """Single outcome in a lottery."""

    value: float = Field(..., description="Monetary outcome value.")
    probability: float = Field(..., ge=0.0, le=1.0, description="Outcome probability.")


class Lottery(BaseModel):
    """A lottery represented by 2-3 outcomes and probabilities."""

    outcomes: list[LotteryOutcome] = Field(..., min_length=2, max_length=3)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "Lottery":
        total = sum(outcome.probability for outcome in self.outcomes)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        return self


class LotteryElicitationAction(_ActionBase):
    """Agent action: two lotteries and optional final parameter estimate."""

    if _ActionBase is BaseModel:
        model_config = ConfigDict(extra="forbid")
        metadata: dict[str, Any] = Field(default_factory=dict)

    lottery_a: Lottery = Field(..., description="First lottery in the pair.")
    lottery_b: Lottery = Field(..., description="Second lottery in the pair.")
    theta_estimate: dict[str, float] | None = Field(
        default=None,
        description='Optional final estimate: {"gamma": float, "lambda": float}.',
    )
    terminate_early: bool = Field(
        default=False,
        description="Set True to terminate the episode early (requires theta_estimate).",
    )


class LotteryElicitationObservation(_ObservationBase):
    """Observation returned to the agent after reset/step."""

    if _ObservationBase is BaseModel:
        model_config = ConfigDict(extra="forbid")
        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: dict[str, Any] = Field(default_factory=dict)

    step_idx: int = Field(..., ge=0, description="Current step index (0-based).")
    steps_remaining: int = Field(..., ge=0, description="How many steps remain.")
    max_steps: int = Field(..., ge=1, description="Maximum steps per episode.")
    history: list[dict[str, Any]] = Field(default_factory=list)
    last_choice: str | None = Field(default=None, description='Most recent choice: "A" or "B".')
    gamma_range: tuple[float, float] = Field(..., description="Prior range for gamma.")
    lambda_range: tuple[float, float] = Field(..., description="Prior range for lambda.")
    min_outcome_value: float = Field(..., description="Minimum allowed lottery outcome value.")
    max_outcome_value: float = Field(..., description="Maximum allowed lottery outcome value.")


class LotteryElicitationState(_StateBase):
    """Internal environment state used for logging/debugging."""

    if _StateBase is BaseModel:
        model_config = ConfigDict(extra="allow")
        episode_id: str | None = Field(default=None)
        step_count: int = Field(default=0, ge=0)

    true_gamma: float = Field(..., description="Hidden ground-truth gamma.")
    true_lambda: float = Field(..., description="Hidden ground-truth lambda.")
    estimated_gamma: float | None = Field(default=None)
    estimated_lambda: float | None = Field(default=None)
    gamma_mse: float | None = Field(default=None, ge=0.0)
    lambda_mse: float | None = Field(default=None, ge=0.0)
    holt_laury_prediction_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    total_reward: float = Field(default=0.0)
