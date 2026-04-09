"""Core OpenEnv environment for lottery preference elicitation."""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from .config import EnvConfig
from .models import (
    Lottery,
    LotteryElicitationAction,
    LotteryElicitationObservation,
    LotteryElicitationState,
)
from .priors import PriorConfig, sample_theta
from .respondent import respondent_choice
from .reward import compute_episode_reward

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from abc import ABC, abstractmethod
    from typing import Generic, TypeVar

    ActT = TypeVar("ActT")
    ObsT = TypeVar("ObsT")
    StateT = TypeVar("StateT")

    class Environment(ABC, Generic[ActT, ObsT, StateT]):
        @abstractmethod
        def reset(self, seed=None, episode_id=None, **kwargs): ...

        @abstractmethod
        def step(self, action, timeout_s=None, **kwargs): ...

        @property
        @abstractmethod
        def state(self): ...


class LotteryElicitationEnvironment(
    Environment[LotteryElicitationAction, LotteryElicitationObservation, LotteryElicitationState]
):
    """Environment where an agent adaptively proposes lottery pairs."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: Optional[EnvConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self._config = config or EnvConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._prior_config = PriorConfig(
            gamma_low=self._config.gamma_range[0],
            gamma_high=self._config.gamma_range[1],
            lambda_low=self._config.lambda_range[0],
            lambda_high=self._config.lambda_range[1],
            distribution="uniform",
        )

        self._episode_id: Optional[str] = None
        self._step_idx: int = 0
        self._history: list[dict] = []
        self._done: bool = False

        self._true_gamma: float = 0.0
        self._true_lambda: float = 0.0

        self._estimated_gamma: float | None = None
        self._estimated_lambda: float | None = None
        self._gamma_mse: float | None = None
        self._lambda_mse: float | None = None
        self._holt_laury_accuracy: float | None = None
        self._total_reward: float = 0.0
        self._last_metadata: dict = {}

    def _build_observation(
        self,
        *,
        last_choice: str | None,
        reward: float | None,
        done: bool,
    ) -> LotteryElicitationObservation:
        return LotteryElicitationObservation(
            step_idx=self._step_idx,
            steps_remaining=max(0, self._config.max_steps - self._step_idx),
            max_steps=self._config.max_steps,
            history=list(self._history),
            last_choice=last_choice,
            gamma_range=self._config.gamma_range,
            lambda_range=self._config.lambda_range,
            min_outcome_value=self._config.min_outcome_value,
            max_outcome_value=self._config.max_outcome_value,
            done=done,
            reward=reward,
            metadata=dict(self._last_metadata),
        )

    def _validate_lottery_values(self, lottery: Lottery) -> None:
        for outcome in lottery.outcomes:
            if outcome.value < self._config.min_outcome_value:
                raise ValueError(
                    f"Outcome value {outcome.value} below min_outcome_value "
                    f"{self._config.min_outcome_value}"
                )
            if outcome.value > self._config.max_outcome_value:
                raise ValueError(
                    f"Outcome value {outcome.value} above max_outcome_value "
                    f"{self._config.max_outcome_value}"
                )

    def _validate_action(self, action: LotteryElicitationAction) -> None:
        self._validate_lottery_values(action.lottery_a)
        self._validate_lottery_values(action.lottery_b)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> LotteryElicitationObservation:
        curriculum_kw = kwargs.pop("curriculum_stage", None)
        kwargs.clear()
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        effective_stage = (
            int(curriculum_kw)
            if curriculum_kw is not None
            else int(self._config.curriculum_stage)
        )
        theta = sample_theta(
            config=self._prior_config,
            rng=self._rng,
            curriculum_stage=effective_stage,
        )

        self._episode_id = episode_id or str(uuid.uuid4())
        self._true_gamma = float(theta["gamma"])
        self._true_lambda = float(theta["lambda"])
        self._step_idx = 0
        self._history = []
        self._done = False

        self._estimated_gamma = None
        self._estimated_lambda = None
        self._gamma_mse = None
        self._lambda_mse = None
        self._holt_laury_accuracy = None
        self._total_reward = 0.0
        self._last_metadata = {}

        return self._build_observation(last_choice=None, reward=0.0, done=False)

    def step(
        self,
        action: LotteryElicitationAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> LotteryElicitationObservation:
        del timeout_s, kwargs
        if self._done:
            self._last_metadata = {"info": "Episode already done."}
            return self._build_observation(last_choice=None, reward=0.0, done=True)

        try:
            self._validate_action(action)
        except ValueError as exc:
            self._last_metadata = {"error": str(exc)}
            return self._build_observation(last_choice=None, reward=-1.0, done=False)

        choice = respondent_choice(
            lottery_a=action.lottery_a,
            lottery_b=action.lottery_b,
            gamma=self._true_gamma,
            lambda_=self._true_lambda,
            noise_std=self._config.noise_std,
            rng=self._rng,
        )

        self._history.append(
            {
                "lottery_a": action.lottery_a.model_dump(),
                "lottery_b": action.lottery_b.model_dump(),
                "choice": choice,
                "step": self._step_idx,
            }
        )
        self._step_idx += 1

        reward = 0.0
        self._last_metadata = {}
        is_final_step = self._step_idx >= self._config.max_steps
        if is_final_step or action.terminate_early:
            if action.theta_estimate is None:
                reward = -2.0
                self._last_metadata = {
                    "error": "theta_estimate is required when terminating the episode."
                }
            else:
                try:
                    reward, breakdown = compute_episode_reward(
                        theta_estimate=action.theta_estimate,
                        true_gamma=self._true_gamma,
                        true_lambda=self._true_lambda,
                        history=self._history,
                        config=self._config,
                        steps_taken=self._step_idx,
                    )
                    self._estimated_gamma = float(action.theta_estimate["gamma"])
                    self._estimated_lambda = float(action.theta_estimate["lambda"])
                    self._gamma_mse = breakdown["gamma_mse"]
                    self._lambda_mse = breakdown["lambda_mse"]
                    self._holt_laury_accuracy = breakdown["holt_laury_accuracy"]
                    self._last_metadata = {"reward_breakdown": breakdown}
                except Exception as exc:
                    reward = -2.0
                    self._last_metadata = {"error": f"Invalid theta_estimate: {exc}"}

            self._done = True

        self._total_reward += reward
        return self._build_observation(last_choice=choice, reward=reward, done=self._done)

    @property
    def state(self) -> LotteryElicitationState:
        return LotteryElicitationState(
            episode_id=self._episode_id,
            step_count=self._step_idx,
            true_gamma=self._true_gamma,
            true_lambda=self._true_lambda,
            estimated_gamma=self._estimated_gamma,
            estimated_lambda=self._estimated_lambda,
            gamma_mse=self._gamma_mse,
            lambda_mse=self._lambda_mse,
            holt_laury_prediction_accuracy=self._holt_laury_accuracy,
            total_reward=self._total_reward,
        )
