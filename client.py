"""Typed OpenEnv client for LotteryElicitationEnv."""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .env.models import (
        LotteryElicitationAction,
        LotteryElicitationObservation,
        LotteryElicitationState,
    )
except ImportError:
    from env.models import (
        LotteryElicitationAction,
        LotteryElicitationObservation,
        LotteryElicitationState,
    )


class LotteryElicitationEnvClient(
    EnvClient[
        LotteryElicitationAction,
        LotteryElicitationObservation,
        LotteryElicitationState,
    ]
):
    """WebSocket client for interacting with LotteryElicitationEnv."""

    def _step_payload(self, action: LotteryElicitationAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "lottery_a": action.lottery_a.model_dump(),
            "lottery_b": action.lottery_b.model_dump(),
        }
        if action.theta_estimate is not None:
            payload["theta_estimate"] = action.theta_estimate
        return payload

    def _parse_result(self, payload: Dict[str, Any]):
        obs_data = payload.get("observation")
        if not isinstance(obs_data, dict):
            obs_data = payload if isinstance(payload, dict) else {}

        done = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward"))

        observation = LotteryElicitationObservation(
            step_idx=obs_data.get("step_idx", 0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            max_steps=obs_data.get("max_steps", 1),
            history=obs_data.get("history", []),
            last_choice=obs_data.get("last_choice"),
            gamma_range=tuple(obs_data.get("gamma_range", (0.2, 1.5))),
            lambda_range=tuple(obs_data.get("lambda_range", (1.0, 4.5))),
            min_outcome_value=float(obs_data.get("min_outcome_value", -50.0)),
            max_outcome_value=float(obs_data.get("max_outcome_value", 100.0)),
            done=done,
            reward=reward,
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]):
        state_data = payload.get("state")
        if not isinstance(state_data, dict):
            state_data = payload if isinstance(payload, dict) else {}

        return LotteryElicitationState(
            episode_id=state_data.get("episode_id"),
            step_count=state_data.get("step_count", 0),
            true_gamma=state_data.get("true_gamma", 0.0),
            true_lambda=state_data.get("true_lambda", 0.0),
            estimated_gamma=state_data.get("estimated_gamma"),
            estimated_lambda=state_data.get("estimated_lambda"),
            gamma_mse=state_data.get("gamma_mse"),
            lambda_mse=state_data.get("lambda_mse"),
            holt_laury_prediction_accuracy=state_data.get("holt_laury_prediction_accuracy"),
            total_reward=state_data.get("total_reward", 0.0),
        )


# Backward-compatible alias from scaffold naming.
LotteryelicitationenvEnv = LotteryElicitationEnvClient
