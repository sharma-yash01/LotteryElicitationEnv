"""FastAPI app exposing LotteryElicitationEnv via OpenEnv create_app."""

try:
    from env.config import EnvConfig
    from env.lottery_env import LotteryElicitationEnvironment
    from models import LotteryElicitationAction, LotteryElicitationObservation
except ImportError:
    from ..env.config import EnvConfig
    from ..env.lottery_env import LotteryElicitationEnvironment
    from ..models import LotteryElicitationAction, LotteryElicitationObservation

try:
    from openenv.core.env_server import create_app
except ImportError:
    create_app = None  # type: ignore


def _env_factory():
    """Create a fresh environment instance per WebSocket session."""
    return LotteryElicitationEnvironment(config=EnvConfig())


if create_app is not None:
    app = create_app(
        _env_factory,
        LotteryElicitationAction,
        LotteryElicitationObservation,
        env_name="lottery-elicitation-env",
        max_concurrent_envs=64,
    )
else:
    from fastapi import FastAPI

    app = FastAPI(title="lottery-elicitation-env")
    app.get("/health")(lambda: {"status": "ok"})


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for local execution."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
