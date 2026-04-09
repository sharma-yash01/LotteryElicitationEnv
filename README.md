---
title: LotteryElicitationEnv
emoji: 🎖️
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Lottery Elicitation Environment

OpenEnv environment for **adaptive lottery preference elicitation**. A policy proposes pairs of lotteries (`lottery_a`, `lottery_b`); a simulated respondent chooses **A** or **B** according to a hidden **CPT-style** parameter pair \((\gamma, \lambda)\). The agent may submit a final **`theta_estimate`**; the environment returns a **verifiable terminal reward** (negative normalized MSE on \((\gamma,\lambda)\), plus Holt–Laury consistency and an efficiency bonus). Intermediate steps return **reward `0.0`** until the episode ends.

- **Transport:** WebSocket session to the FastAPI server (same pattern as other OpenEnv envs).
- **Training:** See sibling package [`LotteryElicitationPT`](../LotteryElicitationPT) for GRPO/TRL with `ENV_BASE_URL` pointing at this server.

## Requirements

- Python **3.11+**
- Dependencies: `openenv-core[core]`, `numpy`, `pydantic`, `fastapi`, `uvicorn` (see `pyproject.toml`).
- **Docker** (optional): for building/running the image locally or for `from_docker_image` in code.

## Server endpoints

| Path | Purpose |
|------|---------|
| `/health` | Liveness for load balancers and Docker `HEALTHCHECK` |
| `/docs` | OpenAPI (HTTP); useful for discovery |
| `/web` | Gradio-style web UI when deployed as an HF Space |
| WebSocket | Used by `EnvClient` / `LotteryElicitationEnvClient` for `reset` / `step` (persistent episode state per connection) |

Base URL examples:

- Local: `http://127.0.0.1:8000`
- Hugging Face Space: `https://<owner>-<space-name>.hf.space` (no trailing slash)

The Python client accepts `http://` or `https://`; OpenEnv converts it to the correct WebSocket URL.

---

## 1. Hugging Face Spaces (deploy + talk over the API)

### Deploy

From **this directory** (where `openenv.yaml` lives), with the OpenEnv CLI and a Hugging Face login:

```bash
cd LotteryElicitationEnv
openenv push
# or, e.g.:
openenv push --repo-id <your-username>/lottery-elicitation-env --private
```

After build, the Space page is at `https://huggingface.co/spaces/<owner>/<space-name>`. For **client code**, use the **direct Space app URL**:

```text
https://<owner>-<space-name>.hf.space
```

That value is the **`base_url`** you pass to the client (same as `ENV_BASE_URL` in `LotteryElicitationPT`).

### Call the environment from Python (HF or any hosted URL)

Run with `PYTHONPATH=.` set to this folder so `client` and `env` resolve (or run from a project that already adds this path):

```python
import asyncio

from client import LotteryElicitationEnvClient
from env.models import Lottery, LotteryElicitationAction, LotteryOutcome

BASE_URL = "https://your-username-your-space.hf.space"  # or http://127.0.0.1:8000


def make_action(*, final: bool) -> LotteryElicitationAction:
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
        terminate_early=final,
    )


async def main() -> None:
    client = LotteryElicitationEnvClient(base_url=BASE_URL)
    async with client:
        result = await client.reset(seed=42, curriculum_stage=1)
        obs = result.observation
        print("After reset:", obs.step_idx, obs.max_steps, obs.gamma_range)

        # Example: a few exploratory steps (reward 0.0), then early stop with estimate.
        for _ in range(2):
            result = await client.step(make_action(final=False))
            obs = result.observation
            print("Choice:", obs.last_choice, "reward:", result.reward, "done:", result.done)

        result = await client.step(make_action(final=True))
        print("Terminal reward:", result.reward, "metadata:", result.observation.metadata)


if __name__ == "__main__":
    asyncio.run(main())
```

**Synchronous** usage against an already-running server (same `base_url` as above):

```python
from client import LotteryElicitationEnvClient
from env.models import Lottery, LotteryElicitationAction, LotteryOutcome

# ... same make_action ...

with LotteryElicitationEnvClient(base_url="http://127.0.0.1:8000").sync() as env:
    r = env.reset(seed=42, curriculum_stage=1)
    r = env.step(make_action(final=False))
    r = env.step(make_action(final=True))
    print(r.reward, r.done)
```

`close()` on the async client stops the WebSocket session only. It does **not** tear down a Space you do not own.

---

## 2. Local Docker image (build, run, rewards)

### Build

The Dockerfile expects the **build context to be this environment directory** (`LotteryElicitationEnv/`), not the monorepo root:

```bash
cd LotteryElicitationEnv
docker build -t lottery-elicitation-env:latest -f server/Dockerfile .
```

### Run

```bash
docker run --rm -p 8000:8000 lottery-elicitation-env:latest
```

Check readiness:

```bash
curl -s http://127.0.0.1:8000/health
```

### Client: connect to the container

Use **`base_url="http://127.0.0.1:8000"`** with the same sync or async examples as in §1.

### Client: start container from Python (`from_docker_image`)

In **openenv-core 0.2.x**, `from_docker_image` is **async** and returns a connected client; use `asyncio`:

```python
import asyncio

from client import LotteryElicitationEnvClient


async def main() -> None:
    client = await LotteryElicitationEnvClient.from_docker_image("lottery-elicitation-env:latest")
    try:
        async with client:
            r = await client.reset(seed=0)
            # ... await client.step(...) ...
            print(r.observation, r.reward)
    finally:
        await client.close()  # disconnects and stops/removes the container when provider is Docker


if __name__ == "__main__":
    asyncio.run(main())
```

Requires a local Docker daemon and the image tag you built.

---

## Action, observation, reward, and config

### `LotteryElicitationAction`

- **`lottery_a` / `lottery_b`:** each a `Lottery` with **2–3** `LotteryOutcome(value, probability)`; probabilities must sum to **1.0** (within tolerance).
- **`theta_estimate`:** optional `{"gamma": float, "lambda": float}`; **required** when the episode terminates (final step or `terminate_early=True`).
- **`terminate_early`:** if `True`, ends the episode on that step (still requires valid `theta_estimate`).

### `LotteryElicitationObservation`

- **`step_idx`**, **`steps_remaining`**, **`max_steps`**, **`history`** (past pairs and choices), **`last_choice`** (`"A"` / `"B"`).
- **`gamma_range`**, **`lambda_range`**, **`min_outcome_value`**, **`max_outcome_value`** (constraints for valid lotteries).
- **`done`**, **`reward`**, **`metadata`** (errors, reward breakdown on terminal steps).

### Reward semantics

- **Non-terminal steps:** reward **`0.0`** (invalid actions can yield a penalty; see tests).
- **Terminal step:** reward from `env/reward.py`: weighted **negative normalized MSE** on \((\gamma,\lambda)\), **Holt–Laury** agreement, and **efficiency** vs `max_steps`.
- Episode ends when **`step_idx` reaches `max_steps`** or **`terminate_early`** with a valid estimate.

### `reset(..., curriculum_stage=...)`

Optional **`curriculum_stage`** (int) changes how latent \((\gamma,\lambda)\) is sampled (see `env/priors.py`). Default server config uses `EnvConfig` in `server/app.py` (`env/config.py`): **`max_steps`**, priors, outcome bounds, reward weights, and default **`curriculum_stage`**.

---

## Run the server without Docker (development)

From this directory:

```bash
# Installs: pip install -e ".[dev]" or uv sync
uvicorn server.app:app --host 0.0.0.0 --port 8000
# or: python -m server.app
```

Entry point: `server.app:app` (`create_app` + factory for **concurrent** WebSocket sessions, `max_concurrent_envs=64`).

---

## Tests

```bash
cd LotteryElicitationEnv
pytest tests/ -q
```

---

## Project layout

```text
LotteryElicitationEnv/
├── openenv.yaml           # OpenEnv manifest (HF / CLI)
├── pyproject.toml
├── client.py              # LotteryElicitationEnvClient (WebSocket)
├── models.py              # Re-exports action/observation/state for tooling
├── env/
│   ├── lottery_env.py     # Core environment
│   ├── models.py          # Lottery, actions, observations
│   ├── config.py          # EnvConfig
│   ├── respondent.py      # Simulated choices
│   ├── reward.py          # Terminal reward
│   └── priors.py          # Theta sampling + curriculum stages
├── server/
│   ├── app.py             # FastAPI + OpenEnv create_app
│   └── Dockerfile         # Production-style image
└── tests/
```

### Import note

`pyproject.toml` currently packages `env`, `server`, etc. The **client** lives at the repo root; for local runs use **`PYTHONPATH=.`** (as in `pytest` config) or import `client` / `env.models` from a parent project that adds this folder to the path.

### Legacy alias

`LotteryelicitationenvEnv` is kept as an alias on `client` for old scaffolds; prefer **`LotteryElicitationEnvClient`**.
