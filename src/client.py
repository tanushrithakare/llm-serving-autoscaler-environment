"""
client.py — Async Docker client for the LLM Serving Autoscaler Environment.

Provides LLMAutoscalerEnv with the standard OpenEnv client interface:
  - from_docker_image(image_name) : start container, return connected client
  - reset(task)                   : POST /reset
  - step(action)                  : POST /step
  - close()                       : stop + remove container
"""

import asyncio
import socket
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import httpx

from src.models import LLMServeObs, LLMServeAction


# ---------------------------------------------------------------------------
# Step result wrapper
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result returned by reset() and step()."""
    observation: LLMServeObs
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Async environment client
# ---------------------------------------------------------------------------

class LLMAutoscalerEnv:
    """
    Async client that manages a Docker container running the
    LLM Serving Autoscaler server and communicates via HTTP.
    """

    def __init__(self, base_url: str, container_id: Optional[str] = None):
        self._base_url = base_url
        self._container_id = container_id
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def from_docker_image(cls, image_name: str) -> "LLMAutoscalerEnv":
        """Start a Docker container from *image_name* and return a connected client."""
        port = _find_free_port()

        result = subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:8000", image_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

        container_id = result.stdout.strip()
        base_url = f"http://localhost:{port}"
        env = cls(base_url=base_url, container_id=container_id)
        await env._wait_for_health()
        return env

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    async def reset(self, task: str = "easy") -> StepResult:
        """POST /reset?task=..."""
        resp = await self._client.post(f"/reset?task={task}")
        resp.raise_for_status()
        obs = LLMServeObs(**resp.json())
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    async def step(self, action: LLMServeAction) -> StepResult:
        """POST /step with JSON action body."""
        resp = await self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        data = resp.json()
        obs = LLMServeObs(**data["observation"])
        return StepResult(
            observation=obs,
            reward=data["reward"],
            done=data["done"],
            info=data.get("info", {}),
        )

    async def close(self) -> None:
        """Stop and remove the Docker container."""
        await self._client.aclose()
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
            subprocess.run(
                ["docker", "rm", self._container_id],
                capture_output=True,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _wait_for_health(self, timeout: int = 60) -> None:
        """Poll GET /health until the container is ready."""
        for _ in range(timeout * 2):
            try:
                resp = await self._client.get("/health")
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadError):
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Container not healthy after {timeout}s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
