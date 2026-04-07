"""
environment.py — LLM Serving Autoscaler RL Environment (OpenEnv-compatible).

Implements LLMServeEnv with:
  - reset(task)  : initialise episode for easy / medium / hard
  - step(action) : advance one timestep, return (obs, reward, done, info)
  - state()      : return current observation without advancing
"""

import math
import numpy as np
import os
import sys
# Ensure the root directory is on the path for local imports
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.append(_root)

from models import LLMServeObs, LLMServeAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPISODE_LENGTH   = 1000
MAX_GPUS         = 100
MAX_QUEUE        = 20000        # hard spike needs room to build up
MAX_LATENCY      = 500.0        # ms — used for reward/grader normalisation
MIN_GPUS         = 1            # always keep at least one GPU running


# ---------------------------------------------------------------------------
# Traffic generators (deterministic via seeded RNG)
# ---------------------------------------------------------------------------

def _traffic_easy(step: int, rng: np.random.Generator) -> float:
    """
    Stable 150 req/s — always comfortably below 4-GPU capacity (256).
    Agent barely needs to scale. Queue stays near zero.
    """
    return float(np.clip(150.0 + rng.normal(0, 8.0), 80, 220))


def _traffic_medium(step: int, rng: np.random.Generator) -> float:
    """
    Sinusoidal 100–2000 req/s — peaks exceed initial capacity,
    agent must scale aggressively, queue builds transiently.
    """
    base = 1050.0 + 950.0 * math.sin(2 * math.pi * step / 200)
    return float(np.clip(base + rng.normal(0, 40.0), 80, 2200))


def _traffic_hard(step: int, rng: np.random.Generator) -> float:
    """
    Stable → massive spike (200–500) at 15000 req/s (exceeds max GPU
    capacity of 12800) → slow elevated cooldown (6000→200).
    Queue is irrecoverable during spike even at 100 GPUs.
    """
    if step < 200:
        base = 150.0                                      # same as easy
    elif step < 500:
        base = 20000.0 + rng.normal(0, 300.0)            # 20K rps - exceeds max capacity
    else:
        # cooldown stays well above easy-level for 500 steps
        progress = (step - 500) / 500
        base = 6000.0 - 5800.0 * progress                # 6000 → 200
    return float(np.clip(base + rng.normal(0, 15.0), 80, 21000))


_TRAFFIC_FN = {
    "easy":   _traffic_easy,
    "medium": _traffic_medium,
    "hard":   _traffic_hard,
}


# ---------------------------------------------------------------------------
# OpenEnv base class (lightweight stub — no external dependency)
# ---------------------------------------------------------------------------

class OpenEnv:
    """Minimal OpenEnv interface contract."""

    def reset(self, task: str):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def state(self):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class LLMServeEnv(OpenEnv):
    """
    LLM Serving Autoscaler Environment.

    Episode length : 1000 steps
    Tasks          : easy | medium | hard
    Seed           : 42 (deterministic)
    """

    def __init__(self):
        self._rng: np.random.Generator = None
        self._task: str = "easy"
        self._step_count: int = 0

        # Mutable state
        self._active_gpus: int = 4
        self._queue_length: int = 0
        self._incoming_rate: float = 0.0
        self._avg_latency: float = 0.0
        self._batch_size: int = 64
        self._cache_load: float = 0.1
        self._spot_gpu_ratio: float = 0.0

        # Episode-level accumulators (used by grader)
        self._total_service_ratio: float = 0.0  # served / incoming each step
        self._total_latency: float = 0.0
        self._total_cost: float = 0.0
        self._steps_done: int = 0

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task: str = "easy") -> LLMServeObs:
        """Reset the environment for a new episode."""
        if task not in _TRAFFIC_FN:
            raise ValueError(f"Unknown task '{task}'. Choose from: easy, medium, hard.")

        self._task       = task
        self._rng        = np.random.default_rng(42)
        self._step_count = 0

        # Initial conditions
        self._active_gpus     = 4
        self._queue_length    = 0
        self._incoming_rate   = 20.0
        self._avg_latency     = 50.0
        self._batch_size      = 64
        self._cache_load      = 0.1
        self._spot_gpu_ratio  = 0.0

        # Accumulators
        self._total_service_ratio = 0.0
        self._total_latency       = 0.0
        self._total_cost          = 0.0
        self._steps_done          = 0

        return self.state()

    def step(self, action: LLMServeAction) -> tuple:
        """
        Advance one timestep.

        Returns:
            (obs, reward, done, info)
        """
        if self._rng is None:
            raise RuntimeError("Call reset() before step().")

        # --- 1. Apply action (with robust clipping for safety/exploits) ---
        new_gpus = int(np.clip(
            self._active_gpus + action.scale,
            MIN_GPUS, MAX_GPUS
        ))
        self._active_gpus    = new_gpus
        self._batch_size     = int(np.clip(action.batch_size, 32, 128))
        self._spot_gpu_ratio = float(np.clip(action.spot_allocation, 0.0, 1.0))

        # --- 1b. Spot GPU preemption (safe, deterministic) ---
        # Cloud provider may reclaim spot instances under load.
        # Only affects medium/hard tasks; uses seeded RNG for reproducibility.
        _preempted = 0
        if self._spot_gpu_ratio > 0 and self._task != "easy":
            if self._rng.random() < 0.03:  # ~3 % chance per step
                lost = max(1, int(self._active_gpus * 0.05))
                lost = min(lost, 2)          # cap at 2 GPUs
                self._active_gpus = max(MIN_GPUS, self._active_gpus - lost)
                _preempted = lost

        # --- 2. Generate incoming traffic ---
        traffic_fn          = _TRAFFIC_FN[self._task]
        self._incoming_rate = traffic_fn(self._step_count, self._rng)

        # --- 3. Compute capacity and throughput ---
        # Each GPU can handle batch_size requests per step; scale linearly
        capacity          = self._active_gpus * self._batch_size
        served            = float(min(capacity, self._queue_length + self._incoming_rate))
        throughput        = served

        # --- 4. Update queue ---
        arrived            = self._incoming_rate
        self._queue_length = int(np.clip(
            self._queue_length + arrived - served,
            0, MAX_QUEUE * 2          # allow slight overflow for penalty
        ))

        # --- 5. Compute average latency (ms) ---
        # Base latency grows with queue depth, shrinks with more capacity
        queue_pressure     = self._queue_length / max(capacity, 1)
        self._avg_latency  = float(np.clip(
            20.0 + queue_pressure * 400.0 + self._rng.normal(0, 2.0),
            0.0, 2000.0
        ))

        # --- 6. Compute cost (normalised 0–1) ---
        # Spot GPUs are cheaper (0.3× cost); regular GPUs = 1.0×
        spot_gpus    = self._active_gpus * self._spot_gpu_ratio
        regular_gpus = self._active_gpus - spot_gpus
        gpu_cost     = (regular_gpus * 1.0 + spot_gpus * 0.3) / MAX_GPUS

        # --- 7. Update cache load ---
        # Cache fills with high queue, drains when serving catches up
        cache_delta       = 0.05 if self._queue_length > 50 else -0.02
        self._cache_load  = float(np.clip(self._cache_load + cache_delta, 0.0, 1.0))

        # --- 8. Accumulate episode stats ---
        # Service ratio: fraction of incoming requests actually served (0–1)
        service_ratio = float(np.clip(
            served / max(self._incoming_rate, 1.0), 0.0, 1.0
        ))
        self._total_service_ratio += service_ratio
        self._total_latency       += self._avg_latency
        self._total_cost          += gpu_cost
        self._steps_done          += 1

        # --- 9. Compute reward ---
        latency_score    = 1.0 - min(self._avg_latency / MAX_LATENCY, 1.0)
        throughput_score = float(np.clip(service_ratio, 0.0, 1.0))
        gpu_penalty      = self._active_gpus / MAX_GPUS
        queue_penalty    = min(self._queue_length / MAX_QUEUE, 1.0)
        batch_penalty    = self._batch_size / 128.0

        # Stronger queue penalty and latency focus to keep baseline scores grounded
        reward = (
              0.6 * latency_score        # increased from 0.5
            + 0.2 * throughput_score     # reduced from 0.3
            - 0.15 * gpu_penalty         # kept same
            - 0.30 * queue_penalty       # increased penalty from 0.25
            - 0.05 * batch_penalty       # lightly penalise large batch usage
        )
        reward = float(np.clip(reward, -1.0, 1.0))

        # --- 10. Advance step counter ---
        self._step_count += 1
        done = self._step_count >= EPISODE_LENGTH

        # Detect incidents for "Incident Report" logging
        incidents = []
        if self._avg_latency > 200:
            incidents.append("SLA_VIOLATION_LATENCY")
        if self._queue_length > MAX_QUEUE:
            incidents.append("QUEUE_OVERFLOW")
        if _preempted > 0:
            incidents.append("SPOT_PREEMPTION_EVENT")

        info = {
            "step":            self._step_count,
            "throughput":      throughput,
            "gpu_cost":        gpu_cost,
            "capacity":        capacity,
            "task":            self._task,
            "preempted_gpus":  _preempted,
            "incidents":       incidents,
        }

        return self.state(), reward, done, info

    def state(self) -> LLMServeObs:
        """Return the current observation without advancing the episode."""
        return LLMServeObs(
            active_gpus      = self._active_gpus,
            queue_length     = self._queue_length,
            incoming_rate    = self._incoming_rate,
            avg_latency      = self._avg_latency,
            batch_size       = self._batch_size,
            cache_load       = self._cache_load,
            spot_gpu_ratio   = self._spot_gpu_ratio,
        )

    # ------------------------------------------------------------------
    # Helpers (used by grader)
    # ------------------------------------------------------------------

    def episode_stats(self) -> dict:
        """Aggregate stats for the completed episode."""
        n = max(self._steps_done, 1)
        return {
            "mean_latency":        self._total_latency        / n,
            "mean_service_ratio":  self._total_service_ratio  / n,  # 0–1, 1=perfect
            "mean_cost":           self._total_cost           / n,
            "steps":               self._steps_done,
            "task":                self._task,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def render(self, reward: float = 0.0) -> str:
        """
        Return a compact ASCII dashboard of the current environment state.

        Parameters
        ----------
        reward : float
            The reward from the most recent step (environment does not
            store it, so the caller passes it in).

        Returns
        -------
        str
            Multi-line dashboard string (also printed to stdout).
        """
        # --- queue bar (30-char max) ---
        q_frac   = min(self._queue_length / MAX_QUEUE, 1.0)
        q_filled = int(q_frac * 30)
        q_bar    = "#" * q_filled + "-" * (30 - q_filled)

        # --- GPU utilisation bar (20-char max) ---
        g_frac   = self._active_gpus / MAX_GPUS
        g_filled = int(g_frac * 20)
        g_bar    = "#" * g_filled + "-" * (20 - g_filled)

        # --- latency indicator ---
        if self._avg_latency < 50:
            lat_icon = "GOOD"
        elif self._avg_latency < 200:
            lat_icon = "WARN"
        else:
            lat_icon = "CRIT"

        # --- ASCII box-drawing characters (safe for all systems) ---
        H  = "-" * 52
        TL, TR, ML, MR, BL, BR, V = "+", "+", "+", "+", "+", "+", "|"

        lines = [
            f"{TL}{H}{TR}",
            f"{V}  Step: {self._step_count:<6d} {V} Task: {self._task:<8s} {V} Reward: {reward:+.4f}  {V}",
            f"{ML}{H}{MR}",
            f"{V}  GPUs:  [{g_bar}]  {self._active_gpus:>3d}/{MAX_GPUS}        {V}",
            f"{V}  Spot:  {self._spot_gpu_ratio:.0%}                                       {V}",
            f"{V}  Queue: [{q_bar}]  {self._queue_length:>6d}  {V}",
            f"{V}  Rate:  {self._incoming_rate:>10.1f} req/s                    {V}",
            f"{V}  Lat:   {self._avg_latency:>8.1f} ms  {lat_icon}                      {V}",
            f"{V}  Cache: {self._cache_load:.0%}   Batch: {self._batch_size}                     {V}",
            f"{BL}{H}{BR}",
        ]
        dashboard = "\n".join(lines)
        print(dashboard)
        return dashboard
