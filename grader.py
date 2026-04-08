"""
grader.py — Deterministic grader for the LLM Serving Autoscaler Environment.

Produces a score in [0.0, 1.0] from episode statistics.

Scoring breakdown:
  - 40 % latency efficiency   (lower avg latency → higher score)
  - 40 % service ratio        (served/incoming per step; 1.0 = perfect)
  - 20 % cost efficiency      (lower GPU cost → higher score)
"""

import numpy as np  # type: ignore
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from environment import LLMServeEnv, MAX_LATENCY  # type: ignore # noqa: E402

# ---------------------------------------------------------------------------
# Reference bounds (used for normalisation)
# ---------------------------------------------------------------------------

# Latency: anything above MAX_LATENCY (500 ms) is treated as worst-case
_MAX_MEAN_LATENCY = MAX_LATENCY    # imported from environment (500.0 ms)
# Service ratio is already normalised [0, 1] — no extra scale needed
# Cost is already normalised [0, 1] by the environment


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class LLMServeGrader:
    """
    Evaluates an agent across a full episode on a given task.

    Usage
    -----
    grader = LLMServeGrader()
    score  = grader.grade(agent_fn, task="hard")
    """

    def grade(self, agent_fn, task: str = "easy") -> float:
        """
        Run a full episode and return a score in [0.0, 1.0].

        Parameters
        ----------
        agent_fn : callable
            Function that accepts LLMServeObs and returns LLMServeAction.
        task : str
            One of "easy", "medium", "hard".

        Returns
        -------
        float
            Score between 0.0 and 1.0.
        """
        env  = LLMServeEnv()
        obs  = env.reset(task)
        done = False

        while not done:
            action          = agent_fn(obs)
            obs, _, done, _ = env.step(action)

        stats = env.episode_stats()
        return self._compute_score(stats)

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _compute_score(self, stats: dict) -> float:
        """
        Convert episode stats into a [0.0, 1.0] score.

        Weights
        -------
        40 % latency       — lower mean latency is better
        40 % service ratio — fraction of incoming requests served (already 0–1)
        20 % cost          — lower GPU cost is better
        """
        mean_latency       = stats["mean_latency"]
        mean_service_ratio = stats["mean_service_ratio"]
        mean_cost          = stats["mean_cost"]

        # Latency score: 1.0 = perfect (0 ms), 0.0 = at or above tightened MAX_LATENCY/2
        latency_score = 1.0 - float(np.clip(mean_latency / (_MAX_MEAN_LATENCY / 2.0), 0.0, 1.0))

        # Service ratio score: directly usable (0–1, 1 = all demand was met)
        throughput_score = float(np.clip(mean_service_ratio, 0.0, 1.0))

        # Cost score: 1.0 = zero cost, 0.0 = fully utilised expensive GPUs
        cost_score = 1.0 - float(np.clip(mean_cost, 0.0, 1.0))

        # Weighted aggregation: latency and throughput equally valued,
        # cost efficiency is a secondary objective (20 %).
        score = (
            0.4 * latency_score      # SLA: fast response time
            + 0.4 * throughput_score  # Capacity: serve all incoming requests
            + 0.2 * cost_score        # Efficiency: minimise GPU spend
        )

        # Clamp to open interval (0, 1) — scores of exactly 0.0 or 1.0 are
        # reserved to indicate system errors (not achievable in normal operation).
        return float(np.clip(score, 0.01, 0.99))

    def grade_all_tasks(self, agent_fn) -> dict:
        """
        Grade the agent on all three tasks.

        Returns
        -------
        dict with keys: easy, medium, hard, overall
        """
        results = {}
        for task in ("easy", "medium", "hard"):
            results[task] = self.grade(agent_fn, task=task)

        results["overall"] = float(np.mean([
            results["easy"],
            results["medium"],
            results["hard"],
        ]))

        return results
