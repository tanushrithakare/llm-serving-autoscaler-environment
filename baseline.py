"""
baseline.py — Rule-based baseline agent for the LLM Serving Autoscaler Environment.

Strategy:
  - Scale up  (+1) when queue > 50  or latency > 200 ms
  - Scale down (-1) when queue < 10 and latency < 100 ms and active_gpus > 4
  - Hold       ( 0) otherwise
  - Use larger batch when queue is growing
  - Allocate spot GPUs proportional to queue level (cost saving)
"""

from models import LLMServeObs, LLMServeAction


class BaselineAgent:
    """
    Simple deterministic rule-based agent.

    Provides a reproducible reference score with no learning.
    """

    def __call__(self, obs: LLMServeObs) -> LLMServeAction:
        return self.act(obs)

    def act(self, obs: LLMServeObs) -> LLMServeAction:
        # --- Scaling decision ---
        if obs.queue_length > 50 or obs.avg_latency > 200:
            scale = 1                           # need more GPUs
        elif obs.queue_length < 10 and obs.avg_latency < 100 and obs.active_gpus > 4:
            scale = -1                          # over-provisioned, save cost
        else:
            scale = 0

        # --- Batch size ---
        # Larger batch when queue is deep to process more per pass
        if obs.queue_length > 100:
            batch_size = 128
        elif obs.queue_length > 30:
            batch_size = 96
        else:
            batch_size = 64

        # --- Spot allocation ---
        # Use more spot instances when system is under light load (safe to risk preemption)
        if obs.queue_length < 20 and obs.avg_latency < 150:
            spot_allocation = 0.5               # safe to use spot
        elif obs.queue_length > 80:
            spot_allocation = 0.1               # spike — prefer stable GPUs
        else:
            spot_allocation = 0.3

        return LLMServeAction(
            scale           = scale,
            batch_size      = batch_size,
            spot_allocation = spot_allocation,
        )
