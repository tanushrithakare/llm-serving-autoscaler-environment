"""
models.py — Pydantic V2 data models for the LLM Serving Autoscaler Environment.

Defines:
  - LLMServeObs    : observation emitted by the environment each step
  - LLMServeAction : action submitted by the agent each step
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class LLMServeObs(BaseModel):
    """Observation returned by the environment after each step / reset."""

    # Number of GPUs currently active (provisioned and running)
    active_gpus: int = Field(..., ge=0, le=100, description="Number of active GPUs (0–100)")

    # Requests waiting to be processed
    queue_length: int = Field(..., ge=0, description="Pending requests in the queue")

    # Requests arriving per second in this step
    incoming_rate: float = Field(..., ge=0.0, description="Incoming request rate (req/s)")

    # Mean end-to-end latency of served requests this step (ms)
    avg_latency: float = Field(..., ge=0.0, description="Average latency in milliseconds")

    # Number of requests packed into a single GPU forward pass
    batch_size: int = Field(..., ge=1, description="Current batch size per forward pass")

    # KV-cache utilisation fraction across active GPUs
    cache_load: float = Field(..., ge=0.0, le=1.0, description="Cache utilisation (0.0–1.0)")

    # Fraction of active GPUs that are spot/preemptible instances
    spot_gpu_ratio: float = Field(..., ge=0.0, le=1.0, description="Spot GPU fraction (0.0–1.0)")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class LLMServeAction(BaseModel):
    """Action submitted by the agent at each step."""

    # GPU scaling decision: -1 = scale down, 0 = hold, +1 = scale up
    scale: int = Field(..., ge=-1, le=1, description="Scale direction: -1, 0, or +1")

    # Target batch size for this step
    batch_size: int = Field(..., ge=32, le=128, description="Batch size per forward pass (32–128)")

    # Fraction of requested GPUs to provision as spot instances
    spot_allocation: float = Field(
        ..., ge=0.0, le=1.0, description="Spot GPU allocation fraction (0.0–1.0)"
    )
