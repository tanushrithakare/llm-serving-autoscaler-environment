"""
Inference Script — LLM Serving Autoscaler Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

from openai import OpenAI

from client import LLMAutoscalerEnv
from src.models import LLMServeAction, LLMServeObs

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("IMAGE_NAME", "llm-env")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "llm-serving-autoscaler"
MAX_STEPS = 1000
LLM_CALL_INTERVAL = 50          # call LLM every N steps (cost-efficient)
TEMPERATURE = 0.3
MAX_TOKENS = 150
SUCCESS_THRESHOLD = 0.3          # score >= this → success

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert GPU autoscaler for a large-scale LLM serving cluster.
You observe the cluster state and decide scaling actions each timestep.

GOALS (in priority order):
1. Keep average latency below 100 ms
2. Serve all incoming requests (minimize queue buildup)
3. Minimize GPU cost

OBSERVATION FIELDS:
- active_gpus   : GPUs currently running (1-100)
- queue_length  : pending unserved requests
- incoming_rate : requests/second arriving now
- avg_latency   : current response latency in ms
- batch_size    : current batch size per GPU forward pass
- cache_load    : KV-cache utilisation (0.0-1.0)
- spot_gpu_ratio: fraction of GPUs that are spot (preemptible)

ACTION (respond as JSON only, no explanation):
- scale          : -1 (remove GPU), 0 (hold), or +1 (add GPU)
- batch_size     : integer 32 to 128
- spot_allocation: float 0.0 to 1.0

STRATEGY:
- Scale UP   when queue > 50 or latency > 200 ms
- Scale DOWN when queue < 10 and latency < 100 ms and GPUs > 4
- Use large batch_size (96-128) when queue is deep
- Use spot GPUs (0.3-0.5) when load is light; avoid spot (0.0-0.1) during spikes
- Default safe action: {"scale": 0, "batch_size": 64, "spot_allocation": 0.3}

Respond with ONLY valid JSON, nothing else.""")


# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation → prompt
# ---------------------------------------------------------------------------

def format_observation(obs: LLMServeObs, step: int) -> str:
    """Build user prompt from the current observation."""
    return textwrap.dedent(f"""\
Step {step} of {MAX_STEPS}.
Current cluster state:
  active_gpus   = {obs.active_gpus}
  queue_length  = {obs.queue_length}
  incoming_rate = {obs.incoming_rate:.1f} req/s
  avg_latency   = {obs.avg_latency:.1f} ms
  batch_size    = {obs.batch_size}
  cache_load    = {obs.cache_load:.2f}
  spot_gpu_ratio= {obs.spot_gpu_ratio:.2f}

Decide the next action as JSON.""")


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

FALLBACK_ACTION = LLMServeAction(scale=0, batch_size=64, spot_allocation=0.3)


def parse_action(text: str) -> LLMServeAction:
    """Parse LLM response text into a validated action (with fallback)."""
    try:
        # Strip markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)
        return LLMServeAction(
            scale=int(data.get("scale", 0)),
            batch_size=int(data.get("batch_size", 64)),
            spot_allocation=float(data.get("spot_allocation", 0.3)),
        )
    except Exception:
        return FALLBACK_ACTION


def get_llm_action(
    client: OpenAI,
    obs: LLMServeObs,
    step: int,
) -> LLMServeAction:
    """Call the LLM to decide the next autoscaling action."""
    user_prompt = format_observation(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return FALLBACK_ACTION


def action_str(action: LLMServeAction) -> str:
    """Format action for the [STEP] log line."""
    return (
        f"scale={action.scale},"
        f"batch={action.batch_size},"
        f"spot={action.spot_allocation:.2f}"
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_task(
    llm_client: OpenAI,
    env: LLMAutoscalerEnv,
    task: str,
) -> None:
    """Run a single task (full episode) and emit mandatory logs."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    current_action = FALLBACK_ACTION

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Call LLM at intervals (every N steps) for cost efficiency
            if (step - 1) % LLM_CALL_INTERVAL == 0:
                current_action = get_llm_action(llm_client, obs, step)

            # Step environment
            result = await env.step(current_action)
            obs = result.observation
            reward = result.reward
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str(current_action),
                reward=reward,
                done=done,
                error=None,
            )

            if done:
                break

        # Score: normalise mean reward from [-1, 1] to [0, 1]
        if rewards:
            mean_r = sum(rewards) / len(rewards)
            score = max(0.0, min(1.0, (mean_r + 1.0) / 2.0))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        log_step(
            step=steps_taken + 1,
            action=action_str(current_action),
            reward=0.0,
            done=True,
            error=str(exc),
        )

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await LLMAutoscalerEnv.from_docker_image(IMAGE_NAME)

    try:
        for task in TASKS:
            await run_task(llm_client, env, task)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
