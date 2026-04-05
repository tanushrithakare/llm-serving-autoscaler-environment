"""
app.py — FastAPI server exposing the LLM Serving Autoscaler Environment.

Endpoints
---------
POST /reset?task=easy|medium|hard  — start a new episode
POST /step                         — submit an action, get obs + reward
GET  /state                        — read current observation
GET  /health                       — liveness check
POST /grade                        — run full grader with baseline
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import sys
import os

# Ensure the root directory is on the path for Docker runtime
sys.path.append(os.getcwd())

from environment import LLMServeEnv
from models import LLMServeAction, LLMServeObs
from grader import LLMServeGrader
from baseline import BaselineAgent

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "LLM Serving Autoscaler Environment",
    description = "OpenEnv-compatible RL environment for GPU autoscaling.",
    version     = "1.0.0",
)

# Singleton environment (one session per server process)
_env     = LLMServeEnv()
_grader  = LLMServeGrader()
_baseline = BaselineAgent()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    observation: LLMServeObs
    reward:      float
    done:        bool
    info:        dict


class GradeRequest(BaseModel):
    task: str = "easy"


class GradeResponse(BaseModel):
    task:  str
    score: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=LLMServeObs, summary="Reset the environment")
def reset(task: str = Query("easy", pattern="^(easy|medium|hard)$")):
    """
    Start a fresh episode.

    - **task**: difficulty level — `easy`, `medium`, or `hard`
    """
    obs = _env.reset(task=task)
    return obs


@app.post("/step", response_model=StepResponse, summary="Submit an action")
def step(action: LLMServeAction):
    """
    Advance one timestep with the given action.

    Returns the new observation, reward, done flag, and debug info.
    """
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return StepResponse(
        observation = obs,
        reward      = reward,
        done        = done,
        info        = info,
    )


@app.get("/state", response_model=LLMServeObs, summary="Read current observation")
def state():
    """Return the current environment observation without advancing the episode."""
    try:
        return _env.state()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/health", summary="Liveness check")
def health():
    """Simple health probe."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/grade", response_model=GradeResponse, summary="Grade the baseline agent")
def grade(request: GradeRequest):
    """
    Run a full episode with the built-in baseline agent and return its score.

    Useful for verifying environment correctness and getting a reference score.
    """
    if request.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task must be easy, medium, or hard")

    score = _grader.grade(_baseline, task=request.task)
    return GradeResponse(task=request.task, score=score)


def main():
    """Entry point for the server (used by [project.scripts])."""
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
