"""
app.py — FastAPI server (Live Ops Center v1.1 - Final Recovery)

Endpoints
---------
POST /reset?task=easy|medium|hard  — start a new episode
POST /step                         — submit an action, get obs + reward
GET  /state                        — read current observation
GET  /health                       — liveness check
POST /grade                        — run full grader with baseline
"""

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
import os
import sys

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
    version     = "1.0.1",
)

# Singleton environment (one session per server process)
_env     = LLMServeEnv()
_grader  = LLMServeGrader()
_baseline = BaselineAgent()

# Track the last action for UI visibility
_last_action = None

# A flag to prevent concurrent demo runs
_is_demo_running = False

# ---------------------------------------------------------------------------
# UI Dashboard (HTML/JS)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Cluster Ops Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;600&display=swap" rel="stylesheet">
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: 'Outfit', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; }
        .stat-value { font-size: 2.5rem; font-weight: 600; color: #58a6ff; }
        .decision-card { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
        .decision-label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
        .decision-value { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; }
        .pulse { animation: pulse-animation 2s infinite; }
        @keyframes pulse-animation {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body class="p-8">
    <div class="max-w-6xl mx-auto">
        <header class="flex justify-between items-center mb-10">
            <div>
                <h1 class="text-3xl font-bold text-white tracking-tight">LLM Cluster Ops <span class="text-blue-500">Center</span></h1>
                <p class="text-gray-400 mt-1">OpenEnv Real-Time Telemetry Gateway</p>
            </div>
            <div class="flex items-center gap-4">
                <select id="task-select" class="bg-gray-800 text-white text-sm rounded-lg border border-gray-700 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer">
                    <option value="easy">🟢 Easy (Stable)</option>
                    <option value="medium" selected>🟡 Medium (Sine Wave)</option>
                    <option value="hard">🔴 Hard (Extreme Spikes)</option>
                </select>
                <button onclick="runLiveDemo()" id="demo-btn" class="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-bold text-sm transition-all shadow-lg hover:scale-105">
                    🚀 Launch Baseline Agent
                </button>
                <div class="flex items-center gap-3">
                    <div id="status-dot" class="w-3 h-3 rounded-full bg-green-500 pulse"></div>
                    <span id="status-text" class="text-sm font-medium uppercase tracking-widest text-green-500">System Ready</span>
                </div>
            </div>
        </header>

        <!-- AGENT DECISION PANEL -->
        <h3 class="text-white text-sm font-bold uppercase tracking-widest mb-4 ml-1">🧠 Agent Decision <span class="text-gray-500 font-normal">(Real-Time)</span></h3>
        <div class="decision-card grid grid-cols-3 gap-8 text-center mb-10 shadow-2xl">
            <div class="border-r border-gray-700">
                <div class="decision-label">Node Action</div>
                <div id="dec-scale" class="decision-value">--</div>
            </div>
            <div class="border-r border-gray-700">
                <div class="decision-label">Target Batch</div>
                <div id="dec-batch" class="decision-value mono">--</div>
            </div>
            <div>
                <div class="decision-label">Spot Allocation</div>
                <div id="dec-spot" class="decision-value mono">--%</div>
            </div>
        </div>

        <h3 class="text-white text-sm font-bold uppercase tracking-widest mb-4 ml-1">📡 Live System State</h3>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10 text-center">
            <div class="card p-6 border-l-4 border-blue-500">
                <p class="text-xs uppercase tracking-widest text-gray-400 mb-2">Active GPUs</p>
                <p id="gpu-count" class="stat-value mono">--</p>
            </div>
            <div class="card p-6 border-l-4 border-purple-500">
                <p class="text-xs uppercase tracking-widest text-gray-400 mb-2">Request Queue</p>
                <p id="queue-len" class="stat-value mono">--</p>
            </div>
            <div class="card p-6 border-l-4 border-yellow-500">
                <p class="text-xs uppercase tracking-widest text-gray-400 mb-2">Avg Latency</p>
                <p id="latency" class="stat-value mono">-- ms</p>
            </div>
            <div class="card p-6 border-l-4 border-green-500">
                <p class="text-xs uppercase tracking-widest text-gray-400 mb-2">Current Load</p>
                <p id="load" class="stat-value mono">-- %</p>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div class="card p-6 min-h-[300px]">
                <h2 class="text-lg font-semibold mb-4 text-white">Latency Over Time (ms)</h2>
                <canvas id="latencyChart"></canvas>
            </div>
            <div class="card p-6 min-h-[300px]">
                <h2 class="text-lg font-semibold mb-4 text-white">Cluster Utilization (GPUs)</h2>
                <canvas id="gpuChart"></canvas>
            </div>
        </div>

        <footer class="text-center text-gray-500 text-sm mt-12 border-t border-gray-800 pt-8">
            <p>Environment: <span class="mono">llm-serving-autoscaler-v1</span> | <span class="mono">Tanushri205</span></p>
        </footer>
    </div>

    <script>
        const MAX_DATA_POINTS = 30;
        const latencyData = [];
        const gpuData = [];
        const labels = [];

        const chartL = new Chart(document.getElementById('latencyChart').getContext('2d'), {
            type: 'line',
            data: { labels, datasets: [{ label: 'Latency', data: latencyData, borderColor: '#58a6ff', tension: 0.4, fill: true, backgroundColor: 'rgba(88, 166, 255, 0.1)' }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, grid: { color: '#30363d' } }, x: { display: false } }, plugins: { legend: { display: false } } }
        });

        const chartG = new Chart(document.getElementById('gpuChart').getContext('2d'), {
            type: 'bar',
            data: { labels, datasets: [{ label: 'GPUs', data: gpuData, backgroundColor: '#238636' }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 100, grid: { color: '#30363d' } }, x: { display: false } }, plugins: { legend: { display: false } } }
        });

        async function updateStats() {
            try {
                const resState = await fetch('/state');
                const data = await resState.json();
                
                document.getElementById('gpu-count').innerText = data.active_gpus;
                document.getElementById('queue-len').innerText = data.queue_length;
                document.getElementById('latency').innerText = data.avg_latency.toFixed(0);
                document.getElementById('load').innerText = (data.cache_load * 100).toFixed(0);

                const resAction = await fetch('/last_action');
                const act = await resAction.json();
                if (act && act.batch_size > 0) {
                    const scaleEl = document.getElementById('dec-scale');
                    scaleEl.innerText = act.scale > 0 ? "+1" : (act.scale < 0 ? "-1" : "0");
                    scaleEl.style.color = act.scale > 0 ? "#22c55e" : (act.scale < 0 ? "#ef4444" : "#facc15");
                    document.getElementById('dec-batch').innerText = act.batch_size;
                    document.getElementById('dec-spot').innerText = (act.spot_allocation * 100).toFixed(0) + "%";
                }

                const now = new Date().toLocaleTimeString();
                labels.push(now);
                latencyData.push(data.avg_latency);
                gpuData.push(data.active_gpus);
                if (labels.length > MAX_DATA_POINTS) { labels.shift(); latencyData.shift(); gpuData.shift(); }
                chartL.update('none'); chartG.update('none');
            } catch (e) { console.error(e); }
        }

        async function runLiveDemo() {
            const btn = document.getElementById('demo-btn');
            const dot = document.getElementById('status-dot');
            const status = document.getElementById('status-text');
            const task = document.getElementById('task-select').value;

            btn.disabled = true;
            btn.innerText = '⚡ AGENT SIMULATING...';
            btn.classList.replace('bg-blue-600', 'bg-gray-700');
            dot.classList.replace('bg-green-500', 'bg-blue-500');
            status.innerText = 'EPISODE IN PROGRESS (' + task.toUpperCase() + ')';
            status.style.color = '#3b82f6';

            try {
                const res = await fetch('/run_live_demo?task=' + task, { method: 'POST' });
                const result = await res.json();
                if (result.status === 'ok') {
                    alert('Simulation Complete! Baseline score achieved: ' + (result.score * 100).toFixed(1) + '%');
                } else {
                    alert('Error: ' + result.detail);
                }
            } catch (e) {
                alert('Connection failure during simulation.');
            } finally {
                btn.disabled = false;
                btn.innerText = '🚀 Launch Baseline Agent';
                btn.classList.replace('bg-gray-700', 'bg-blue-600');
                dot.classList.replace('bg-blue-500', 'bg-green-500');
                status.innerText = 'SYSTEM READY';
                status.style.color = '#22c55e';
            }
        }

        setInterval(updateStats, 1000); updateStats();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML

@app.get("/last_action", response_model=LLMServeAction, summary="Read last submitted action")
def get_last_action():
    """Return the last action processed by the server."""
    if _last_action is None:
        return LLMServeAction(scale=0, batch_size=0, spot_allocation=0.0)
    return _last_action

@app.post("/run_live_demo", summary="Start a non-blocking animated simulation")
async def run_live_demo(task: str = Query("easy", pattern="^(easy|medium|hard)$")):
    """
    Runs a live episode update using the baseline agent. 
    Pauses between steps to allow the dashboard to poll for real-time data.
    """
    global _is_demo_running, _last_action
    
    if _is_demo_running:
        return {"status": "error", "detail": "A demo is already in progress."}
    
    _is_demo_running = True
    try:
        # 1. Reset Global Env with selected task
        obs = _env.reset(task=task)
        steps = 0
        max_steps = 200 # Run for 200 steps for a snappy 20-second demo
        
        while steps < max_steps:
            # 2. Get baseline action
            action = _baseline(obs)
            _last_action = action
            
            # 3. Step Global Env
            obs, reward, done, info = _env.step(action)
            steps += 1
            
            # 4. Critical: Yield control to the event loop so /state can be served
            await asyncio.sleep(0.05) # ~20 steps per second pace
            
            if done:
                break
        
        stats = _env.episode_stats()
        score = _grader._compute_score(stats)
        return {"status": "ok", "score": score}
        
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        _is_demo_running = False


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
    global _last_action
    _last_action = action
    
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
    return {"status": "ok", "version": "1.0.1"}


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
