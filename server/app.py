"""
app.py — FastAPI server for the LLM Serving Autoscaler Environment.

Endpoints
---------
POST /reset?task=easy|medium|hard  — start a new episode
POST /step                         — submit an action, get obs + reward
GET  /state                        — read current observation
GET  /health                       — liveness check
POST /grade                        — run full graded episode with baseline agent
"""

from fastapi import FastAPI, HTTPException, Query, Response  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from pydantic import BaseModel  # type: ignore
import asyncio
import os
import sys
from typing import Optional

from environment import LLMServeEnv  # type: ignore # noqa: E402
from models import LLMServeAction, LLMServeObs  # type: ignore # noqa: E402
from grader import LLMServeGrader  # type: ignore # noqa: E402
from baseline import BaselineHeuristicAgent  # type: ignore # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "LLM Serving Autoscaler Environment",
    description = "OpenEnv-compatible RL environment for GPU autoscaling.",
    version     = "1.0.1",
)


# ---------------------------------------------------------------------------
# Centralised application state — avoids mutable module-level globals
# ---------------------------------------------------------------------------

class AppState:
    """
    Holds all mutable server-side state in one place with thread-safety.
    A single instance (_state) is created at startup and shared across
    request handlers via closure — no global keyword required.
    """

    def __init__(self) -> None:
        self.env             = LLMServeEnv()
        self.baseline        = BaselineHeuristicAgent()
        self.last_action: Optional[LLMServeAction] = None
        self.last_reward: float = 0.0
        self.history: list   = []
        self.is_demo_running: bool = False
        self.lock            = asyncio.Lock()

    MAX_HISTORY: int = 1000

    def record(self, step: int, obs: LLMServeObs, reward: float) -> None:
        """Append a telemetry snapshot and keep the ring-buffer bounded."""
        self.history.append({
            "step":    step,
            "latency": obs.avg_latency,
            "gpus":    obs.active_gpus,
            "queue":   obs.queue_length,
            "reward":  reward,
        })
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)

    def reset_episode(self, task: str) -> LLMServeObs:
        """Clear transient per-episode state and reset the underlying environment."""
        self.last_action  = None
        self.last_reward  = 0.0
        self.history      = []
        # Total isolation: instantiate fresh environment
        self.env          = LLMServeEnv()
        return self.env.reset(task=task)


_state = AppState()


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
                <button onclick="resetEpisode()" id="reset-btn" class="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg font-bold text-xs transition-all border border-gray-700">
                    🔄 Reset Episode
                </button>
                <div class="flex items-center gap-3 ml-4">
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
                <p class="text-xs uppercase tracking-widest text-gray-400 mb-2">Cluster Efficiency</p>
                <p id="load" class="stat-value mono">-- %</p>
            </div>
        </div>

        <!-- LIVE REWARD PANEL -->
        <div class="card p-4 bg-black border-dashed border-2 border-gray-700 text-center mb-10">
            <p class="text-[10px] uppercase tracking-[0.2em] text-gray-500 mb-1">Live Agent Reinforcement Signal</p>
            <div class="flex justify-center items-center gap-4">
                <span id="rew-label" class="text-xs font-bold text-gray-400">Step Reward:</span>
                <span id="rew-value" class="text-2xl font-black mono text-white">0.00</span>
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

                const resRew = await fetch('/live_reward');
                const rewData = await resRew.json();
                const rVal = rewData.reward;
                const rEl = document.getElementById('rew-value');
                rEl.innerText = (rVal >= 0 ? "+" : "") + rVal.toFixed(2);
                rEl.style.color = rVal > 0.1 ? "#22c55e" : (rVal < -0.1 ? "#ef4444" : "#e2e8f0");
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

        async function resetEpisode() {
            const task = document.getElementById('task-select').value;
            try {
                const res = await fetch('/reset?task=' + task, { method: 'POST' });
                if (res.ok) {
                    latencyData.length = 0;
                    gpuData.length = 0;
                    labels.length = 0;
                    chartL.update(); 
                    chartG.update();
                    updateStats();
                    alert('Episode Reset to Step 0 (Task: ' + task.toUpperCase() + ')');
                }
            } catch (e) { alert('Reset failed.'); }
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
                if (result.status === 'started') {
                    // Poll for completion via /demo_status
                    const poll = setInterval(async () => {
                        const st = await fetch('/demo_status');
                        const s = await st.json();
                        if (!s.running) {
                            clearInterval(poll);
                            btn.disabled = false;
                            btn.innerText = '🚀 Launch Baseline Agent';
                            btn.classList.replace('bg-gray-700', 'bg-blue-600');
                            dot.classList.replace('bg-blue-500', 'bg-green-500');
                            status.innerText = 'SYSTEM READY';
                            status.style.color = '#22c55e';
                        }
                    }, 2000);
                } else {
                    alert('Error: ' + result.detail);
                    btn.disabled = false;
                    btn.innerText = '🚀 Launch Baseline Agent';
                    btn.classList.replace('bg-gray-700', 'bg-blue-600');
                }
            } catch (e) {
                alert('Connection failure during simulation.');
                btn.disabled = false;
            }
        }

        setInterval(updateStats, 2000); updateStats();
    </script>
</body>
</html>
"""

VIZ_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Cluster Telemetry | /viz</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;600&display=swap" rel="stylesheet">
    <style>
        body { background-color: #0b0e14; color: #e2e8f0; font-family: 'Outfit', sans-serif; }
        .chart-card { background-color: #151921; border: 1px solid #1f2937; border-radius: 16px; padding: 15px; }
    </style>
</head>
<body class="p-6">
    <div class="max-w-7xl mx-auto">
        <div class="flex justify-between items-center mb-8">
            <h1 class="text-3xl font-bold text-white tracking-widest uppercase">Live <span class="text-blue-500">Telemetry</span> Dashboard</h1>
            <div class="text-sm text-gray-500">Update frequency: 1Hz</div>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div class="chart-card shadow-lg"><div id="latency-chart" style="height:350px;"></div></div>
            <div class="chart-card shadow-lg"><div id="gpu-chart" style="height:350px;"></div></div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="chart-card shadow-lg"><div id="queue-chart" style="height:350px;"></div></div>
            <div class="chart-card shadow-lg"><div id="reward-chart" style="height:350px;"></div></div>
        </div>
    </div>

    <script>
        const layout_base = { 
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', family: 'Outfit' },
            margin: { t: 40, b: 40, l: 50, r: 20 },
            xaxis: { gridcolor: '#1f2937', zeroline: false },
            yaxis: { gridcolor: '#1f2937', zeroline: false }
        };

        async function updateGraphs() {
            const res = await fetch('/history');
            const history = await res.json();
            
            const x = history.map(h => h.step);
            const latency = history.map(h => h.latency);
            const gpus = history.map(h => h.gpus);
            const queue = history.map(h => h.queue);
            const reward = history.map(h => h.reward);

            Plotly.react('latency-chart', [{ x, y: latency, type: 'scatter', mode: 'lines', fill: 'tozeroy', line: { color: '#3b82f6', width: 3 }, name: 'Latency (ms)' }], { ...layout_base, title: 'Live Latency Evolution' });
            Plotly.react('gpu-chart', [{ x, y: gpus, type: 'bar', marker: { color: '#10b981' }, name: 'Active GPUs' }], { ...layout_base, title: 'GPU Cluster Capacity' });
            Plotly.react('queue-chart', [{ x, y: queue, type: 'scatter', mode: 'lines', line: { color: '#8b5cf6', dash: 'dot' }, name: 'Queue Depth' }], { ...layout_base, title: 'Pending Request Backlog' });
            Plotly.react('reward-chart', [{ x, y: reward, type: 'scatter', mode: 'lines+markers', line: { color: '#f43f5e' }, name: 'Step Reward' }], { ...layout_base, title: 'Agent Performance Signal' });
        }

        setInterval(updateGraphs, 2000);
        updateGraphs();
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes — UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/viz", response_class=HTMLResponse)
async def viz_dashboard():
    return VIZ_HTML


@app.get("/history")
async def get_history():
    return _state.history


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
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=LLMServeObs, summary="Reset the environment")
async def reset(task: str = Query("easy", pattern="^(easy|medium|hard)$")):
    """
    Start a fresh episode.

    - **task**: difficulty level — `easy`, `medium`, or `hard`
    """
    try:
        async with _state.lock:
            obs = _state.reset_episode(task=task)
            return obs
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse, summary="Submit an action")
async def step(action: LLMServeAction):
    """
    Advance one timestep with the given action.

    Returns the new observation, reward, done flag, and debug info.
    """
    try:
        async with _state.lock:
            _state.last_action = action
            obs, reward, done, info = _state.env.step(action)
            _state.last_reward = reward
            _state.record(step=_state.env._step_count, obs=obs, reward=reward)
            return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=LLMServeObs, summary="Read current observation")
async def state():
    """Return the current environment observation without advancing the episode."""
    try:
        return _state.env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", summary="Liveness check")
async def health():
    """Simple health probe — always returns 200 when the server is up."""
    return {"status": "ok", "version": "1.0.1"}


@app.get("/healthz", summary="HF liveness check")
async def healthz():
    """HuggingFace Spaces health probe."""
    return {"ok": True}


@app.post("/grade", response_model=GradeResponse, summary="Grade the baseline agent")
def grade(request: GradeRequest):
    """
    Run a full 1000-step episode with the built-in baseline agent and return
    a normalised score in (0.01, 0.99).
    """
    try:
        if request.task not in ("easy", "medium", "hard"):
            raise HTTPException(status_code=400, detail="task must be easy, medium, or hard")
            
        # Create fresh grader instance (full isolation)
        grader = LLMServeGrader()
        score = grader.grade(_state.baseline, task=request.task)
        
        return GradeResponse(task=request.task, score=score)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Supplementary endpoints (dashboard helpers)
# ---------------------------------------------------------------------------

@app.get("/last_action", response_model=LLMServeAction, summary="Read last submitted action")
async def get_last_action():
    """Return the last action processed by the server."""
    if _state.last_action is None:
        return LLMServeAction(scale=0, batch_size=64, spot_allocation=0.0)
    return _state.last_action


@app.get("/live_reward", summary="Read last reward signal")
async def get_live_reward():
    """Return the last step reward for dashboard display."""
    return {"reward": _state.last_reward}


@app.get("/demo_status", summary="Check whether a live demo is running")
async def demo_status():
    """Polled by the dashboard to detect when a background demo finishes."""
    return {"running": _state.is_demo_running}


# ---------------------------------------------------------------------------
# Background demo task — runs simulation without blocking the event loop
# ---------------------------------------------------------------------------

async def _run_demo_background(task: str, max_steps: int = 500) -> None:
    """
    Simulate up to *max_steps* using the baseline agent.
    Runs as a background asyncio task so /run_live_demo returns immediately.
    """
    _state.is_demo_running = True
    _state.last_reward = 0.0
    try:
        async with _state.lock:
            obs = _state.reset_episode(task=task)
            
        for step_num in range(max_steps):
            # Compute action outside the lock (baseline does not mutate state)
            action = _state.baseline(obs)
            
            async with _state.lock:
                _state.last_action = action
                obs, reward, done, _ = _state.env.step(action)
                _state.last_reward = reward
                _state.record(step=step_num, obs=obs, reward=reward)
                
            # Yield to the event loop so /state and /live_reward can be served
            await asyncio.sleep(0.1)
            
            if done:
                break
    except Exception as exc:
        print(f"[demo] error: {exc}")
    finally:
        _state.is_demo_running = False


@app.post("/run_live_demo", summary="Start a non-blocking animated simulation")
async def run_live_demo(task: str = Query("easy", pattern="^(easy|medium|hard)$")):
    """
    Kick off a background simulation of the baseline agent.
    Returns immediately with ``{"status": "started"}``; poll ``/demo_status``
    to detect completion.
    """
    if _state.is_demo_running:
        return {"status": "error", "detail": "A demo is already in progress."}

    asyncio.create_task(_run_demo_background(task))
    return {"status": "started"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
