import sys
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Ensure project root (one level up) is in path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from environment import SentinelSOCEnv
from models import IncidentAction, IncidentObs
import gradio as gr
from server.gradio_ui import create_gradio_ui, CSS, JS_FORCE_DARK, THEME

app = FastAPI()

# Per-request environment storage (prevents state sharing between concurrent requests)
_env_instances = {}  # Maps session_id to environment


def get_or_create_env(session_id: str = "default") -> SentinelSOCEnv:
    """Get or create environment instance for this session."""
    if session_id not in _env_instances:
        _env_instances[session_id] = SentinelSOCEnv()
    return _env_instances[session_id]


@app.get("/health")
def health():
    return {"status": "healthy", "service": "sentinel-soc"}


@app.post("/reset", response_model=IncidentObs)
def reset(task: str = Query("easy", enum=["easy", "medium", "hard"])):
    # Create fresh environment for this reset (prevents state sharing between concurrent users)
    env = SentinelSOCEnv()
    _env_instances["default"] = env
    return env.reset(task=task)


@app.post("/step")
def step(action: IncidentAction):
    env = get_or_create_env()
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }


@app.get("/state", response_model=IncidentObs)
def state():
    env = get_or_create_env()
    return env._get_obs()


@app.post("/grade")
def grade():
    env = get_or_create_env()
    score = env.grade()
    return {"score": score}


@app.get("/history")
def get_history():
    env = get_or_create_env()
    return {"history": env.history}

# --- Gradio UI Integration ---
ui_app = create_gradio_ui(server_url="http://localhost:7860")
app = gr.mount_gradio_app(
    app, 
    ui_app, 
    path="/",
    app_kwargs={
        "theme": THEME,
        "css": CSS,
        "js": JS_FORCE_DARK
    }
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
