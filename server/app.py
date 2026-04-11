import sys
import os

# Ensure project root (one level up) is in path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from environment import SentinelSOCEnv
from models import IncidentAction, IncidentObs

app = FastAPI()

# Mount static files (isolated)
os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
@app.get("/dashboard")
def read_dashboard():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "dashboard.html"))

# Singleton environment instance
_env_instance = None


def get_env(task: str = "leak-investigation"):
    global _env_instance
    if _env_instance is None:
        _env_instance = SentinelSOCEnv()
    return _env_instance


@app.get("/health")
def health():
    return {"status": "healthy", "service": "sentinel-soc"}


@app.post("/reset", response_model=IncidentObs)
def reset(task: str = Query("easy", enum=["easy", "medium", "hard"])):
    env = get_env(task)
    return env.reset(task=task)


@app.post("/step")
def step(action: IncidentAction):
    env = get_env()
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }


@app.get("/state", response_model=IncidentObs)
def state():
    env = get_env()
    return env._get_obs()


@app.post("/grade")
def grade():
    env = get_env()
    score = env.grade()
    return {"score": score}


@app.get("/history")
def get_history():
    env = get_env()
    return {"history": env.history}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
