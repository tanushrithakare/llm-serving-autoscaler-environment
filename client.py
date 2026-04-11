import httpx
from models import IncidentAction, IncidentObs

class SentinelSOCClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task: str = "leak-investigation") -> IncidentObs:
        response = self.client.post(f"{self.base_url}/reset", params={"task": task})
        response.raise_for_status()
        return IncidentObs(**response.json())

    def step(self, action: IncidentAction) -> dict:
        response = self.client.post(f"{self.base_url}/step", json=action.model_dump())
        response.raise_for_status()
        return response.json()

    def state(self) -> IncidentObs:
        response = self.client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return IncidentObs(**response.json())

    def grade(self) -> float:
        response = self.client.post(f"{self.base_url}/grade")
        response.raise_for_status()
        return response.json()["score"]

    def close(self):
        self.client.close()
