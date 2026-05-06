from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class IncidentAction(BaseModel):
    reasoning: str    
    tool: str         
    parameters: str   

class IncidentObs(BaseModel):
    logs: str                
    code_snippet: str        
    incident_thread: str     
    status: str              
    steps_remaining: int     
    reward_signal: float
    severity_score: float = 0.0  # Normalized urgency score (0.0 – 1.0)

class KillChainPhaseStats(BaseModel):
    phase: str
    completed: bool
    tool_used: str
    reward_earned: float

class AnalyticsReport(BaseModel):
    """Session analytics returned by GET /analytics."""
    task: str
    total_steps: int
    steps_remaining: int
    kill_chain_phases: List[KillChainPhaseStats]
    total_reward: float
    efficiency_score: float          # reward / steps ratio (higher is better)
    success_rate: float              # fraction of steps with positive reward
    action_breakdown: Dict[str, int] # tool → call count
    incident_resolved: bool
    final_grade: Optional[float] = None
