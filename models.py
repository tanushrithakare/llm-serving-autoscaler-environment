from pydantic import BaseModel

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
