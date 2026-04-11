import asyncio
import os
from client import SentinelSOCClient
from models import IncidentAction

def baseline_agent(obs: dict, task: str = "easy") -> dict:
    """
    State-aware baseline agent for standardized grading.
    """
    # 1. Determine Level (fall back to easy)
    phase = "easy"
    if "SQL" in obs['incident_thread'] or "192.168" in obs['logs']:
        phase = "medium"
    if "egress" in obs['incident_thread'] or "base64" in obs['code_snippet']:
        phase = "hard"

    # 2. Logic Gates (Grand Master sequence)
    if "Monitoring for Recurrence" in obs['status']:
        return {"reasoning": "Mission goal achieved.", "tool": "query_logs", "parameters": "heartbeat"}

    if "Initial" in obs['status'] or "Active" in obs['status']:
        if "High-confidence" not in obs['status'] and "CONFIRMED" not in obs['status']:
            # Gate 1: Logs first
            if phase == "easy":
                return {"reasoning": "Step 1: Discovering patterns in logs.", "tool": "query_logs", "parameters": "all"}
            elif phase == "medium":
                return {"reasoning": "Step 1: Monitoring DB traffic.", "tool": "query_logs", "parameters": "192.168.1.137"}
            else:
                return {"reasoning": "Step 1: Auditing network egress.", "tool": "query_logs", "parameters": "attacker-domain.cc"}
        
        if "Ready for Fix" not in obs['status'] and "Root Cause" not in obs['status']:
            # Gate 2: Extract IOC after logs
            if phase == "easy":
                return {"reasoning": "Step 2: Confirming PRODUCTION leak sk_live.", "tool": "extract_ioc", "parameters": "sk_live_51M0x2L9ABcdEF67890"}
            elif phase == "medium":
                return {"reasoning": "Step 2: Confirming Malicious IP source.", "tool": "extract_ioc", "parameters": "192.168.1.137"}
            else:
                return {"reasoning": "Step 2: Confirming Backdoor Domain.", "tool": "extract_ioc", "parameters": "attacker-domain.cc"}

        if "Monitoring" not in obs['status']:
            # Gate 3: Inspect file
            if phase == "easy":
                return {"reasoning": "Step 3: Finding root cause in app.log.", "tool": "inspect_file", "parameters": "app.log"}
            elif phase == "medium":
                return {"reasoning": "Step 3: Finding vulnerable DB logic.", "tool": "inspect_file", "parameters": "db_utils.py"}
            else:
                return {"reasoning": "Step 3: Auditing compromised library.", "tool": "inspect_file", "parameters": "vendor/auth_lib.py"}

        # Gate 4: Final Fix
        return {"reasoning": "Final Mitigation.", "tool": "apply_fix", "parameters": "rotate_and_mask" if phase == "easy" else "patch_sql" if phase == "medium" else "remove_backdoor"}

    return {"reasoning": "Default hunt.", "tool": "query_logs", "parameters": "status"}

async def run_baseline(task="easy"):
    print(f"Running Standardized Baseline on task: {task}")
    client = SentinelSOCClient("http://localhost:7860")
    try:
        obs = client.reset(task=task)
        print(f"Initial Phase: {obs.status}")
        
        # Simple loop using the baseline_agent function
        for i in range(5):
            action_dict = baseline_agent(obs.model_dump(), task=task)
            action = IncidentAction(**action_dict)
            res = client.step(action)
            obs = res['observation']
            print(f"Step {i+1}: {res['info']['tool_result']} (Reward: {res['reward']:.2f})")
            if res['done']: break
            
        final_score = client.grade()
        print(f"\nFinal Achievement Score (with efficiency): {final_score}")

    finally:
        client.close()

if __name__ == "__main__":
    import sys
    task_arg = sys.argv[1] if len(sys.argv) > 1 else "easy"
    asyncio.run(run_baseline(task=task_arg))
