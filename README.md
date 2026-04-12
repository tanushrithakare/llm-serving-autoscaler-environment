---
title: Sentinel-SOC (Forensic Env)
emoji: 🛡️
colorFrom: green
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
short_description: High-fidelity forensic environment for security incident response.
tags:
- openenv
- security
- forensics
- pytorch
---

# Sentinel-SOC: Security Incident Analyzer 🛡️

**Sentinel-SOC** is a high-utility, real-world forensic environment where AI agents act as **Senior Security Analysts**. Unlike basic game environments, Sentinel-SOC requires multi-step deduction, noise filtering, and obfuscation decoding to mitigate critical security threats.

## Description
Sentinel-SOC provides a simulated Security Operations Center (SOC) where agents investigate supply-chain attacks, SQL injections, and production data leaks. The environment is designed to test the **forensic reasoning** of LLMs, forcing them to distinguish between decoy "test" data and malicious "live" payloads.

## Action Space
Agents interact with the environment using the `IncidentAction` model:

- **`reasoning`** (str): The analyst's internal chain-of-thought explaining the forensic logic.
- **`tool`** (str): The forensic tool to execute. Options include:
    - `query_logs`: Fetch system and access logs.
    - `inspect_file`: Read the contents of a specific source file.
    - `decode_payload`: Decode suspected malicious strings (Base64).
    - `remediate`: Apply a patch or rotate a compromised secret.
- **`parameters`** (str): The specific target for the tool (e.g., "access.log", "config.py", or a base64 string).

## Observation Space
Agents receive an `IncidentObs` model after every step:

- **`logs`** (str): Raw output from log queries or tool execution.
- **`code_snippet`** (str): Snippets of code retrieved during inspection.
- **`incident_thread`** (str): A summary of past actions and findings for context.
- **`status`** (str): Current investigation state (`In Progress`, `Success`, `Failed`).
- **`steps_remaining`** (int): How many actions are left in the budget.
- **`reward_signal`** (float): Incremental progress score toward resolution.

## Rewards
The environment uses a **Forensic Efficiency Rubric**:
- **Step Cost**: -0.05 per action (incentivizes speed).
- **Milestone Reward**: +0.2 for identifying a true IOC (Indicator of Compromise).
- **Final Reward**: +1.0 for successful remediation of the root cause.
- **Failure Penalty**: -1.0 for applying the wrong fix or exhausting the budget.

## Tasks
The environment currently supports three distinct scenarios:
1. **`leak-investigation` (Easy)**: Distinguish between test and production keys in noisy logs.
2. **`sqli-detection` (Medium)**: Identify SQL injection patterns and trace back to an internal IP.
3. **`backdoor-hunt` (Hard)**: Detect obfuscated Base64 backdoors in vendor dependencies.

## Quick Start

### Local Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Environment Server**:
   ```bash
   python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
   ```
3. **Run Baseline Agent**:
   ```bash
   export HF_TOKEN="your_token"
   python inference.py
   ```

---
*Developed for the Meta PyTorch Hackathon (OpenEnv).*
