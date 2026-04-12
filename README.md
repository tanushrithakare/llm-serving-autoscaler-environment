---
title: Sentinel-SOC (Forensic Env)
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Forensic AI for Security Incident Response
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
- **Step cost**: -0.05 per action (incentivizes efficiency)
- **Log investigation**: +0.10 for initial reconnaissance
- **IOC confirmed**: +0.30 for identifying the true indicator of compromise
- **File identified**: +0.20 for locating the root cause file
- **Incident resolved**: +0.40 for successful remediation
- **Wrong action penalty**: -0.10 to -0.20 for acting on decoy data

## Tasks
The environment currently supports three distinct scenarios:
1. **`easy`** — Detect and contain a production secret key leak in noisy application logs
2. **`medium`** — Identify and trace a SQL injection attack back to its source IP
3. **`hard`** — Detect and remove an obfuscated Base64 backdoor in a vendor dependency

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
*Developed for the Meta PyTorch Hackathon 2026 (OpenEnv).*
