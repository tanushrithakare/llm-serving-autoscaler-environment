---
title: Sentinel-SOC (Forensic Env)
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: AI Security Analyst — Forensic Incident Response Environment
tags:
  - openenv
  - security
  - forensics
  - cybersecurity
  - agent
  - rl
  - reinforcement-learning
  - benchmark
  - pytorch
  - open-env
---

**Sentinel-SOC** is an AI-powered Security Operations Center where LLM agents act as autonomous security analysts — investigating, reasoning through adversarial noise, and mitigating real-world cyber threats using structured forensic tools.

Unlike traditional benchmarks that only evaluate whether an agent reaches the correct answer, Sentinel-SOC evaluates **how** the agent reasons: every tool call, every decision, and every mistake is tracked, scored, and explainable.

The environment follows current OpenEnv client/server conventions:

- `SentinelSOCEnv` is the server-side environment
- `SentinelSOCClient` is the HTTP client for remote usage
- `inference.py` is the standard OpenEnv baseline inference runner
- Scoring follows the **Cyber Kill Chain** methodology (Reconnaissance → Identification → Containment → Remediation)

## Overview

Inside the SOC, the agent can:

- call `query_logs` to ingest raw system telemetry
- call `extract_ioc` to identify Indicators of Compromise (IOCs)
- call `inspect_file` to locate malicious source files
- call `apply_fix` to remediate the confirmed threat
- navigate ambiguous, noise-injected telemetry where decoy data competes with real signals

Every episode is **procedurally generated** — unique API keys, attacker IPs, C2 domains, and log timestamps are synthesized per `reset()`, making the environment non-memorizable and genuinely useful for RL training.

## Why Sentinel-SOC Matters

Traditional benchmarks evaluate whether an agent reaches the correct answer. Sentinel-SOC evaluates the **reliability of the reasoning path** itself.

### 💡 Key Innovation
**This makes Sentinel-SOC not only a benchmark, but a debugging tool for agent reasoning failures.** By observing where an agent incorrectly identifies a decoy as an IOC, researchers can pinpoint exactly where a model's deductive logic breaks down under adversarial noise.

### 🌍 Real-World Impact
1. **SOC Team Augmentation**: Provides a safe, reproducible sandbox to evaluate whether AI agents can handle Tier-1/Tier-2 analyst workloads before deployment in production networks.
2. **Adversarial Resilience**: Unlike static datasets, our procedural engine allows teams to test if agents can resist "noise-injection" attacks (common in real-world log spoofing).
3. **Auditable Intelligence**: The Cyber Kill Chain enforcement ensures that AI actions are not just "correct" but **procedurally compliant** with security industry standards.

## Environment Specification

| Property | Value |
|---|---|
| **API Compliance** | OpenEnv v1.1 |
| **Action Space** | Discrete (4 Key Tools + Logic) |
| **Observation Space** | Multi-modal (Textual Logs + Code Snippets) |
| **State Transitions** | Cyber Kill Chain State Machine |
| **Reward Distribution** | Shaped (Partial Success + Budget Penalties) |
| **Scenarios** | Procedural (Random Seeded Episodes) |

## Diagnostic Toolkit

Sentinel-SOC evaluates the agent's ability to select the correct forensic tool for the current kill-chain phase:

| Tool | Industrial Equivalent | Purpose |
|---|---|---|
| `query_logs` | ELK / Splunk | Ingest raw system telemetry to find initial signals. |
| `extract_ioc` | Threat Intelligence | Validate suspicious strings against intelligence databases. |
| `inspect_file` | EDR / Carbon Black | Locally inspect file content for malicious logic or backdoors. |
| `apply_fix` | SOAR / Remediation | Execute containment and neutralization actions. |

## Core Features

- Evaluates **how** the agent reasons — not just outcomes
- Simulates **real SOC workflows** with multi-step forensic methodology
- Introduces **adversarial noise and decoy IOCs** to force genuine deduction
- Enforces **structured investigation** via Cyber Kill Chain phase gating
- Produces **fully auditable traces** — every decision is logged, scored, and diagnosable

This makes it suitable for **training, evaluating, and debugging AI systems** intended for real-world security operations — a use case with direct industry relevance.

## Key Innovation

Sentinel-SOC is not just an evaluation environment.

It is an **auditable intelligence system** where:

- Every agent decision is tracked and logged
- Every reasoning step is surfaced in the UI
- Every failure mode is attributable to a specific kill chain phase

This enables not only benchmark scoring, but also **training signal generation and failure diagnosis for AI security analysts** — closing the gap between academic RL environments and deployable SOC tooling.

This makes Sentinel-SOC not only a benchmark, but a **debugging tool for agent reasoning failures**.

## Current Architecture

Main modules:

- [`environment.py`](environment.py): procedural scenario engine, kill chain enforcement, and grading logic
- [`models.py`](models.py): typed `IncidentAction` and `IncidentObs` Pydantic models
- [`client.py`](client.py): synchronous HTTP client for remote usage
- [`inference.py`](inference.py): OpenEnv-compliant baseline inference script (`[START]`/`[STEP]`/`[END]`)
- [`server/app.py`](server/app.py): FastAPI server exposing `/reset`, `/step`, `/state`, `/grade`, `/history`
- [`server/gradio_ui.py`](server/gradio_ui.py): Professional Gradio forensic dashboard
- [`tests/test_environment.py`](tests/test_environment.py): 9-test validation suite

## Rewards

Rewards are structured to enforce **kill chain methodology**. Skipping phases or acting on decoy data is penalized.

| Phase | Tool | Condition | Reward |
|---|---|---|---|
| Reconnaissance | `query_logs` | First call | `+0.10` |
| Identification | `extract_ioc` | Correct IOC submitted | `+0.30` |
| Containment | `inspect_file` | Correct file submitted | `+0.20` |
| Remediation | `apply_fix` | After IOC + file confirmed | `+0.40` |
| Step cost | any | Per action taken | `−0.05` |
| Decoy penalty | `extract_ioc` | Wrong IOC selected | `−0.15` |
| Skip penalty | `apply_fix` | Premature remediation | `−0.15` |

Final score is clipped to `[0.01, 0.99]` for OpenEnv boundary compliance.

## Quick Start

### Remote (HTTP API)

```python
import httpx

SERVER = "https://tanushri205-llm-serving-autoscaler-enviornment-server.hf.space"

# Reset the environment
obs = httpx.post(f"{SERVER}/reset", params={"task": "easy"}).json()
print(obs["incident_thread"])   # → live incident alert with situational report
print(obs["logs"])              # → noisy telemetry with mixed signals

# Reconnaissance — query logs
resp = httpx.post(f"{SERVER}/step", json={
    "reasoning": "Ingesting all telemetry to identify credential anomalies",
    "tool": "query_logs",
    "parameters": "all"
}).json()
print(resp["reward"])   # → 0.10

# Identification — extract the IOC
resp = httpx.post(f"{SERVER}/step", json={
    "reasoning": "sk_live_ prefix indicates production credential leak",
    "tool": "extract_ioc",
    "parameters": "sk_live_51M0xABcdEF67"
}).json()
print(resp["reward"])   # → 0.30

# Containment — inspect the source file
resp = httpx.post(f"{SERVER}/step", json={
    "reasoning": "Credential observed in app.log — inspecting root cause",
    "tool": "inspect_file",
    "parameters": "app.log"
}).json()
print(resp["reward"])   # → 0.20

# Remediation — apply fix
resp = httpx.post(f"{SERVER}/step", json={
    "reasoning": "Rotating and masking all exposed production credentials",
    "tool": "apply_fix",
    "parameters": "remediate"
}).json()
print(resp["reward"])   # → 0.40
print(resp["done"])     # → True

# Final grade
score = httpx.post(f"{SERVER}/grade").json()
print(score)            # → {"score": 0.93}
```

### Local Usage (Direct Python)

```python
from environment import SentinelSOCEnv
from models import IncidentAction

env = SentinelSOCEnv()
obs = env.reset(task="easy")

action = IncidentAction(
    reasoning="Ingesting raw telemetry for initial reconnaissance",
    tool="query_logs",
    parameters="all"
)
obs, reward, done, info = env.step(action)
print(reward)               # → 0.10
print(info["tool_result"])  # → "Log analysis complete. Credential pattern observed in..."

score = env.grade()
print(score)                # → 0.0 – 1.0
```

### Baseline Inference

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

python inference.py
```

Output follows the mandatory OpenEnv log format:
```
[START] task=easy env=sentinel-soc model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=query_logs(all) reward=0.10 done=false error=null
[STEP] step=2 action=extract_ioc(sk_live_...) reward=0.30 done=false error=null
[STEP] step=3 action=inspect_file(app.log) reward=0.20 done=false error=null
[STEP] step=4 action=apply_fix(remediate) reward=0.40 done=true error=null
[END] success=true steps=4 score=0.930 rewards=0.10,0.30,0.20,0.40
```

## Tasks & Scenarios

Every `reset()` generates a **unique episode** — IPs, API keys, domains, filenames, and log timestamps are seeded fresh each time.

| Task | Difficulty | Scenario | Noise | Max Steps |
|---|---|---|---|---|
| `easy` | 🟢 Tier 1 | Production secret key exposed in application logs | 10% | 10 |
| `medium` | 🟡 Tier 2 | SQL injection attack traced to an external attacker IP | 30% | 15 |
| `hard` | 🔴 Tier 3 | Obfuscated Base64 C2 backdoor in vendor dependency | 50% | 20 |

Hard tier introduces multiple suspicious—but incorrect—domains and IPs alongside the real C2 callback. The agent must reason through a **multi-step deduction chain**:

1. Identify suspicious outbound network traffic in noisy telemetry
2. Inspect obfuscated vendor source code containing a Base64-encoded payload
3. Decode the payload to reveal the embedded C2 domain
4. Correlate the decoded domain against observed UNAUTHORIZED network connections
5. Distinguish the true C2 from decoy domains that appear in both logs and code

Single-step heuristics and pattern-matching fail in this scenario. The agent must perform genuine multi-source deductive reasoning.

## 🔍 IOC Extraction Design

Sentinel-SOC evaluates whether an AI agent can identify the correct **Indicator of Compromise (IOC)** among realistic noise and decoys. Each task introduces ambiguity that requires contextual reasoning — not pattern matching.

### 🟢 Easy — Credential Leak

Logs contain both a real production key and a decoy test key:

```
sk_live_...   ← real credential (production exposure)
sk_test_...   ← decoy (development key, harmless)
```

The agent must **distinguish production vs test credentials** and select only the sensitive one.

### 🟡 Medium — SQL Injection

Multiple external IPs appear in access logs — only one is associated with a SQL injection payload:

```
94.103.42.17  → GET /user?id=1' UNION SELECT password_hash FROM credentials --   ← target
45.227.18.203 → POST /api/login   (401 failed auth — suspicious but not the source)
```

The agent must **analyze query behavior** and correlate attack patterns with the correct attacker IP.

### 🔴 Hard — Backdoor Detection

The malicious C2 domain appears in two places simultaneously:

```
# In network logs:
NETWORK: 172.16.x.x → malware-cdn.xyz:443  (UNAUTHORIZED)     ← target
NETWORK: 172.16.x.x → attacker-domain.io:8080  (BLOCKED)      ← decoy

# In source code (Base64-obfuscated):
beacon_url = base64.b64decode("bWFsd2FyZS1jZG4ueHl6").decode()  ← encoded target
```

The agent must **decode the obfuscated payload**, correlate it with network activity, and distinguish the true C2 endpoint from decoys that appear in both logs and source code.

### 🧠 Why This Is Challenging

- Multiple valid-looking candidates appear (decoys intentionally resemble real IOCs)
- Correct identification requires **cross-source reasoning** (logs + code)
- Cannot be solved by single-step extraction or pattern matching alone

### 🎯 Evaluation Objective

The agent is not rewarded for extraction alone, but for **selecting the correct IOC based on contextual understanding** within a structured investigation workflow. Wrong selections are penalized.

## Actions and Observations


### `IncidentAction`

```python
reasoning: str    # analyst chain-of-thought
tool: str         # one of: query_logs, extract_ioc, inspect_file, apply_fix
parameters: str   # target value (filename, IOC string, etc.)
```

### `IncidentObs`

```python
logs: str              # raw system/access/network telemetry
code_snippet: str      # source code of relevant file
incident_thread: str   # incident alert + situational phase report
status: str            # current investigation status
steps_remaining: int   # budget remaining
reward_signal: float   # cumulative reward so far
```

## What Works Today

- ✅ Procedural scenario generation (non-memorizable, seeded per episode)
- ✅ Cyber Kill Chain enforcement with per-phase rewards and penalties
- ✅ Adversarial noise scaling (10% / 30% / 50% by difficulty)
- ✅ Hard task decoy ambiguity (multiple false-positive IOC candidates)
- ✅ Neutral situational awareness — no hand-holding guidance to agents
- ✅ Professional Gradio dashboard with 8-tab forensic analyst UI
- ✅ Investigation Timeline, Agent Reasoning, and Evaluation panels
- ✅ Plain-English "Explain Simply" mode for accessibility
- ✅ OpenEnv-compliant `[START]`/`[STEP]`/`[END]` logging
- ✅ Scores clipped to `[0.01, 0.99]` for boundary compliance

## Baseline Scores

Evaluation via `inference.py` with `Qwen/Qwen2.5-72B-Instruct` at `temperature=0.0`:

| Task | Max Steps | Score | Status |
|---|---|---|---|
| `easy` | 10 | **0.93** | ✅ Mitigated in 4 steps |
| `medium` | 15 | **0.95** | ✅ Mitigated in 4 steps |
| `hard` | 20 | **0.96** | ✅ Mitigated in 4 steps |

## Test Suite

```bash
python tests/test_environment.py

# ✔ test_reset_produces_unique_scenarios PASSED
# ✔ test_kill_chain_ordering PASSED
# ✔ test_easy_full_solve PASSED (score: 0.93)
# ✔ test_decoy_penalty PASSED
# ✔ test_noise_scaling PASSED (easy: 6 lines, hard: 8 lines)
# ✔ test_grade_boundaries PASSED (min: 0.01, max: 0.93)
# ✔ test_all_tasks [easy] PASSED (score: 0.93)
# ✔ test_all_tasks [medium] PASSED (score: 0.95)
# ✔ test_all_tasks [hard] PASSED (score: 0.96)
# 🏆 ALL TESTS PASSED
```

## Local Setup

```bash
# Clone and install
git clone https://github.com/tanushri205/sentinel-soc.git
cd sentinel-soc
pip install -r requirements.txt

# Run the server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or run tests directly
python tests/test_environment.py
```

### Docker

```bash
docker build -t sentinel-soc:latest .
docker run -p 7860:7860 \
  -e HF_TOKEN="hf_..." \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  sentinel-soc:latest
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_TOKEN` | Hugging Face / OpenAI-compatible API key | — |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `SPACE_HOST` | Auto-set by HF Spaces for HTTPS routing | `localhost:7860` |

## HTTP Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reset?task={easy\|medium\|hard}` | Start a new episode |
| `POST` | `/step` | Execute a forensic tool action |
| `GET` | `/state` | Current observation |
| `GET` | `/history` | Full step history |
| `POST` | `/grade` | Final deterministic score |

---

Sentinel-SOC demonstrates how LLMs can transition from passive assistants to **active, auditable security analysts** — reducing mean time to detection, enforcing structured investigation methodology, and making AI-driven incident response transparent and trustworthy.

*Developed for the Meta × HuggingFace × Scaler OpenEnv Hackathon 2026.*
