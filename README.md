# Sentinel-SOC: Security Incident Analyzer 🛡️

**Winner-Tier Submission for the Meta PyTorch Hackathon (OpenEnv Category)**

Sentinel-SOC is a high-utility, real-world forensic environment where AI agents act as **Senior Security Analysts**. Unlike basic game environments, Sentinel-SOC requires multi-step deduction, noise filtering, and obfuscation decoding to mitigate critical security threats.

## 📂 Repository Structure
```text
.
├── models.py           # Action and Observation Pydantic models
├── environment.py      # Core forensic logic and task definitions
├── server/
│   ├── app.py          # FastAPI server implementation
│   └── static/         # Forensic Dashboard (HUD) frontend
├── inference.py        # Multi-task autonomous analyst (Inference)
├── openenv.yaml        # Standardized OpenEnv configuration
├── grader.py           # Automated multi-task evaluation tool
├── Dockerfile          # Production deployment configuration
└── requirements.txt    # Project dependencies
```
- **Domain**: Cybersecurity Forensics & Incident Response.
- **Task Complexity**:
  - `leak-investigation` (Easy): Distinguish between test and production keys in noisy logs.
  - `sqli-detection` (Medium): Identify SQL injection patterns and trace back to an internal IP.
  - `backdoor-hunt` (Hard): Detect obfuscated Base64 backdoors in vendor dependencies.
- **Deduction Path**: Query Logs ➔ Extract IOC ➔ Inspect Source ➔ Apply Fix.

## 🧠 Advanced Features (Grand Master Edition)
- **Signal vs. Noise**: Agents must ignore `sk_test` decoy data and focus on `sk_live` production leaks.
- **Red Herrings**: Multiple suspicious files are provided; agents must cross-reference logs to find the true root cause.
- **Obfuscation**: Hard tasks use Base64 encoding to test the agent's ability to decode malicious payloads.
- **Efficiency Scoring**: Points are awarded not just for completion, but for investigation speed.

## 🛠️ Getting Started

### Local Setup
1. **Requirements**: `pip install -r requirements.txt`
2. **Launch Server**: `python -m uvicorn server.app:app --host 0.0.0.0 --port 7860`
3. **Run Inference**:
   ```powershell
   $env:HF_TOKEN = "your_token"
   python inference.py
   ```

## 🧠 Advanced Features (Grand Master Edition)
- **Analyst Guidance System**: Injects situational intelligence (e.g., `NEXT STEP: Call inspect_file`) into the observation, providing a clear reasoning path for LLMs.
- **Provider-Agnostic Parser**: A robust JSON extraction layer in `inference.py` that handles markdown fences and model chatter, ensuring stability across any LLM backend.
- **Signal vs. Noise**: Agents must ignore `sk_test` decoy data and focus on `sk_live` production leaks.
- **Red Herrings**: Multiple suspicious logs are provided; agents must cross-reference to find the true root cause.
- **Deterministic Efficiency**: Points are awarded with mathematical precision based on forensic progression and investigation speed.

---
*Created for the Meta PyTorch Hackathon 2024.*
