---
title: LLM Serving Autoscaler
emoji: 🏎️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
- openenv
- rl
- engineering
---

# LLM Serving Autoscaler Environment (OpenEnv)

## 🚀 Overview

This project implements an OpenEnv-compatible reinforcement learning environment that simulates large-scale AI request serving systems.

The environment models how incoming requests are processed using GPU resources under dynamic traffic conditions. An agent learns to manage resources efficiently to maintain fast response times while minimizing operational cost.

---

## 🧠 Problem Statement

Modern AI serving systems must handle highly variable traffic:

* Sudden spikes in incoming requests
* Limited GPU resources
* Trade-offs between speed and cost

Poor resource management can lead to:

* High latency (slow responses)
* Long request queues
* Wasted compute resources

This environment simulates these challenges and enables agents to learn effective decision-making strategies.

---

## 🎯 Objective

Design an agent that can:

* Maintain low response latency
* Maximize request throughput
* Minimize GPU usage cost
* Adapt to changing traffic patterns

---

## ⚙️ Environment Design

### Observation (State)

The agent observes:

* Active GPU count
* Request queue length
* Incoming request rate
* Average response latency
* Batch size
* Cache load (memory usage)
* Spot GPU availability

---

### Action Space

The agent can:

* Scale GPUs up or down
* Adjust batch size
* Allocate between spot and regular GPUs

---

### Reward Function

The reward balances performance and cost:

* Positive reward for low latency and high throughput
* Penalty for high GPU usage cost
* Penalty for queue buildup
* Penalty for high cache load

Final scores are normalized between **0.0 and 1.0**.

---

## 📊 Tasks

| Level  | Description                                  |
| ------ | -------------------------------------------- |
| Easy   | Stable traffic with predictable load         |
| Medium | Variable traffic with moderate spikes        |
| Hard   | Sudden high-traffic bursts and system stress |

All tasks are:

* Deterministic
* Reproducible
* Evaluated consistently

---

## 🧪 Grading

Agents are evaluated based on:

* Average latency
* Throughput (requests served)
* Resource efficiency (GPU usage)

Scores are normalized between **0.0 – 1.0** for fair comparison.

---

## 🌐 API Endpoints

* `POST /reset?task=easy|medium|hard`
* `POST /step`
* `GET /state`
* `GET /health`

---

## 🛠️ Setup

### Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/llm-serving-autoscaler
cd llm-serving-autoscaler
```

### Run with Docker

```bash
docker build -t llm-env .
docker run -p 8000:8000 llm-env
```

### Check service

```bash
curl http://localhost:8000/health
```

---

## 🚀 Quick Submission Guide

Follow these 6 steps to complete your hackathon entry:

1. **Scaffold:** (Completed) Project structure is ready.
2. **Build:** (Completed) Environment, Grader, and Baseline are implemented in `src/`.
3. **Test Locally:** Run the server to verify:
   ```bash
   uv run server
   ```
4. **Validate:** Confirm spec compliance:
   ```bash
   openenv validate .
   ```
5. **Deploy:** Push your environment to Hugging Face:
   ```bash
   openenv push --repo-id your-huggingface-username/llm-serving-autoscaler
   ```
6. **Submit:** Copy your Hugging Face Space URL and paste it into the official hackathon submission portal.

---

## 📁 Project Structure

```
llm-serving-autoscaler/
├── src/
│   ├── models.py
│   ├── environment.py
│   ├── grader.py
│   └── baseline.py
├── app.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── README.md
└── assets/
```

---

### 📈 Baseline Scores
The following scores were achieved using the provided `src/baseline.py` agent across all three tasks:

| Task   | Score (%) | Performance Description |
| ------ | --------- | ------------------------ |
| **Easy**   | 95.2%      | Perfect scaling; zero queue backlog. |
| **Medium** | 69.6%      | Handles sinusoidal waves; slight latency during peaks. |
| **Hard**   | 44.1%      | Struggles with massive burst traffic; builds queue backlog. |
| **Overall**| **69.6%**    | Strong starting baseline performance. |

---

## 🎨 Visualization
The environment provides two ways to visualize:

1. **Terminal Dashboard**: Built-in ASCII dashboard (run `python dashboard.py`).
2. **Streamlit UI**: Full professional observability dashboard (run `uv run streamlit run app_visual.py`).

These help interpret how the AI agent manages resource allocation in real-time.

---

## ✅ OpenEnv Compliance

* Implements `reset()`, `step()`, `state()`
* Uses typed models for observations and actions
* Provides deterministic graders
* Supports reproducible evaluation

---

## 📌 Notes

This is a simplified simulation of large-scale AI serving systems designed for experimentation and benchmarking.

---

## 🏁 Summary

This project provides a structured environment to study:

* Resource allocation under dynamic demand
* Trade-offs between latency, throughput, and cost
* Adaptive decision-making in real-time systems

---

Built for OpenEnv Hackathon 2026
