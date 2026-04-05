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

## 📈 Baseline

A simple deterministic baseline is included:

* rule-based scaling strategy
* reproducible performance

This provides a reference score for evaluation.

---

## 🎨 Visualization

The environment can visualize:

* Request queue behavior
* GPU scaling decisions
* Latency trends

These help interpret how the agent adapts to changing traffic.

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
